"""
PDF Processing Module

This module provides PDF processing capabilities including:
- Text extraction using multiple backends (pymupdf4llm, docling, marker, markitdown)
- Image extraction and processing
- Intelligent text chunking with two strategies:
  - Fixed size chunking: Splits text into chunks of specified word count with overlap
  - Document structure chunking: Respects markdown hierarchy and document structure
- Embedding generation for both text and images
- Vector storage and retrieval
"""

import base64
import hashlib
import io
import logging
import os
import re
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import chromadb
import pymupdf4llm
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from PIL import Image

from models import ModelProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtractionBackend(Enum):
    """Supported PDF extraction backends"""

    PYMUPDF = "pymupdf"
    DOCLING = "docling"


class ChunkingStrategy(Enum):
    """Text chunking strategies"""

    FIXED_SIZE = "fixed_size"
    DOCUMENT_STRUCTURE = "document_structure"


@dataclass
class TextChunk:
    """Container for a text chunk with metadata"""

    content: str
    chunk_index: int = 0


@dataclass
class ImageChunk:
    """Container for an image chunk with metadata"""

    image_data: bytes
    caption: str
    description: str = ""
    image_index: int = 0
    image_format: str = "PNG"


@dataclass
class ImageElement:
    """Container for image and metadata extracted from a file"""

    image: Image.Image
    caption: str
    page_number: int


@dataclass
class ProcessedDocument:
    """Container for processed document with text and images"""

    text_chunks: list[TextChunk]
    image_chunks: list[ImageChunk]
    raw_text: str
    metadata: dict[str, Any]
    document_hash: str


def _extract_docling(pdf_path) -> tuple[str, list[ImageElement]]:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    # pipeline_options.do_formula_enrichment = True  # Disabled to avoid memory issues
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    logger.info("Docling backend initialized")
    result = converter.convert(str(pdf_path))
    text = result.document.export_to_markdown()
    images = _extract_images_docling(result)
    return text, images


def _extract_images_docling(docling_result: ConversionResult) -> list[ImageElement]:
    """Extract images using docling with captions."""
    images_and_text = []
    doc = docling_result.document

    for picture in doc.pictures:
        try:
            pil_image = picture.get_image(doc)
            if pil_image is None:
                logger.warning(
                    f"No image data available for picture {picture.get_ref()}"
                )
                continue

            caption_text = picture.caption_text(doc)
            page_number = 1
            if hasattr(picture, "prov") and picture.prov:
                page_number = picture.prov[0].page_no

            images_and_text.append(
                ImageElement(pil_image, caption_text.strip(), page_number)
            )

            logger.debug(
                f"Extracted image from page {page_number} with {len(caption_text)} chars of caption"
            )

        except Exception as e:
            logger.warning(f"Failed to extract image: {e}")

    return images_and_text


def _extract_images_pymupdf(doc_source) -> list[ImageElement]:
    """Extract images using PyMuPDF"""
    import fitz  # PyMuPDF

    images_and_text = []
    doc = fitz.open(str(doc_source))

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # Get text blocks with positioning information
        text_blocks = page.get_text("dict")
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)

                img_rect = (
                    page.get_image_rects(xref)[0]
                    if page.get_image_rects(xref)
                    else None
                )

                caption_text = ""
                if img_rect:
                    caption_blocks = []

                    for block in text_blocks["blocks"]:
                        if "lines" in block:
                            block_rect = fitz.Rect(block["bbox"])

                            if (
                                block_rect.y0 >= img_rect.y1
                                and block_rect.y0 - img_rect.y1 < 50
                            ):
                                caption_blocks.append(block)

                    caption_blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
                    for block in caption_blocks[:2]:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                caption_text += span["text"] + " "

                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    img_data = pix.tobytes("png")
                    images_and_text.append(
                        ImageElement(
                            Image.open(io.BytesIO(img_data)),
                            caption_text.strip(),
                            page_num + 1,
                        )
                    )
                    logger.debug(
                        f"Extracted image from page {page_num + 1} with {len(caption_text)} chars of caption"
                    )
                pix = None
            except Exception as e:
                logger.warning(
                    f"Failed to extract image {img_index} from page {page_num}: {e}"
                )
    doc.close()
    return images_and_text


def _extract_pymupdf(doc_source):
    text = pymupdf4llm.to_markdown(str(doc_source))
    images = _extract_images_pymupdf(doc_source)
    return text, images


class PDFProcessor:
    """Main PDF processing class"""

    def __init__(
        self,
        image_captioning_model: ModelProvider,
        extraction_backend: ExtractionBackend = ExtractionBackend.DOCLING,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.DOCUMENT_STRUCTURE,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        extract_images: bool = True,
    ):
        """
        Initialize PDF processor

        Args:
            image_captioning_model: Model for image captioning. Should be a multimodal model.
            extraction_backend: Backend for PDF extraction
            chunking_strategy: Strategy for text chunking
            chunk_size: Size of text chunks (for fixed_size strategy)
            chunk_overlap: Overlap between chunks
            extract_images: Whether to extract images
        """
        self.image_captioning_model = image_captioning_model
        self.extraction_backend = extraction_backend
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_images = extract_images

    def extract_content(self, pdf_path: str | Path) -> tuple[str, list[ImageElement]]:
        """
        Extract text and images from PDF using selected backend

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (extracted_text, list_of_images)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(
            f"Extracting text from {pdf_path} using {self.extraction_backend.value}"
        )

        images = []

        if self.extraction_backend == ExtractionBackend.PYMUPDF:
            text, images = _extract_pymupdf(pdf_path)

        elif self.extraction_backend == ExtractionBackend.DOCLING:
            text, images = _extract_docling(pdf_path)

        else:
            raise ValueError(
                f"Unsupported extraction backend: {self.extraction_backend}"
            )

        logger.info(f"Extracted {len(text)} characters and {len(images)} images")
        return text, images

    def chunk_text(self, text: str) -> list[TextChunk]:
        """
        Chunk text according to selected strategy

        Args:
            text: Text to chunk

        Returns:
            List of TextChunk objects
        """
        logger.info(f"Chunking text using {self.chunking_strategy.value} strategy")

        if self.chunking_strategy == ChunkingStrategy.FIXED_SIZE:
            return self._chunk_by_fixed_size(text)
        elif self.chunking_strategy == ChunkingStrategy.DOCUMENT_STRUCTURE:
            return self._chunk_by_document_structure(text)
        else:
            raise ValueError(f"Unsupported chunking strategy: {self.chunking_strategy}")

    def _chunk_by_document_structure(self, text: str) -> list[TextChunk]:
        """
        Chunk text by document structure (markdown hierarchy)

        This method recursively chunks content based on markdown structure,
        respecting headers, paragraphs, and other structural elements while
        maintaining the maximum chunk size limit.
        """
        chunks = []

        sections = self._split_by_headers(text)

        for section in sections:
            if len(section["content"].split()) <= self.chunk_size:
                chunks.append(
                    TextChunk(
                        content=section["content"].strip(),
                        chunk_index=len(chunks),
                    )
                )
            else:
                sub_chunks = self._recursive_structure_chunk(
                    section["content"], section["level"]
                )
                chunks.extend(sub_chunks)

        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        return chunks

    def _split_by_headers(self, text: str) -> list[dict[str, Any]]:
        """Split text into sections based on markdown headers"""
        lines = text.split("\n")
        sections = []
        current_section = {"content": "", "level": 0}

        for line in lines:
            header_match = re.match(r"^(#{1,6})\s+(.+)", line)
            if header_match:
                if current_section["content"].strip():
                    sections.append(current_section)

                level = len(header_match.group(1))
                current_section = {"content": line + "\n", "level": level}
            else:
                current_section["content"] += line + "\n"

        if current_section["content"].strip():
            sections.append(current_section)

        return sections

    def _recursive_structure_chunk(
        self, text: str, header_level: int
    ) -> list[TextChunk]:
        """Recursively chunk text while respecting structure"""
        chunks = []

        # Try to split by next level headers first
        next_level_pattern = f"^#{{{header_level + 1},{header_level + 3}}}\\s+"
        parts = re.split(f"({next_level_pattern})", text, flags=re.MULTILINE)

        if len(parts) > 1:
            # Found sub-headers, chunk by those
            current_chunk = ""
            for part in parts:
                if re.match(next_level_pattern, part, re.MULTILINE):
                    current_chunk += part
                else:
                    current_chunk += part
                    if len(current_chunk.split()) >= self.chunk_size:
                        if current_chunk.strip():
                            chunks.append(
                                TextChunk(
                                    content=current_chunk.strip(),
                                    chunk_index=len(chunks),
                                )
                            )
                        current_chunk = ""

            # Add remaining content
            if current_chunk.strip():
                chunks.append(
                    TextChunk(
                        content=current_chunk.strip(),
                        chunk_index=len(chunks),
                    )
                )
        else:
            # No sub-headers, split by paragraphs
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            current_chunk = ""

            for paragraph in paragraphs:
                # Check if single paragraph is too large
                if len(paragraph.split()) > self.chunk_size:
                    # Save current chunk if it exists
                    if current_chunk.strip():
                        chunks.append(
                            TextChunk(
                                content=current_chunk.strip(),
                                chunk_index=len(chunks),
                            )
                        )
                        current_chunk = ""

                    # Split large paragraph by sentences
                    sentences = re.split(r"[.!?]+", paragraph)
                    sentences = [s.strip() for s in sentences if s.strip()]

                    for sentence in sentences:
                        test_chunk = (
                            current_chunk + ". " + sentence
                            if current_chunk
                            else sentence
                        )
                        if len(test_chunk.split()) > self.chunk_size and current_chunk:
                            chunks.append(
                                TextChunk(
                                    content=current_chunk.strip(),
                                    chunk_index=len(chunks),
                                )
                            )
                            current_chunk = sentence
                        else:
                            current_chunk = test_chunk
                else:
                    # Check if adding this paragraph would exceed limit
                    test_chunk = (
                        current_chunk + "\n\n" + paragraph
                        if current_chunk
                        else paragraph
                    )
                    if len(test_chunk.split()) > self.chunk_size and current_chunk:
                        # Save current chunk and start new one
                        chunks.append(
                            TextChunk(
                                content=current_chunk.strip(),
                                chunk_index=len(chunks),
                            )
                        )
                        current_chunk = paragraph
                    else:
                        current_chunk = test_chunk

            # Add final chunk
            if current_chunk.strip():
                chunks.append(
                    TextChunk(
                        content=current_chunk.strip(),
                        chunk_index=len(chunks),
                    )
                )

        return chunks

    def _chunk_by_fixed_size(self, text: str) -> list[TextChunk]:
        """Chunk text by fixed size with overlap"""
        chunks = []
        words = text.split()

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)

            chunks.append(
                TextChunk(
                    content=chunk_text,
                    chunk_index=len(chunks),
                )
            )

        return chunks

    def caption_images(self, image_chunks: list[ImageChunk]) -> list[ImageChunk]:
        """Generate captions for image chunks"""
        logger.info(f"Generating captions for {len(image_chunks)} image chunks")

        for i, chunk in enumerate(image_chunks):
            try:
                # Convert image to base64 for the model
                image_b64 = base64.b64encode(chunk.image_data).decode("utf-8")

                # Generate caption using the multimodal model
                caption_prompt = """Describe this image in three sentences.
                It could be a technical diagram, a chart, a graph, or an illustration from a scientific or academic paper.
                Be concise and accurate.
                Do not write any introduction like "Here is the description", write _only_ the description.
                """
                description = self.image_captioning_model.generate_with_images(
                    caption_prompt, [image_b64]
                )
                chunk.description = description
                logger.info(
                    f"Generated caption for image {i + 1}/{len(image_chunks)}: {description[:100]}..."
                )

            except Exception as e:
                logger.error(f"Failed to process image {i}: {e}")
                chunk.description = None

        return image_chunks

    def process_pdf(self, pdf_path: str | Path) -> ProcessedDocument:
        """
        Process a PDF file end-to-end

        Args:
            pdf_path: Path to PDF file

        Returns:
            ProcessedDocument with text and image chunks
        """
        logger.info(f"Processing PDF: {pdf_path}")

        hash_obj = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)
        document_hash = hash_obj.hexdigest()[:16]

        raw_text, images = self.extract_content(pdf_path)
        text_chunks = self.chunk_text(raw_text)

        image_chunks = []
        if self.extract_images and images:
            for i, img_element in enumerate(images):
                img_bytes = io.BytesIO()
                img_element.image.save(img_bytes, format="PNG")
                img_bytes = img_bytes.getvalue()  # TODO wasteful saving and loading

                image_chunks.append(
                    ImageChunk(
                        image_data=img_bytes,
                        caption=img_element.caption,
                        image_index=i,
                        image_format="PNG",
                    )
                )

            if image_chunks:
                image_chunks = self.caption_images(image_chunks)

        processed_doc = ProcessedDocument(
            text_chunks=text_chunks,
            image_chunks=image_chunks,
            raw_text=raw_text,
            metadata={
                "extraction_backend": self.extraction_backend.value,
                "chunking_strategy": self.chunking_strategy.value,
                "num_text_chunks": len(text_chunks),
                "num_image_chunks": len(image_chunks),
                "image_captioning_model": self.image_captioning_model,
            },
            document_hash=document_hash,
        )

        logger.info(
            f"PDF processing complete: {len(text_chunks)} text chunks, {len(image_chunks)} image chunks"
        )
        return processed_doc


class VectorStore:
    """Vector store for similarity search using ChromaDB"""

    def __init__(
        self, embedding_function: chromadb.EmbeddingFunction, persist_directory: str
    ):
        self.persist_directory = persist_directory
        self._volatile_image_store: dict[str, bytes] = {}
        self.embedding_function = embedding_function

        settings = Settings(anonymized_telemetry=False)
        self.client = chromadb.PersistentClient(
            path=persist_directory, settings=settings
        )
        self.image_directory = os.path.join(persist_directory, "images")
        os.makedirs(self.image_directory, exist_ok=True)

        self._collections: dict[str, chromadb.Collection] = {}

    def _get_collection(self, document_hash: str) -> chromadb.Collection:
        """Get or create collection for a specific document"""
        if document_hash not in self._collections:
            collection_name = f"doc_{document_hash}"
            self._collections[document_hash] = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                configuration={"hnsw": {"space": "cosine"}},
            )
        return self._collections[document_hash]

    def document_exists(self, document_hash: str) -> bool:
        """Check if a document with the given hash already exists"""
        try:
            collection_name = f"doc_{document_hash}"
            collection = self.client.get_collection(name=collection_name)
            count = collection.count()
            return count > 0
        except NotFoundError:
            return False

    def add_document(self, processed_doc: ProcessedDocument) -> None:
        """Add a processed document to the vector store"""
        if self.document_exists(processed_doc.document_hash):
            logger.info(
                f"Document with hash {processed_doc.document_hash} already exists, skipping..."
            )
            return

        collection = self._get_collection(processed_doc.document_hash)

        # Add text chunks
        text_ids = []
        text_contents = []
        text_metadatas = []

        for chunk in processed_doc.text_chunks:
            chunk_id = str(uuid.uuid4())
            text_ids.append(chunk_id)
            text_contents.append(chunk.content)
            metadata = asdict(chunk)
            metadata.pop("content")
            text_metadatas.append(
                {
                    **metadata,
                    "type": "text",
                    "document_hash": processed_doc.document_hash,
                }
            )

        if text_ids:
            collection.add(
                ids=text_ids, documents=text_contents, metadatas=text_metadatas
            )

        # Add image chunks
        image_ids = []
        image_contents = []
        image_metadatas = []

        for chunk in processed_doc.image_chunks:
            chunk_id = str(uuid.uuid4())
            emb_input = f"Image description: {chunk.description}\nImage caption: {chunk.caption}"
            image_ids.append(chunk_id)
            image_contents.append(emb_input)
            metadata = asdict(chunk)
            image_data = metadata.pop("image_data")

            image_filename = f"{chunk_id}.{chunk.image_format.lower()}"
            image_path = os.path.join(self.image_directory, image_filename)

            with open(image_path, "wb") as f:
                f.write(image_data)

            metadata["image_path"] = image_path
            image_metadatas.append(
                {
                    **metadata,
                    "type": "image",
                    "document_hash": processed_doc.document_hash,
                }
            )

        if image_ids:
            collection.add(
                ids=image_ids,
                documents=image_contents,
                metadatas=image_metadatas,
            )

    def _search_collection(
        self,
        collection: chromadb.Collection,
        query: str,
        where: chromadb.Where | None,
        top_k: int = 5,
    ):
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=where,
        )
        return results

    def search_document(
        self,
        document_hash: str,
        query: str,
        where: chromadb.Where | None = None,
        top_k: int = 5,
    ) -> chromadb.QueryResult:
        """Search within a specific document"""
        if not self.document_exists(document_hash):
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "embeddings": None,
                "uris": None,
                "data": None,
                "included": ["documents", "metadatas", "distances"],
            }

        collection = self._get_collection(document_hash)
        return self._search_collection(collection, query, where, top_k)

    def _retrieve_chunks(
        self, results: chromadb.QueryResult
    ) -> list[tuple[TextChunk | ImageChunk, float]]:
        if not results["ids"] or len(results["ids"]) == 0:
            return []

        if results["distances"] is None or results["metadatas"] is None:
            return []

        out = []
        for i, distance in enumerate(results["distances"][0]):
            metadata = results["metadatas"][0][i]
            if metadata is None:
                continue

            metadata_copy = dict(metadata)
            data_type = metadata_copy.pop("type", None)

            if data_type == "text" and results["documents"] is not None:
                chunk_index = metadata_copy.get("chunk_index", 0)
                if not isinstance(chunk_index, int):
                    chunk_index = int(chunk_index) if chunk_index is not None else 0

                chunk = TextChunk(
                    content=results["documents"][0][i],
                    chunk_index=chunk_index,
                )
                out.append((chunk, 1.0 - distance))
            elif data_type == "image":
                image_path = str(metadata_copy.pop("image_path", ""))
                if image_path and os.path.exists(image_path):
                    with open(image_path, "rb") as f:
                        image_data = f.read()

                    image_index = metadata_copy.get("image_index", 0)
                    if not isinstance(image_index, int):
                        image_index = int(image_index) if image_index is not None else 0

                    image_format = metadata_copy.get("image_format", "PNG")
                    if not isinstance(image_format, str):
                        image_format = (
                            str(image_format) if image_format is not None else "PNG"
                        )

                    chunk = ImageChunk(
                        image_data=image_data,
                        caption=str(metadata_copy.get("caption", "")),
                        description=str(metadata_copy.get("description", "")),
                        image_index=image_index,
                        image_format=image_format,
                    )
                    out.append((chunk, 1.0 - distance))

        return out

    def search_text(self, query: str, document_hash: str, top_k: int = 5):
        """Search for similar text chunks"""
        results = self.search_document(document_hash, query, {"type": "text"}, top_k)
        return self._retrieve_chunks(results)

    def search_image(self, query: str, document_hash: str, top_k: int = 5):
        """Search for similar image chunks"""
        results = self.search_document(document_hash, query, {"type": "image"}, top_k)
        return self._retrieve_chunks(results)

    def search_combined(self, query: str, document_hash: str, top_k: int = 5):
        """Search both text and images, ranking them together"""
        results = self.search_document(document_hash, query, None, top_k)
        return self._retrieve_chunks(results)

    def list_documents(self) -> list[str]:
        """List all document hashes stored in the vector store"""
        collections = self.client.list_collections()
        document_hashes = []
        for collection_info in collections:
            if collection_info.name.startswith("doc_"):
                document_hash = collection_info.name[4:]
                document_hashes.append(document_hash)
        return document_hashes

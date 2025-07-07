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
import io
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import ollama
from pydantic.deprecated.config import Extra
import pymupdf4llm
from docling.document_converter import DocumentConverter
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from markitdown import MarkItDown
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtractionBackend(Enum):
    """Supported PDF extraction backends"""

    PYMUPDF = "pymupdf"
    DOCLING = "docling"
    MARKER = "marker"
    MARKITDOWN = "markitdown"


class ChunkingStrategy(Enum):
    """Text chunking strategies"""

    FIXED_SIZE = "fixed_size"
    DOCUMENT_STRUCTURE = "document_structure"


@dataclass
class TextChunk:
    """Container for a text chunk with metadata"""

    content: str
    page_number: Optional[int] = None
    chunk_index: int = 0
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class ImageChunk:
    """Container for an image chunk with metadata"""

    image_data: bytes
    description: Optional[str] = None
    page_number: Optional[int] = None
    image_index: int = 0
    image_format: str = "PNG"
    embedding: Optional[np.ndarray] = None


@dataclass
class ProcessedDocument:
    """Container for processed document with text and images"""

    text_chunks: List[TextChunk]
    image_chunks: List[ImageChunk]
    raw_text: str
    metadata: Dict[str, Any]


class PDFProcessor:
    """Main PDF processing class"""

    def __init__(
        self,
        text_embedding_model: str = "nomic-embed-text",
        image_captioning_model: str = "qwen2.5vl:3b",
        extraction_backend: ExtractionBackend = ExtractionBackend.DOCLING,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.DOCUMENT_STRUCTURE,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        extract_images: bool = True,
    ):
        """
        Initialize PDF processor

        Args:
            text_embedding_model: Model for text embeddings
            image_captioning_model: Model for image captioning. Should be a multimodal model.
            extraction_backend: Backend for PDF extraction
            chunking_strategy: Strategy for text chunking
            chunk_size: Size of text chunks (for fixed_size strategy)
            chunk_overlap: Overlap between chunks
            extract_images: Whether to extract images
        """
        self.text_embedding_model = text_embedding_model
        self.image_captioning_model = image_captioning_model
        self.extraction_backend = extraction_backend
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_images = extract_images

        self._init_backends()

    def _init_backends(self):
        """Initialize PDF extraction backends"""
        self.backends = {}

        if self.extraction_backend == ExtractionBackend.DOCLING:
            self.backends[ExtractionBackend.DOCLING] = DocumentConverter()
            logger.info("Docling backend initialized")
        elif self.extraction_backend == ExtractionBackend.MARKER:
            self.backends[ExtractionBackend.MARKER] = PdfConverter(
                artifact_dict=create_model_dict()
            )
            logger.info("Marker backend initialized")
        elif self.extraction_backend == ExtractionBackend.MARKITDOWN:
            self.backends[ExtractionBackend.MARKITDOWN] = MarkItDown()
            logger.info("MarkItDown backend initialized")

    def extract_content(
        self, pdf_path: Union[str, Path]
    ) -> Tuple[str, List[Image.Image]]:
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
            text = pymupdf4llm.to_markdown(str(pdf_path))
            if self.extract_images:
                images = self._extract_images_pymupdf(pdf_path)

        elif self.extraction_backend == ExtractionBackend.DOCLING:
            converter = self.backends[ExtractionBackend.DOCLING]
            result = converter.convert(str(pdf_path))
            text = result.document.export_to_markdown()
            if self.extract_images:
                images = self._extract_images_docling(result)

        elif self.extraction_backend == ExtractionBackend.MARKER:
            converter = self.backends[ExtractionBackend.MARKER]
            rendered = converter(str(pdf_path))
            text, _, marker_images = text_from_rendered(rendered)
            if self.extract_images and marker_images:
                images = marker_images

        elif self.extraction_backend == ExtractionBackend.MARKITDOWN:
            converter = self.backends[ExtractionBackend.MARKITDOWN]
            result = converter.convert(str(pdf_path))
            text = result.markdown
            if self.extract_images:
                images = self._extract_images_markitdown(pdf_path)

        else:
            raise ValueError(
                f"Unsupported extraction backend: {self.extraction_backend}"
            )

        logger.info(f"Extracted {len(text)} characters and {len(images)} images")
        return text, images

    def _extract_images_pymupdf(self, pdf_path: Path) -> List[Image.Image]:
        """Extract images using PyMuPDF"""
        import fitz  # PyMuPDF

        images = []
        doc = fitz.open(str(pdf_path))

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)

                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        images.append(Image.open(io.BytesIO(img_data)))
                    pix = None
                except Exception as e:
                    logger.warning(
                        f"Failed to extract image {img_index} from page {page_num}: {e}"
                    )

        doc.close()
        return images

    def _extract_images_docling(self, docling_result) -> List[Image.Image]:
        """Extract images from Docling result"""
        images = []
        # TODO: Implement image extraction from Docling result
        return images

    def _extract_images_markitdown(self, pdf_path: Path) -> List[Image.Image]:
        """Extract images using alternative method for MarkItDown"""
        # MarkItDown doesn't directly provide image extraction
        # Fall back to PyMuPDF for image extraction
        return self._extract_images_pymupdf(pdf_path)

    def chunk_text(self, text: str) -> List[TextChunk]:
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

    def _chunk_by_document_structure(self, text: str) -> List[TextChunk]:
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
                        start_char=None,
                        end_char=None,
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

    def _split_by_headers(self, text: str) -> List[Dict[str, Any]]:
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
    ) -> List[TextChunk]:
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
                                    start_char=None,
                                    end_char=None,
                                )
                            )
                        current_chunk = ""

            # Add remaining content
            if current_chunk.strip():
                chunks.append(
                    TextChunk(
                        content=current_chunk.strip(),
                        chunk_index=len(chunks),
                        start_char=None,
                        end_char=None,
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
                                start_char=None,
                                end_char=None,
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
                                    start_char=None,
                                    end_char=None,
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
                                start_char=None,
                                end_char=None,
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
                        start_char=None,
                        end_char=None,
                    )
                )

        return chunks

    def _chunk_by_fixed_size(self, text: str) -> List[TextChunk]:
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
                    start_char=None,
                    end_char=None,
                )
            )

        return chunks

    def generate_text_embeddings(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Generate embeddings for text chunks

        Args:
            chunks: List of TextChunk objects

        Returns:
            List of TextChunk objects with embeddings
        """
        logger.info(f"Generating embeddings for {len(chunks)} text chunks")

        for i, chunk in enumerate(chunks):
            try:
                embedding = ollama.embed(
                    model=self.text_embedding_model, input=chunk.content
                )["embeddings"][0]
                chunk.embedding = np.array(embedding)

                if (i + 1) % 10 == 0:
                    logger.info(
                        f"Generated embeddings for {i + 1}/{len(chunks)} chunks"
                    )

            except Exception as e:
                logger.error(f"Failed to generate embedding for chunk {i}: {e}")
                chunk.embedding = None

        return chunks

    def caption_and_embed_images(
        self, image_chunks: List[ImageChunk]
    ) -> List[ImageChunk]:
        """
        Generate captions and embeddings for image chunks.

        Args:
            image_chunks: List of ImageChunk objects

        Returns:
            List of ImageChunk objects with captions and embeddings
        """
        logger.info(
            f"Generating captions and embeddings for {len(image_chunks)} image chunks"
        )

        for i, chunk in enumerate(image_chunks):
            try:
                # Convert image to base64 for the model
                image_b64 = base64.b64encode(chunk.image_data).decode("utf-8")

                # Generate caption using the multimodal model
                caption_prompt = """Describe this image in detail.
                It could be a technical diagram, a chart, a graph, or an illustration from a scientific or academic paper.
                Focus on the key information presented.
                Do not write any introduction like "Here is the description", write _only_ the description.
                """
                response = ollama.generate(
                    model=self.image_captioning_model,
                    prompt=caption_prompt,
                    images=[image_b64],
                )

                chunk.description = response["response"].strip()
                logger.info(
                    f"Generated caption for image {i + 1}/{len(image_chunks)}: {chunk.description[:100]}..."
                )

                # Generate embedding from the caption using the text embedding model
                embedding = ollama.embed(
                    model=self.text_embedding_model, input=chunk.description
                )["embeddings"][0]
                chunk.embedding = np.array(embedding)

                logger.info(
                    f"Generated embedding for image caption {i + 1}/{len(image_chunks)}"
                )

            except Exception as e:
                logger.error(f"Failed to process image {i}: {e}")
                chunk.description = None
                chunk.embedding = None

        return image_chunks

    def process_pdf(self, pdf_path: Union[str, Path]) -> ProcessedDocument:
        """
        Process a PDF file end-to-end

        Args:
            pdf_path: Path to PDF file

        Returns:
            ProcessedDocument with text and image chunks
        """
        logger.info(f"Processing PDF: {pdf_path}")

        raw_text, images = self.extract_content(pdf_path)
        text_chunks = self.chunk_text(raw_text)
        text_chunks = self.generate_text_embeddings(text_chunks)

        image_chunks = []
        if self.extract_images and images:
            for i, img in enumerate(images):
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                img_bytes = img_bytes.getvalue()

                image_chunks.append(
                    ImageChunk(image_data=img_bytes, image_index=i, image_format="PNG")
                )

            if image_chunks:
                image_chunks = self.caption_and_embed_images(image_chunks)

        processed_doc = ProcessedDocument(
            text_chunks=text_chunks,
            image_chunks=image_chunks,
            raw_text=raw_text,
            metadata={
                "extraction_backend": self.extraction_backend.value,
                "chunking_strategy": self.chunking_strategy.value,
                "num_text_chunks": len(text_chunks),
                "num_image_chunks": len(image_chunks),
                "text_embedding_model": self.text_embedding_model,
                "image_captioning_model": self.image_captioning_model,
            },
        )

        logger.info(
            f"PDF processing complete: {len(text_chunks)} text chunks, {len(image_chunks)} image chunks"
        )
        return processed_doc


class VectorStore:
    """Simple vector store for similarity search"""

    def __init__(self):
        self.text_chunks: List[TextChunk] = []
        self.image_chunks: List[ImageChunk] = []
        self.text_embeddings: Optional[np.ndarray] = None
        self.image_embeddings: Optional[np.ndarray] = None

    def add_document(self, processed_doc: ProcessedDocument):
        """Add a processed document to the vector store"""
        self.text_chunks.extend(processed_doc.text_chunks)
        self.image_chunks.extend(processed_doc.image_chunks)

        # Rebuild embedding matrices
        self._rebuild_embeddings()

    def _rebuild_embeddings(self):
        """Rebuild embedding matrices from chunks"""
        text_embeddings = []
        for chunk in self.text_chunks:
            if chunk.embedding is not None:
                text_embeddings.append(chunk.embedding)

        if text_embeddings:
            self.text_embeddings = np.vstack(text_embeddings)

        image_embeddings = []
        for chunk in self.image_chunks:
            if chunk.embedding is not None:
                image_embeddings.append(chunk.embedding)

        if image_embeddings:
            self.image_embeddings = np.vstack(image_embeddings)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between vectors"""
        a = a / np.linalg.norm(a, axis=-1, keepdims=True)
        b = b / np.linalg.norm(b, axis=-1, keepdims=True)
        return a @ b.T

    def search_text(
        self, query: str, top_k: int = 5, embedding_model: str = "nomic-embed-text"
    ) -> List[Tuple[TextChunk, float]]:
        """Search for similar text chunks"""
        if self.text_embeddings is None or len(self.text_chunks) == 0:
            return []

        query_embedding = np.array(
            ollama.embed(model=embedding_model, input=query)["embeddings"][0]
        )

        similarities = self.cosine_similarity(
            query_embedding.reshape(1, -1), self.text_embeddings
        )[0]

        top_indices = np.argsort(-similarities)[:top_k]
        results = []

        for idx in top_indices:
            if idx < len(self.text_chunks):
                results.append((self.text_chunks[idx], similarities[idx]))

        return results

    def search_images(
        self, query: str, top_k: int = 5, embedding_model: str = "nomic-embed-text"
    ) -> List[Tuple[ImageChunk, float]]:
        """Search for similar image chunks based on their descriptions"""
        if self.image_embeddings is None or len(self.image_chunks) == 0:
            return []

        query_embedding = np.array(
            ollama.embed(model=embedding_model, input=query)["embeddings"][0]
        )

        similarities = self.cosine_similarity(
            query_embedding.reshape(1, -1), self.image_embeddings
        )[0]

        top_indices = np.argsort(-similarities)[:top_k]
        results = []

        for idx in top_indices:
            if idx < len(self.image_chunks):
                results.append((self.image_chunks[idx], similarities[idx]))

        return results

    def search_combined(
        self, query: str, top_k: int = 5, text_weight: float = 0.7
    ) -> List[Tuple[Union[TextChunk, ImageChunk], float, str]]:
        """Search both text and images with weighted results"""
        text_results = self.search_text(query, top_k)
        image_results = self.search_images(query, top_k)

        combined_results = []

        for chunk, score in text_results:
            combined_results.append((chunk, score * text_weight, "text"))

        for chunk, score in image_results:
            combined_results.append((chunk, score * (1 - text_weight), "image"))

        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:top_k]

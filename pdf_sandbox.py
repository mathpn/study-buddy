"""
Sandbox script to use the PDF processor module
"""

import logging
from pathlib import Path

from chromadb.utils.embedding_functions import (
    OllamaEmbeddingFunction,
    OpenAIEmbeddingFunction,
)

from models import OllamaModel, OpenAIModel
from pdf_processor import (
    ChunkingStrategy,
    ExtractionBackend,
    PDFProcessor,
    TextChunk,
    VectorStore,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Example usage of the PDF processor"""

    processor = PDFProcessor(
        # image_captioning_model=OpenAIModel("gpt-4.1-mini"),
        image_captioning_model=OllamaModel("granite3.2-vision"),
        extraction_backend=ExtractionBackend.DOCLING,
        chunking_strategy=ChunkingStrategy.DOCUMENT_STRUCTURE,
        chunk_size=500,
        chunk_overlap=50,
        extract_images=True,
    )

    pdf_path = "input_test_small.pdf"
    if not Path(pdf_path).exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return

    processed_doc = processor.process_pdf(pdf_path)
    print("\nProcessing Results:")
    print(f"- Extraction backend: {processed_doc.metadata['extraction_backend']}")
    print(f"- Chunking strategy: {processed_doc.metadata['chunking_strategy']}")
    print(f"- Text chunks: {processed_doc.metadata['num_text_chunks']}")
    print(f"- Image chunks: {processed_doc.metadata['num_image_chunks']}")
    print(f"- Raw text length: {len(processed_doc.raw_text)} characters")

    print("\nSample text chunks:")
    for i, chunk in enumerate(processed_doc.text_chunks[:3]):
        print(f"Chunk {i + 1}:")
        print(f"  Content: {chunk.content[:100]}...")
        print()

    if processed_doc.image_chunks:
        print("\nSample image chunks:")
        for i, chunk in enumerate(processed_doc.image_chunks[:2]):
            print(f"Image {i + 1}:")
            print(f"  Format: {chunk.image_format}")
            print(f"  Data size: {len(chunk.image_data)} bytes")
            print(
                f"  Description: {chunk.description[:100] if chunk.description else 'None'}..."
            )
            print(f"  Caption: {chunk.caption[:100] if chunk.caption else 'None'}")
            print()

    vector_store = VectorStore(
        OllamaEmbeddingFunction(model_name="nomic-embed-text"),
        persist_directory="./vector_store",
    )
    vector_store.add_document(processed_doc)

    print("\n" + "=" * 50)
    print("SEARCH EXAMPLES")
    print("=" * 50)

    query = "statistical regularity"
    print(f"\nSearching for: '{query}'")
    text_results = vector_store.search_text(query, top_k=3)

    print("\nTop text results:")
    for i, (chunk, score) in enumerate(text_results):
        print(f"{i + 1}. Score: {score:.3f}")
        print(f"   Content: {chunk.content[:150]}...")
        print()

    combined_query = "optimization pressure"
    print(f"\nCombined search for: '{combined_query}'")
    combined_results = vector_store.search_combined(combined_query, top_k=5)

    print("\nTop combined results:")
    for i, (chunk, score) in enumerate(combined_results):
        chunk_type = "text" if isinstance(chunk, TextChunk) else "image"
        print(f"{i + 1}. Type: {chunk_type}, Score: {score:.3f}")
        if isinstance(chunk, TextChunk):
            print(f"   Content: {chunk.content[:100]}...")
        else:
            print(
                f"   Description: {chunk.description[:100] if chunk.description else 'None'}..."
            )
        print()


def compare_extraction_backends():
    """Compare different extraction backends"""

    pdf_path = "input_test.pdf"
    if not Path(pdf_path).exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return

    backends = [
        ExtractionBackend.PYMUPDF,
        ExtractionBackend.DOCLING,
    ]

    print("Comparing extraction backends...")
    print("=" * 50)

    for backend in backends:
        try:
            processor = PDFProcessor(
                image_captioning_model=OllamaModel("qwen2.5-vl"),
                extraction_backend=backend,
                extract_images=False,  # Skip images for quick comparison
            )

            text, images = processor.extract_content(pdf_path)
            chunks = processor.chunk_text(text)

            print(f"\n{backend.value}:")
            print(f"  Text length: {len(text)} characters")
            print(f"  Number of chunks: {len(chunks)}")
            print(f"  Sample text: {text[:100]}...")

        except Exception as e:
            print(f"\n{backend.value}: FAILED - {e}")


if __name__ == "__main__":
    main()
    # print("\n" + "=" * 50)
    # compare_extraction_backends()

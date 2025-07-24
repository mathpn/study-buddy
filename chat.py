import logging
from pathlib import Path

from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from models import OllamaModel
from pdf_processor import (
    ChunkingStrategy,
    ExtractionBackend,
    ImageChunk,
    PDFProcessor,
    TextChunk,
    VectorStore,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_enhanced_query(
    current_query: str, conversation_history: list, max_history_tokens: int = 100
) -> str:
    """
    Enhance the current query with relevant context from conversation history.

    Args:
        current_query: The user's current question
        conversation_history: List of previous user/assistant messages
        max_history_tokens: Rough token limit for history context (approximate)

    Returns:
        Enhanced query string for better retrieval
    """
    if not conversation_history:
        return current_query

    recent_user_queries = []
    char_count = 0

    for message in reversed(conversation_history):
        if message["role"] == "user":
            query_text = message["content"]
            # Rough token estimation: ~4 chars per token
            if char_count + len(query_text) > max_history_tokens * 4:
                break
            recent_user_queries.insert(0, query_text)
            char_count += len(query_text)

    if recent_user_queries:
        context_queries = (
            recent_user_queries[-2:]
            if len(recent_user_queries) > 2
            else recent_user_queries
        )
        enhanced_query = f"{' '.join(context_queries)} {current_query}"
        return enhanced_query

    return current_query


def chat(vector_store: VectorStore, model: OllamaModel):
    """
    Handles the interactive RAG chat interface, maintaining conversation history.
    Retrieved chunks are placed in system messages and not persisted in history.

    Args:
        vector_store: An instance of VectorStore to retrieve context from.
        model: An instance of a language model to generate responses.
    """
    logger.info("\n--- RAG Chat Interface ---")

    base_system_prompt = (
        "You are a helpful assistant whose goal is to help the users in their study. "
        "Answer the user's questions based on the provided context. "
        "The context is retrieved from a document and may include text and image captions. "
        "If the information is not in the context, say you don't know. "
        "Keep your answers concise and relevant to the question."
    )

    conversation_history = []

    while True:
        try:
            query = input("\nYou: ")
            if not query.strip():
                continue

            enhanced_query = _build_enhanced_query(query, conversation_history)
            retrieved_chunks = vector_store.search_combined(enhanced_query, top_k=3)

            messages = []

            system_content = base_system_prompt
            if retrieved_chunks:
                context_parts = []
                for chunk, score in retrieved_chunks:
                    if isinstance(chunk, TextChunk):
                        context_parts.append(chunk.content)
                    elif isinstance(chunk, ImageChunk):
                        context_parts.append(f"Image Caption: {chunk.caption}")

                context = "\n---\n".join(context_parts)
                system_content += f"\n\nRelevant context from the document:\n{context}"
            else:
                logger.info("No relevant context found in the document for this query.")

            messages.append({"role": "system", "content": system_content})
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": query})

            response_content = model.chat(messages=messages)

            conversation_history.append({"role": "user", "content": query})
            conversation_history.append(
                {"role": "assistant", "content": response_content}
            )

            print(f"\nAssistant: {response_content}")

        except KeyboardInterrupt:
            print("\nExiting chat.")
            break


def main():
    vector_store_dir = Path("./vector_store")
    pdf_path = "input_test_small.pdf"

    if not vector_store_dir.exists() or not any(vector_store_dir.iterdir()):
        logger.info(f"Vector store not found. Processing PDF: {pdf_path}")
        if not Path(pdf_path).exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return

        processor = PDFProcessor(
            image_captioning_model=OllamaModel("granite3.2-vision"),
            extraction_backend=ExtractionBackend.DOCLING,
            chunking_strategy=ChunkingStrategy.DOCUMENT_STRUCTURE,
            chunk_size=500,
            chunk_overlap=50,
            extract_images=True,
        )
        processed_doc = processor.process_pdf(pdf_path)
        vector_store = VectorStore(
            OllamaEmbeddingFunction(model_name="nomic-embed-text"),
            persist_directory=str(vector_store_dir),
        )
        vector_store.add_document(processed_doc)
        logger.info("PDF processed and vector store created.")
    else:
        logger.info(f"Loading existing vector store from: {vector_store_dir}")

    vector_store = VectorStore(
        OllamaEmbeddingFunction(model_name="nomic-embed-text"),
        persist_directory=str(vector_store_dir),
    )
    chat_model = OllamaModel("qwen3:4b")
    chat(vector_store, chat_model)


if __name__ == "__main__":
    main()

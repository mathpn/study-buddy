import logging
from pathlib import Path
from typing import Set

from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from graph import KnowledgeGraph, build_knowledge_graph
from models import ModelProvider, OllamaModel, OpenAIModel
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


def build_enhanced_query(
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


def merge_knowledge_graphs(
    main_graph: KnowledgeGraph, new_graph: KnowledgeGraph
) -> None:
    """
    Merge a new knowledge graph into the main cumulative graph.
    Avoids duplicate nodes (by id) and relationships.

    Args:
        main_graph: The cumulative session graph to merge into
        new_graph: The new graph fragment to merge
    """
    existing_node_ids = {node.id for node in main_graph.nodes}
    existing_relationships = {
        (rel.source, rel.target, rel.relationship) for rel in main_graph.relationships
    }

    for node in new_graph.nodes:
        if node.id not in existing_node_ids:
            main_graph.nodes.append(node)
            existing_node_ids.add(node.id)

    for relationship in new_graph.relationships:
        rel_key = (
            relationship.source,
            relationship.target,
            relationship.relationship,
        )
        if rel_key not in existing_relationships:
            main_graph.relationships.append(relationship)
            existing_relationships.add(rel_key)


def display_knowledge_graph(graph: KnowledgeGraph) -> None:
    """
    Print a simple text representation of the knowledge graph.

    Args:
        graph: The knowledge graph to display
    """
    if not graph.nodes:
        print("\n--- Knowledge Graph ---")
        print("No relevant concepts found yet.")
        return

    print("\n--- Knowledge Graph ---")
    print(
        f"Concepts ({len(graph.nodes)} nodes, {len(graph.relationships)} relationships):"
    )

    # Display nodes grouped by type
    node_types = {}
    for node in graph.nodes:
        if node.type not in node_types:
            node_types[node.type] = []
        node_types[node.type].append(node)

    for node_type, nodes in node_types.items():
        print(f"\n{node_type.upper()}S:")
        for node in nodes:
            print(f"  • {node.label}: {node.description}")

    # Display relationships
    if graph.relationships:
        print("\nRELATIONSHIPS:")
        for rel in graph.relationships:
            # Find the actual node labels for better readability
            source_label = next(
                (n.label for n in graph.nodes if n.id == rel.source), rel.source
            )
            target_label = next(
                (n.label for n in graph.nodes if n.id == rel.target), rel.target
            )
            print(f"  • {source_label} --[{rel.relationship}]--> {target_label}")
            if rel.description:
                print(f"    ({rel.description})")

    print("--- End Knowledge Graph ---\n")


def generate_chunk_id(chunk) -> str:
    """
    Generate a unique identifier for a chunk.
    This is a simple hash based on content.
    """
    if isinstance(chunk, TextChunk):
        content = chunk.content
    elif isinstance(chunk, ImageChunk):
        content = chunk.caption
    else:
        content = str(chunk)

    return str(hash(content))


def chat(vector_store: VectorStore, model: ModelProvider):
    """
    Handles the interactive RAG chat interface, maintaining conversation history
    and a cumulative knowledge graph.

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

    session_graph = KnowledgeGraph(nodes=[], relationships=[])
    processed_chunks: Set[str] = set()

    while True:
        try:
            query = input("\nYou: ")
            if not query.strip():
                continue

            enhanced_query = build_enhanced_query(query, conversation_history)
            retrieved_chunks = vector_store.search_combined(enhanced_query, top_k=3)

            new_chunks_content = []
            for chunk, _ in retrieved_chunks:
                chunk_id = generate_chunk_id(chunk)

                if chunk_id not in processed_chunks:
                    if isinstance(chunk, TextChunk):
                        chunk_content = chunk.content
                    elif isinstance(chunk, ImageChunk):
                        chunk_content = f"Image Caption: {chunk.caption}"
                    else:
                        chunk_content = str(chunk)

                    new_chunks_content.append(chunk_content)
                    processed_chunks.add(chunk_id)

            if new_chunks_content:
                combined_new_content = "\n\n---\n\n".join(new_chunks_content)

                try:
                    new_graph = build_knowledge_graph(combined_new_content, model)
                    if new_graph:
                        merge_knowledge_graphs(session_graph, new_graph)
                        logger.info(
                            f"Added {len(new_graph.nodes)} new nodes and {len(new_graph.relationships)} new relationships to session graph"
                        )
                except Exception as e:
                    logger.warning(f"Failed to generate knowledge graph: {e}")

            display_knowledge_graph(session_graph)

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

        except (KeyboardInterrupt, EOFError):
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

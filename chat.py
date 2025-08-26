import logging
from typing import Set

from graph import KnowledgeGraph, build_knowledge_graph
from models import ModelProvider
from pdf_processor import (
    ImageChunk,
    TextChunk,
    VectorStore,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

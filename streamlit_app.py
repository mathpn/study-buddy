import logging
import os
import tempfile

import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from chat import (
    build_enhanced_query,
    generate_chunk_id,
    merge_knowledge_graphs,
)
from graph import KnowledgeGraph, build_knowledge_graph
from models import ModelProvider, OllamaModel, OpenAIModel
from pdf_processor import (
    ExtractionBackend,
    ImageChunk,
    PDFProcessor,
    TextChunk,
    VectorStore,
    hash_file,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state():
    """Initialize session state variables"""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "session_graph" not in st.session_state:
        st.session_state.session_graph = KnowledgeGraph(nodes=[], relationships=[])
    if "processed_chunks" not in st.session_state:
        st.session_state.processed_chunks = set()
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = ""


def create_graph_visualization(graph: KnowledgeGraph):
    """Create an interactive plotly visualization of the knowledge graph"""
    if not graph.nodes:
        return go.Figure().add_annotation(
            text="No knowledge graph data available yet",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="black"),
        )

    # Create networkx graph
    G = nx.Graph()

    # Add nodes
    node_colors = {
        "concept": "#FF6B6B",
        "method": "#4ECDC4",
        "entity": "#45B7D1",
        "metric": "#96CEB4",
        "application": "#FECA57",
    }

    for node in graph.nodes:
        G.add_node(
            node.id, label=node.label, type=node.type, description=node.description
        )

    # Add edges
    for rel in graph.relationships:
        G.add_edge(
            rel.source,
            rel.target,
            relationship=rel.relationship,
            description=rel.description,
        )

    # Generate layout
    pos = nx.spring_layout(G, k=3, iterations=50)

    # Create edge traces
    edge_x = []
    edge_y = []
    edge_info = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        # Get relationship info
        rel_data = next(
            (
                r
                for r in graph.relationships
                if (r.source == edge[0] and r.target == edge[1])
                or (r.source == edge[1] and r.target == edge[0])
            ),
            None,
        )
        if rel_data:
            edge_info.append(f"{rel_data.relationship}: {rel_data.description}")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=2, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_info = []
    node_colors_list = []

    for node_id in G.nodes():
        x, y = pos[node_id]
        node_x.append(x)
        node_y.append(y)

        node_data = next(n for n in graph.nodes if n.id == node_id)
        node_text.append(node_data.label)
        node_info.append(
            f"<b>{node_data.label}</b><br>Type: {node_data.type}<br>Description: {node_data.description}"
        )
        node_colors_list.append(node_colors.get(node_data.type, "#888"))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        textposition="middle center",
        hovertext=node_info,
        marker=dict(size=30, color=node_colors_list, line=dict(width=2, color="white")),
        textfont=dict(color="black"),  # Make node text font black for higher contrast
    )

    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text="Knowledge Graph", font=dict(size=16, color="black")),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Hover over nodes for details",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(color="black", size=12),
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
        ),
    )

    return fig


def process_pdf_file(uploaded_file, model_choice, extraction_backend):
    """Process uploaded PDF file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        st.session_state.processing_status = "Processing PDF..."

        if model_choice == "granite3.2-vision (Ollama)":
            image_model = OllamaModel("granite3.2-vision")
        elif model_choice == "gpt-4.1-mini (OpenAI)":
            image_model = OpenAIModel("gpt-4.1-mini")
        else:
            raise NotImplementedError(f"model not supported: {model_choice}")

        if st.session_state.vector_store is None:
            vector_store = VectorStore(
                OllamaEmbeddingFunction(model_name="nomic-embed-text"),
                persist_directory="./vector_store_streamlit",  # FIXME better path
            )
            st.session_state.vector_store = vector_store

        vector_store: VectorStore = st.session_state.vector_store
        document_hash = hash_file(tmp_file_path)
        if not st.session_state.vector_store.document_exists(document_hash):
            processor = PDFProcessor(
                image_captioning_model=image_model,
                extraction_backend=extraction_backend,
                extract_images=True,
            )

            st.session_state.processing_status = "Extracting content from PDF..."
            processed_doc = processor.process_pdf(tmp_file_path)

            st.session_state.processing_status = "Creating vector store..."
            vector_store.add_document(document_hash, processed_doc)
        else:
            logger.info("Document with hash %s already processed", document_hash)

        st.session_state.document_hash = document_hash
        st.session_state.pdf_processed = True
        st.session_state.processing_status = "PDF processed successfully!"

        os.unlink(tmp_file_path)
        return True

    except Exception as e:
        st.session_state.processing_status = f"Error processing PDF: {str(e)}"
        logger.error(f"Error processing PDF: {e}", exc_info=True)
        return False


def handle_chat_message(query: str, model: ModelProvider):
    """Handle a chat message and update the knowledge graph"""
    if not st.session_state.document_hash:
        return "Please upload and process a PDF file first."

    # TODO better way to have types?
    vector_store: VectorStore = st.session_state.vector_store
    try:
        enhanced_query = build_enhanced_query(
            query, st.session_state.conversation_history
        )

        retrieved_chunks = vector_store.search_combined(
            enhanced_query, st.session_state.document_hash, top_k=3
        )

        new_chunks_content = []
        for chunk, _ in retrieved_chunks:
            chunk_id = generate_chunk_id(chunk)

            if chunk_id not in st.session_state.processed_chunks:
                if isinstance(chunk, TextChunk):
                    chunk_content = chunk.content
                elif isinstance(chunk, ImageChunk):
                    chunk_content = f"Image Caption: {chunk.caption}"
                else:
                    chunk_content = str(chunk)

                new_chunks_content.append(chunk_content)
                st.session_state.processed_chunks.add(chunk_id)

        if new_chunks_content:
            combined_new_content = "\n\n---\n\n".join(new_chunks_content)

            try:
                new_graph = build_knowledge_graph(combined_new_content, model)
                if new_graph:
                    merge_knowledge_graphs(st.session_state.session_graph, new_graph)
                    logger.info(
                        f"Added {len(new_graph.nodes)} new nodes and {len(new_graph.relationships)} new relationships"
                    )
            except Exception as e:
                logger.warning(f"Failed to generate knowledge graph: {e}")

        base_system_prompt = (
            "You are a helpful assistant whose goal is to help the users in their study. "
            "Answer the user's questions based on the provided context. "
            "The context is retrieved from a document and may include text and image captions. "
            "If the information is not in the context, say you don't know. "
            "Keep your answers concise and relevant to the question."
        )

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

        messages.append({"role": "system", "content": system_content})
        messages.extend(st.session_state.conversation_history)
        messages.append({"role": "user", "content": query})

        logger.debug("current messages: %s", messages)
        response_content = model.chat(messages=messages)

        st.session_state.conversation_history.append({"role": "user", "content": query})
        st.session_state.conversation_history.append(
            {"role": "assistant", "content": response_content}
        )

        return response_content

    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return f"Sorry, I encountered an error: {str(e)}"


def main():
    """Main Streamlit app"""
    initialize_session_state()

    st.title("📚 Study Assistant")
    st.markdown(
        "Upload a PDF document and chat with an AI assistant enhanced with knowledge graphs!"
    )

    with st.sidebar:
        st.header("📁 Document Upload")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to analyze and chat about",
        )

        st.header("⚙️ Processing Settings")

        model_choice = st.selectbox(
            "Image Captioning Model",
            ["gpt-4.1-mini (OpenAI)", "granite3.2-vision (Ollama)"],
            help="Model used for describing images in the PDF",
        )

        chat_model_choice = st.selectbox(
            "Chat Model",
            [
                "gpt-4.1 (OpenAI)",
                "gpt-4.1-mini (OpenAI)",
                "o4-mini (OpenAI)",
                "qwen3 (Ollama)",
                "gemma3 (Ollama)",
            ],
            help="Model used for answering questions",
        )

        extraction_backend = st.selectbox(
            "Extraction Backend",
            [ExtractionBackend.DOCLING, ExtractionBackend.PYMUPDF],
            format_func=lambda x: x.value.title(),
            help="Method for extracting content from PDF",
        )

        if uploaded_file is not None and not st.session_state.pdf_processed:
            if st.button("🚀 Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    success = process_pdf_file(
                        uploaded_file,
                        model_choice,
                        extraction_backend,
                    )
                    if success:
                        st.success("PDF processed successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to process PDF. Check the logs for details.")
        elif st.session_state.pdf_processed:
            st.success("✅ PDF processed and ready!")
            if st.button("🔄 Upload New PDF"):
                st.session_state.pdf_processed = False
                st.session_state.document_hash = None
                st.session_state.conversation_history = []
                st.session_state.session_graph = KnowledgeGraph(
                    nodes=[], relationships=[]
                )
                st.session_state.processed_chunks = set()
                st.rerun()

        if st.session_state.processing_status:
            st.info(st.session_state.processing_status)

    if st.session_state.pdf_processed:
        col1, col2 = st.columns([3, 2])

        with col1:
            st.header("💬 Chat")

            chat_container = st.container()
            with chat_container:
                for i, message in enumerate(st.session_state.conversation_history):
                    if message["role"] == "user":
                        st.chat_message("user").write(message["content"])
                    else:
                        st.chat_message("assistant").write(message["content"])

            if query := st.chat_input("Ask a question about the document..."):
                if chat_model_choice == "gpt-4.1 (OpenAI)":
                    chat_model = OpenAIModel("gpt-4.1")
                elif chat_model_choice == "gpt-4.1-mini (OpenAI)":
                    chat_model = OpenAIModel("gpt-4.1-mini")
                elif chat_model_choice == "o4-mini (OpenAI)":
                    chat_model = OpenAIModel("o4-mini")
                elif chat_model_choice == "qwen3 (Ollama)":
                    chat_model = OllamaModel("qwen3")
                elif chat_model_choice == "gemma3 (Ollama)":
                    chat_model = OllamaModel("gemma3")
                else:
                    raise NotImplementedError(f"unsupported model: {chat_model_choice}")

                st.chat_message("user").write(query)
                with st.spinner("Thinking..."):
                    response = handle_chat_message(query, chat_model)

                st.chat_message("assistant").write(response)
                st.rerun()

        with col2:
            st.header("🧠 Knowledge Graph")

            graph_fig = create_graph_visualization(st.session_state.session_graph)
            st.plotly_chart(graph_fig, use_container_width=True)

            if st.session_state.session_graph.nodes:
                st.metric("Concepts", len(st.session_state.session_graph.nodes))
                st.metric(
                    "Relationships", len(st.session_state.session_graph.relationships)
                )

                node_types = {}
                for node in st.session_state.session_graph.nodes:
                    node_types[node.type] = node_types.get(node.type, 0) + 1

                st.subheader("Node Types")
                for node_type, count in node_types.items():
                    st.write(f"**{node_type.title()}**: {count}")
            else:
                st.info("Start chatting to build the knowledge graph!")

    else:
        st.markdown(
            """
        ## Welcome to Study Assistant! 🎓

        This tool helps you study PDF documents by providing:

        - **📖 Smart Chat**: Ask questions about your document with RAG-enhanced responses
        - **🧠 Knowledge Graph**: Automatically generated visual representation of key concepts
        - **🔍 Context-Aware**: Uses conversation history for better understanding
        - **🖼️ Image Support**: Processes and describes images in your PDFs

        ### How to get started:
        1. Upload a PDF document using the sidebar
        2. Configure your processing settings
        3. Click "Process PDF" and wait for completion
        4. Start asking questions about your document!

        ### Tips:
        - Use specific questions for better results
        - The knowledge graph updates as you chat
        - Images in PDFs are automatically captioned
        - Conversation history provides context for follow-up questions
        """
        )


if __name__ == "__main__":
    main()

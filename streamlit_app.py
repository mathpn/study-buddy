import os
import random
import tempfile

import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from chromadb.utils.embedding_functions import (
    OllamaEmbeddingFunction,
    OpenAIEmbeddingFunction,
)

from graph import KnowledgeGraph
from logger import setup_logger
from models import AnthropicModel, ModelProvider, OllamaModel, OpenAIModel
from pdf_processor import (
    ExtractionBackend,
    PDFProcessor,
    VectorStore,
    hash_file,
)
from study_buddy import (
    StudyBuddy,
    chat,
    generate_question,
    generate_topic_based_question,
    get_or_extract_study_topics,
    retrieve_topic_content,
    select_next_quiz_topic,
)

logger = setup_logger(__name__)

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
    if "quiz_question" not in st.session_state:
        st.session_state.quiz_question = None
    if "user_answer" not in st.session_state:
        st.session_state.user_answer = None
    if "show_feedback" not in st.session_state:
        st.session_state.show_feedback = False
    if "app_state" not in st.session_state:
        st.session_state.app_state = "WAITING_FOR_GOALS"
    if "study_buddy" not in st.session_state:
        st.session_state.study_buddy = None
    if "document_hash" not in st.session_state:
        st.session_state.document_hash = None
    if "study_topics" not in st.session_state:
        st.session_state.study_topics = None
    if "covered_quiz_topics" not in st.session_state:
        st.session_state.covered_quiz_topics = set()
    if "quiz_topic_performance" not in st.session_state:
        st.session_state.quiz_topic_performance = {}


def reset_quiz_state():
    """Reset quiz-related session state variables."""
    st.session_state.covered_quiz_topics = set()
    st.session_state.quiz_topic_performance = {}
    st.session_state.quiz_question = None
    st.session_state.user_answer = None
    st.session_state.show_feedback = False


def display_study_topics():
    """Display current study topics."""
    if st.session_state.study_topics:
        total_count = len(st.session_state.study_topics)

        with st.expander(f"📚 Study Topics ({total_count} identified)", expanded=False):
            st.write("**All Study Topics:**")
            for i, topic in enumerate(st.session_state.study_topics, 1):
                st.write(f"{i}. {topic}")


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

        # FIXME this fails sometimes
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


_chat_model_mapping = {
    "GPT 4.1 (OpenAI)": ("openai", "gpt-4.1"),
    "GPT 4.1 mini (OpenAI)": ("openai", "gpt-4.1-mini"),
    "o4 mini (OpenAI)": ("openai", "o4-mini"),
    "Qwen3 (Ollama)": ("ollama", "qwen3:latest"),
    "Gemma3 (Ollama)": ("ollama", "gemma3:latest"),
    "Claude Haiku (Anthropic)": ("anthropic", "claude-3-5-haiku-latest"),
    "Claude Sonnet 4 (Anthropic)": ("anthropic", "claude-sonnet-4-20250514"),
}

_embedding_model_mapping = {
    "text-embedding-3-small (OpenAI)": ("openai", "text-embedding-3-small"),
    "text-embedding-3-large (OpenAI)": ("openai", "text-embedding-3-large"),
    "nomic-embed-text (Ollama)": ("openai", "nomic-embed-text"),
}

_captioning_model_mapping = {
    "gpt-4.1-mini (OpenAI)": ("openai", "gpt-4.1-mini"),
    "granite3.2-vision (Ollama)": ("ollama", "granite3.2-vision"),
    "Claude Haiku (Anthropic)": ("anthropic", "claude-3-5-haiku-latest"),
    "Claude Sonnet 4 (Anthropic)": ("anthropic", "claude-sonnet-4-20250514"),
}


def _get_model(alias: str, mapping: dict[str, tuple[str, str]]) -> ModelProvider:
    provider, model_name = mapping[alias]
    if provider == "ollama":
        return OllamaModel(model_name)
    if provider == "openai":
        return OpenAIModel(model_name)
    if provider == "anthropic":
        return AnthropicModel(model_name)
    raise NotImplementedError(f"model not supported: {alias}")


def process_pdf_file(
    uploaded_file,
    captioning_model_alias: str,
    embedding_model_alias: str,
    extraction_backend: ExtractionBackend,
):
    """Process uploaded PDF file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        st.session_state.processing_status = "Processing PDF..."

        captioning_model = _get_model(captioning_model_alias, _captioning_model_mapping)

        provider, emb_model_name = _embedding_model_mapping[embedding_model_alias]
        if provider == "openai":
            embedding_fn = OpenAIEmbeddingFunction(
                os.environ["OPENAI_API_KEY"], emb_model_name
            )
        elif provider == "ollama":
            embedding_fn = OllamaEmbeddingFunction(model_name=emb_model_name)
        else:
            raise NotImplementedError(
                f"embedding model not supported: {embedding_model_alias}"
            )

        if st.session_state.vector_store is None:
            vector_store = VectorStore(
                embedding_fn,
                persist_directory="./vector_store_streamlit",  # FIXME better path
            )
            st.session_state.vector_store = vector_store

        vector_store: VectorStore = st.session_state.vector_store
        document_hash = hash_file(tmp_file_path)
        if not st.session_state.vector_store.document_exists(document_hash):
            processor = PDFProcessor(
                image_captioning_model=captioning_model,
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
        logger.error("Error processing PDF: %s", e, exc_info=True)
        return False


def handle_chat_message(query: str, model: ModelProvider) -> str:
    """Handle a chat message and update the knowledge graph"""
    if not st.session_state.document_hash:
        return "Please upload and process a PDF file first."

    try:
        response_content, conversation_history, session_graph, processed_chunks = chat(
            query,
            st.session_state.document_hash,
            st.session_state.vector_store,
            st.session_state.conversation_history,
            st.session_state.processed_chunks,
            st.session_state.session_graph,
            model,
        )
        st.session_state.conversation_history = conversation_history
        st.session_state.processed_chunks = processed_chunks
        st.session_state.session_graph = session_graph
        return response_content

    except Exception as e:
        logger.error("Error in chat: %s", e, exc_info=True)
        return "Sorry, I've encountered an error, please try again"


def main():
    """Main Streamlit app with guided study experience"""
    initialize_session_state()

    st.title("📚 Study Assistant")
    st.markdown("Upload a PDF document for a personalized, guided study experience!")

    with st.sidebar:
        st.header("📁 Document Upload")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to analyze and chat about",
        )

        st.header("⚙️ Processing Settings")

        embedding_model_alias = st.selectbox(
            "Embedding Model",
            _embedding_model_mapping.keys(),
            help="Model used for text embeddings",
        )

        captioning_model_alias = st.selectbox(
            "Image Captioning Model",
            _captioning_model_mapping.keys(),
            help="Model used for describing images in the PDF",
        )

        chat_model_alias = st.selectbox(
            "Chat Model",
            _chat_model_mapping.keys(),
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
                        captioning_model_alias,
                        embedding_model_alias,
                        extraction_backend,
                    )
                    if success:
                        st.success("PDF processed successfully!")
                        # Reset app state when new PDF is processed
                        st.session_state.app_state = "WAITING_FOR_GOALS"
                        st.session_state.study_buddy = None
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
                st.session_state.app_state = "WAITING_FOR_GOALS"
                st.session_state.study_buddy = None
                st.rerun()

        if st.session_state.processing_status:
            st.info(st.session_state.processing_status)

    # Main content area
    if st.session_state.pdf_processed:
        # State: WAITING_FOR_GOALS
        if st.session_state.app_state == "WAITING_FOR_GOALS":
            st.header("🎯 Let's Set Your Study Goals")
            st.markdown(
                "Before we begin, tell me what you want to learn from this document. "
                "This will help me create a personalized study plan just for you!"
            )

            study_goals = st.text_area(
                "What are your study goals?",
                placeholder="For example: I want to understand the key concepts of machine learning, particularly supervised learning algorithms and how they work...",
                height=120,
            )

            if st.button("🚀 Start My Study Journey", type="primary"):
                if study_goals.strip():
                    with st.spinner("Setting up your personalized study session..."):
                        chat_model = _get_model(chat_model_alias, _chat_model_mapping)
                        st.session_state.study_buddy = StudyBuddy(
                            model=chat_model,
                            vector_store=st.session_state.vector_store,
                            document_hash=st.session_state.document_hash,
                        )
                        st.session_state.study_buddy.set_study_goals(study_goals)

                        st.session_state.study_buddy.generate_assessment_questions()
                        st.session_state.app_state = "ASSESSING"
                        st.rerun()
                else:
                    st.warning("Please enter your study goals to continue.")

        # State: ASSESSING
        elif st.session_state.app_state == "ASSESSING":
            st.header("📝 Quick Knowledge Assessment")
            st.markdown(
                "Let me ask you a few questions to understand your current knowledge level. "
                "This will help me create the best study plan for you!"
            )

            questions = st.session_state.study_buddy.get_assessment_questions()

            with st.form("assessment_form"):
                answers = []
                for i, question in enumerate(questions):
                    st.subheader(f"Question {i + 1}")
                    st.write(question)
                    answer = st.text_area(
                        "Your answer:",
                        key=f"assessment_answer_{i}",
                        placeholder="Share what you know, even if you're not sure. It's okay to say 'I don't know' too!",
                        height=80,
                    )
                    answers.append(answer)

                if st.form_submit_button("Submit Assessment", type="primary"):
                    # Store all answers
                    for i, answer in enumerate(answers):
                        st.session_state.study_buddy.answer_assessment_question(
                            i, answer
                        )

                    # Check if all questions are answered
                    if st.session_state.study_buddy.all_questions_answered():
                        with st.spinner("Creating your study plan..."):
                            st.session_state.study_buddy.generate_study_plan()
                            st.session_state.app_state = "STUDYING"
                            st.rerun()
                    else:
                        st.warning("Please answer all questions to continue.")

        # State: STUDYING
        elif st.session_state.app_state == "STUDYING":
            col1, col2 = st.columns([3, 2])

            with col1:
                # Show study plan
                st.header("Your Study Plan")
                with st.expander("📋 View Study Plan", expanded=True):
                    study_plan = st.session_state.study_buddy.get_study_plan()
                    st.markdown(study_plan)

                # Chat and Quiz tabs
                chat_tab, quiz_tab = st.tabs(["💬 Study Chat", "🧠 Practice Quiz"])

                with chat_tab:
                    st.markdown(
                        "**Ask questions about your study topics or request clarifications on the study plan!**"
                    )

                    # Display conversation history
                    chat_container = st.container()
                    with chat_container:
                        for message in st.session_state.conversation_history:
                            if message["role"] == "user":
                                st.chat_message("user").write(message["content"])
                            else:
                                st.chat_message("assistant").write(message["content"])

                    # Chat input
                    if query := st.chat_input(
                        "Ask a question about the document or your study plan..."
                    ):
                        chat_model = _get_model(chat_model_alias, _chat_model_mapping)
                        st.chat_message("user").write(query)
                        with st.spinner("Thinking..."):
                            response = handle_chat_message(query, chat_model)
                        st.chat_message("assistant").write(response)
                        st.rerun()

                with quiz_tab:
                    st.markdown(
                        "**Test your understanding with questions based on your study plan!**"
                    )

                    display_study_topics()

                    if st.button("Generate New Question"):
                        st.session_state.quiz_question = None
                        st.session_state.user_answer = None
                        st.session_state.show_feedback = False

                        if (
                            not st.session_state.study_buddy
                            or not st.session_state.study_buddy.get_study_plan()
                        ):
                            st.warning(
                                "Please complete the study planning process first."
                            )
                        else:
                            with st.spinner("Generating topic-focused question..."):
                                chat_model = _get_model(
                                    chat_model_alias, _chat_model_mapping
                                )

                                if st.session_state.study_topics is None:
                                    st.info(
                                        "Extracting key topics from your study plan..."
                                    )
                                    study_plan = (
                                        st.session_state.study_buddy.get_study_plan()
                                    )
                                    st.session_state.study_topics = (
                                        get_or_extract_study_topics(
                                            study_plan, chat_model
                                        )
                                    )
                                    logger.info(
                                        "Cached %d study topics: %s",
                                        len(st.session_state.study_topics),
                                        st.session_state.study_topics,
                                    )
                                    st.success(
                                        f"Identified {len(st.session_state.study_topics)} study topics!"
                                    )

                                if not st.session_state.study_topics:
                                    st.error(
                                        "Unable to extract study topics from your study plan. Please ensure your study plan contains clear learning objectives."
                                    )
                                else:
                                    selected_topic = select_next_quiz_topic(
                                        st.session_state.study_topics
                                    )

                                    st.info(f"Focusing on topic: **{selected_topic}**")

                                    with st.spinner(
                                        f"Searching for content related to {selected_topic}..."
                                    ):
                                        retrieved_chunks = retrieve_topic_content(
                                            selected_topic,
                                            st.session_state.vector_store,
                                            st.session_state.document_hash,
                                            top_k=5,
                                        )

                                    logger.info(
                                        f"Retrieved {len(retrieved_chunks)} chunks for topic: {selected_topic}"
                                    )

                                    if not retrieved_chunks:
                                        st.warning(
                                            f"No relevant content found for topic: **{selected_topic}**. Using fallback method."
                                        )
                                        # Fallback method if no content retrieved
                                        if st.session_state.processed_chunks:
                                            num_chunks = min(
                                                3,
                                                len(st.session_state.processed_chunks),
                                            )
                                            random_chunks = random.sample(
                                                list(st.session_state.processed_chunks),
                                                num_chunks,
                                            )
                                            question_data = generate_question(
                                                text_chunks=random_chunks,
                                                graph=st.session_state.session_graph,
                                                model=chat_model,
                                            )
                                        else:
                                            question_data = None
                                    else:
                                        with st.spinner(
                                            f"Creating question for {selected_topic}..."
                                        ):
                                            study_plan = (
                                                st.session_state.study_buddy.get_study_plan()
                                            )
                                            question_data = (
                                                generate_topic_based_question(
                                                    selected_topic,
                                                    retrieved_chunks,
                                                    chat_model,
                                                )
                                            )

                                        if question_data:
                                            st.session_state.covered_quiz_topics.add(
                                                selected_topic
                                            )
                                            logger.info(
                                                f"Added topic to covered set: {selected_topic}"
                                            )

                                    if question_data is None:
                                        st.error(
                                            "❌ Failed to generate a quiz question. This might be due to insufficient content or model issues. Please try again."
                                        )
                                    else:
                                        st.session_state.quiz_question = question_data
                                        if "topic" in question_data:
                                            st.success(
                                                f"✅ **Generated question for:** {question_data['topic']}"
                                            )

                                st.rerun()

                    if (
                        st.session_state.quiz_question
                        and not st.session_state.show_feedback
                    ):
                        q_data = st.session_state.quiz_question
                        with st.form(key="quiz_form"):
                            st.markdown(f"#### {q_data['question']}")
                            options = q_data["options"]
                            user_answer = st.radio(
                                "Select your answer:",
                                options=options,
                                index=None,
                            )
                            submitted = st.form_submit_button("Submit")
                            if submitted:
                                st.session_state.user_answer = user_answer
                                st.session_state.show_feedback = True
                                st.rerun()

                    if (
                        st.session_state.show_feedback
                        and st.session_state.quiz_question
                    ):
                        q_data = st.session_state.quiz_question
                        user_answer = st.session_state.user_answer
                        correct_answer = q_data["answer"]

                        st.markdown(f"#### {q_data['question']}")
                        st.write("Options:", ", ".join(q_data["options"]))

                        if user_answer is None:
                            st.warning("You did not select an answer.")
                        elif user_answer == correct_answer:
                            st.success(
                                f"**Correct!** The right answer is **{correct_answer}**."
                            )
                            if "topic" in q_data:
                                topic = q_data["topic"]
                                if topic not in st.session_state.quiz_topic_performance:
                                    st.session_state.quiz_topic_performance[topic] = {
                                        "correct": 0,
                                        "total": 0,
                                    }
                                st.session_state.quiz_topic_performance[topic][
                                    "correct"
                                ] += 1
                                st.session_state.quiz_topic_performance[topic][
                                    "total"
                                ] += 1
                        else:
                            st.error(
                                f"**Incorrect.** You chose: *{user_answer}*. The correct answer is **{correct_answer}**."
                            )
                            # Track topic performance
                            if "topic" in q_data:
                                topic = q_data["topic"]
                                if topic not in st.session_state.quiz_topic_performance:
                                    st.session_state.quiz_topic_performance[topic] = {
                                        "correct": 0,
                                        "total": 0,
                                    }
                                st.session_state.quiz_topic_performance[topic][
                                    "total"
                                ] += 1

                        # Display topic performance if available
                        if (
                            "topic" in q_data
                            and st.session_state.quiz_topic_performance
                        ):
                            topic = q_data["topic"]
                            if topic in st.session_state.quiz_topic_performance:
                                perf = st.session_state.quiz_topic_performance[topic]
                                accuracy = (
                                    perf["correct"] / perf["total"] * 100
                                    if perf["total"] > 0
                                    else 0
                                )
                                st.info(
                                    f"📈 **{topic}** accuracy: {perf['correct']}/{perf['total']} ({accuracy:.1f}%)"
                                )

            with col2:
                st.header("🧠 Knowledge Graph")
                graph_fig = create_graph_visualization(st.session_state.session_graph)
                st.plotly_chart(graph_fig, use_container_width=True)

                # Display overall quiz performance summary
                if st.session_state.quiz_topic_performance:
                    st.subheader("📊 Quiz Performance Summary")
                    total_questions = sum(
                        perf["total"]
                        for perf in st.session_state.quiz_topic_performance.values()
                    )
                    total_correct = sum(
                        perf["correct"]
                        for perf in st.session_state.quiz_topic_performance.values()
                    )
                    overall_accuracy = (
                        (total_correct / total_questions * 100)
                        if total_questions > 0
                        else 0
                    )

                    st.metric(
                        "Overall Performance",
                        f"{total_correct}/{total_questions} ({overall_accuracy:.1f}%)",
                    )

                    for topic, perf in st.session_state.quiz_topic_performance.items():
                        accuracy = (
                            perf["correct"] / perf["total"] * 100
                            if perf["total"] > 0
                            else 0
                        )
                        if accuracy >= 80:
                            st.success(
                                f"🟢 **{topic}**: {perf['correct']}/{perf['total']} ({accuracy:.1f}%)"
                            )
                        elif accuracy >= 50:
                            st.warning(
                                f"🟡 **{topic}**: {perf['correct']}/{perf['total']} ({accuracy:.1f}%)"
                            )
                        else:
                            st.error(
                                f"🔴 **{topic}**: {perf['correct']}/{perf['total']} ({accuracy:.1f}%)"
                            )

                    if st.button("🔄 Reset Quiz Progress"):
                        reset_quiz_state()
                        st.success("✅ Quiz progress has been reset!")
                        st.rerun()
                elif st.session_state.study_topics:
                    st.info(
                        "📝 No quiz attempts yet. Generate your first question to start tracking performance!"
                    )

                # Option to restart study session
                if st.button("🔄 Start New Study Session"):
                    st.session_state.app_state = "WAITING_FOR_GOALS"
                    st.session_state.study_buddy = None
                    st.session_state.conversation_history = []
                    st.session_state.study_topics = None
                    st.session_state.covered_quiz_topics = set()
                    st.session_state.quiz_topic_performance = {}
                    st.rerun()

    else:
        st.markdown(
            """
        ## Welcome to Study Assistant! 🎓

        This tool provides a **personalized, guided study experience** with your PDF documents:

        ### 🌟 What makes this special:
        - **🎯 Goal-Oriented**: Tell me what you want to learn
        - **📊 Knowledge Assessment**: I'll evaluate your current understanding
        - **📋 Personalized Study Plan**: Get a custom roadmap based on your needs
        - **💬 Smart Chat**: Ask questions with RAG-enhanced responses
        - **🧠 Knowledge Graph**: Visual representation of key concepts
        - **🧩 Practice Quizzes**: Test your understanding

        ### 🚀 How it works:
        1. **Upload** a PDF document using the sidebar
        2. **Set your study goals** - what do you want to learn?
        3. **Take a quick assessment** to gauge your current knowledge
        4. **Get your personalized study plan** with topics and guidance questions
        5. **Study with AI assistance** - chat, explore, and practice!

        ### 💡 Perfect for:
        - Students preparing for exams
        - Professionals learning new skills
        - Researchers exploring new topics
        - Anyone who wants structured, effective learning
        """
        )


if __name__ == "__main__":
    main()

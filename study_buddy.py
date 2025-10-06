import logging
import random
from textwrap import dedent

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from graph import KnowledgeGraph, build_knowledge_graph, merge_knowledge_graphs
from models import ModelProvider
from pdf_processor import ImageChunk, TextChunk, VectorStore

logger = logging.getLogger(__name__)


class QuestionSchema(BaseModel):
    """Schema for a multiple-choice question."""

    question: str
    options: list[str]
    answer: str


class AssessmentSchema(BaseModel):
    """Schema for a list of questions."""

    questions: list[str]


class StudyTopicsSchema(BaseModel):
    """Schema for extracted study topics."""

    topics: list[str]


class TopicSelectionSchema(BaseModel):
    """Schema for topic selection priority."""

    selected_topic: str
    reasoning: str


def get_or_extract_study_topics(
    study_plan: str, model: ModelProvider, limit: int = 8
) -> list[str]:
    """Extract and cache study topics from study plan."""
    fallback_topics = [
        "Core concepts",
        "Key principles",
        "Important methods",
        "Main theories",
    ]
    system_prompt = """
    You are an educational assistant that extracts key study topics from a study plan.
    Analyze the provided study plan and extract 5-8 core topics that should be covered in quizzes.
    Focus on concrete concepts, theories, methods, or subject areas that can be tested.
    Return topics that are specific enough to generate meaningful questions but broad enough to cover substantial content.
    """

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Study Plan:\n{study_plan}"}],
        },
    ]

    try:
        response = model.chat_with_schema(messages=messages, schema=StudyTopicsSchema)
        if response is None:
            logger.warning("Failed to extract topics from study plan, using fallback")
            return fallback_topics

        topics = response.topics[:limit]
        logger.info("Extracted %d study topics from study plan", len(topics))
        return topics

    except Exception as e:
        logger.error("Error extracting study topics: %s", e, exc_info=True)
        return fallback_topics


def select_next_quiz_topic(study_topics: list[str]) -> str:
    return random.choice(study_topics)


def retrieve_topic_content(
    topic: str, vector_store: VectorStore, document_hash: str, top_k: int = 5
) -> list:
    """Perform targeted RAG retrieval for a specific topic."""
    try:
        focused_queries = [
            topic,
            f"concepts related to {topic}",
            f"definition of {topic}",
            f"examples of {topic}",
        ]

        all_chunks = []
        seen_content = set()

        for query in focused_queries:
            chunks = vector_store.search_combined(query, document_hash, top_k=3)
            for chunk, score in chunks:
                if isinstance(chunk, TextChunk):
                    content_hash = hash(chunk.content)
                elif isinstance(chunk, ImageChunk):
                    content_hash = hash(chunk.caption)
                else:
                    content_hash = hash(str(chunk))

                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_chunks.append((chunk, score))

        if not all_chunks:
            logger.warning("No chunks found for any query related to topic '%s'", topic)
            return []

        all_chunks.sort(key=lambda x: x[1], reverse=True)
        return all_chunks[:top_k]

    except Exception as e:
        logger.error(
            "Error retrieving topic content for '%s': %s", topic, e, exc_info=True
        )
        return []


def calc_question_difficulty(
    correct_answers: int, streak: int, n_questions: int
) -> int:
    """Calculate question difficulty based on previous quiz performance."""
    if n_questions == 0:
        return 5

    accuracy = correct_answers / n_questions
    difficulty = round(
        max(
            1,
            (max(0, 10 - n_questions) * 0.5 + min(10, n_questions) * accuracy),
        )
    )
    difficulty += max(-3, min(3, streak))
    difficulty = min(10, max(1, difficulty))
    return difficulty


def generate_topic_based_question(
    topic: str,
    retrieved_chunks: list,
    model: ModelProvider,
    previous_questions: list[dict],
    difficulty: int = 5,
) -> dict | None:
    """Generate quiz question focused on specific topic using retrieved content."""
    try:
        if not retrieved_chunks:
            logger.warning("No retrieved chunks available for topic '%s'", topic)
            return None

        content_parts = []
        for chunk, _ in retrieved_chunks:
            if isinstance(chunk, TextChunk):
                content_parts.append(chunk.content)
            elif isinstance(chunk, ImageChunk):
                content_parts.append(f"Image Caption: {chunk.caption}")

        text_context = "\n---\n".join(content_parts)

        previous_questions_str = "\n".join(
            [f"- {q['question']}" for q in previous_questions if "question" in q]
        )
        previous_questions_prompt = (
            f"These questions were already asked:\n{previous_questions_str}"
            if previous_questions_str
            else "No previous questions."
        )
        system_prompt = f"""
        You are a helpful assistant designed to create focused study questions.
        Generate a single multiple-choice question specifically about: "{topic}"

        Use the provided content and knowledge graph to create a question that:
        1. Tests understanding of the specified topic
        2. Is based on the actual document content provided
        3. Has 4 plausible options with one clearly correct answer
        4. Aligns with the study objectives shown in the study plan excerpt
        5. Do not repeat questions that have already been asked

        {previous_questions_prompt}
        
        Consider a difficulty index in which 1 refers to a question that requires only basic
        text comprehension, and 10 requires deep knowledge about the topic.
        Questions of level 8 and above must present a significant challenge for the student.
        For difficult questions, prefer creating a context in which the concepts are applied.
        Generate a question with a difficulty of {difficulty}.
        The question should be in JSON format with the following keys: "question", "options", "answer".
        The question MUST be grounded in the provided text content and focus on "{topic}".
        """

        user_prompt = f"""
        Topic to focus on: {topic}

        Relevant Content:
        ---
        {text_context}
        ---

        Create a multiple-choice question focused specifically on "{topic}" using the above content.
        The user does not see the content, only the question.
        """

        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            },
        ]

        logger.info(
            "generating a question for topic %s with difficulty %d", topic, difficulty
        )
        response = model.chat_with_schema(messages, schema=QuestionSchema)
        if response is None:
            logger.error("Failed to generate topic-based question for '%s'", topic)
            return None

        result = response.model_dump()
        result["topic"] = topic
        return result

    except Exception as e:
        logger.error(
            "Error generating topic-based question for '%s': %s",
            topic,
            e,
            exc_info=True,
        )
        return None


def generate_question(
    text_chunks: list[str], graph: KnowledgeGraph, model: ModelProvider
) -> dict | None:
    """
    Generates a multiple-choice question from text chunks and a knowledge graph.

    Args:
        text_chunks (List[str]): A list of text chunks for context.
        graph (KnowledgeGraph): The knowledge graph to base the question on.
        model (ModelProvider): The language model to use for generation.

    Returns:
        dict: A dictionary containing the question, options, and correct answer, or None on failure.
    """
    graph_json = graph.model_dump_json(indent=2)
    text_context = "\n---\n".join(text_chunks)
    system_prompt = """
    You are a helpful assistant designed to create study questions.
    Based on the provided knowledge graph and text excerpts from a document, generate a single multiple-choice question.
    The question MUST be related to the concepts and relationships in the knowledge graph.
    The provided text excerpts are for additional context.

    The question should be in JSON format with the following keys: "question", "options", "answer".
    The "options" should be a list of 4 strings, where one is the correct answer.
    The "answer" should be the string of the correct option.
    """

    user_prompt = f"""
    Knowledge Graph:
    ---
    {graph_json}
    ---

    Text Excerpts:
    ---
    {text_context}
    ---
    """

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": dedent(system_prompt)}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": dedent(user_prompt)}],
        },
    ]
    response = model.chat_with_schema(messages, schema=QuestionSchema)
    if response is None:
        logger.error("Failed to generate question.")
        return None

    return response.model_dump()


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


def chat(
    query: str,
    document_hash: str,
    vector_store: VectorStore,
    conversation_history,
    processed_chunks,
    session_graph: KnowledgeGraph,
    model: ModelProvider,
):
    enhanced_query = build_enhanced_query(query, conversation_history)

    retrieved_chunks = vector_store.search_combined(
        enhanced_query, document_hash, top_k=3
    )

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
                    "Added %d new nodes and %d new relationships",
                    len(new_graph.nodes),
                    len(new_graph.relationships),
                )
        except Exception as e:
            logger.warning("Failed to generate knowledge graph: %s", e, exc_info=True)

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
        for chunk, _ in retrieved_chunks:
            if isinstance(chunk, TextChunk):
                context_parts.append(chunk.content)
            elif isinstance(chunk, ImageChunk):
                context_parts.append(f"Image Caption: {chunk.caption}")

        context = "\n---\n".join(context_parts)
        system_content += f"\n\nRelevant context from the document:\n{context}"

    messages: list[ChatCompletionMessageParam] = []
    messages.append({"role": "system", "content": system_content})
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": query})

    logger.debug("current messages: %s", messages)
    response_content = model.chat(messages=messages)

    conversation_history.append({"role": "user", "content": query})
    conversation_history.append({"role": "assistant", "content": response_content})

    return response_content, conversation_history, session_graph, processed_chunks


class StudyBuddy:
    """Manages personalized study sessions with assessment and planning."""

    def __init__(
        self, model: ModelProvider, vector_store: VectorStore, document_hash: str
    ):
        """Initialize StudyBuddy with model and document context."""

        self.model = model
        self.vector_store = vector_store
        self.document_hash = document_hash
        self.conversation_history = []
        self.processed_chunks = set()
        self.session_graph = KnowledgeGraph(nodes=[], relationships=[])
        self.study_goals = ""
        self.assessment_questions = []
        self.assessment_answers = []
        self.study_plan = ""

    def set_study_goals(self, study_goals: str) -> None:
        """Set the user's study goals."""
        self.study_goals = study_goals
        logger.info("Study goals set: %s...", study_goals[:100])

    def generate_assessment_questions(self, num_questions: int = 3) -> None:
        """Generate assessment questions based on study goals and document content."""
        try:
            # Use study goals to retrieve relevant chunks
            retrieved_chunks = self.vector_store.search_combined(
                self.study_goals, self.document_hash, top_k=5
            )

            # Prepare context from retrieved chunks
            context_parts = []
            for chunk, _ in retrieved_chunks:
                if isinstance(chunk, TextChunk):
                    context_parts.append(chunk.content)
                elif isinstance(chunk, ImageChunk):
                    context_parts.append(f"Image Caption: {chunk.caption}")

            context = "\n---\n".join(context_parts)

            # Generate assessment questions
            system_prompt = (
                "You are an educational assistant that creates assessment questions. "
                "Based on the user's study goals and the provided document context, "
                f"generate exactly {num_questions} questions to assess their current knowledge level. "
                "These should be open-ended questions that help determine what the student already knows "
                "and identify knowledge gaps. Format your response as a JSON list of strings."
            )

            messages: list[ChatCompletionMessageParam] = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Study goals: {self.study_goals}\n\nDocument context:\n{context}",
                        }
                    ],
                },
            ]

            response = self.model.chat_with_schema(
                messages=messages, schema=AssessmentSchema
            )
            if response is None:
                # Fallback questions
                self.assessment_questions = [
                    "What do you already know about the main topics in this document?",
                    "What specific aspects would you like to understand better?",
                    "Are there any concepts or terms that are completely new to you?",
                ]
                self.assessment_answers = [""] * len(self.assessment_questions)
                return

            questions = response.questions
            self.assessment_questions = questions[:num_questions]
            self.assessment_answers = [""] * len(self.assessment_questions)
            logger.info(
                "Generated %d assessment questions", len(self.assessment_questions)
            )

        except Exception as e:
            logger.error("Error generating assessment questions: %s", e, exc_info=True)
            # Fallback questions
            self.assessment_questions = [
                "What do you already know about the main topics in this document?",
                "What specific aspects would you like to understand better?",
                "Are there any concepts or terms that are completely new to you?",
            ]
            self.assessment_answers = [""] * len(self.assessment_questions)

    def answer_assessment_question(self, question_idx: int, answer: str) -> None:
        """Store the user's answer to an assessment question."""
        if 0 <= question_idx < len(self.assessment_answers):
            self.assessment_answers[question_idx] = answer
            logger.debug("Recorded answer for question %s", question_idx)

    def all_questions_answered(self) -> bool:
        """Check if all assessment questions have been answered."""
        return all(answer.strip() for answer in self.assessment_answers)

    def generate_study_plan(self) -> None:
        """Generate a personalized study plan based on goals and assessment answers."""
        try:
            # Retrieve relevant chunks again for study plan context
            retrieved_chunks = self.vector_store.search_combined(
                self.study_goals, self.document_hash, top_k=7
            )

            # Prepare context from retrieved chunks
            context_parts = []
            for chunk, _ in retrieved_chunks:
                if isinstance(chunk, TextChunk):
                    context_parts.append(chunk.content)
                elif isinstance(chunk, ImageChunk):
                    context_parts.append(f"Image Caption: {chunk.caption}")

            context = "\n---\n".join(context_parts)

            # Prepare assessment summary
            qa_summary = []
            for i, (question, answer) in enumerate(
                zip(self.assessment_questions, self.assessment_answers)
            ):
                qa_summary.append(f"Q{i + 1}: {question}\nA{i + 1}: {answer}")

            assessment_text = "\n\n".join(qa_summary)

            system_prompt = (
                "You are an expert study coach. Based on the user's study goals, their assessment answers, "
                "and the document content, create a personalized study plan. The plan should:\n"
                "1. Identify the student's current knowledge level\n"
                "2. Highlight specific knowledge gaps\n"
                "3. Provide a structured list of topics to focus on\n"
                "4. Suggest specific questions the student should be able to answer after studying\n"
                "5. Recommend a study sequence (what to learn first, second, etc.)\n\n"
                "Format your response in clear markdown with headers and bullet points."
            )

            user_message = (
                f"**Study Goals:**\n{self.study_goals}\n\n"
                f"**Assessment Results:**\n{assessment_text}\n\n"
                f"**Document Context:**\n{context}"
            )

            messages: list[ChatCompletionMessageParam] = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_message}],
                },
            ]

            self.study_plan = self.model.chat(messages=messages)
            logger.info("Generated personalized study plan")

        except Exception as e:
            logger.error("Error generating study plan: %s", e, exc_info=True)
            self.study_plan = (
                "# Your Study Plan\n\n"
                "I encountered an error generating your personalized study plan. "
                "Please use the chat feature to ask specific questions about the topics you want to learn."
            )

    def get_study_plan(self) -> str:
        """Get the generated study plan."""
        return self.study_plan

    def get_assessment_questions(self) -> list[str]:
        """Get the list of assessment questions."""
        return self.assessment_questions

    def get_assessment_answers(self) -> list[str]:
        """Get the list of assessment answers."""
        return self.assessment_answers

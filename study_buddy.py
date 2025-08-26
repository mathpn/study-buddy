import json
import logging

from pydantic import BaseModel

from models import ModelProvider
from pdf_processor import ImageChunk, TextChunk, VectorStore
from graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class QuestionSchema(BaseModel):
    """Schema for a multiple-choice question."""

    question: str
    options: list[str]
    answer: str


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
    prompt = f"""
    You are a helpful assistant designed to create study questions.
    Based on the provided Knowledge Graph and text excerpts from a document, generate a single multiple-choice question.
    The question MUST be related to the concepts and relationships in the Knowledge Graph.
    The provided text excerpts are for additional context.

    The question should be in JSON format with the following keys: "question", "options", "answer".
    The "options" should be a list of 4 strings, where one is the correct answer.
    The "answer" should be the string of the correct option.

    Knowledge Graph:
    ---
    {graph_json}
    ---

    Text Excerpts:
    ---
    {text_context}
    ---

    JSON output:
    """
    response = model.generate_with_schema(prompt, schema=QuestionSchema)
    if response is None:
        logger.error("Failed to generate question.")
        return None

    return response.model_dump()


class StudyBuddy:
    """Manages personalized study sessions with assessment and planning."""

    def __init__(
        self, model: ModelProvider, vector_store: VectorStore, document_hash: str
    ):
        """Initialize StudyBuddy with model and document context."""
        self.model = model
        self.vector_store = vector_store
        self.document_hash = document_hash
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

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Study goals: {self.study_goals}\n\nDocument context:\n{context}",
                },
            ]

            response = self.model.chat(messages=messages)

            try:
                questions = json.loads(response)
                if isinstance(questions, list):
                    self.assessment_questions = questions[:num_questions]
                    self.assessment_answers = [""] * len(self.assessment_questions)
                else:
                    raise ValueError("Response is not a list")
            except (json.JSONDecodeError, ValueError):
                # Fallback: extract questions from text response
                lines = response.strip().split("\n")
                questions = []
                for line in lines:
                    line = line.strip()
                    if line and (
                        line.startswith('"')
                        or line.startswith("1.")
                        or line.startswith("-")
                    ):
                        # Clean up the line
                        question = line.strip('"').strip("1234567890.-").strip()
                        if question:
                            questions.append(question)

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

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
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

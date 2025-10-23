# study-buddy

A personalized AI-powered study tool that helps you learn from PDF documents through goal-based learning, assessments, and interactive practice.

## Features

- **PDF Processing**: Upload and process PDF documents with text and image extraction
- **Goal-Based Learning**: Set your study goals and get a personalized study plan
- **Knowledge Assessment**: Initial assessment to gauge your current understanding
- **RAG-Powered Chat**: Ask questions about your documents with context-aware responses
- **Knowledge Graph**: Visual representation of concepts and their relationships
- **Practice Quizzes**: Topic-focused quizzes with adaptive difficulty
- **Performance Tracking**: Monitor your progress across different topics

## Installation

This project requires Python 3.13+. Install dependencies using uv:

```bash
uv sync
```

## Usage

Run the Streamlit app:

```bash
uv run run_app.py
```

Then:

1. Upload a PDF document
1. Configure your model settings (OpenAI, Anthropic, or Ollama)
1. Set your study goals
1. Complete a quick assessment
1. Start studying with your personalized plan

## Configuration

You'll need API keys for the models you want to use:

- **OpenAI**: Set `OPENAI_API_KEY` environment variable
- **Anthropic**: Set `ANTHROPIC_API_KEY` environment variable
- **Ollama**: Install and run Ollama locally

## How It Works

The app uses:

- **Vector embeddings** to store and retrieve document content
- **LLMs** for chat, question generation, and assessments
- **Knowledge graphs** to map relationships between concepts
- **ChromaDB** for persistent vector storage
- **Docling or PyMuPDF** for PDF extraction

## Project Purpose

This is a learning project built to explore LLM application development and RAG systems.

import os
from abc import ABC, abstractmethod
from typing import Type, TypeVar

import anthropic
import ollama
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ModelProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: ...

    @abstractmethod
    def chat(self, messages: list[ChatCompletionMessageParam]) -> str: ...

    @abstractmethod
    def generate_with_images(self, prompt: str, images_b64: list[str]) -> str: ...

    @abstractmethod
    def chat_with_schema(
        self, messages: list[ChatCompletionMessageParam], schema: Type[T]
    ) -> T | None: ...


class OllamaModel(ModelProvider):
    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.client = ollama.Client(host)

    def generate(self, prompt: str) -> str:
        response = self.client.generate(model=self.model_name, prompt=prompt)
        return response["response"].strip()

    def chat(self, messages: list[ChatCompletionMessageParam]) -> str:
        """Handles chat with conversation history."""
        response = self.client.chat(model=self.model_name, messages=messages)
        return response["message"]["content"].strip()

    def chat_with_schema(
        self, messages: list[ChatCompletionMessageParam], schema: Type[T]
    ) -> T | None:
        response = self.client.chat(
            model=self.model_name, messages=messages, format=schema.model_json_schema()
        )
        return schema.model_validate_json(response["response"])

    def generate_with_images(self, prompt: str, images_b64: list[str]) -> str:
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            images=images_b64,
        )
        return response["response"].strip()


class OpenAIModel(ModelProvider):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    def chat(self, messages: list[ChatCompletionMessageParam]) -> str:
        """Handles chat with conversation history."""
        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages
        )
        return response.choices[0].message.content.strip()

    def chat_with_schema(
        self, messages: list[ChatCompletionMessageParam], schema: Type[T]
    ) -> T | None:
        response = self.client.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            response_format=schema,
        )
        return response.choices[0].message.parsed

    def generate_with_images(self, prompt: str, images_b64: list[str]) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        for image_b64 in images_b64:
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                }
            )

        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages, max_tokens=1000
        )

        return response.choices[0].message.content.strip()


class AnthropicModel(ModelProvider):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def chat(self, messages: list[ChatCompletionMessageParam]) -> str:
        """Handles chat with conversation history."""
        system_message = anthropic.NotGiven()
        anthropic_messages = []

        for msg in messages:
            role = msg["role"]
            if role == "system":
                system_message = str(msg.get("content", ""))
            elif role in ["assistant", "user"]:
                anthropic_messages.append(
                    {"role": role, "content": msg.get("content", "")}
                )

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            system=system_message,
            messages=anthropic_messages,
        )
        return response.content[0].text.strip()

    def chat_with_schema(
        self, messages: list[ChatCompletionMessageParam], schema: Type[T]
    ) -> T | None:
        system_message = anthropic.NotGiven()
        anthropic_messages = []

        for msg in messages:
            role = msg["role"]
            if role == "system":
                system_message = str(msg.get("content", ""))
            elif role in ["assistant", "user"]:
                anthropic_messages.append(
                    {"role": role, "content": msg.get("content", "")}
                )

        # Add schema instruction to the last user message
        if anthropic_messages and anthropic_messages[-1]["role"] == "user":
            schema_instruction = f"\n\nPlease respond with valid JSON that matches this schema: {schema.model_json_schema()}"
            anthropic_messages[-1]["content"] += schema_instruction

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            system=system_message,
            messages=anthropic_messages,
        )

        try:
            return schema.model_validate_json(response.content[0].text)
        except Exception:
            return None

    def generate_with_images(self, prompt: str, images_b64: list[str]) -> str:
        content = [{"type": "text", "text": prompt}]

        for image_b64 in images_b64:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64,
                    },
                }
            )

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )

        return response.content[0].text.strip()

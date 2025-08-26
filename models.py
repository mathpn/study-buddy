import os
from abc import ABC, abstractmethod
from typing import Type, TypeVar

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

    def chat(self, messages: list[dict[str, str]]) -> str:
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

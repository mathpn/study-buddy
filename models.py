import os
from abc import ABC, abstractmethod

import numpy as np
import ollama
from openai import OpenAI


class ModelProvider(ABC):
    @abstractmethod
    def generate_with_images(self, prompt: str, images_b64: list[str]) -> str: ...
    def embed(self, input: str) -> np.ndarray: ...


class OllamaModel(ModelProvider):
    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.client = ollama.Client(host)

    def generate_with_images(self, prompt: str, images_b64: list[str]) -> str:
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            images=images_b64,
        )
        return response["response"].strip()

    def embed(self, input: str) -> np.ndarray:
        return np.array(
            self.client.embed(model=self.model_name, input=input)["embeddings"][0]
        )


class OpenAIModel(ModelProvider):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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

    # TODO multiple inputs in one request
    def embed(self, input: str) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model_name, input=input)
        return np.array(response.data[0].embedding)

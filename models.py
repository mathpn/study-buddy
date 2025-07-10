from abc import ABC, abstractmethod
import numpy as np

import ollama


class ModelProvider(ABC):
    @abstractmethod
    def generate_with_images(self, prompt: str, images_b64: list[str]) -> str: ...
    def embed(self, input: str) -> np.ndarray: ...


class OllamaModel(ModelProvider):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_with_images(self, prompt: str, images_b64: list[str]) -> str:
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            images=images_b64,
        )
        return response["response"].strip()

    def embed(self, input: str) -> np.ndarray:
        return np.array(
            ollama.embed(model=self.model_name, input=input)["embeddings"][0]
        )

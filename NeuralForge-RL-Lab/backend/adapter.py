"""Adapter contract for user-supplied models. DemoAdapter is not a learning model."""
from abc import ABC, abstractmethod
from typing import Any

class ModelAdapter(ABC):
    name = 'custom-adapter'
    @abstractmethod
    def generate(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        pass
    def train_step(self, batch: list[dict[str, Any]], rewards: list[float]) -> dict[str, Any]:
        raise NotImplementedError('This adapter does not implement weight updates.')
    def save_checkpoint(self, destination: str) -> dict[str, Any]:
        raise NotImplementedError('Checkpoint saving is not implemented.')
    def load_checkpoint(self, source: str) -> dict[str, Any]:
        raise NotImplementedError('Checkpoint loading is not implemented.')

class DemoAdapter(ModelAdapter):
    name = 'Demo workflow simulator'
    def generate(self, prompt: str, context=None) -> str:
        return '-- DEMO OUTPUT: connect your own model to generate real Luau.\n-- Task: ' + prompt[:300]
    def train_step(self, batch, rewards):
        raise NotImplementedError('Demo mode records rewards but does not update weights.')

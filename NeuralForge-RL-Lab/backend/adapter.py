"""Adapter contract used by NeuralForge.

The lab imports the user's configured adapter factory. The demo adapter is only
for interface exploration and intentionally cannot train or save model weights.
"""
from abc import ABC, abstractmethod
from typing import Any

class ModelAdapter(ABC):
    name = "custom-adapter"

    @abstractmethod
    def generate(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def train_step(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Perform a real reward-based optimizer update."""
        raise NotImplementedError

    def supervised_step(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Optional teacher-forced update on imported reference examples."""
        return {"updates": 0, "loss": None, "note": "Supervised updates are not implemented."}

    @abstractmethod
    def save_checkpoint(self, destination: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, source: str) -> dict[str, Any]:
        raise NotImplementedError

class DemoAdapter(ModelAdapter):
    name = "Demo workflow simulator"

    def generate(self, prompt: str, context=None) -> str:
        return "-- DEMO OUTPUT: connect a real model adapter to generate Luau.\n-- " + prompt[:200]

    def train_step(self, batch):
        raise NotImplementedError("Demo mode does not update neural-network weights.")

    def save_checkpoint(self, destination):
        raise NotImplementedError("Demo mode has no model weights to save.")

    def load_checkpoint(self, source):
        raise NotImplementedError("Demo mode has no model weights to restore.")

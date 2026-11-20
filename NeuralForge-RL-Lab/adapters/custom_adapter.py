"""Template for connecting a scratch-built Python AI.

Copy this file and replace ScratchModelStub with your actual model code.
Configure the lab with adapter_spec = adapters.custom_adapter:load_adapter.

Important: train_step must perform a real optimizer/weight update. Returning a
success value without changing model parameters is not learning.
"""
from pathlib import Path
from typing import Any

class ScratchModelAdapter:
    name = "My scratch Luau AI"

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        # TODO: import and initialize your model/tokenizer here.
        # Example: self.model = MyModel.load(config["model_path"])
        self.model = None
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)

    def generate(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        if self.model is None:
            raise RuntimeError("Replace ScratchModelStub with your actual model.")
        # TODO: return the generated Luau text from your model.
        return self.model.generate_text(prompt)

    def supervised_step(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Optional: supervised update using imported reference completions."""
        if self.model is None:
            raise RuntimeError("Connect your model before training.")
        # TODO: calculate next-token / teacher-forced loss on prompt + completion,
        # mask prompt tokens from the loss, and call optimizer.step().
        result = self.model.supervised_learn(batch)
        return {"updates": 1, "loss": float(result["loss"])}

    def train_step(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Connect your model before training.")
        # Each item contains prompt, completion, reward (-1 to +1), and task_id.
        # TODO: calculate policy-gradient / preference loss and call optimizer.step().
        # This must update real trainable parameters.
        result = self.model.learn_from_rewards(batch)
        return {"updates": 1, "loss": float(result["loss"])}

    def save_checkpoint(self, destination: str) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("No model is loaded.")
        self.model.save(destination)
        return {"saved_to": destination}

    def load_checkpoint(self, source: str) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("No model is loaded.")
        self.model.load(source)
        return {"loaded_from": source}

def load_adapter(config: dict[str, Any] | None = None):
    return ScratchModelAdapter(config or {})

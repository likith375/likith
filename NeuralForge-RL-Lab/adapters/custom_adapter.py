"""Domain-neutral adapter template for a user's own Python AI.

Configure NeuralForge with:
    adapters.custom_adapter:load_adapter

Replace the model loading and learning hooks with your model's real APIs.
Never return updates=1 unless optimizer.step() actually changed trainable weights.
The lab rejects a reward train_step that does not explicitly report a real update.
"""
from pathlib import Path
from typing import Any


class ScratchModelAdapter:
    name = "My Scratch AI"

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.model = None
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Load your model/tokenizer/optimizer here, for example:
        # self.model = MyModel.load(config["model_path"])

    def generate(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        if self.model is None:
            raise RuntimeError("Load your actual model in ScratchModelAdapter.__init__ first.")
        return self.model.generate_text(prompt, context=context or {})

    def supervised_step(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Optional teacher-forced update using prompt/completion examples."""
        if self.model is None:
            raise RuntimeError("No model is loaded.")
        # Your implementation must calculate a supervised loss and call optimizer.step().
        result = self.model.supervised_learn(batch)
        return {"updates": int(result["updates"]), "loss": float(result["loss"])}

    def train_step(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Required reward update. Each item has prompt, completion, reward and task_id."""
        if self.model is None:
            raise RuntimeError("No model is loaded.")
        # Implement a policy-gradient, preference, or other documented reward update.
        # This must really update parameters; do not return a fake success value.
        result = self.model.learn_from_rewards(batch)
        return {"updates": int(result["updates"]), "loss": float(result["loss"])}

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

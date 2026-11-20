"""Optional real REINFORCE adapter for local Hugging Face causal language models.

Install requirements-ml.txt and point the Adapter settings at a local model folder.
The model must be a causal language model with a compatible tokenizer.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

class HuggingFaceREINFORCEAdapter:
    name = "Hugging Face causal LM · REINFORCE"

    def __init__(self, config: dict[str, Any]):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install requirements-ml.txt to use the Hugging Face adapter.") from exc
        self.torch = torch
        self.path = str(config.get("model_path", "")).strip()
        if not self.path or not Path(self.path).exists():
            raise ValueError("Model path must point to an existing local model directory.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.path, local_files_only=True)
        self.model.to(self.device)
        self.model.train()
        self.learning_rate = float(config.get("learning_rate", 1e-5))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.max_new_tokens = int(config.get("max_new_tokens", 384))
        self.baseline = 0.0
        self.baseline_momentum = 0.9
        self.max_length = int(getattr(self.model.config, "max_position_embeddings", 2048) or 2048)

    def generate(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        torch = self.torch
        limit = int((context or {}).get("max_new_tokens", self.max_new_tokens))
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                 max_length=max(128, self.max_length - limit)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                **encoded, max_new_tokens=limit, do_sample=True, temperature=0.8,
                top_p=0.92, repetition_penalty=1.08,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id)
        new_tokens = generated[0, encoded["input_ids"].shape[1]:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        self.model.train()
        return result

    def train_step(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        torch = self.torch
        if not batch:
            return {"updates": 0, "loss": None}
        losses = []
        self.optimizer.zero_grad(set_to_none=True)
        for sample in batch:
            prompt = str(sample.get("prompt", ""))
            completion = str(sample.get("completion", ""))
            reward = float(sample.get("reward", 0.0))
            if not completion.strip():
                continue
            prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
            completion_ids = self.tokenizer(completion, add_special_tokens=False)["input_ids"]
            if not completion_ids:
                continue
            # Keep the response intact and trim only the oldest prompt tokens.
            completion_ids = completion_ids[: max(1, self.max_length // 2)]
            prompt_ids = prompt_ids[-max(1, self.max_length - len(completion_ids) - 1):]
            ids = prompt_ids + completion_ids
            if len(ids) < 2:
                continue
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            attention = torch.ones_like(input_ids)
            output = self.model(input_ids=input_ids, attention_mask=attention)
            logits = output.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            log_probs = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            response_start = max(0, len(prompt_ids) - 1)
            response_log_probs = log_probs[:, response_start:]
            if response_log_probs.numel() == 0:
                continue
            advantage = reward - self.baseline
            losses.append(-advantage * response_log_probs.mean())
        if not losses:
            return {"updates": 0, "loss": None, "note": "No trainable response tokens in this batch."}
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        mean_reward = sum(float(x.get("reward", 0.0)) for x in batch) / len(batch)
        self.baseline = self.baseline_momentum * self.baseline + (1-self.baseline_momentum) * mean_reward
        return {"updates": 1, "loss": float(loss.detach().cpu()), "baseline": self.baseline}

    def supervised_step(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Supervised next-token update on imported prompt/completion examples."""
        torch = self.torch
        losses = []
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        for sample in batch:
            prompt = str(sample.get("prompt", ""))
            completion = str(sample.get("completion", ""))
            if not completion.strip():
                continue
            prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
            completion_ids = self.tokenizer(completion, add_special_tokens=False)["input_ids"]
            if not completion_ids:
                continue
            completion_ids = completion_ids[: max(1, self.max_length // 2)]
            prompt_ids = prompt_ids[-max(1, self.max_length - len(completion_ids) - 1):]
            ids = prompt_ids + completion_ids
            labels = ([-100] * len(prompt_ids)) + completion_ids
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            label_ids = torch.tensor([labels], dtype=torch.long, device=self.device)
            attention = torch.ones_like(input_ids)
            output = self.model(input_ids=input_ids, attention_mask=attention, labels=label_ids)
            losses.append(output.loss)
        if not losses:
            return {"updates": 0, "loss": None, "note": "No usable supervised examples in this batch."}
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return {"updates": 1, "loss": float(loss.detach().cpu()), "method": "supervised_reference_examples"}

    def save_checkpoint(self, destination: str) -> dict[str, Any]:
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.torch.save({"optimizer": self.optimizer.state_dict(), "baseline": self.baseline}, path / "neuralforge_training.pt")
        metadata = {"adapter": self.name, "model_path": str(path), "device": str(self.device),
                    "baseline": self.baseline, "learning_rate": self.learning_rate}
        (path / "neuralforge_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    def load_checkpoint(self, source: str) -> dict[str, Any]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(source)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        state_path = path / "neuralforge_training.pt"
        if state_path.exists():
            state = self.torch.load(state_path, map_location=self.device, weights_only=False)
            self.optimizer = self.torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
            self.optimizer.load_state_dict(state["optimizer"])
            self.baseline = float(state.get("baseline", 0.0))
        self.model.train()
        return {"restored": True, "path": str(path), "device": str(self.device)}

def load_adapter(config: dict[str, Any] | None = None):
    return HuggingFaceREINFORCEAdapter(config or {})

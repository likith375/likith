"""Optional local Hugging Face adapter with LoRA-first reward and supervised tuning.

The default path uses parameter-efficient LoRA so a modest GPU can tune a small
fraction of a model's weights. Set use_lora=False only when full fine-tuning is
intended and the machine has enough memory. Generated code is never executed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class HuggingFaceREINFORCEAdapter:
    name = "Hugging Face causal LM · LoRA/REINFORCE"

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
        self.base_model_path = self.path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer needs a pad token or EOS token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.use_lora = str(config.get("use_lora", "true")).lower() not in {"0", "false", "no"}
        # LoRA can keep frozen base weights in FP16 on CUDA to reduce VRAM.
        # Full fine-tuning stays FP32 by default for safer optimizer behavior.
        self.load_dtype = torch.float16 if self.device.type == "cuda" and self.use_lora else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            self.path, local_files_only=True, torch_dtype=self.load_dtype,
            low_cpu_mem_usage=True,
        )
        self.lora_r = int(config.get("lora_r", 8))
        self.lora_alpha = int(config.get("lora_alpha", 16))
        self.lora_dropout = float(config.get("lora_dropout", 0.05))
        target_setting = str(config.get("lora_target_modules", "")).strip()

        if self.use_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as exc:
                raise RuntimeError("LoRA is enabled. Install requirements-ml.txt (includes peft) or disable LoRA.") from exc
            if not 1 <= self.lora_r <= 256 or not 1 <= self.lora_alpha <= 1024:
                raise ValueError("LoRA rank must be 1..256 and alpha 1..1024.")
            if not 0.0 <= self.lora_dropout < 1.0:
                raise ValueError("LoRA dropout must be in the range [0, 1).")
            if target_setting:
                target_modules = [part.strip() for part in target_setting.split(",") if part.strip()]
            else:
                module_names = {name.rsplit(".", 1)[-1] for name, _ in base_model.named_modules() if name}
                candidates = (
                    "q_proj", "k_proj", "v_proj", "o_proj", "query", "key", "value",
                    "query_key_value", "c_attn", "c_proj", "Wqkv", "gate_proj",
                    "up_proj", "down_proj", "dense_h_to_4h", "dense_4h_to_h",
                )
                target_modules = [name for name in candidates if name in module_names]
            if not target_modules:
                raise ValueError(
                    "Could not detect LoRA target modules for this architecture. "
                    "Set lora_target_modules to comma-separated layer names, or disable LoRA."
                )
            lora_config = LoraConfig(
                r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout,
                target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(base_model, lora_config)
            self.lora_target_modules = target_modules
        else:
            self.model = base_model
            self.lora_target_modules = []

        self.model.to(self.device)
        self.model.train()
        self.learning_rate = float(config.get("learning_rate", 1e-5))
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 0.1:
            raise ValueError("Learning rate must be finite and in the range (0, 0.1].")
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("The selected model has no trainable parameters.")
        self.optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate)
        self.max_new_tokens = int(config.get("max_new_tokens", 384))
        self.baseline = 0.0
        self.baseline_momentum = 0.9
        configured_length = (
            getattr(self.model.config, "max_position_embeddings", None)
            or getattr(self.model.config, "n_positions", None)
            or getattr(self.model.config, "max_sequence_length", None)
            or 2048
        )
        self.max_length = max(128, min(int(configured_length), 4096))

    def diagnostics(self) -> dict[str, Any]:
        """Expose model and trainability details without running a training step."""
        total=sum(parameter.numel() for parameter in self.model.parameters())
        trainable=sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        return {
            "device":str(self.device),
            "torch_dtype":str(self.load_dtype).replace("torch.",""),
            "model_type":str(getattr(self.model.config,"model_type","unknown")),
            "total_parameters":total,
            "trainable_parameters":trainable,
            "trainable_percent":round((100.0*trainable/total),4) if total else 0.0,
            "use_lora":self.use_lora,
            "lora_rank":self.lora_r if self.use_lora else None,
            "lora_target_modules":self.lora_target_modules,
            "learning_rate":self.learning_rate,
            "max_sequence_length":self.max_length,
        }

    def _response_tensors(self, prompt: str, completion: str):
        torch = self.torch
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        completion_ids = self.tokenizer(completion, add_special_tokens=False)["input_ids"]
        if not completion_ids:
            return None
        eos_id=self.tokenizer.eos_token_id
        completion_limit=max(1,self.max_length // 2)
        if eos_id is not None:
            completion_ids=completion_ids[:max(0,completion_limit-1)]+[eos_id]
        else:
            completion_ids=completion_ids[:completion_limit]
        prompt_ids = prompt_ids[-max(1, self.max_length - len(completion_ids) - 1):]
        ids = prompt_ids + completion_ids
        if len(ids) < 2:
            return None
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention = torch.ones_like(input_ids)
        response_start = max(0, len(prompt_ids) - 1)
        return input_ids, attention, response_start

    def _has_nonzero_grad(self) -> bool:
        torch = self.torch
        return any(
            parameter.grad is not None and bool(torch.any(parameter.grad.detach() != 0).item())
            for parameter in self.model.parameters() if parameter.requires_grad
        )

    def generate(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        torch = self.torch
        context = context or {}
        limit = int(context.get("max_new_tokens", self.max_new_tokens))
        limit = max(1, min(limit, self.max_new_tokens, self.max_length - 1))
        encoded = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=max(1, self.max_length - limit),
        ).to(self.device)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                generated = self.model.generate(
                    **encoded, max_new_tokens=limit, do_sample=True, temperature=0.8,
                    top_p=0.92, repetition_penalty=1.08,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = generated[0, encoded["input_ids"].shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        finally:
            self.model.train(was_training)

    def train_step(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        torch = self.torch
        if not batch:
            return {"updates": 0, "loss": None, "note": "Empty reward batch."}
        self.model.train()
        losses = []
        rewards = []
        self.optimizer.zero_grad(set_to_none=True)
        for sample in batch:
            prompt = str(sample.get("prompt", ""))
            completion = str(sample.get("completion", ""))
            reward = float(sample.get("reward", 0.0))
            if not math.isfinite(reward):
                raise ValueError("Reward must be finite.")
            reward = max(-1.0, min(1.0, reward))
            if not completion.strip():
                continue
            tensors = self._response_tensors(prompt, completion)
            if tensors is None:
                continue
            input_ids, attention, response_start = tensors
            output = self.model(input_ids=input_ids, attention_mask=attention)
            logits = output.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            log_probs = torch.log_softmax(logits, dim=-1).gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)
            response_log_probs = log_probs[:, response_start:]
            if response_log_probs.numel() == 0:
                continue
            advantage = reward - self.baseline
            losses.append(-advantage * response_log_probs.mean())
            rewards.append(reward)

        if not losses:
            self.optimizer.zero_grad(set_to_none=True)
            return {"updates": 0, "loss": None, "note": "No trainable response tokens in this batch."}
        loss = torch.stack(losses).mean()
        if not bool(torch.isfinite(loss).item()):
            self.optimizer.zero_grad(set_to_none=True)
            raise RuntimeError("REINFORCE loss is non-finite.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0
        )
        if not self._has_nonzero_grad():
            self.optimizer.zero_grad(set_to_none=True)
            return {"updates": 0, "loss": float(loss.detach().cpu()),
                    "note": "Reward produced zero gradients; optimizer was not stepped."}
        self.optimizer.step()
        mean_reward = sum(rewards) / len(rewards)
        self.baseline = self.baseline_momentum * self.baseline + (1 - self.baseline_momentum) * mean_reward
        return {"updates": 1, "loss": float(loss.detach().cpu()), "baseline": self.baseline,
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)}

    def supervised_step(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Supervised next-token update on imported prompt/completion examples."""
        torch = self.torch
        losses = []
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        for sample in batch:
            tensors = self._response_tensors(
                str(sample.get("prompt", "")), str(sample.get("completion", ""))
            )
            if tensors is None:
                continue
            input_ids, attention, response_start = tensors
            # Mask prompt positions but keep every completion token as a target.
            labels = input_ids.clone()
            labels[:, :response_start + 1] = -100
            output = self.model(input_ids=input_ids, attention_mask=attention, labels=labels)
            losses.append(output.loss)
        if not losses:
            return {"updates": 0, "loss": None, "note": "No usable supervised examples in this batch."}
        loss = torch.stack(losses).mean()
        if not bool(torch.isfinite(loss).item()):
            self.optimizer.zero_grad(set_to_none=True)
            raise RuntimeError("Supervised loss is non-finite.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0
        )
        if not self._has_nonzero_grad():
            self.optimizer.zero_grad(set_to_none=True)
            return {"updates": 0, "loss": float(loss.detach().cpu()),
                    "note": "Supervised batch produced zero gradients; optimizer was not stepped."}
        self.optimizer.step()
        return {"updates": 1, "loss": float(loss.detach().cpu()),
                "method": "supervised_reference_examples"}

    def save_checkpoint(self, destination: str) -> dict[str, Any]:
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.torch.save(
            {"optimizer": self.optimizer.state_dict(), "baseline": self.baseline},
            path / "neuralforge_training.pt",
        )
        metadata = {
            "adapter": self.name, "model_path": str(path),
            "base_model_path": self.base_model_path, "device": str(self.device),
            "torch_dtype": str(self.load_dtype).replace("torch.", ""),
            "baseline": self.baseline, "learning_rate": self.learning_rate,
            "use_lora": self.use_lora, "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha, "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
        }
        (path / "neuralforge_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return metadata

    def load_checkpoint(self, source: str) -> dict[str, Any]:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(source)
        metadata_path = path / "neuralforge_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        self.use_lora = bool(metadata.get("use_lora", self.use_lora))
        self.lora_r = int(metadata.get("lora_r", self.lora_r))
        self.lora_alpha = int(metadata.get("lora_alpha", self.lora_alpha))
        self.lora_dropout = float(metadata.get("lora_dropout", self.lora_dropout))
        self.lora_target_modules = list(metadata.get("lora_target_modules", self.lora_target_modules))
        self.baseline = float(metadata.get("baseline", 0.0))
        dtype_name=str(metadata.get("torch_dtype","float16" if self.device.type=="cuda" and self.use_lora else "float32"))
        if dtype_name not in {"float16","bfloat16","float32","float64"}:
            raise ValueError("Checkpoint contains an unsupported torch dtype.")
        self.load_dtype=getattr(self.torch,dtype_name)
        base_path = str(metadata.get("base_model_path") or self.base_model_path)
        if self.use_lora:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Restoring this LoRA checkpoint requires peft from requirements-ml.txt.") from exc
            base = AutoModelForCausalLM.from_pretrained(
                base_path, local_files_only=True, torch_dtype=self.load_dtype,
                low_cpu_mem_usage=True,
            )
            self.model = PeftModel.from_pretrained(base, path, is_trainable=True).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                path, local_files_only=True, torch_dtype=self.load_dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.learning_rate = float(metadata.get("learning_rate", self.learning_rate))
        self.optimizer = self.torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.learning_rate
        )
        state_path = path / "neuralforge_training.pt"
        if state_path.exists():
            state = self.torch.load(state_path, map_location=self.device, weights_only=True)
            self.optimizer.load_state_dict(state["optimizer"])
            self.baseline = float(state.get("baseline", 0.0))
        self.model.train()
        return {"restored": True, "path": str(path), "device": str(self.device),
                "use_lora": self.use_lora}

def load_adapter(config: dict[str, Any] | None = None):
    return HuggingFaceREINFORCEAdapter(config or {})

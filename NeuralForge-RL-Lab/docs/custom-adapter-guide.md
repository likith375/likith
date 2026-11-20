# Connect your own scratch-built AI

NeuralForge does not replace your model. It calls a Python adapter that you provide. The adapter is the bridge between the lab and your AI's actual tokenizer, model, optimizer and checkpoint code.

## 1. Configure the adapter

Place your adapter module inside the project, for example:

```text
NeuralForge-RL-Lab/
  adapters/
    my_ai_adapter.py
```

In Model connection, use:

```text
adapters.my_ai_adapter:load_adapter
```

The factory is called with one settings dictionary. It should return an adapter instance.

## 2. Required methods

### `generate(prompt, context)`

Return a string containing the model's generated response. The context may contain the subject/category, task ID and generation limit.

### `train_step(batch)`

Perform a real reward-based optimizer update. Every batch item contains:

| Key | Meaning |
|---|---|
| `prompt` | The prompt supplied to the model |
| `completion` | The model's generated answer |
| `reward` | Reward between -1 and +1 |
| `task_id` | Source task identifier |

Return a dictionary such as:

```python
{"updates": 1, "loss": 0.42}
```

The `updates` count must be an explicit non-negative integer and must report optimizer steps that actually happened. A zero count is valid when no trainable tokens or non-zero policy gradient exist; it is not counted as learning. The run can continue without claiming a reward update. Do not return a fake success value.

### `save_checkpoint(path)` and `load_checkpoint(path)`

Save and restore the model weights and any state needed to continue training, such as optimizer state, tokenizer configuration, reward baseline and scheduler state. A checkpoint must create at least one non-empty file.

## 3. Optional supervised-learning method

Implement `supervised_step(batch)` if your model can learn from reference answers. Each item contains:

- `prompt`: the question/instruction
- `completion`: the reference answer or verified correction
- `category`: the subject/domain

Return `{"updates": 1, "loss": 0.25}` only after a real optimizer update. Return `{"updates": 0, "loss": None}` if no usable examples were available.

NeuralForge sends imported train-split examples and evidence-verified mistake corrections to this hook. Unverified corrections are never sent as supervised targets.

## 4. Optional domain evaluator

Implement:

```python
def evaluate(self, prompt, output, expected, domain):
    return {
        "score": 82,             # 0..100
        "reward": 0.64,          # -1..+1
        "checks": {"rubric": True},
        "evaluator": "my-domain-checker",
        "warning": "Explain what this score does and does not prove."
    }
```

Use real checks appropriate to the subject. For example, use syntax parsing or isolated tests for code, symbolic equivalence for mathematics, source-backed fact checks for science, and an explicit rubric for writing. Never execute untrusted generated code inside the NeuralForge API process.

## 5. Verify before training

1. Save the adapter configuration.
2. Run **Test model connection**. This only generates one sample; it does not update weights.
3. Check adapter diagnostics and confirm that the intended model and trainable parameters are loaded.
4. Import data and create tasks.
5. Run a short training session first.
6. Confirm that the run reports real optimizer updates and that both automatic checkpoints contain files.

A model can generate text successfully but still have no trainable parameters. Generation success alone does not prove that learning is connected.

## 6. Feedback loop

Rate a saved attempt from 1 to 5. Ratings map to reward values -1, -0.5, 0, 0.5 and +1. The next training run replays pending ratings through `train_step`. Feedback is marked consumed only after the adapter reports a real reward update and the updated model has been checkpointed. A zero-gradient feedback batch remains pending.

If a reviewer supplies a correction, NeuralForge stores it as an unverified mistake candidate. Review its evidence and explicitly verify it before it can enter `supervised_step`.

## 7. Important trust boundary

The adapter module is imported and executed as trusted Python code in the NeuralForge process. Only configure adapter code that you wrote or reviewed. Uploaded datasets and generated code are treated as data and are not executed by the lab.


## 8. Failure and restore safety

Before training, NeuralForge saves a pre-training checkpoint. If an optimizer operation or final checkpoint fails, it attempts to reload that snapshot and records whether rollback succeeded. Before a manual restore, it saves a safety copy of the currently loaded model; if the selected checkpoint cannot load, it attempts to restore the previous model. Keep these checkpoints until you have validated the new model.

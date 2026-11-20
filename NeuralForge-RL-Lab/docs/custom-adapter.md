# Connect your own scratch-built AI

NeuralForge does not replace your model. It calls your model through a small Python adapter. The adapter is the bridge between your model's existing Python code and the lab's training, evaluation and checkpoint workflows.

## 1. Configure the adapter

In the dashboard, open **Model connection** and set:

```text
adapters.custom_adapter:load_adapter
```

The factory receives one dictionary containing the saved settings, including `model_path`, `learning_rate`, `batch_size`, and LoRA options. The template currently raises a clear error until you connect your actual model.

## 2. Required model methods

Your adapter object must implement these four methods:

| Method | Input | Expected behavior |
|---|---|---|
| `generate(prompt, context)` | Task prompt and metadata | Return the model's generated text as a string |
| `train_step(batch)` | Prompt, generated completion, reward and task ID | Perform a real reward/preference/ policy-gradient update |
| `save_checkpoint(destination)` | A folder path | Save actual model weights and any state needed to resume training |
| `load_checkpoint(source)` | A saved folder path | Restore the model and training state |

It may also implement:

- `supervised_step(batch)`: optional supervised learning from train-split examples and verified corrections.
- `evaluate(prompt, output, expected, domain)`: optional domain-specific evaluator. Return a dictionary with `score` from 0–100, `reward` from -1 to 1, and a `checks` dictionary.

## 3. Batch formats

A reward batch contains items like:

```python
{
    "prompt": "Solve this physics problem...",
    "completion": "The model's generated answer...",
    "reward": 0.5,
    "task_id": "..."
}
```

A supervised batch contains:

```python
{
    "prompt": "Explain photosynthesis",
    "completion": "The reviewed reference answer...",
    "category": "biology"
}
```

A rating of 5 maps to reward +1, 4 to +0.5, 3 to 0, 2 to -0.5, and 1 to -1. Neutral rewards do not create a policy-gradient update by themselves.

## 4. Report updates honestly

Both training methods must return a dictionary with an explicit non-negative integer `updates` count. For example:

```python
return {"updates": optimizer_steps, "loss": average_loss}
```

- Return `updates: 1` only after a real optimizer step has happened.
- Return `updates: 0` when there were no trainable tokens or no gradient. This is valid and is not counted as learning.
- Never return a hard-coded update count to make a run appear successful.
- Loss must be finite when supplied.

The lab checks the report format, but it cannot independently prove that arbitrary custom adapter code changed the model's weights. Only connect adapter code you own and trust.

## 5. Checkpoints

`save_checkpoint` must write at least one non-empty file under the destination. Save everything needed to restore the model, including optimizer state and reward baseline when applicable. `load_checkpoint` must genuinely restore it. NeuralForge creates a pre-training backup, rolls back after training failures when possible, and creates a safety backup before manual restores.

## 6. Evaluation

The default evaluator is deliberately conservative. It does not execute generated code or prove scientific/factual correctness. Implement `evaluate` when you have a trusted evaluator for your subject—for example, a deterministic math checker, a domain-specific rubric, or an isolated test runner. Do not run untrusted generated code inside the NeuralForge API process.

## 7. Test before training

1. Save your adapter module.
2. Configure its module path in Model connection.
3. Click **Test model connection**. This tests generation only and does not change weights.
4. Run the training preflight check.
5. Start with a small episode count and inspect the run report and validation benchmark.
6. Keep the automatic pre-training checkpoint until you have confirmed the new model works.

See `adapters/custom_adapter.py` for the minimal implementation template and `docs/rl-training.md` for the training lifecycle.

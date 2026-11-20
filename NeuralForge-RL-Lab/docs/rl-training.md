# Training and reward design

## Combined learning loop

The training controller uses an adaptive task curriculum and combines two kinds of learning when the selected adapter supports both:

Before each episode, the curriculum weights tasks by recent performance (the last 20 attempts); tasks with lower recent scores receive more practice, while every enabled task retains a non-zero selection chance.

1. **Supervised reference learning:** samples are drawn only from the dataset rows labelled `train`. The optional `supervised_step(batch)` hook teaches the model to predict the reference completion with teacher-forced next-token loss. Prompt tokens are masked out of the loss.
2. **Reward-based policy learning:** the model generates a response to a task. The evaluator assigns a scalar reward. The adapter receives prompt, completion, reward and task ID, then performs a policy update through `train_step(batch)`.

The built-in Hugging Face adapter uses LoRA by default so only a small adapter parameter set is trainable. Rank, alpha, dropout and target modules are configurable. Full fine-tuning is available by disabling LoRA, but requires substantially more memory. The adapter implements both supervised next-token updates and a minimal REINFORCE-style policy-gradient baseline. The supervised and policy updates both count toward the run's update metric.

The custom adapter may implement `supervised_step(batch)` as an optional hook. If it is absent, the lab still performs reward-based updates through `train_step(batch)`. Reward updates must return a dictionary with an explicit positive integer `updates` count. The lab no longer assumes that a missing return value means learning happened. Non-finite losses are rejected. Supervised updates may explicitly report zero when no examples were usable.

## REINFORCE baseline

The optional Hugging Face adapter:

1. Samples a completion from the current policy.
2. Computes a scalar reward from the configured evaluator.
3. Subtracts an exponential moving-average reward baseline.
4. Computes the mean log probability of sampled completion tokens.
5. Minimizes `-advantage * log_probability`.
6. Clips gradients and performs an AdamW optimizer step.

This is a minimal REINFORCE-style policy-gradient baseline. It is not PPO, RLHF with a learned reward model, or a guarantee of useful code generation.

## How imported examples are used

- Dataset examples with split `train` are used for supervised updates if the adapter implements `supervised_step`.
- Relevant examples are retrieved into task prompts using subject matching and lightweight keyword-overlap ranking rather than simply taking the newest records.
- Verified mistake corrections are filtered by the task's subject so unrelated corrections are not injected into the prompt.
- Validation and test rows are not used for gradient updates or prompt examples.
- Human-reviewed corrections are added to future prompts only when their latest review event is `verified`.
- Task attempts are stored immutably with scores, rewards and evaluator metadata; task performance history drives the adaptive curriculum.

Keep train, validation and test examples distinct. Do not train on test examples or feed test answers into prompts. Remove duplicated or near-duplicated examples across splits.

## Evaluation contract

The built-in evaluator is deliberately domain-neutral. It checks basic text properties such as non-empty output, placeholder markers, delimiter balance, and optional reference matching. If no expected reference is supplied, it returns a neutral score of 50 and reward 0 rather than claiming success. With an expected reference, exact reference matching is required for a high score. It is a **text heuristic**, not a fact checker, code parser, type checker, or runtime test. It can be gamed by outputting plausible-looking text. Treat it as a smoke test only.

For stronger evaluation, a trusted adapter may implement `evaluate(prompt, output, expected, domain)` and return a dictionary with `score` from 0 to 100 and `reward` from -1 to 1, plus optional `checks`, `evaluator`, and `warning` fields. This allows different evaluators for Python, mathematics, science, writing, engineering, and other subjects. Keep evaluation deterministic where possible, document its evidence, and never execute untrusted code in the API process.

Python-domain tasks also receive a built-in AST syntax check when the response looks like Python code. This parses syntax without importing or executing the generated program; a syntax pass is not proof of runtime correctness.

## Evaluator examples

- Python: run static analysis or unit tests inside a disposable container with strict CPU, memory, file-system and time limits.
- Mathematics: use symbolic/numeric equivalence checks and tolerances where appropriate.
- Science: compare claims against curated, cited reference material; do not treat a language-model judge as ground truth.
- Writing: evaluate a transparent rubric such as relevance, structure, clarity and requested constraints.
- Game scripting: parse the target language and run isolated tests with mocked engine APIs where possible.

Never run untrusted generated code in the main API process.

## Training safety

NeuralForge saves an automatic restorable checkpoint before each run; the adapter must actually create the checkpoint directory or training will not begin. Keep these backups and also maintain an external copy of your original model. Use a small learning rate and small batches. Save a checkpoint before training, run a short experiment, then compare validation results. Stop if validation quality drops, the loss becomes non-finite, or the model starts repeating or emitting malformed code.

## Long training sessions

A run can be limited by a maximum episode count and an optional time budget of up to 24 hours. The time budget is an upper bound, not a promise that the run will consume the full duration. Set a sufficiently high episode cap if you want the clock to be the limiting condition. The Stop button requests a graceful stop after the current batch. The computer or Codespace must remain available; closing a browser tab alone does not stop the server-side worker, but a sleeping or stopped Codespace will interrupt the process.


## Human-rated replay learning

Every saved task attempt has a stable attempt ID. Submit reviewer feedback with:

```http
POST /api/attempts/{attempt_id}/feedback
Content-Type: application/json

{"rating": 5, "note": "Correct and clear", "evidence": "Checked against the supplied reference"}
```

Ratings map to reward: 1 → -1, 2 → -0.5, 3 → 0, 4 → 0.5, 5 → 1. A neutral 3 is stored for history but is not placed in the reward replay queue. The next training run replays up to 64 pending rated attempts through the adapter's reward-learning hook before generating new samples. Feedback is marked consumed only after a successful, validated positive update count and a durable post-training checkpoint. If the adapter fails, the feedback remains pending. If a reviewer supplies a correction, it is stored as an unverified mistake candidate; it must pass the separate evidence-review workflow before it can enter supervised training. This makes reviewer feedback a durable learning signal rather than a dashboard-only comment.

Use `GET /api/attempts` to inspect attempts and `GET /api/feedback?pending_only=true` to inspect the replay queue. Task deletion preserves attempt and feedback records.

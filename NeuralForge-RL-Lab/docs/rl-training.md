# Training and reward design

## Combined learning loop

The training controller combines two kinds of learning when the selected adapter supports both:

1. **Supervised reference learning:** samples are drawn only from the dataset rows labelled `train`. The optional `supervised_step(batch)` hook teaches the model to predict the reference completion with teacher-forced next-token loss. Prompt tokens are masked out of the loss.
2. **Reward-based policy learning:** the model generates a response to a task. The evaluator assigns a scalar reward. The adapter receives prompt, completion, reward and task ID, then performs a policy update through `train_step(batch)`.

The built-in Hugging Face adapter implements both supervised next-token updates and a minimal REINFORCE-style policy-gradient baseline. The supervised and policy updates both count toward the run's update metric.

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
- Relevant examples are also retrieved into task prompts as few-shot context.
- Validation and test rows are not used for gradient updates or prompt examples.
- Human-reviewed corrections are added to future prompts only when their latest review event is `verified`.

Keep train, validation and test examples distinct. Do not train on test examples or feed test answers into prompts. Remove duplicated or near-duplicated examples across splits.

## Evaluation contract

The built-in evaluator is deliberately domain-neutral. It checks basic text properties such as non-empty output, placeholder markers, delimiter balance, and optional reference matching. It is a **text heuristic**, not a fact checker, code parser, type checker, or runtime test. It can be gamed by outputting plausible-looking text. Treat it as a smoke test only.

For stronger evaluation, a trusted adapter may implement `evaluate(prompt, output, expected, domain)` and return a dictionary with `score` from 0 to 100 and `reward` from -1 to 1, plus optional `checks`, `evaluator`, and `warning` fields. This allows different evaluators for Python, mathematics, science, writing, engineering, and other subjects. Keep evaluation deterministic where possible, document its evidence, and never execute untrusted code in the API process.

## Evaluator examples

- Python: run static analysis or unit tests inside a disposable container with strict CPU, memory, file-system and time limits.
- Mathematics: use symbolic/numeric equivalence checks and tolerances where appropriate.
- Science: compare claims against curated, cited reference material; do not treat a language-model judge as ground truth.
- Writing: evaluate a transparent rubric such as relevance, structure, clarity and requested constraints.
- Game scripting: parse the target language and run isolated tests with mocked engine APIs where possible.

Never run untrusted generated code in the main API process.

## Training safety

Start with a copy of your model. Use a small learning rate and small batches. Save a checkpoint before training, run a short experiment, then compare validation results. Stop if validation quality drops, the loss becomes non-finite, or the model starts repeating or emitting malformed code.

## Long training sessions

A run can be limited by a maximum episode count and an optional time budget of up to 24 hours. The time budget is an upper bound, not a promise that the run will consume the full duration. Set a sufficiently high episode cap if you want the clock to be the limiting condition. The Stop button requests a graceful stop after the current batch. The computer or Codespace must remain available; closing a browser tab alone does not stop the server-side worker, but a sleeping or stopped Codespace will interrupt the process.

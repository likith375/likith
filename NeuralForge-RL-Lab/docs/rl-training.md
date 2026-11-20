# Training and reward design

## Combined learning loop

The training controller combines two kinds of learning when the selected adapter supports both:

1. **Supervised reference learning:** samples are drawn only from the dataset rows labelled `train`. The optional `supervised_step(batch)` hook teaches the model to predict the reference completion with teacher-forced next-token loss. Prompt tokens are masked out of the loss.
2. **Reward-based policy learning:** the model generates a response to a task. The evaluator assigns a scalar reward. The adapter receives prompt, completion, reward and task ID, then performs a policy update through `train_step(batch)`.

The built-in Hugging Face adapter implements both supervised next-token updates and a minimal REINFORCE-style policy-gradient baseline. The supervised and policy updates both count toward the run's update metric.

The custom adapter may implement `supervised_step(batch)` as an optional hook. If it is absent, the lab still performs reward-based updates through `train_step(batch)`.

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

## Current reward function

The built-in evaluator checks whether output is non-empty, contains common Luau constructs, avoids placeholder markers, has roughly balanced delimiters, and optionally contains a supplied expected-output string.

The score is a **text heuristic**, not a Luau parser, type checker, Roblox API validator or runtime test. It can be gamed by outputting code-shaped text. Treat it as a smoke test only.

## Recommended next evaluator

For serious Roblox Luau training, build a separate isolated evaluator that can:

- Parse Luau syntax with a real Luau parser.
- Check required identifiers, function signatures and structural properties.
- Run unit tests against mocked Roblox services where possible.
- Run integration tests in a disposable Roblox Studio environment when appropriate.
- Enforce CPU, memory, file-system and time limits.
- Return structured test results, not just a single score.

Never run untrusted generated code in the main API process.

## Training safety

Start with a copy of your model. Use a small learning rate and small batches. Save a checkpoint before training, run a short experiment, then compare validation results. Stop if validation quality drops, the loss becomes non-finite, or the model starts repeating or emitting malformed code.

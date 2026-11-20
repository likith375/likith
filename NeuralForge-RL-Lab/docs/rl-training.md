# Training and reward design

## Current learning algorithm

The optional Hugging Face adapter is a minimal REINFORCE-style policy-gradient baseline:

1. Sample a completion from the current policy.
2. Compute a scalar reward from the configured evaluator.
3. Subtract an exponential moving-average reward baseline.
4. Compute the mean log probability of sampled completion tokens.
5. Minimize `-advantage * log_probability`.
6. Clip gradients and perform an AdamW optimizer step.

This is genuine parameter optimization when the adapter successfully runs its optimizer. It is not PPO, RLHF with a learned reward model, or a guarantee of useful code generation.

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

## Data quality rules

- Keep train, validation and test examples distinct.
- Do not train on test examples or feed test answers into prompts.
- Remove duplicated or near-duplicated examples across splits.
- Keep API names and behavior grounded in the Roblox/Luau version you target.
- Label the evidence for every correction.
- Human-review corrections before marking them verified.
- Prefer multiple tests and behavioral assertions over exact string matching.

## Training safety

Start with a copy of your model. Use a small learning rate and small batches. Save a checkpoint before training, run a short experiment, then compare validation results. Stop if validation quality drops, the loss becomes non-finite, or the model starts repeating or emitting malformed code.

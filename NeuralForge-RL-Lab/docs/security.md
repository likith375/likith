# Security notes

- Uploaded Luau examples are treated as text and are not executed.
- The built-in evaluator only performs static text checks.
- The custom model adapter is Python code imported into the API process. It is trusted code and can access the process environment and files. Only configure code you wrote or reviewed.
- Keep the application bound to localhost or a private Codespace. Do not expose it to the public internet without authentication, authorization, rate limits, and a security review.
- Do not store API keys or secrets in dataset examples, prompts, logs or source control.
- Use a dedicated, low-privilege operating-system account for local training.
- Use a separate disposable sandbox for any future Luau execution evaluator. Do not execute generated scripts inside the API server.
- Review and back up model checkpoints before training. A bad reward function can degrade model behavior.

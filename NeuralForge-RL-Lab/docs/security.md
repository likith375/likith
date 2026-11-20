# Security notes

- Uploaded documents, code and datasets are treated as data and are never executed by the API server.
- The built-in evaluator performs static checks and a safe Python AST syntax parse; it does not execute generated code or verify factual correctness.
- The custom model adapter is Python code imported into the API process. It is trusted code and can access the process environment and files. Only configure code you wrote or reviewed.
- Keep the application bound to localhost or a private Codespace. Do not expose it to the public internet without authentication, authorization, rate limits, and a security review.
- Do not store API keys or secrets in dataset examples, prompts, logs or source control.
- Use a dedicated, low-privilege operating-system account for local training.
- Any future code execution evaluator must run in a separate disposable sandbox with strict CPU, memory, filesystem and time limits. Do not execute generated code inside the API server.
- Review and back up model checkpoints before training. A bad reward function can degrade model behavior.


Office files (DOCX, XLSX and PPTX) are checked as ZIP containers before parsing. NeuralForge rejects malformed archives, more than 5,000 entries, expanded content over 100 MB, individual entries over 30 MB, and suspicious compression ratios. The upload limit is 15 MB compressed; these checks reduce, but do not eliminate, parser risk.


Run NeuralForge as a single Uvicorn worker/process. Its training scheduler and model-operation locks are in-process locks; multiple workers would not share them and could attempt concurrent runs against the same local model/database. The supplied Windows and Linux launchers use one worker.

# NeuralForge RL Lab

**A domain-independent, local-first training workbench for connecting your own AI and teaching it across subjects.**

NeuralForge is the training software, not a prebuilt AI assistant. You bring a model adapter; the lab organizes material from programming, science, mathematics, writing, engineering, languages, and other fields; generates tasks; scores attempts; preserves mistakes; supplies reviewed examples; orchestrates training updates; and manages checkpoints.

## What is included

- FastAPI backend, SQLite persistence, and a responsive browser dashboard.
- JSONL, JSON, CSV, Jupyter notebooks, source code, markup, configuration, text, PDF (selectable text), and DOCX import, filtering, and JSONL export.
- Train / validation / test split support.
- Task studio with difficulty, category, expected behavior, and enable/disable controls.
- Append-only mistake records with a separate review-event history. Only verified corrections are added to future task prompts.
- Domain-neutral static response heuristics with explicit warnings, a safe Python AST syntax check, and an optional trusted adapter evaluation hook for subject-specific verification.
- Adapter dry-run testing, one-shot model generation, system diagnostics, and dataset/task deletion APIs.
- Background training runs, live run polling, reward and loss metrics.
- Optional training time budget up to 24 hours, episode cap, and graceful stop control.
- Combined supervised fine-tuning on imported train-split examples plus reward-based updates when the adapter supports it.
- A watched `data/inbox/` folder for dropping supported source files, notebooks, datasets and documentation from any subject.
- Dedicated `data/mistakes/mistakes.jsonl` append-only audit log.
- Held-out validation and test benchmarks with stored per-example results.
- A model-adapter contract for your own scratch-built Python AI.
- An optional Hugging Face causal-LM adapter with LoRA-first parameter-efficient tuning, supervised updates, and REINFORCE-style policy-gradient updates.
- Immutable per-task attempt history and an adaptive curriculum that prioritizes less-mastered tasks.
- Model checkpoint save and restore through the adapter.
- API documentation at `/docs`.
- Automated tests and GitHub Actions workflow.
- GitHub Codespaces development-container setup.

## Run in GitHub Codespaces

1. Open this repository on GitHub.
2. Open the `NeuralForge-RL-Lab` folder, then choose **Code → Codespaces → Create codespace** if Codespaces is available to your account.
3. Wait for dependencies to install and the development server to start.
4. Open the forwarded port **8000**.

If you open the repository root instead, the root `.devcontainer/devcontainer.json` also configures the app folder and port.

Codespaces quotas and billing depend on your GitHub plan and settings.

## Run on your own computer

Python 3.10+ is required.

```bash
cd NeuralForge-RL-Lab
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000.

## Connect your own scratch-built AI

1. Put your model integration in a Python module inside the project.
2. Implement the adapter methods shown in `adapters/custom_adapter.py`.
3. In **Model connection**, set the module factory, for example:

   `adapters.custom_adapter:load_adapter`

4. Save settings and verify the connection status.
5. Create tasks, import data, and start a training run.

The required adapter contract:

- `generate(prompt, context)` returns an answer or artifact appropriate to the selected subject.
- `train_step(batch)` performs an actual optimizer update and returns update/loss information.
- `save_checkpoint(path)` saves model weights and training state.
- `load_checkpoint(path)` restores them.

A successful-looking return value is not enough: `train_step` must really update trainable parameters and return an explicit positive integer `updates` count. Missing update counts, zero reward updates, and non-finite losses fail the run instead of being counted as learning.

## Optional built-in Hugging Face adapter

Install its optional dependencies:

```bash
pip install -r requirements-ml.txt
```

Download or prepare a **local** causal language model directory that is compatible with Transformers. In Model connection, set:

- Adapter module: `backend.hf_adapter:load_adapter`
- Local model folder: path to your local model directory

The built-in adapter uses LoRA by default (configurable rank, alpha, dropout and target modules) to tune a small fraction of model parameters; disable it only when you intentionally want full fine-tuning and have enough memory. It combines supervised reference learning with sampled-response REINFORCE updates, independent of any single programming language. It is an experimental baseline for learning workflows, not a state-of-the-art coding model. It does not download model weights or call paid APIs.

## Supported upload formats

JSONL is recommended. Each line is one JSON object:

```json
{"prompt":"Create a part","category":"instances","content":"local part = Instance.new(\"Part\")","split":"train"}
```

Accepted content fields: `content`, `code`, `completion`, `output`, `response`, `answer`, `target`, `assistant`, or `text`. Prompt aliases include `prompt`, `instruction`, `question`, `input`, `user`, `title`, and `name`. OpenAI `messages` and ShareGPT `conversations` rows are also normalized. Plain-text source and documentation files are imported as reference examples; Jupyter notebooks are split into cells. PDF files import selectable text page by page (scanned-image OCR is not included); DOCX imports paragraphs and tables.

Accepted prompt fields: `prompt`, `instruction`, or `title`.

If `split` is omitted, structured datasets are split across train/validation/test rows. A single document is kept entirely in train to prevent its chunks leaking into validation/test. Long prose documents are chunked on paragraph boundaries. For serious experiments, provide your own splits and keep near-duplicate examples in the same split to avoid leakage.

The optional Luau starter corpus remains available as one subject-specific example; it is not required for other subjects.

## Reinforcement-learning loop

1. The scheduler selects an enabled task.
2. The lab builds a prompt with a small set of human-verified corrections.
3. The connected adapter generates a candidate completion.
4. The evaluator computes transparent static checks and a reward.
5. Weak attempts are preserved in mistake memory for human review.
6. The adapter receives prompt, completion, and reward samples. Imported train-split examples also go to the optional supervised-training hook.
7. The adapter performs a real update; run metrics record updates and loss.
8. The model can be checkpointed and restored.

Read `docs/rl-training.md` before interpreting the metrics.

## Important limitations

- **Evaluation depends on the configured evaluator:** the default evaluator checks only basic text structure and reference matching; without an answer key it gives a neutral score rather than inventing a pass. It cannot prove code works or factual claims are true. Python-domain tasks also receive a safe AST syntax check (without execution). Add a domain-specific `evaluate(prompt, output, expected, domain)` method to a trusted adapter for stronger checks. Never treat a generic heuristic score as a runtime pass.
- **Reward quality matters:** the built-in static reward is a baseline. Better results require a carefully designed evaluator and reviewed task/reference data.
- **REINFORCE is experimental:** it may be unstable, especially with small datasets, small models, or noisy rewards. Keep backups and compare held-out validation/test results.
- **Local resources:** training speed and memory use depend on your model size, hardware, and batch settings.
- **No model included:** the project does not bundle or secretly train a foundation model. You must connect one.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Security and privacy

The server does not execute uploaded Luau or arbitrary uploaded Python. Only load adapter code that you own and trust: a configured Python adapter executes inside the server process and therefore has the same permissions as that process. Do not expose the server publicly without authentication, network restrictions, and a security review. See `docs/security.md`.

## License

No license has been added yet. Choose a license before redistributing or publishing this project.

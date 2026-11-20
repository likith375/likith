# Troubleshooting

## The dashboard does not open

1. Run `run_windows.bat` on Windows or `./run_linux.sh` on Linux/macOS.
2. Keep the terminal window open while using the lab.
3. Open `http://127.0.0.1:8000`.
4. If the port is already in use, stop the older NeuralForge process or change the port in the launcher.
5. Check `GET /api/health` and `GET /api/system/diagnostics` if the API is reachable.

## Python dependencies fail to install

- Use Python 3.10 or newer.
- Activate the project's virtual environment before installing packages.
- Run `python -m pip install -r requirements.txt`.
- For local Transformers training, also install `requirements-ml.txt`.
- On a machine with a CUDA GPU, use the PyTorch install command appropriate for that CUDA runtime. Do not replace a working CUDA-enabled PyTorch build with a CPU-only wheel.

## Adapter module cannot be imported

- The module path must be a Python import path, not a file path.
- Example: `adapters.my_ai_adapter:load_adapter`.
- The file must be inside the project and the factory function must exist.
- The factory receives a settings dictionary and must return an adapter object.
- Only connect Python code you trust; adapter code runs inside the API process.

## Generation works but training reports zero updates

- Generation alone does not prove the model can learn.
- Confirm `train_step(batch)` really computes a loss, calls `backward()`, and calls `optimizer.step()`.
- Return a dictionary with a positive integer `updates` count and a finite `loss`.
- If the evaluator gives reward zero, NeuralForge records the attempt but skips it in the reward hook. Add a reference answer, a domain evaluator, or human ratings to provide a learning signal.
- Implement `supervised_step(batch)` if your model learns from prompt/reference pairs.

## CUDA out-of-memory

- Keep LoRA enabled; the built-in adapter loads frozen CUDA base weights in FP16.
- Reduce the model size or sequence length in the model configuration. NeuralForge caps training sequences at 4096 tokens.
- Reduce batch size to 1.
- Close other GPU applications.
- Full fine-tuning uses FP32 by default and needs substantially more memory than LoRA.

## Training fails to save a checkpoint

- Your adapter must write at least one non-empty file inside the destination directory.
- Save model weights and the state required to continue training, such as optimizer state and reward baseline.
- The lab saves a pre-training backup and a post-training checkpoint after real updates.
- If the post-training save fails, the run is marked failed and rated feedback remains pending.

## PDF or document import fails

- Install all dependencies from `requirements.txt`.
- Scanned-image PDFs need OCR; only selectable PDF text is supported.
- Password-protected or damaged files may not parse.
- XLSX question/answer columns are imported as paired examples. PPTX slide text and tables are imported slide by slide.

## Dataset readiness warnings

- Check `GET /api/datasets/stats` for empty examples, duplicates and split conflicts.
- Use `PATCH /api/datasets/{dataset_id}/split` to repair a split. Set `apply_to_exact_duplicates: true` to move the whole identical-content group together.
- Moving a generated task's source out of the train split disables that task.
- Keep near-duplicate examples in the same split; exact duplicate detection cannot detect semantic similarity.

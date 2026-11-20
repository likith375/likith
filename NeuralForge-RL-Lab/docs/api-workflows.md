# API workflow examples

The interactive API reference is available at `http://127.0.0.1:8000/docs` after the server starts.

## 1. Import a dataset

```bash
curl -X POST http://127.0.0.1:8000/api/datasets/import \
  -F "file=@./my-training-data.jsonl"
```

Inspect the split and quality summary:

```bash
curl http://127.0.0.1:8000/api/datasets/stats
curl http://127.0.0.1:8000/api/training/preview
```

Generate tasks from train-split examples:

```bash
curl -X POST "http://127.0.0.1:8000/api/tasks/generate-from-dataset?limit=200"
```

## 2. Connect your model

For a custom Python adapter:

```http
PUT /api/adapter/config
Content-Type: application/json

{
  "adapter_spec": "adapters.my_ai_adapter:load_adapter",
  "model_path": "",
  "learning_rate": 0.00001,
  "max_new_tokens": 384,
  "batch_size": 2
}
```

For the built-in local Transformers adapter, leave `adapter_spec` empty and set `model_path` to the local model directory. Check the adapter guide for supported settings.

Test generation without changing weights:

```bash
curl -X POST http://127.0.0.1:8000/api/adapter/test
```

Inspect model details:

```bash
curl http://127.0.0.1:8000/api/adapter/status
```

## 3. Run training

```bash
curl -X POST http://127.0.0.1:8000/api/runs/start \
  -H "Content-Type: application/json" \
  -d '{"episodes": 100, "batch_size": 2, "duration_hours": 0}'
```

The response includes a run ID. Poll it:

```bash
curl http://127.0.0.1:8000/api/runs/RUN_ID
curl http://127.0.0.1:8000/api/runs/RUN_ID/report
```

Stop an active run:

```bash
curl -X POST http://127.0.0.1:8000/api/runs/RUN_ID/stop
```

A run writes a pre-training backup and, after real optimizer updates, a post-training checkpoint. The report includes attempt metrics, feedback usage and checkpoint records.

## 4. Rate a model attempt

Use the `attempt_id` returned by `POST /api/tasks/{task_id}/try`:

```http
POST /api/attempts/ATTEMPT_ID/feedback
Content-Type: application/json

{
  "rating": 5,
  "note": "Correct and clear",
  "evidence": "Checked against the supplied reference"
}
```

A rating of 3 is neutral and is stored for history but not queued for reward learning. Ratings 1, 2, 4 and 5 become reward samples. If you provide a correction, it must pass the separate evidence-review workflow before supervised training can use it.

Inspect or export feedback:

```bash
curl "http://127.0.0.1:8000/api/feedback?pending_only=true"
curl http://127.0.0.1:8000/api/feedback/export
```

## 5. Benchmark without training

```bash
curl -X POST "http://127.0.0.1:8000/api/benchmarks/run?split=validation&limit=100"
curl http://127.0.0.1:8000/api/benchmarks
```

Benchmarking does not call the optimizer. It evaluates the current model against the selected validation or test split.

## 6. Restore a checkpoint

List available checkpoints:

```bash
curl http://127.0.0.1:8000/api/checkpoints
```

Restore one:

```http
POST /api/checkpoints/CHECKPOINT_ID/restore
```

Restoring a checkpoint changes the connected model state. Do not restore a checkpoint while another operation is using the model; model operations are serialized by the service.

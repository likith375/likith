from fastapi.testclient import TestClient
from pathlib import Path
from backend import main

def setup_client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DB", tmp_path / "test.sqlite3")
    monkeypatch.setattr(main, "MISTAKE_LOG", tmp_path / "mistakes.jsonl")
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    inbox = tmp_path / "inbox"
    examples = tmp_path / "examples"
    inbox.mkdir()
    examples.mkdir()
    monkeypatch.setattr(main, "DATA_INBOX", inbox)
    monkeypatch.setattr(main, "DATA_EXAMPLES_DIR", examples)
    main.init_db()
    return TestClient(main.app)

def test_health_and_seed_data(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    assert client.get("/health").json()["ok"] is True
    assert len(client.get("/api/datasets").json()) >= 4
    assert len(client.get("/api/tasks").json()) >= 1

def test_static_evaluator_is_explicitly_heuristic():
    result = main.static_evaluate("local coins = 10", "local coins = 10")
    assert result["score"] == 100
    assert "prove factual correctness" in result["warning"]

def test_task_creation(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    response = client.post("/api/tasks", json={
        "title": "Make a door", "prompt": "Create a door that opens",
        "expected": "local door = Instance.new(\"Part\")",
        "difficulty": "medium", "category": "instances"
    })
    assert response.status_code == 200
    assert any(t["title"] == "Make a door" for t in client.get("/api/tasks").json())

def test_mistake_review_is_append_only(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    created = client.post("/api/mistakes", json={
        "prompt": "Create a part", "output": "wrong output",
        "category": "failed_tasks", "correction": "local p = Instance.new(\"Part\")"
    }).json()
    before = client.get("/api/mistakes").json()[0]
    reviewed = client.patch(f"/api/mistakes/{created['id']}", json={
        "verification": "verified", "note": "Checked manually", "evidence": "Confirmed against Luau API reference"
    })
    assert reviewed.status_code == 200
    after = client.get("/api/mistakes?verification=verified").json()[0]
    assert before["output"] == after["output"] == "wrong output"
    assert after["verification"] == "verified"
    assert after["evidence"] == "Confirmed against Luau API reference"
    assert main.verified_examples()[0]["correction"] == 'local p = Instance.new("Part")'
    assert client.get("/api/overview").json()["verified"] == 1
    with main.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM mistake_events WHERE mistake_id=?", (created["id"],)).fetchone()[0] == 1

def test_failed_run_without_adapter_is_not_reported_as_success(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    # Directly exercise the worker to avoid timing assumptions in a background task.
    run_id = "test-run"
    with main.connect() as db:
        db.execute("INSERT INTO runs(id,status,episodes,created_at) VALUES(?,?,?,?)",
                   (run_id, "queued", 1, main.now()))
    main.execute_training(run_id, {"episodes": 1, "batch_size": 1})
    run = client.get(f"/api/runs/{run_id}").json()
    assert run["status"] == "failed"
    assert "No model is connected" in run["error"]
    assert run["real_updates"] == 0

def test_inbox_folder_import_is_deduplicated(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    (main.DATA_INBOX / "small.luau").write_text("local coins = 10", encoding="utf-8")
    first = client.post("/api/datasets/import-folder").json()
    second = client.post("/api/datasets/import-folder").json()
    assert first["added"] == 1
    assert second["added"] == 0
    assert any(x["name"] == "Reference file: small.luau" for x in client.get("/api/datasets").json())

def test_mistake_audit_file_is_jsonl(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    client.post("/api/mistakes", json={"prompt":"bad","output":"bad code"})
    lines = main.MISTAKE_LOG.read_text(encoding="utf-8").splitlines()
    import json
    assert json.loads(lines[0])["event"] == "mistake_created"

def test_generate_tasks_from_train_split(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    response = client.post("/api/tasks/generate-from-dataset?limit=10")
    assert response.status_code == 200
    result = response.json()
    assert result["created"] >= 1
    assert any(t["title"].startswith("Practice:") for t in client.get("/api/tasks").json())

def test_training_uses_supervised_and_reward_updates(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    class FakeAdapter:
        name = "test-adapter"
        def __init__(self):
            self.sft = 0
            self.rl = 0
        def generate(self, prompt, context=None):
            return "local health = 100"
        def supervised_step(self, batch):
            self.sft += 1
            assert batch
            return {"updates": 1, "loss": 0.2}
        def train_step(self, batch):
            self.rl += 1
            assert batch and "reward" in batch[0]
            return {"updates": 1, "loss": 0.1}
        def save_checkpoint(self, destination):
            Path(destination).mkdir(parents=True, exist_ok=True)
            (Path(destination) / "weights.bin").write_bytes(b"test weights")
            return {"saved": destination}
        def load_checkpoint(self, source):
            return {"loaded": source}
    adapter = FakeAdapter()
    monkeypatch.setattr(main, "get_adapter", lambda: adapter)
    with main.connect() as db:
        db.execute("UPDATE tasks SET expected='local health = 100' WHERE id=(SELECT id FROM tasks LIMIT 1)")
        db.execute("INSERT INTO runs(id,status,episodes,created_at) VALUES(?,?,?,?)",
                   ("fake-run", "queued", 2, main.now()))
    main.execute_training("fake-run", {"episodes": 2, "batch_size": 2, "duration_hours": 0})
    run = client.get("/api/runs/fake-run").json()
    assert run["status"] == "completed"
    assert run["episodes_completed"] == 2
    assert run["real_updates"] == 2
    assert adapter.sft == 1 and adapter.rl == 1
    checkpoints = client.get("/api/checkpoints").json()
    assert len(checkpoints) == 2
    assert {item["label"] for item in checkpoints} == {
        "Automatic backup before training run",
        "Automatic checkpoint after training run"
    }
    with main.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM task_attempts WHERE run_id='fake-run'").fetchone()[0] == 2

def test_validation_benchmark_never_calls_train_step(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    class FakeAdapter:
        name = "benchmark-test-adapter"
        def generate(self, prompt, context=None):
            return 'local part = Instance.new("Part")'
        def train_step(self, batch):
            raise AssertionError("Benchmark must not train")
        def save_checkpoint(self, destination):
            return {}
        def load_checkpoint(self, source):
            return {}
    monkeypatch.setattr(main, "get_adapter", lambda: FakeAdapter())
    with main.connect() as db:
        db.execute("DELETE FROM datasets WHERE split='validation'")
        db.execute("INSERT INTO datasets(id,name,category,split,content,created_at) VALUES(?,?,?,?,?,?)",
                   ("val-1","Create a part","instances","validation",'local part = Instance.new("Part")',main.now()))
    response = client.post("/api/benchmarks/run?split=validation")
    assert response.status_code == 200
    assert response.json()["average_score"] == 100
    assert client.get("/api/benchmarks").json()[0]["split"] == "validation"

def test_queued_run_can_be_stopped(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    with main.connect() as db:
        db.execute("INSERT INTO runs(id,status,episodes,created_at) VALUES(?,?,?,?)",
                   ("stop-me", "queued", 50, main.now()))
    response = client.post("/api/runs/stop-me/stop")
    assert response.status_code == 200
    assert response.json()["stop_requested"] is True
    with main.connect() as db:
        assert db.execute("SELECT stop_requested FROM runs WHERE id='stop-me'").fetchone()[0] == 1

def test_verification_requires_correction_and_evidence(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    created = client.post("/api/mistakes", json={
        "prompt": "Make a door", "output": "broken",
        "category": "failed_tasks"
    }).json()
    response = client.patch(f"/api/mistakes/{created['id']}", json={
        "verification": "verified", "correction": "local door = Instance.new(\"Part\")"
    })
    assert response.status_code == 400
    assert client.get("/api/mistakes?verification=unverified").json()

def test_python_file_imported_as_general_training_material(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    response = client.post("/api/datasets/import", files={
        "file": ("example.py", b"def add(a, b):\\n    return a + b\\n", "text/x-python")
    })
    assert response.status_code == 200
    item = next(x for x in client.get("/api/datasets").json() if x["name"] == "Reference material: example.py")
    assert item["category"] == "py"
    assert "def add" in item["content"]

def test_notebook_cells_imported_separately(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    notebook = b'{"cells":[{"cell_type":"code","source":["x = 1\\n","print(x)"]}]}'
    response = client.post("/api/datasets/import", files={
        "file": ("lesson.ipynb", notebook, "application/json")
    })
    assert response.status_code == 200
    assert any("lesson.ipynb cell 1" in x["name"] for x in client.get("/api/datasets").json())

def test_generic_evaluator_does_not_require_luau_syntax():
    result = main.static_evaluate("Photosynthesis converts light energy into chemical energy.")
    assert result["checks"]["non_empty"] is True
    assert "does not execute code" in result["warning"]


def test_reward_update_must_be_explicit_and_positive():
    import pytest
    assert main.validate_update_result({"updates": 2, "loss": 0.5}, "train_step") == (2, 0.5)
    with pytest.raises(ValueError):
        main.validate_update_result({"loss": 0.5}, "train_step")
    with pytest.raises(ValueError):
        main.validate_update_result({"updates": True}, "train_step")
    assert main.validate_update_result({"updates": 0, "loss": 0.5}, "train_step") == (0, 0.5)
    with pytest.raises(ValueError):
        main.validate_update_result({"updates": 1, "loss": float("nan")}, "train_step")


def test_run_fails_if_adapter_does_not_report_real_updates(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)

    class BadAdapter:
        name = "bad-update-adapter"
        def generate(self, prompt, context=None):
            return "A useful-looking answer."
        def train_step(self, batch):
            return {"loss": 0.25}  # No explicit update count.
        def save_checkpoint(self, destination):
            Path(destination).mkdir(parents=True, exist_ok=True)
            (Path(destination) / "weights.bin").write_bytes(b"test weights")
            return {"saved": destination}
        def load_checkpoint(self, source):
            return {}

    monkeypatch.setattr(main, "get_adapter", lambda: BadAdapter())
    with main.connect() as db:
        db.execute("UPDATE tasks SET expected='reference answer' WHERE id=(SELECT id FROM tasks LIMIT 1)")
        db.execute("INSERT INTO runs(id,status,episodes,created_at) VALUES(?,?,?,?)",
                   ("fake-no-update", "queued", 1, main.now()))
    main.execute_training("fake-no-update", {"episodes": 1, "batch_size": 1})
    run = client.get("/api/runs/fake-no-update").json()
    assert run["status"] == "failed"
    assert run["real_updates"] == 0
    assert "updates field" in run["error"]


def test_adapter_domain_evaluator_is_used():
    class EvaluatorAdapter:
        name = "science-evaluator"
        def evaluate(self, prompt, output, expected, domain):
            assert domain == "science"
            return {"score": 88, "reward": 0.76, "checks": {"evidence": True}}
    result = main.evaluate_with_adapter(
        EvaluatorAdapter(), "Explain photosynthesis", "Answer", "", "science"
    )
    assert result["score"] == 88
    assert result["reward"] == 0.76
    assert result["evaluator"] == "EvaluatorAdapter"


def test_system_diagnostics_endpoint(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    response = client.get("/api/system/diagnostics")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["counts"]["datasets"] >= 4


def test_checkpoint_failure_is_not_saved_as_fake_metadata(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)

    class NoCheckpointAdapter:
        def save_checkpoint(self, destination):
            Path(destination).mkdir(parents=True, exist_ok=True)
            return {"saved": destination}
    monkeypatch.setattr(main, "get_adapter", lambda: NoCheckpointAdapter())
    response = client.post("/api/checkpoints", json={"label": "should fail"})
    assert response.status_code == 400
    assert client.get("/api/checkpoints").json() == []


def test_docx_upload_extracts_paragraphs_and_tables(tmp_path, monkeypatch):
    from io import BytesIO
    from docx import Document
    client = setup_client(tmp_path, monkeypatch)
    document = Document()
    document.add_paragraph("Neural networks learn patterns from examples.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Input"
    table.rows[0].cells[1].text = "Output"
    buffer = BytesIO()
    document.save(buffer)
    response = client.post("/api/datasets/import", files={
        "file": ("lesson.docx", buffer.getvalue(),
                 "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    })
    assert response.status_code == 200
    item = next(x for x in client.get("/api/datasets").json()
                if x["name"] == "Reference material: lesson.docx")
    assert "Neural networks learn patterns" in item["content"]
    assert "Input | Output" in item["content"]


def test_python_domain_checks_syntax_without_execution():
    valid = main.evaluate_python_static("def add(a, b):" + chr(10) + "    return a + b")
    invalid = main.evaluate_python_static("def add(a, b)" + chr(10) + "    return a + b")
    assert valid["checks"]["python_syntax_valid"] is True
    assert invalid["checks"]["python_syntax_valid"] is False
    assert invalid["score"] <= 25
    assert "not executed" in invalid["warning"]
    assignment = main.evaluate_python_static("x = 1")
    broken_assignment = main.evaluate_python_static("x =")
    assert assignment["checks"]["python_syntax_valid"] is True
    assert broken_assignment["checks"]["python_syntax_valid"] is False
    fenced = main.evaluate_python_static("Here is the code:" + chr(10) + chr(10) + "```python3" + chr(10) + "def add(a, b):" + chr(10) + "    return a + b" + chr(10) + "```")
    assert fenced["checks"]["python_syntax_valid"] is True


def test_instruction_dataset_schema_imports_question_and_answer_fields(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    payload = b'{"examples":[{"question":"What is gravity?","answer":"Gravity attracts objects with mass.","category":"physics"},{"input":"Solve 2+2","target":"4","category":"mathematics"}]}'
    response = client.post("/api/datasets/import", files={
        "file": ("mixed.json", payload, "application/json")
    })
    assert response.status_code == 200
    rows = client.get("/api/datasets").json()
    gravity = next(x for x in rows if x["name"] == "What is gravity?")
    arithmetic = next(x for x in rows if x["name"] == "Solve 2+2")
    assert gravity["content"] == "Gravity attracts objects with mass."
    assert gravity["category"] == "physics"
    assert arithmetic["content"] == "4"
    assert arithmetic["category"] == "mathematics"


def test_long_document_chunks_stay_in_training_split(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    text = ("Paragraph about biology and cells. " * 100 + chr(10) + chr(10) +
            "Paragraph about genetics and DNA. " * 100)
    response = client.post("/api/datasets/import", files={
        "file": ("biology.md", text.encode(), "text/markdown")
    })
    assert response.status_code == 200
    imported = [x for x in client.get("/api/datasets").json()
                if x["name"].startswith("Reference material: biology.md")]
    assert len(imported) > 1
    assert {x["split"] for x in imported} == {"train"}
    assert all(len(x["content"]) <= 3200 for x in imported)


def test_dataset_normalizer_accepts_common_chat_fields():
    row = main.normalize_dataset_row({
        "user": "Explain energy", "assistant": "Energy is the ability to do work.",
        "difficulty": "beginner"
    }, 0, "science")
    assert row["prompt"] == "Explain energy"
    assert row["content"] == "Energy is the ability to do work."
    assert row["category"] == "science"
    assert row["metadata"]["difficulty"] == "beginner"


def test_task_attempt_history_and_performance(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    task = client.get("/api/tasks").json()[0]
    class FakeAdapter:
        name = "attempt-recorder"
        def generate(self, prompt, context=None):
            return "A complete answer with a clear explanation."
    monkeypatch.setattr(main, "get_adapter", lambda: FakeAdapter())
    response = client.post(f"/api/tasks/{task['id']}/try")
    assert response.status_code == 200
    refreshed = next(x for x in client.get("/api/tasks").json() if x["id"] == task["id"])
    assert refreshed["attempt_count"] == 1
    assert refreshed["latest_score"] is not None
    performance = client.get("/api/tasks/performance").json()
    assert next(x for x in performance if x["id"] == task["id"])["attempts"] == 1


def test_task_delete_preserves_its_attempt_history(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    task = client.get("/api/tasks").json()[0]
    main.record_attempt(task["id"], "", "test prompt", "test output", {
        "score": 50, "reward": 0, "evaluator": "test", "checks": {}
    })
    response = client.delete(f"/api/tasks/{task['id']}")
    assert response.status_code == 200
    with main.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM task_attempts WHERE task_id=?", (task["id"],)).fetchone()[0] == 1


def test_openai_and_sharegpt_conversation_dataset_formats():
    openai_row = main.normalize_dataset_row({
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is inertia?"},
            {"role": "assistant", "content": "Inertia is resistance to changes in motion."}
        ]
    }, 0, "physics")
    sharegpt_row = main.normalize_dataset_row({
        "conversations": [
            {"from": "human", "value": "What is 3 times 4?"},
            {"from": "gpt", "value": "12"}
        ]
    }, 1, "mathematics")
    assert "What is inertia?" in openai_row["prompt"]
    assert openai_row["content"] == "Inertia is resistance to changes in motion."
    assert sharegpt_row["prompt"] == "human: What is 3 times 4?"
    assert sharegpt_row["content"] == "12"


def test_adaptive_curriculum_weights_low_scoring_tasks_higher(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    first = client.get("/api/tasks").json()[0]
    second = client.post("/api/tasks", json={
        "title": "Second subject task", "prompt": "Explain a concept.",
        "category": "science"
    }).json()["id"]
    for _ in range(3):
        main.record_attempt(first["id"], "", "task", "output", {
            "score": 20, "reward": -0.6, "evaluator": "test", "checks": {}
        })
        main.record_attempt(second, "", "task", "output", {
            "score": 95, "reward": 0.9, "evaluator": "test", "checks": {}
        })
    captured = {}
    def choose(population, weights, k):
        captured.update({item["id"]: weight for item, weight in zip(population, weights)})
        return [next(item for item in population if item["id"] == first["id"])]
    monkeypatch.setattr(main.random, "choices", choose)

    class TrainingAdapter:
        name = "curriculum-test"
        def generate(self, prompt, context=None):
            return "A clear answer."
        def train_step(self, batch):
            return {"updates": 1, "loss": 0.1}
        def save_checkpoint(self, destination):
            Path(destination).mkdir(parents=True, exist_ok=True)
            (Path(destination) / "weights.bin").write_bytes(b"test weights")
            return {"saved": destination}
        def load_checkpoint(self, source):
            return {}
    monkeypatch.setattr(main, "get_adapter", lambda: TrainingAdapter())
    with main.connect() as db:
        db.execute("INSERT INTO runs(id,status,episodes,created_at) VALUES(?,?,?,?)",
                   ("curriculum-run", "queued", 1, main.now()))
    main.execute_training("curriculum-run", {"episodes": 1, "batch_size": 1})
    assert captured[first["id"]] > captured[second]


def test_lora_adapter_settings_round_trip(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    response = client.put("/api/adapter/config", json={
        "adapter_spec": "backend.hf_adapter:load_adapter",
        "model_path": "/models/local",
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_target_modules": "q_proj,v_proj",
        "learning_rate": 0.0002,
        "max_new_tokens": 256,
        "batch_size": 4,
        "reward_threshold": 0.5
    })
    assert response.status_code == 200
    config = client.get("/api/adapter/config").json()
    assert config["use_lora"] is True
    assert config["lora_r"] == 16
    assert config["lora_alpha"] == 32
    assert config["lora_target_modules"] == "q_proj,v_proj"
    assert config["learning_rate"] == 0.0002


def test_lora_settings_reject_invalid_rank(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    response = client.put("/api/adapter/config", json={"lora_r": 0})
    assert response.status_code == 422


def test_static_evaluator_never_claims_success_without_reference():
    result = main.static_evaluate("A detailed explanation with balanced (text).")
    assert result["score"] == 50
    assert result["reward"] == 0
    assert result["checks"]["reference_match"] is None
    assert "neutral" in result["warning"]


def test_dataset_retrieval_prefers_matching_subject_and_prompt(tmp_path, monkeypatch):
    setup_client(tmp_path, monkeypatch)
    with main.connect() as db:
        db.execute("INSERT INTO datasets(id,name,category,split,content,created_at) VALUES(?,?,?,?,?,?)",
                   ("newest-writing","Recent writing","writing","train",
                    "How to write a persuasive paragraph.",main.now()))
        db.execute("INSERT INTO datasets(id,name,category,split,content,created_at) VALUES(?,?,?,?,?,?)",
                   ("science-reference","Photosynthesis process","science","train",
                    "Photosynthesis converts light energy into chemical energy in plants.",main.now()))
    selected = main.dataset_examples("science", "Explain photosynthesis", limit=1)
    assert selected[0]["prompt"] == "Photosynthesis process"


def test_verified_corrections_are_filtered_by_task_domain(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    science_task = client.post("/api/tasks", json={
        "title": "Science task", "prompt": "Explain photosynthesis", "category": "science"
    }).json()["id"]
    python_task = client.post("/api/tasks", json={
        "title": "Python task", "prompt": "Write a Python function", "category": "python"
    }).json()["id"]
    for task_id, prompt, correction in [
        (science_task, "Explain photosynthesis", "Plants convert light into stored chemical energy."),
        (python_task, "Write a Python function", "def add(a, b): return a + b")
    ]:
        created = client.post("/api/mistakes", json={
            "task_id": task_id, "prompt": prompt, "output": "wrong",
            "correction": correction, "evidence": "Reviewed against reference"
        }).json()
        reviewed = client.patch(f"/api/mistakes/{created['id']}", json={
            "verification": "verified", "correction": correction, "evidence": "Reviewed against reference"
        })
        assert reviewed.status_code == 200
    science_memory = main.verified_examples(category="science")
    assert len(science_memory) == 1
    assert science_memory[0]["correction"].startswith("Plants convert")



def test_task_attempt_feedback_is_saved_as_training_reward(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    task = client.get("/api/tasks").json()[0]
    attempt_id = main.record_attempt(task["id"], "", "Explain a concept", "A generated answer", {
        "score": 50, "reward": 0, "evaluator": "neutral", "checks": {}
    })
    response = client.post(f"/api/attempts/{attempt_id}/feedback", json={
        "rating": 5, "note": "Correct and clear", "evidence": "Compared with class notes"
    })
    assert response.status_code == 200
    assert response.json()["reward"] == 1
    assert response.json()["queued_for_training"] is True
    pending = client.get("/api/feedback?pending_only=true").json()
    assert len(pending) == 1
    assert pending[0]["attempt_id"] == attempt_id
    assert pending[0]["rating"] == 5


def test_training_consumes_human_ratings_only_after_real_update(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    task = client.get("/api/tasks").json()[0]
    attempt_id = main.record_attempt(task["id"], "", "Explain gravity", "Gravity attracts masses.", {
        "score": 50, "reward": 0, "evaluator": "neutral", "checks": {}
    })
    client.post(f"/api/attempts/{attempt_id}/feedback", json={
        "rating": 1, "note": "Incorrect explanation"
    })
    class FeedbackAdapter:
        name = "human-feedback-test"
        def __init__(self): self.batches = []
        def generate(self, prompt, context=None): return "A generated answer with a useful explanation."
        def train_step(self, batch):
            self.batches.append(batch)
            return {"updates": 1, "loss": 0.2}
        def save_checkpoint(self, destination):
            Path(destination).mkdir(parents=True, exist_ok=True)
            (Path(destination) / "weights.bin").write_bytes(b"test weights")
            return {"saved": destination}
        def load_checkpoint(self, source): return {}
    adapter = FeedbackAdapter()
    monkeypatch.setattr(main, "get_adapter", lambda: adapter)
    with main.connect() as db:
        db.execute("INSERT INTO runs(id,status,episodes,created_at) VALUES(?,?,?,?)",
                   ("feedback-run", "queued", 1, main.now()))
    main.execute_training("feedback-run", {"episodes": 1, "batch_size": 1})
    completed_run = client.get("/api/runs/feedback-run").json()
    assert completed_run["status"] == "completed"
    assert completed_run["human_feedback_examples"] == 1
    assert completed_run["feedback_updates"] == 1
    assert any(batch[0].get("task_id") == task["id"] and batch[0]["reward"] == -1
               for batch in adapter.batches)
    assert client.get("/api/feedback?pending_only=true").json() == []
    with main.connect() as db:
        row = db.execute("SELECT consumed_at,consumed_run_id FROM human_feedback").fetchone()
        assert row["consumed_at"] is not None
        assert row["consumed_run_id"] == "feedback-run"


def test_invalid_feedback_rating_is_rejected(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    attempt_id = main.record_attempt(client.get("/api/tasks").json()[0]["id"], "", "p", "o", {
        "score": 50, "reward": 0, "evaluator": "test", "checks": {}
    })
    response = client.post(f"/api/attempts/{attempt_id}/feedback", json={"rating": 6})
    assert response.status_code == 422


def test_deleting_task_keeps_human_feedback_history(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    task = client.get("/api/tasks").json()[0]
    attempt_id = main.record_attempt(task["id"], "", "p", "o", {
        "score": 50, "reward": 0, "evaluator": "test", "checks": {}
    })
    client.post(f"/api/attempts/{attempt_id}/feedback", json={"rating": 4})
    deleted = client.delete(f"/api/tasks/{task['id']}")
    assert deleted.json()["history_preserved"] is True
    assert len(client.get("/api/attempts").json()) == 1
    assert len(client.get("/api/feedback").json()) == 1



def test_failed_human_feedback_update_stays_pending(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    task = client.get("/api/tasks").json()[0]
    attempt_id = main.record_attempt(task["id"], "", "prompt", "completion", {
        "score": 50, "reward": 0, "evaluator": "test", "checks": {}
    })
    client.post(f"/api/attempts/{attempt_id}/feedback", json={"rating": 5})
    class FailingAdapter:
        name = "failing-feedback-adapter"
        def generate(self, prompt, context=None): return "generated output"
        def train_step(self, batch): raise RuntimeError("optimizer failure")
        def save_checkpoint(self, destination):
            Path(destination).mkdir(parents=True, exist_ok=True)
            (Path(destination) / "weights.bin").write_bytes(b"test weights")
            return {}
        def load_checkpoint(self, source): return {}
    monkeypatch.setattr(main, "get_adapter", lambda: FailingAdapter())
    with main.connect() as db:
        db.execute("INSERT INTO runs(id,status,episodes,created_at) VALUES(?,?,?,?)",
                   ("failed-feedback-run", "queued", 1, main.now()))
    main.execute_training("failed-feedback-run", {"episodes": 1, "batch_size": 1})
    assert client.get("/api/runs/failed-feedback-run").json()["status"] == "failed"
    assert len(client.get("/api/feedback?pending_only=true").json()) == 1



def test_dataset_stats_report_splits_categories_and_duplicates(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    with main.connect() as db:
        db.execute("INSERT INTO datasets(id,name,category,split,content,created_at) VALUES(?,?,?,?,?,?)",
                   ("dup-a", "Duplicate A", "physics", "train", "same reference", main.now()))
        db.execute("INSERT INTO datasets(id,name,category,split,content,created_at) VALUES(?,?,?,?,?,?)",
                   ("dup-b", "Duplicate B", "physics", "train", "same reference", main.now()))
        db.execute("INSERT INTO datasets(id,name,category,split,content,created_at) VALUES(?,?,?,?,?,?)",
                   ("dup-c", "Duplicate C", "physics", "test", "same reference", main.now()))
    stats = client.get("/api/datasets/stats").json()
    assert stats["total"] >= 6
    assert stats["by_split"]["train"] >= 3
    assert any(x["category"] == "physics" for x in stats["by_category"])
    assert stats["duplicate_content_groups"] == 1
    assert stats["duplicate_split_conflicts"] == 1
    assert stats["content_characters"]["max"] >= len("same reference")


def test_training_preview_explains_missing_model_without_loading_it(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    preview = client.get("/api/training/preview").json()
    assert preview["enabled_tasks"] >= 1
    assert preview["adapter_configured"] is False
    assert preview["ready"] is False
    assert any("Connect a model" in warning for warning in preview["warnings"])



def test_verified_mistake_corrections_are_reused_for_supervised_training(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    task = client.get("/api/tasks").json()[0]
    correction = "A correct explanation reviewed against the official reference."
    mistake = client.post("/api/mistakes", json={
        "task_id": task["id"], "prompt": "Explain the concept",
        "output": "Incorrect explanation", "correction": correction,
        "evidence": "Checked against official reference"
    }).json()
    reviewed = client.patch(f"/api/mistakes/{mistake['id']}", json={
        "verification": "verified", "correction": correction,
        "evidence": "Checked against official reference"
    })
    assert reviewed.status_code == 200
    class CorrectionAdapter:
        name = "verified-correction-test"
        def __init__(self): self.supervised = []
        def generate(self, prompt, context=None): return "Generated answer."
        def supervised_step(self, batch):
            self.supervised.extend(batch)
            return {"updates": 1, "loss": 0.2}
        def train_step(self, batch): return {"updates": 1, "loss": 0.1}
        def save_checkpoint(self, destination):
            Path(destination).mkdir(parents=True, exist_ok=True)
            (Path(destination) / "weights.bin").write_bytes(b"test weights")
            return {}
        def load_checkpoint(self, source): return {}
    adapter = CorrectionAdapter()
    monkeypatch.setattr(main, "get_adapter", lambda: adapter)
    with main.connect() as db:
        db.execute("INSERT INTO runs(id,status,episodes,created_at) VALUES(?,?,?,?)",
                   ("verified-correction-run", "queued", 1, main.now()))
    main.execute_training("verified-correction-run", {"episodes": 1, "batch_size": 1})
    assert client.get("/api/runs/verified-correction-run").json()["status"] == "completed"
    assert adapter.supervised[0]["completion"] == correction



def test_adapter_status_includes_non_mutating_diagnostics(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    client.put("/api/adapter/config", json={"adapter_spec": "tests.fake:load_adapter"})
    class DiagnosticAdapter:
        name = "diagnostic-adapter"
        def generate(self, prompt, context=None): return "sample"
        def train_step(self, batch): return {"updates": 1, "loss": 0.1}
        def save_checkpoint(self, destination): return {}
        def load_checkpoint(self, source): return {}
        def diagnostics(self):
            return {"device": "cpu", "trainable_parameters": 1234, "trainable_percent": 2.5}
    monkeypatch.setattr(main, "get_adapter", lambda: DiagnosticAdapter())
    status = client.get("/api/adapter/status").json()
    assert status["connected"] is True
    assert status["diagnostics"]["device"] == "cpu"
    assert status["diagnostics"]["trainable_parameters"] == 1234


def test_adapter_test_reports_generation_without_weight_updates(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    class TestAdapter:
        name = "test-only-adapter"
        def __init__(self): self.updates = 0
        def generate(self, prompt, context=None): return "A generated sample."
        def train_step(self, batch):
            self.updates += 1
            return {"updates": 1, "loss": 0.1}
        def save_checkpoint(self, destination): return {}
        def load_checkpoint(self, source): return {}
    adapter = TestAdapter()
    monkeypatch.setattr(main, "get_adapter", lambda: adapter)
    result = client.post("/api/adapter/test").json()
    assert result["generation_ok"] is True
    assert result["sample_output"] == "A generated sample."
    assert adapter.updates == 0



def test_post_training_checkpoint_failure_is_reported(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    task = client.get("/api/tasks").json()[0]
    attempt_id = main.record_attempt(task["id"], "", "prompt", "answer", {
        "score": 50, "reward": 0, "evaluator": "test", "checks": {}
    })
    client.post(f"/api/attempts/{attempt_id}/feedback", json={"rating": 5})
    class CheckpointFailureAdapter:
        name = "checkpoint-failure-adapter"
        def __init__(self): self.saves = 0
        def generate(self, prompt, context=None): return "A generated answer."
        def train_step(self, batch): return {"updates": 1, "loss": 0.1}
        def save_checkpoint(self, destination):
            self.saves += 1
            if self.saves == 1:
                Path(destination).mkdir(parents=True, exist_ok=True)
                (Path(destination) / "weights.bin").write_bytes(b"test weights")
                return {"saved": destination}
            return {"error": "simulated final save failure"}
        def load_checkpoint(self, source): return {}
    adapter = CheckpointFailureAdapter()
    monkeypatch.setattr(main, "get_adapter", lambda: adapter)
    with main.connect() as db:
        db.execute("INSERT INTO runs(id,status,episodes,created_at) VALUES(?,?,?,?)",
                   ("final-checkpoint-failure", "queued", 1, main.now()))
    main.execute_training("final-checkpoint-failure", {"episodes": 1, "batch_size": 1})
    run = client.get("/api/runs/final-checkpoint-failure").json()
    assert run["status"] == "failed"
    assert "Post-training checkpoint failed" in run["error"]
    checkpoints = client.get("/api/checkpoints").json()
    assert len(checkpoints) == 1
    assert checkpoints[0]["label"] == "Automatic backup before training run"
    assert len(client.get("/api/feedback?pending_only=true").json()) == 1



def test_human_correction_requires_review_before_sft_reuse(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    task = client.get("/api/tasks").json()[0]
    attempt_id = main.record_attempt(task["id"], "", "Explain the topic", "Wrong answer", {
        "score": 25, "reward": -0.5, "evaluator": "test", "checks": {}
    })
    correction = "The corrected answer, checked against a trusted source."
    feedback = client.post(f"/api/attempts/{attempt_id}/feedback", json={
        "rating": 1, "correction": correction, "evidence": "Official reference section 2"
    }).json()
    assert feedback["mistake_id"]
    assert feedback["correction_requires_review"] is True
    candidate = next(x for x in client.get("/api/mistakes").json()
                     if x["id"] == feedback["mistake_id"])
    assert candidate["verification"] == "unverified"
    assert all(x["correction"] != correction for x in main.verified_examples())
    reviewed = client.patch(f"/api/mistakes/{feedback['mistake_id']}", json={
        "verification": "verified", "correction": correction,
        "evidence": "Confirmed in official reference section 2"
    })
    assert reviewed.status_code == 200
    assert any(x["correction"] == correction for x in main.verified_examples())



def test_custom_evaluator_without_expected_match_works_in_benchmark(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    with main.connect() as db:
        db.execute("DELETE FROM datasets WHERE split='validation'")
        db.execute("INSERT INTO datasets(id,name,category,split,content,created_at) VALUES(?,?,?,?,?,?)",
                   ("custom-eval-val", "Explain a physics concept", "physics", "validation",
                    "A reference about motion.", main.now()))
        db.execute("INSERT INTO datasets(id,name,category,split,content,created_at) VALUES(?,?,?,?,?,?)",
                   ("custom-eval-train-duplicate", "Similar motion answer", "physics", "train",
                    "A reference about motion.", main.now()))
    class CustomEvaluatorAdapter:
        name = "custom-evaluator"
        def __init__(self): self.prompts = []
        def generate(self, prompt, context=None):
            self.prompts.append(prompt)
            return "Motion describes a change in position."
        def evaluate(self, prompt, output, expected, domain):
            return {"score": 82, "reward": 0.64, "checks": {"rubric": True},
                    "evaluator": "physics-rubric", "warning": "Rubric score; not a scientific proof."}
        def train_step(self, batch): raise AssertionError("Benchmark must not train")
        def save_checkpoint(self, destination): return {}
        def load_checkpoint(self, source): return {}
    adapter = CustomEvaluatorAdapter()
    monkeypatch.setattr(main, "get_adapter", lambda: adapter)
    response = client.post("/api/benchmarks/run?split=validation")
    assert response.status_code == 200
    detail = response.json()["details"][0]
    assert detail["score"] == 82
    assert detail["expected_match"] is None
    assert "A reference about motion." not in adapter.prompts[0]



def test_adapter_configuration_cannot_change_during_active_training(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    before = client.get("/api/adapter/config").json()
    with main.connect() as db:
        db.execute("INSERT INTO runs(id,status,episodes,created_at) VALUES(?,?,?,?)",
                   ("active-config-run", "running", 10, main.now()))
    response = client.put("/api/adapter/config", json={"learning_rate": 0.0005})
    assert response.status_code == 409
    after = client.get("/api/adapter/config").json()
    assert after["learning_rate"] == before["learning_rate"]



def test_mistake_api_rejects_oversized_text_fields(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    response = client.post("/api/mistakes", json={
        "prompt": "p" * 12001, "output": "short"
    })
    assert response.status_code == 422


def test_feedback_rating_requires_integer_between_one_and_five(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    attempt_id = main.record_attempt(client.get("/api/tasks").json()[0]["id"], "", "p", "o", {
        "score": 50, "reward": 0, "evaluator": "test", "checks": {}
    })
    fractional = client.post(f"/api/attempts/{attempt_id}/feedback", json={"rating": 3.5})
    too_high = client.post(f"/api/attempts/{attempt_id}/feedback", json={"rating": 6})
    assert fractional.status_code == 422
    assert too_high.status_code == 422



def test_xlsx_import_extracts_question_answer_rows(tmp_path, monkeypatch):
    from io import BytesIO
    from openpyxl import Workbook
    client = setup_client(tmp_path, monkeypatch)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Physics"
    sheet.append(["question", "answer", "category"])
    sheet.append(["What is inertia?", "Resistance to a change in motion.", "physics"])
    buffer = BytesIO()
    workbook.save(buffer)
    response = client.post("/api/datasets/import", files={
        "file": ("physics.xlsx", buffer.getvalue(),
                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    })
    assert response.status_code == 200
    item = next(x for x in client.get("/api/datasets").json()
                if x["name"] == "What is inertia?")
    assert item["content"] == "Resistance to a change in motion."
    assert item["category"] == "physics"


def test_pptx_import_extracts_slide_text(tmp_path, monkeypatch):
    from io import BytesIO
    from pptx import Presentation
    from pptx.util import Inches
    client = setup_client(tmp_path, monkeypatch)
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    textbox = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(5), Inches(2))
    textbox.text_frame.text = "Newton's first law describes inertia."
    buffer = BytesIO()
    presentation.save(buffer)
    response = client.post("/api/datasets/import", files={
        "file": ("physics.pptx", buffer.getvalue(),
                 "application/vnd.openxmlformats-officedocument.presentationml.presentation")
    })
    assert response.status_code == 200
    item = next(x for x in client.get("/api/datasets").json()
                if "physics.pptx slide 1" in x["name"])
    assert "Newton's first law" in item["content"]
    assert item["split"] == "train"



def test_dataset_export_preserves_metadata_and_ndjson_rows(tmp_path, monkeypatch):
    import json
    client = setup_client(tmp_path, monkeypatch)
    with main.connect() as db:
        db.execute("""INSERT INTO datasets
          (id,name,category,split,content,metadata_json,created_at)
          VALUES(?,?,?,?,?,?,?)""",
          ("export-row", "Question", "science", "train", "Answer",
           json.dumps({"source_file": "lesson.xlsx", "sheet": "Physics"}), main.now()))
    response = client.get("/api/datasets/export")
    assert response.status_code == 200
    rows = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    exported = next(row for row in rows if row["prompt"] == "Question")
    assert exported["metadata"]["source_file"] == "lesson.xlsx"
    assert exported["metadata"]["sheet"] == "Physics"



def test_dataset_metadata_round_trips_through_normalizer():
    row = main.normalize_dataset_row({
        "prompt": "Question", "content": "Answer", "category": "science",
        "split": "train", "metadata": {"source_file": "lesson.xlsx", "sheet": "Physics"}
    }, 0)
    assert row["metadata"]["source_file"] == "lesson.xlsx"
    assert row["metadata"]["sheet"] == "Physics"



def test_direct_upload_records_sha256_provenance(tmp_path, monkeypatch):
    import hashlib
    import json
    client = setup_client(tmp_path, monkeypatch)
    raw = b"def square(x):\n    return x * x\n"
    response = client.post("/api/datasets/import", files={
        "file": ("square.py", raw, "text/x-python")
    })
    assert response.status_code == 200
    with main.connect() as db:
        row = db.execute("SELECT metadata_json FROM datasets WHERE name='Reference material: square.py'").fetchone()
    metadata = json.loads(row["metadata_json"])
    assert metadata["sha256"] == hashlib.sha256(raw).hexdigest()



def test_long_source_file_is_chunked_without_split_leakage(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    source = ("def function_" + "x" * 10 + "():\n    return 1\n\n") * 180
    response = client.post("/api/datasets/import", files={
        "file": ("large_module.py", source.encode(), "text/x-python")
    })
    assert response.status_code == 200
    rows = [x for x in client.get("/api/datasets").json()
            if x["name"].startswith("Reference material: large_module.py")]
    assert len(rows) > 1
    assert {x["split"] for x in rows} == {"train"}
    assert all(len(x["content"]) <= 3200 for x in rows)



def test_run_report_collects_attempts_and_checkpoints(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    task = client.get("/api/tasks").json()[0]
    with main.connect() as db:
        db.execute("INSERT INTO runs(id,status,episodes,created_at) VALUES(?,?,?,?)",
                   ("report-run", "completed", 1, main.now()))
    main.record_attempt(task["id"], "report-run", "Explain a concept", "A clear answer", {
        "score": 80, "reward": 0.6, "evaluator": "rubric", "checks": {"clarity": True}
    })
    with main.connect() as db:
        db.execute("""INSERT INTO checkpoints
          (id,run_id,label,kind,created_at,path,metadata_json)
          VALUES(?,?,?,?,?,?,?)""",
          ("report-checkpoint", "report-run", "After run", "model_weights",
           main.now(), "/tmp/checkpoint", "{}"))
    report = client.get("/api/runs/report-run/report").json()
    assert report["attempt_stats"]["attempts"] == 1
    assert report["attempt_stats"]["average_score"] == 80
    assert report["recent_attempts"][0]["task_title"] == task["title"]
    assert report["checkpoints"][0]["label"] == "After run"



def test_training_time_budget_marks_run_stopped(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    class TimeBudgetAdapter:
        name = "time-budget-test"
        def generate(self, prompt, context=None): return "Should not be called."
        def train_step(self, batch): return {"updates": 1, "loss": 0.1}
        def save_checkpoint(self, destination):
            Path(destination).mkdir(parents=True, exist_ok=True)
            (Path(destination) / "weights.bin").write_bytes(b"test weights")
            return {}
        def load_checkpoint(self, source): return {}
    monkeypatch.setattr(main, "get_adapter", lambda: TimeBudgetAdapter())
    with main.connect() as db:
        db.execute("INSERT INTO runs(id,status,episodes,created_at) VALUES(?,?,?,?)",
                   ("time-budget-run", "queued", 10, main.now()))
    main.execute_training("time-budget-run", {
        "episodes": 10, "batch_size": 1, "duration_hours": 0.000000000001
    })
    run = client.get("/api/runs/time-budget-run").json()
    assert run["status"] == "stopped"
    assert run["episodes_completed"] == 0
    assert "Time budget expired before the first training batch" in run["notes"]



def test_stop_requested_before_worker_starts_prevents_model_updates(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    class StopBeforeStartAdapter:
        name = "stop-before-start"
        def __init__(self): self.reward_calls = 0
        def generate(self, prompt, context=None): raise AssertionError("Generation must not start")
        def train_step(self, batch):
            self.reward_calls += 1
            return {"updates": 1, "loss": 0.1}
        def save_checkpoint(self, destination):
            Path(destination).mkdir(parents=True, exist_ok=True)
            (Path(destination) / "weights.bin").write_bytes(b"test weights")
            return {}
        def load_checkpoint(self, source): return {}
    adapter = StopBeforeStartAdapter()
    monkeypatch.setattr(main, "get_adapter", lambda: adapter)
    with main.connect() as db:
        db.execute("INSERT INTO runs(id,status,episodes,stop_requested,created_at) VALUES(?,?,?,?,?)",
                   ("pre-stopped-run", "queued", 10, 1, main.now()))
    main.execute_training("pre-stopped-run", {"episodes": 10, "batch_size": 1})
    run = client.get("/api/runs/pre-stopped-run").json()
    assert run["status"] == "stopped"
    assert run["real_updates"] == 0
    assert adapter.reward_calls == 0



def test_generic_evaluator_rejects_unmatched_delimiter_counts():
    result = main.static_evaluate("A calculation with an unclosed ( bracket")
    assert result["checks"]["balanced_delimiters"] is False
    assert result["score"] == 25
    assert result["reward"] == -0.5



def test_dashboard_and_openapi_are_served(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    dashboard = client.get("/")
    schema = client.get("/openapi.json")
    assert dashboard.status_code == 200
    assert "NeuralForge RL Lab" in dashboard.text
    assert schema.status_code == 200
    paths = schema.json()["paths"]
    assert "/api/attempts/{attempt_id}/feedback" in paths
    assert "/api/training/preview" in paths
    assert "/api/runs/{run_id}/report" in paths



def test_neutral_rating_is_recorded_but_not_queued_for_training(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    attempt_id = main.record_attempt(client.get("/api/tasks").json()[0]["id"], "", "p", "o", {
        "score": 50, "reward": 0, "evaluator": "test", "checks": {}
    })
    result = client.post(f"/api/attempts/{attempt_id}/feedback", json={"rating": 3}).json()
    assert result["reward"] == 0
    assert result["queued_for_training"] is False
    assert len(client.get("/api/feedback").json()) == 1
    assert client.get("/api/feedback?pending_only=true").json() == []
    assert client.get("/api/training/preview").json()["pending_human_feedback"] == 0



def test_evaluate_endpoint_uses_python_syntax_checker_without_model(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    response = client.post("/api/evaluate", json={
        "domain": "python", "output": "def broken(:"
    })
    assert response.status_code == 200
    assert response.json()["checks"]["python_syntax_valid"] is False


def test_evaluate_endpoint_uses_configured_domain_evaluator(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    client.put("/api/adapter/config", json={"adapter_spec": "tests.fake:load_adapter"})
    class DomainAdapter:
        name = "domain-evaluator"
        def evaluate(self, prompt, output, expected, domain):
            return {"score": 91, "reward": 0.82, "checks": {"verified": True}}
        def generate(self, prompt, context=None): return "unused"
        def train_step(self, batch): return {"updates": 1, "loss": 0.1}
        def save_checkpoint(self, destination): return {}
        def load_checkpoint(self, source): return {}
    monkeypatch.setattr(main, "get_adapter", lambda: DomainAdapter())
    response = client.post("/api/evaluate", json={
        "domain": "science", "prompt": "Explain gravity", "output": "Gravity answer"
    })
    assert response.status_code == 200
    assert response.json()["score"] == 91
    assert response.json()["checks"]["verified"] is True



def test_feedback_export_preserves_rating_reward_and_attempt_context(tmp_path, monkeypatch):
    import json
    client = setup_client(tmp_path, monkeypatch)
    task = client.get("/api/tasks").json()[0]
    attempt_id = main.record_attempt(task["id"], "", "Explain energy", "Energy is conserved.", {
        "score": 80, "reward": 0.6, "evaluator": "test", "checks": {}
    })
    client.post(f"/api/attempts/{attempt_id}/feedback", json={
        "rating": 5, "note": "Good answer", "evidence": "Verified source"
    })
    response = client.get("/api/feedback/export")
    rows = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    exported = rows[0]
    assert exported["rating"] == 5
    assert exported["reward"] == 1
    assert exported["prompt"] == "Explain energy"
    assert exported["output"] == "Energy is conserved."



def test_generated_task_does_not_retrieve_its_own_reference_answer(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    with main.connect() as db:
        db.execute("DELETE FROM datasets")
        db.execute("""INSERT INTO datasets
          (id,name,category,split,content,created_at)
          VALUES(?,?,?,?,?,?)""",
          ("only-source", "What is inertia?", "physics", "train",
           "INERTIA_SECRET_REFERENCE_12345", main.now()))
        db.execute("""INSERT INTO datasets
          (id,name,category,split,content,created_at)
          VALUES(?,?,?,?,?,?)""",
          ("duplicate-source", "Inertia duplicate", "physics", "train",
           "INERTIA_SECRET_REFERENCE_12345", main.now()))
    generated = client.post("/api/tasks/generate-from-dataset?limit=10").json()
    assert generated["created"] == 1
    task = next(x for x in client.get("/api/tasks").json()
                if x.get("source_dataset_id") in {"only-source", "duplicate-source"})
    assert task["source_dataset_id"] in {"only-source", "duplicate-source"}
    prompt = main.build_prompt(task)
    assert "INERTIA_SECRET_REFERENCE_12345" not in prompt



def test_prompt_builder_excludes_same_task_verified_correction(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    client.post("/api/tasks", json={
        "title": "Leakage guard task", "prompt": "Explain the conservation law",
        "expected": "LEAKED_CORRECTION_SECRET", "category": "physics"
    })
    task = next(x for x in client.get("/api/tasks").json() if x["title"] == "Leakage guard task")
    mistake = client.post("/api/mistakes", json={
        "task_id": task["id"], "prompt": task["prompt"], "output": "Wrong",
        "correction": "LEAKED_CORRECTION_SECRET", "evidence": "Trusted source"
    }).json()
    client.patch(f"/api/mistakes/{mistake['id']}", json={
        "verification": "verified", "correction": "LEAKED_CORRECTION_SECRET",
        "evidence": "Trusted source"
    })
    prompt = main.build_prompt(task)
    assert "LEAKED_CORRECTION_SECRET" not in prompt



def test_exact_duplicate_examples_share_the_same_dataset_split():
    rows = [
        {"prompt": "First copy", "content": "Identical reference"},
        {"prompt": "Another item", "content": "Different reference"},
        {"prompt": "Second copy", "content": "Identical reference"},
    ]
    prepared = main.prepared_dataset_rows(rows, "general", "duplicates.jsonl")
    assert prepared[0][3] == prepared[2][3]


def test_conflicting_explicit_splits_for_duplicate_content_are_rejected():
    rows = [
        {"prompt": "Train copy", "content": "Same answer", "split": "train"},
        {"prompt": "Test copy", "content": "Same answer", "split": "test"},
    ]
    import pytest
    with pytest.raises(ValueError, match="Identical content"):
        main.prepared_dataset_rows(rows, "general", "leaky.jsonl")



def test_training_preview_warns_about_duplicate_split_leakage(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    with main.connect() as db:
        db.execute("INSERT INTO datasets(id,name,category,split,content,created_at) VALUES(?,?,?,?,?,?)",
                   ("leak-train", "Train copy", "science", "train", "same exact answer", main.now()))
        db.execute("INSERT INTO datasets(id,name,category,split,content,created_at) VALUES(?,?,?,?,?,?)",
                   ("leak-test", "Test copy", "science", "test", "same exact answer", main.now()))
    preview = client.get("/api/training/preview").json()
    assert any("benchmark leakage is possible" in warning for warning in preview["warnings"])



def test_dataset_split_update_can_resolve_exact_duplicate_leakage(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    with main.connect() as db:
        db.execute("INSERT INTO datasets(id,name,category,split,content,created_at) VALUES(?,?,?,?,?,?)",
                   ("split-train", "Train", "science", "train", "same content", main.now()))
        db.execute("INSERT INTO datasets(id,name,category,split,content,created_at) VALUES(?,?,?,?,?,?)",
                   ("split-test", "Test", "science", "test", "same content", main.now()))
        db.execute("""INSERT INTO tasks
          (id,title,prompt,expected,difficulty,category,source_dataset_id,created_at)
          VALUES(?,?,?,?,?,?,?,?)""",
          ("split-task", "Generated task", "Prompt", "Answer", "easy", "science",
           "split-test", main.now()))
    conflict = client.patch("/api/datasets/split-test/split", json={"split": "validation"})
    assert conflict.status_code == 409
    fixed = client.patch("/api/datasets/split-test/split", json={"split": "train"})
    assert fixed.status_code == 200
    assert client.get("/api/datasets/stats").json()["duplicate_split_conflicts"] == 0
    client.delete("/api/datasets/split-train")
    moved_out = client.patch("/api/datasets/split-test/split", json={"split": "validation"})
    assert moved_out.status_code == 200
    assert moved_out.json()["disabled_generated_tasks"] == 1
    task = next(x for x in client.get("/api/tasks").json() if x["id"] == "split-task")
    assert task["enabled"] == 0



def test_duplicate_content_in_new_import_inherits_existing_split(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    with main.connect() as db:
        db.execute("INSERT INTO datasets(id,name,category,split,content,created_at) VALUES(?,?,?,?,?,?)",
                   ("existing-copy", "Existing", "science", "validation", "shared reference", main.now()))
    mapping = main.existing_dataset_content_splits()
    prepared = main.prepared_dataset_rows(
        [{"prompt": "New copy", "content": "shared reference"}],
        "science", "second.jsonl", existing_content_splits=mapping
    )
    assert prepared[0][3] == "validation"


def test_new_import_rejects_explicit_split_conflicting_with_existing_duplicate(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    with main.connect() as db:
        db.execute("INSERT INTO datasets(id,name,category,split,content,created_at) VALUES(?,?,?,?,?,?)",
                   ("existing-copy", "Existing", "science", "validation", "shared reference", main.now()))
    mapping = main.existing_dataset_content_splits()
    import pytest
    with pytest.raises(ValueError, match="Identical content"):
        main.prepared_dataset_rows(
            [{"prompt": "New copy", "content": "shared reference", "split": "test"}],
            "science", "second.jsonl", existing_content_splits=mapping
        )



def test_bulk_split_repair_moves_all_exact_duplicates_together(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    with main.connect() as db:
        for dataset_id, split in [
            ("group-train", "train"), ("group-validation", "validation"), ("group-test", "test")
        ]:
            db.execute("INSERT INTO datasets(id,name,category,split,content,created_at) VALUES(?,?,?,?,?,?)",
                       (dataset_id, dataset_id, "science", split, "same exact content", main.now()))
        db.execute("""INSERT INTO tasks
          (id,title,prompt,expected,difficulty,category,source_dataset_id,created_at)
          VALUES(?,?,?,?,?,?,?,?)""",
          ("group-task", "Generated task", "Prompt", "Answer", "easy", "science",
           "group-train", main.now()))
    response = client.patch("/api/datasets/group-train/split", json={
        "split": "validation", "apply_to_exact_duplicates": True
    })
    assert response.status_code == 200
    assert response.json()["updated_examples"] == 3
    assert response.json()["disabled_generated_tasks"] == 1
    stats = client.get("/api/datasets/stats").json()
    assert stats["duplicate_split_conflicts"] == 0



def test_neutral_automatic_reward_skips_reward_hook_and_completes_honestly(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    class NeutralAdapter:
        name = "neutral-reward-adapter"
        def generate(self, prompt, context=None): return "A reasonable general answer."
        def train_step(self, batch):
            raise AssertionError("Neutral reward batches must not be sent to train_step")
        def save_checkpoint(self, destination):
            Path(destination).mkdir(parents=True, exist_ok=True)
            (Path(destination) / "weights.bin").write_bytes(b"unchanged model snapshot")
            return {}
        def load_checkpoint(self, source): return {}
    monkeypatch.setattr(main, "get_adapter", lambda: NeutralAdapter())
    with main.connect() as db:
        db.execute("INSERT INTO runs(id,status,episodes,created_at) VALUES(?,?,?,?)",
                   ("neutral-run", "queued", 1, main.now()))
    main.execute_training("neutral-run", {"episodes": 1, "batch_size": 1})
    run = client.get("/api/runs/neutral-run").json()
    assert run["status"] == "completed"
    assert run["real_updates"] == 0
    assert "No optimizer updates were reported" in run["notes"]
    checkpoints = client.get("/api/checkpoints").json()
    assert len(checkpoints) == 1
    assert checkpoints[0]["label"] == "Automatic backup before training run"



def test_zero_gradient_feedback_is_not_marked_consumed(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    task = client.get("/api/tasks").json()[0]
    attempt_id = main.record_attempt(task["id"], "", "Original prompt", "Old model answer", {
        "score": 50, "reward": 0.0, "evaluator": "test", "checks": {}
    })
    feedback = client.post(f"/api/attempts/{attempt_id}/feedback", json={
        "rating": 5, "note": "Please learn from this", "evidence": "Human review"
    }).json()
    class NoGradientAdapter:
        name = "no-gradient-adapter"
        def generate(self, prompt, context=None): return "A response with no trainable signal."
        def train_step(self, batch): return {"updates": 0, "loss": 0.0, "note": "No gradient"}
        def save_checkpoint(self, destination):
            Path(destination).mkdir(parents=True, exist_ok=True)
            (Path(destination) / "weights.bin").write_bytes(b"snapshot")
            return {}
        def load_checkpoint(self, source): return {}
    monkeypatch.setattr(main, "get_adapter", lambda: NoGradientAdapter())
    with main.connect() as db:
        db.execute("INSERT INTO runs(id,status,episodes,created_at) VALUES(?,?,?,?)",
                   ("no-gradient-run", "queued", 1, main.now()))
    main.execute_training("no-gradient-run", {"episodes": 1, "batch_size": 1})
    run = client.get("/api/runs/no-gradient-run").json()
    assert run["status"] == "completed"
    assert run["real_updates"] == 0
    pending = client.get("/api/feedback?pending_only=true").json()
    assert any(row["id"] == feedback["id"] for row in pending)
    assert run["feedback_updates"] == 0



def test_deleting_dataset_source_disables_generated_task_but_keeps_history(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    source = next(row for row in client.get("/api/datasets").json() if row["split"] == "train")
    generated = client.post("/api/tasks/generate-from-dataset?limit=100").json()
    assert generated["created"] >= 1
    task = next(row for row in client.get("/api/tasks").json()
                if row.get("source_dataset_id") == source["id"])
    response = client.delete(f"/api/datasets/{source['id']}")
    assert response.status_code == 200
    assert response.json()["disabled_generated_tasks"] == 1
    stored_task = next(row for row in client.get("/api/tasks").json() if row["id"] == task["id"])
    assert stored_task["enabled"] == 0
    assert stored_task["expected"] == source["content"]



def test_failed_checkpoint_restore_rolls_back_to_current_model(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    class RestorableAdapter:
        name = "restorable-test-adapter"
        def __init__(self): self.state = "current-model"
        def generate(self, prompt, context=None): return "test output"
        def train_step(self, batch): return {"updates": 1, "loss": 0.1}
        def save_checkpoint(self, destination):
            folder = Path(destination)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "state.txt").write_text(self.state, encoding="utf-8")
            return {"state": self.state}
        def load_checkpoint(self, source):
            restored = (Path(source) / "state.txt").read_text(encoding="utf-8")
            if restored == "corrupt-checkpoint":
                raise ValueError("checkpoint weights are corrupt")
            self.state = restored
            return {"state": self.state}
    adapter = RestorableAdapter()
    monkeypatch.setattr(main, "get_adapter", lambda: adapter)
    bad_path = tmp_path / "bad-checkpoint"
    bad_path.mkdir()
    (bad_path / "state.txt").write_text("corrupt-checkpoint", encoding="utf-8")
    with main.connect() as db:
        db.execute("""INSERT INTO checkpoints
          (id,run_id,label,kind,created_at,path,metadata_json)
          VALUES(?,?,?,?,?,?,?)""",
          ("bad-restore", "", "Corrupt checkpoint", "model_weights",
           main.now(), str(bad_path), "{}"))
    response = client.post("/api/checkpoints/bad-restore/restore")
    assert response.status_code == 400
    assert "previous model was restored" in response.json()["detail"]
    assert adapter.state == "current-model"
    safety = client.get("/api/checkpoints").json()
    assert any(row["label"] == "Automatic safety backup before restore" for row in safety)



def test_failed_training_restores_pretraining_model_weights(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    task = client.get("/api/tasks").json()[0]
    class CrashingAdapter:
        name = "crashing-adapter"
        def __init__(self): self.state = "before-training"
        def generate(self, prompt, context=None): return "deliberately incorrect output"
        def evaluate(self, prompt, output, expected, domain):
            return {"score": 0, "reward": -1, "checks": {}, "evaluator": "test"}
        def train_step(self, batch):
            self.state = "partially-updated"
            raise RuntimeError("simulated optimizer crash")
        def save_checkpoint(self, destination):
            folder = Path(destination)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "state.txt").write_text(self.state, encoding="utf-8")
            return {}
        def load_checkpoint(self, source):
            self.state = (Path(source) / "state.txt").read_text(encoding="utf-8")
            return {"state": self.state}
    adapter = CrashingAdapter()
    monkeypatch.setattr(main, "get_adapter", lambda: adapter)
    with main.connect() as db:
        db.execute("INSERT INTO runs(id,status,episodes,created_at) VALUES(?,?,?,?)",
                   ("crash-rollback-run", "queued", 1, main.now()))
    main.execute_training("crash-rollback-run", {"episodes": 1, "batch_size": 1})
    run = client.get("/api/runs/crash-rollback-run").json()
    assert run["status"] == "failed"
    assert adapter.state == "before-training"
    assert "restored automatically" in run["error"]

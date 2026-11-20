from fastapi.testclient import TestClient
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
    assert "prove code/factual correctness" in result["warning"]

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
            return {"saved": destination}
        def load_checkpoint(self, source):
            return {"loaded": source}
    adapter = FakeAdapter()
    monkeypatch.setattr(main, "get_adapter", lambda: adapter)
    with main.connect() as db:
        db.execute("INSERT INTO runs(id,status,episodes,created_at) VALUES(?,?,?,?)",
                   ("fake-run", "queued", 2, main.now()))
    main.execute_training("fake-run", {"episodes": 2, "batch_size": 2, "duration_hours": 0})
    run = client.get("/api/runs/fake-run").json()
    assert run["status"] == "completed"
    assert run["episodes_completed"] == 2
    assert run["real_updates"] == 2
    assert adapter.sft == 1 and adapter.rl == 1
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
    with pytest.raises(RuntimeError):
        main.validate_update_result({"updates": 0, "loss": 0.5}, "train_step")
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
            return {}
        def load_checkpoint(self, source):
            return {}

    monkeypatch.setattr(main, "get_adapter", lambda: BadAdapter())
    with main.connect() as db:
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


def test_task_delete_removes_its_attempt_history(tmp_path, monkeypatch):
    client = setup_client(tmp_path, monkeypatch)
    task = client.get("/api/tasks").json()[0]
    main.record_attempt(task["id"], "", "test prompt", "test output", {
        "score": 50, "reward": 0, "evaluator": "test", "checks": {}
    })
    response = client.delete(f"/api/tasks/{task['id']}")
    assert response.status_code == 200
    with main.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM task_attempts WHERE task_id=?", (task["id"],)).fetchone()[0] == 0


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
            return {}
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

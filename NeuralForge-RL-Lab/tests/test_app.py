from fastapi.testclient import TestClient
from backend import main

def setup_client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DB", tmp_path / "test.sqlite3")
    monkeypatch.setattr(main, "MISTAKE_LOG", tmp_path / "mistakes.jsonl")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    monkeypatch.setattr(main, "DATA_INBOX", inbox)
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
    assert "not prove" in result["warning"]

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

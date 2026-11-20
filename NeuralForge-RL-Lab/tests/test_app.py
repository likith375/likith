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
        "verification": "verified", "note": "Checked manually"
    })
    assert reviewed.status_code == 200
    after = client.get("/api/mistakes?verification=verified").json()[0]
    assert before["output"] == after["output"] == "wrong output"
    assert after["verification"] == "verified"
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

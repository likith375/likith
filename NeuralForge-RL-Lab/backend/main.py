"""NeuralForge RL Lab — local-first, domain-independent API and training orchestration.

Uploaded/generated code is never executed by this server. The default evaluator
uses static heuristics (plus Python AST syntax checks); trusted adapters may
supply domain-specific evaluation without executing untrusted output.
"""
from __future__ import annotations
import ast, csv, hashlib, importlib, io, json, os, random, re, sqlite3, threading, time, uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
DB = ROOT / "neuralforge.sqlite3"
DATA_DIR.mkdir(exist_ok=True)
DATA_INBOX = DATA_DIR / "inbox"
DATA_INBOX.mkdir(exist_ok=True)
DATA_EXAMPLES_DIR = DATA_DIR / "examples"
DATA_EXAMPLES_DIR.mkdir(exist_ok=True)
MISTAKES_DIR = DATA_DIR / "mistakes"
MISTAKES_DIR.mkdir(exist_ok=True)
MISTAKE_LOG = MISTAKES_DIR / "mistakes.jsonl"
MISTAKE_LOG_LOCK = threading.Lock()
CHECKPOINT_DIR.mkdir(exist_ok=True)
app = FastAPI(title="NeuralForge RL Lab", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")
_adapter_lock = threading.RLock()
_adapter_instance = None
_adapter_fingerprint = None
_adapter_error = None
RUN_SCHEDULER_LOCK = threading.Lock()
MODEL_EXECUTION_LOCK = threading.RLock()

def now() -> str:
    return datetime.now(timezone.utc).isoformat()

def append_mistake_log(event: str, payload: dict[str, Any]):
    """Append an audit event; SQLite remains the queryable source of truth."""
    record={"event":event,"timestamp":now(),**payload}
    with MISTAKE_LOG_LOCK:
        with MISTAKE_LOG.open("a",encoding="utf-8") as handle:
            handle.write(json.dumps(record,ensure_ascii=False)+"\n")

@contextmanager
def connect():
    c = sqlite3.connect(DB, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()

def columns(c, table):
    return {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}

def init_db():
    with connect() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS datasets(
          id TEXT PRIMARY KEY,name TEXT NOT NULL,category TEXT NOT NULL,split TEXT NOT NULL,
          content TEXT NOT NULL,metadata_json TEXT DEFAULT '{}',created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tasks(
          id TEXT PRIMARY KEY,title TEXT NOT NULL,prompt TEXT NOT NULL,expected TEXT DEFAULT '',
          difficulty TEXT DEFAULT 'easy',category TEXT DEFAULT 'general',enabled INTEGER DEFAULT 1,
          created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS mistakes(
          id TEXT PRIMARY KEY,task_id TEXT DEFAULT '',prompt TEXT NOT NULL,output TEXT NOT NULL,
          category TEXT NOT NULL,evidence TEXT DEFAULT '',correction TEXT DEFAULT '',
          verification TEXT DEFAULT 'unverified',evaluator TEXT DEFAULT 'static',created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS mistake_events(
          id TEXT PRIMARY KEY,mistake_id TEXT NOT NULL,action TEXT NOT NULL,note TEXT DEFAULT '',
          correction TEXT DEFAULT '',evidence TEXT DEFAULT '',created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS runs(
          id TEXT PRIMARY KEY,status TEXT NOT NULL,episodes INTEGER NOT NULL,reward REAL DEFAULT 0,
          success_rate REAL DEFAULT 0,real_updates INTEGER DEFAULT 0,notes TEXT DEFAULT '',
          created_at TEXT NOT NULL,started_at TEXT,finished_at TEXT,error TEXT,avg_loss REAL DEFAULT 0,
          model_name TEXT DEFAULT '',stop_requested INTEGER DEFAULT 0,
          duration_hours REAL DEFAULT 0,episodes_completed INTEGER DEFAULT 0,
          human_feedback_examples INTEGER DEFAULT 0,feedback_updates INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS checkpoints(
          id TEXT PRIMARY KEY,run_id TEXT DEFAULT '',label TEXT NOT NULL,kind TEXT NOT NULL,
          created_at TEXT NOT NULL,path TEXT DEFAULT '',metadata_json TEXT DEFAULT '{}');
        CREATE TABLE IF NOT EXISTS adapter_settings(
          key TEXT PRIMARY KEY,value TEXT NOT NULL,updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS benchmarks(
          id TEXT PRIMARY KEY,split TEXT NOT NULL,examples INTEGER NOT NULL,avg_score REAL DEFAULT 0,
          pass_rate REAL DEFAULT 0,details_json TEXT DEFAULT '[]',created_at TEXT NOT NULL,model_name TEXT DEFAULT '');
        CREATE TABLE IF NOT EXISTS task_attempts(
          id TEXT PRIMARY KEY,task_id TEXT NOT NULL,run_id TEXT DEFAULT '',prompt TEXT NOT NULL,
          output TEXT NOT NULL,score REAL NOT NULL,reward REAL NOT NULL,evaluator TEXT DEFAULT '',
          checks_json TEXT DEFAULT '{}',created_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_task_attempts_task_created
          ON task_attempts(task_id,created_at DESC);
        CREATE TABLE IF NOT EXISTS human_feedback(
          id TEXT PRIMARY KEY,attempt_id TEXT NOT NULL,rating INTEGER NOT NULL,
          reward REAL NOT NULL,note TEXT DEFAULT '',correction TEXT DEFAULT '',
          evidence TEXT DEFAULT '',consumed_at TEXT,created_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_human_feedback_pending
          ON human_feedback(consumed_at,created_at);
        CREATE INDEX IF NOT EXISTS idx_human_feedback_attempt
          ON human_feedback(attempt_id,created_at DESC);
        """)
        # Upgrade the early prototype database without destroying user data.
        migrations = {
          "datasets": {"metadata_json": "TEXT DEFAULT '{}'"},
          "tasks": {"category": "TEXT DEFAULT 'general'", "enabled": "INTEGER DEFAULT 1"},
          "mistake_events": {"correction": "TEXT DEFAULT ''", "evidence": "TEXT DEFAULT ''"},
          "runs": {"started_at": "TEXT", "finished_at": "TEXT", "error": "TEXT",
                   "avg_loss": "REAL DEFAULT 0", "model_name": "TEXT DEFAULT ''",
                   "stop_requested": "INTEGER DEFAULT 0", "duration_hours": "REAL DEFAULT 0",
                   "episodes_completed": "INTEGER DEFAULT 0",
                   "human_feedback_examples": "INTEGER DEFAULT 0",
                   "feedback_updates": "INTEGER DEFAULT 0"},
          "checkpoints": {"path": "TEXT DEFAULT ''", "metadata_json": "TEXT DEFAULT '{}'"},
        }
        for table, wanted in migrations.items():
            existing = columns(c, table)
            for name, decl in wanted.items():
                if name not in existing:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
        # Mark runs left active by a previous server process as interrupted.
        c.execute("UPDATE runs SET status='interrupted',finished_at=?,error=COALESCE(error,'Server restarted before this run finished.'),notes='Run interrupted by server restart; review checkpoint and resume with a new run.' WHERE status IN ('queued','running')",(now(),))
        if c.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 0:
            seed = [
              ("Python example","programming","def add(a, b):\n    return a + b","train"),
              ("Science example","science","Photosynthesis converts light energy into chemical energy.","train"),
              ("Mathematics example","mathematics","The area of a rectangle is length multiplied by width.","validation"),
              ("Writing example","writing","A clear paragraph begins with a focused topic sentence.","test"),
            ]
            for name, cat, code, split in seed:
                c.execute("INSERT INTO datasets(id,name,category,split,content,created_at) VALUES(?,?,?,?,?,?)",
                          (str(uuid.uuid4()),name,cat,split,code,now()))
        if c.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0:
            c.execute("INSERT INTO tasks(id,title,prompt,expected,difficulty,category,created_at) VALUES(?,?,?,?,?,?,?)",
                      (str(uuid.uuid4()),"Explore a subject","Explain one important concept from the subject being studied, using a clear example.",
                       "","easy","general",now()))
        defaults = {"adapter_spec":"", "model_path":"", "max_new_tokens":"384",
                    "learning_rate":"0.00001", "batch_size":"2", "reward_threshold":"0.45",
                    "use_lora":"true", "lora_r":"8", "lora_alpha":"16",
                    "lora_dropout":"0.05", "lora_target_modules":""}
        for k,v in defaults.items():
            c.execute("INSERT OR IGNORE INTO adapter_settings(key,value,updated_at) VALUES(?,?,?)",(k,v,now()))

init_db()

class TaskIn(BaseModel):
    title: str = Field(min_length=1, max_length=180)
    prompt: str = Field(min_length=1, max_length=12000)
    expected: str = Field(default="", max_length=20000)
    difficulty: str = Field(default="easy", max_length=40)
    category: str = Field(default="general", max_length=80)

class MistakeIn(BaseModel):
    task_id: str = Field(default="", max_length=80)
    prompt: str = Field(min_length=1, max_length=12000)
    output: str = Field(min_length=1, max_length=30000)
    category: str = Field(default="failed_tasks", max_length=80)
    evidence: str = Field(default="", max_length=4000)
    correction: str = Field(default="", max_length=20000)

class ReviewIn(BaseModel):
    verification: str = Field(min_length=1, max_length=40)
    note: str = Field(default="", max_length=4000)
    correction: str = Field(default="", max_length=20000)
    evidence: str = Field(default="", max_length=4000)

class RunIn(BaseModel):
    episodes: int = Field(default=5000, ge=1, le=1000000)
    batch_size: int | None = Field(default=None, ge=1, le=32)
    duration_hours: float = Field(default=0, ge=0, le=24)

class FeedbackIn(BaseModel):
    rating: int = Field(ge=1, le=5, strict=True)
    note: str = Field(default="", max_length=4000)
    correction: str = Field(default="", max_length=20000)
    evidence: str = Field(default="", max_length=4000)

class AdapterConfig(BaseModel):
    adapter_spec: str = Field(default="", max_length=300)
    model_path: str = Field(default="", max_length=1000)
    max_new_tokens: int = Field(default=384, ge=16, le=4096)
    learning_rate: float = Field(default=0.00001, gt=0, le=0.1)
    batch_size: int = Field(default=2, ge=1, le=32)
    reward_threshold: float = Field(default=0.45, ge=0, le=1)
    use_lora: bool = True
    lora_r: int = Field(default=8, ge=1, le=256)
    lora_alpha: int = Field(default=16, ge=1, le=1024)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    lora_target_modules: str = Field(default="", max_length=500)

def setting_map():
    with connect() as c:
        return {r["key"]:r["value"] for r in c.execute("SELECT key,value FROM adapter_settings")}

def get_adapter(force_reload=False):
    global _adapter_instance, _adapter_fingerprint, _adapter_error
    cfg = setting_map()
    spec = cfg.get("adapter_spec","").strip()
    model_path = cfg.get("model_path","").strip()
    fingerprint = (spec, model_path, tuple(sorted((k, str(v)) for k, v in cfg.items())))
    if not spec and model_path:
        spec = "backend.hf_adapter:load_adapter"
        fingerprint = (spec, model_path)
    if not spec:
        return None
    with _adapter_lock:
        if _adapter_instance is not None and _adapter_fingerprint == fingerprint and not force_reload:
            return _adapter_instance
        try:
            module_name, sep, factory_name = spec.partition(":")
            if not sep:
                raise ValueError("Adapter spec must look like package.module:factory")
            factory = getattr(importlib.import_module(module_name), factory_name)
            # Adapter factories have one explicit contract: accept the settings dictionary.
            # Do not catch TypeError here; it may be a real bug inside the adapter constructor. 
            adapter = factory({"model_path":model_path, **cfg})
            for method in ("generate","train_step","save_checkpoint","load_checkpoint"):
                if not callable(getattr(adapter,method,None)):
                    raise TypeError(f"Adapter is missing required method: {method}")
            _adapter_instance, _adapter_fingerprint, _adapter_error = adapter, fingerprint, None
            return adapter
        except Exception as exc:
            _adapter_instance, _adapter_fingerprint, _adapter_error = None, fingerprint, f"{type(exc).__name__}: {exc}"
            raise RuntimeError(_adapter_error) from exc

def static_evaluate(output: str, expected: str = "", domain: str = "general") -> dict[str, Any]:
    """Conservative text heuristic: without a reference, return neutral rather than fake success."""
    text = (output or "").strip()
    if not text:
        return {"score":0.0,"reward":-1.0,"checks":{"non_empty":False},
                "expected_match":False,"evaluator":"generic_static_heuristic",
                "warning":"Generic static heuristic only; no code or claim was executed."}
    checks = {
      "non_empty": True,
      "no_placeholder": not any(x in text.lower() for x in ("todo", "demo output", "insert code here", "lorem ipsum")),
      "balanced_delimiters": all(text.count(x) >= text.count(y) for x,y in (("(",")"),("{","}"),("[","]"))),
    }
    has_reference=bool(expected.strip())
    expected_match=has_reference and expected.strip() in text
    checks["reference_match"]=expected_match if has_reference else None
    structurally_ok=checks["no_placeholder"] and checks["balanced_delimiters"]
    if has_reference:
        score=100.0 if expected_match and structurally_ok else 25.0
    else:
        # No answer key means correctness is unknown. 50 is neutral, not a pass.
        score=50.0 if structurally_ok else 25.0
    return {"score":score,"reward":round((score/50.0)-1.0,4),"checks":checks,
            "expected_match":expected_match,"evaluator":"generic_static_heuristic",
            "warning":"Generic static heuristic only. Without a reference, score 50 is neutral; it does not execute code or prove factual correctness."}


def evaluate_python_static(output: str, expected: str = "") -> dict[str, Any]:
    """Check Python syntax without executing generated code."""
    result = static_evaluate(output, expected, "python")
    text = (output or "").strip()
    fences = re.findall(r"```(?:python|py)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidate = fences[0].strip() if fences else text
    looks_like_code = bool(re.search(
        r"(?m)^\s*(async\s+def|def|class|from|import|@|if\s+.+:|for\s+.+:|while\s+.+:|try:|with\s+.+:|return\b|[A-Za-z_]\w*\s*=(?!=))|\b(print|return|yield)\s*\(",
        candidate
    ))
    if not looks_like_code:
        return result
    try:
        ast.parse(candidate)
        result["checks"]["python_syntax_valid"] = True
        result["evaluator"] = "python_ast_static"
        result["warning"] = "Python AST syntax check only; code was not executed and correctness is not proven."
    except SyntaxError as exc:
        result["checks"]["python_syntax_valid"] = False
        result["syntax_error"] = {"line": exc.lineno, "column": exc.offset, "message": exc.msg}
        result["score"] = min(float(result["score"]), 25.0)
        result["reward"] = round((result["score"] / 50.0) - 1.0, 4)
        result["evaluator"] = "python_ast_static"
        result["warning"] = "Python syntax error detected by AST parsing; code was not executed."
    return result


def evaluate_with_adapter(adapter, prompt: str, output: str, expected: str = "", domain: str = "general") -> dict[str, Any]:
    """Use an adapter's domain-specific evaluator when supplied; otherwise be explicit about heuristic limits."""
    evaluator = getattr(adapter, "evaluate", None)
    if not callable(evaluator):
        if str(domain).strip().lower() in {"python", "py", "python3"}:
            return evaluate_python_static(output, expected)
        return static_evaluate(output, expected, domain)
    result = evaluator(prompt, output, expected, domain)
    if not isinstance(result, dict):
        raise TypeError("Adapter evaluate() must return a dictionary.")
    score = float(result.get("score"))
    reward = float(result.get("reward", (score / 50.0) - 1.0))
    if not (0.0 <= score <= 100.0) or not (-1.0 <= reward <= 1.0):
        raise ValueError("Adapter evaluator score must be 0..100 and reward must be -1..1.")
    checks=result.get("checks",{})
    if not isinstance(checks,dict):
        raise ValueError("Adapter evaluator 'checks' must be a dictionary.")
    return {**result, "score": round(score, 2), "reward": round(reward, 4),
            "expected_match": result.get("expected_match"),
            "checks": checks,
            "evaluator": result.get("evaluator", type(adapter).__name__),
            "warning": result.get("warning", "Domain-specific adapter evaluator; inspect its implementation and evidence.")}


def checkpoint_has_artifacts(path: str | Path) -> bool:
    """A directory alone is not proof that model weights were checkpointed."""
    root=Path(path)
    return root.is_dir() and any(item.is_file() and item.stat().st_size>0 for item in root.rglob("*"))


def validate_update_result(result: Any, operation: str) -> tuple[int, float | None]:
    """Never count a missing/empty adapter response as a real model update."""
    if not isinstance(result, dict):
        raise TypeError(f"{operation} must return a dictionary containing an explicit updates count.")
    updates = result.get("updates")
    if type(updates) is not int or updates < 0:
        raise ValueError(f"{operation} must return a non-negative integer updates field.")
    if operation == "train_step" and updates < 1:
        raise RuntimeError("The adapter reported zero reward-learning updates; the run cannot claim learning.")
    loss = result.get("loss")
    if loss is not None:
        loss = float(loss)
        if not (loss == loss and abs(loss) != float("inf")):
            raise ValueError(f"{operation} returned a non-finite loss.")
    return updates, loss


def retrieval_terms(value: str) -> set[str]:
    stop={"the","and","for","with","from","this","that","what","when","where","which","your","you","are","was","were","how","why","into","about","using","use","make","create","explain","write","show","give"}
    return {token for token in re.findall(r"[a-z0-9_+#.-]{2,}",str(value or "").lower()) if token not in stop}


def verified_examples(limit=6, category: str | None=None):
    with connect() as c:
        rows = c.execute("""
          SELECT m.prompt,
            COALESCE(NULLIF((SELECT e.correction FROM mistake_events e WHERE e.mistake_id=m.id AND e.correction<>''
                            ORDER BY e.created_at DESC,e.id DESC LIMIT 1),''),m.correction) AS correction,
            COALESCE(NULLIF(t.category,''),m.category) AS domain
          FROM mistakes m LEFT JOIN tasks t ON t.id=m.task_id
          WHERE COALESCE((SELECT e.action FROM mistake_events e WHERE e.mistake_id=m.id
                          ORDER BY e.created_at DESC,e.id DESC LIMIT 1),m.verification)='verified'
            AND COALESCE(NULLIF((SELECT e.correction FROM mistake_events e WHERE e.mistake_id=m.id AND e.correction<>''
                          ORDER BY e.created_at DESC,e.id DESC LIMIT 1),''),m.correction)<>''
          ORDER BY m.created_at DESC LIMIT 500
        """).fetchall()
    if category:
        matching=[row for row in rows if row["domain"] in (category,"general")]
        rows=matching
    return [{"prompt":r["prompt"],"correction":r["correction"],"category":r["domain"]} for r in rows[:limit]]


def dataset_examples(category="general", prompt: str="", limit=4):
    with connect() as c:
        rows=c.execute("""
          SELECT name,category,content,created_at FROM datasets
          WHERE split='train' ORDER BY created_at DESC LIMIT 1000
        """).fetchall()
    query_terms=retrieval_terms(str(category)+" "+str(prompt))
    scored=[]
    for row in rows:
        name_terms=retrieval_terms(row["name"])
        content_terms=retrieval_terms(row["content"][:5000])
        example_category=str(row["category"]).lower()
        score=(5.0 if example_category==str(category).lower() else
               1.0 if example_category=="general" else 0.0)
        score += 3.0*len(query_terms & name_terms)
        score += 0.35*len(query_terms & content_terms)
        scored.append((score,row["created_at"],row))
    scored.sort(key=lambda item:(item[0],item[1]),reverse=True)
    # If there is no lexical/category match, recent examples remain a small fallback.
    return [{"prompt":r["name"],"category":r["category"],"content":r["content"]}
            for _,_,r in scored[:limit]]

def build_prompt(task):
    domain = task.get("domain") or task.get("category") or "general"
    verified = verified_examples(category=domain)
    references = dataset_examples(domain, task.get("prompt",""))
    parts=[task["prompt"], "\n\nWork in the subject/domain: " + str(domain) + ". Use supplied context only when relevant. Follow the requested output format, state assumptions, and do not invent facts."]
    if references:
        parts.append("\n\nRelevant reference examples from uploaded training material:")
        for i,e in enumerate(references,1):
            parts.append("Reference " + str(i) + " [" + e["category"] + "] " + e["prompt"] + ":\n" + e["content"][:1400])
    if verified:
        parts.append("\n\nVerified examples from human-reviewed mistake corrections:")
        for i,e in enumerate(verified,1):
            parts.append("Correction " + str(i) + " task: " + e["prompt"] + "\nVerified correction:\n" + e["correction"][:1400])
    parts.append("\nNow solve the new task. Do not copy an unrelated example.")
    return "\n".join(parts)

def record_attempt(task_id: str, run_id: str, prompt: str, output: str, evaluation: dict[str, Any]):
    """Persist immutable task-level evaluation data for curriculum scheduling and progress."""
    attempt_id=str(uuid.uuid4())
    with connect() as c:
        c.execute(
            "INSERT INTO task_attempts(id,task_id,run_id,prompt,output,score,reward,evaluator,checks_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (attempt_id,task_id,run_id,prompt,output,float(evaluation["score"]),
             float(evaluation["reward"]),str(evaluation.get("evaluator","static_heuristic")),
             json.dumps(evaluation.get("checks",{}),ensure_ascii=False),now())
        )
    return attempt_id


def save_mistake(task_id,prompt,output,evaluation,category="failed_tasks"):
    mid=str(uuid.uuid4())
    with connect() as c:
        c.execute("INSERT INTO mistakes(id,task_id,prompt,output,category,evidence,correction,verification,evaluator,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                  (mid,task_id,prompt,output,category,json.dumps(evaluation), "", "unverified",str(evaluation.get("evaluator","generic_static_heuristic")),now()))
    append_mistake_log("mistake_created",{"id":mid,"task_id":task_id,"prompt":prompt,"output":output,"category":category,"evidence":evaluation})
    return mid

def execute_training(run_id, request):
    """Serialize training against generation and checkpoint operations."""
    with MODEL_EXECUTION_LOCK:
        return _execute_training_locked(run_id, request)


def _execute_training_locked(run_id, request):
    started=now()
    with connect() as c:
        c.execute("UPDATE runs SET status='running',started_at=? WHERE id=?",(started,run_id))
        task_rows=c.execute("SELECT * FROM tasks WHERE enabled=1 ORDER BY created_at").fetchall()
        train_rows=c.execute("SELECT name,category,content FROM datasets WHERE split='train' ORDER BY created_at LIMIT 10000").fetchall()
        task_recent_scores={}
        for task_row in task_rows:
            recent=c.execute("SELECT AVG(score) FROM (SELECT score FROM task_attempts WHERE task_id=? ORDER BY created_at DESC LIMIT 20)",(task_row["id"],)).fetchone()[0]
            task_recent_scores[task_row["id"]]=float(recent) if recent is not None else 50.0
    if not task_rows:
        with connect() as c:c.execute("UPDATE runs SET status='failed',error=?,finished_at=? WHERE id=?",("Create at least one enabled task.",now(),run_id))
        return
    try:
        adapter=get_adapter()
        if adapter is None:
            raise RuntimeError("No model is connected. Configure an adapter or a Hugging Face model path first.")
        model_name=getattr(adapter,"name",type(adapter).__name__)
        verified_rows=verified_examples(limit=500)
        # Put verified corrections first so they are not starved by a large raw corpus.
        supervised_rows=[{
            "name":"Verified correction: "+row["prompt"],
            "category":row["category"],
            "content":row["correction"]
        } for row in verified_rows]
        supervised_rows.extend(dict(row) for row in train_rows)
        # Preserve a restorable model snapshot before the first optimizer update.
        backup_id=str(uuid.uuid4())
        backup_path=str(CHECKPOINT_DIR/backup_id)
        backup_metadata=adapter.save_checkpoint(backup_path) or {}
        if not checkpoint_has_artifacts(backup_path):
            raise RuntimeError("Automatic pre-training checkpoint failed: adapter did not write any non-empty checkpoint files.")
        with connect() as c:
            c.execute("INSERT INTO checkpoints(id,run_id,label,kind,created_at,path,metadata_json) VALUES(?,?,?,?,?,?,?)",
                      (backup_id,run_id,"Automatic backup before training run","model_weights",now(),backup_path,json.dumps(backup_metadata)))
            c.execute("UPDATE runs SET model_name=? WHERE id=?",(model_name,run_id))
        cfg=setting_map()
        episodes=request["episodes"]
        batch_size=request.get("batch_size") or int(cfg.get("batch_size","2"))
        duration_hours=float(request.get("duration_hours",0))
        deadline=time.monotonic()+duration_hours*3600 if duration_hours>0 else None
        updates=0; losses=[]; feedback_updates=0; feedback_examples=0; feedback_ids_to_consume=[]
        with connect() as c:
            initial_state=c.execute("SELECT stop_requested FROM runs WHERE id=?",(run_id,)).fetchone()
        if (initial_state and initial_state["stop_requested"]) or (deadline and time.monotonic()>=deadline):
            reason="Stop requested before the first training batch." if initial_state and initial_state["stop_requested"] else "Time budget expired before the first training batch."
            with connect() as c:
                c.execute("UPDATE runs SET status='stopped',finished_at=?,notes=?,episodes_completed=0 WHERE id=?",
                          (now(),reason,run_id))
            return
        # Human ratings are the strongest available reward signal. They are
        # consumed only after the updated weights have been checkpointed.
        with connect() as c:
            pending_feedback=c.execute("""
              SELECT f.id,f.reward,a.prompt,a.output,a.task_id
              FROM human_feedback f JOIN task_attempts a ON a.id=f.attempt_id
              WHERE f.consumed_at IS NULL
              ORDER BY f.created_at,f.id LIMIT 64
            """).fetchall()
        feedback_batch=[{"prompt":row["prompt"],"completion":row["output"],
                         "reward":float(row["reward"]),"task_id":row["task_id"],
                         "feedback_id":row["id"]} for row in pending_feedback]
        if feedback_batch:
            feedback_result=adapter.train_step(feedback_batch)
            feedback_updates,feedback_loss=validate_update_result(feedback_result,"train_step")
            updates+=feedback_updates
            feedback_examples=len(pending_feedback)
            if feedback_loss is not None: losses.append(feedback_loss)
            feedback_ids_to_consume=[row["id"] for row in pending_feedback]
            with connect() as c:
                c.execute("UPDATE runs SET human_feedback_examples=?,feedback_updates=?,real_updates=? WHERE id=?",
                          (feedback_examples,feedback_updates,updates,run_id))
        samples=[]; total_reward=0.0; successes=0; supervised_cursor=0; completed_episodes=0
        for episode in range(episodes):
            with connect() as c:
                run_state=c.execute("SELECT stop_requested FROM runs WHERE id=?",(run_id,)).fetchone()
            if (run_state and run_state["stop_requested"]) or (deadline and time.monotonic()>=deadline):
                break
            # Adaptive curriculum: weakly mastered tasks receive more practice,
            # while every enabled task retains a non-zero chance of selection.
            weights=[max(0.2, 1.0 + (70.0-task_recent_scores.get(t["id"],50.0))/70.0) for t in task_rows]
            task=random.choices(task_rows,weights=weights,k=1)[0]
            prompt=build_prompt(dict(task))
            try:
                generated=adapter.generate(prompt, {"task_id":task["id"],"difficulty":task["difficulty"],"category":task["category"],
                                                   "max_new_tokens":int(cfg.get("max_new_tokens","384"))})
                output=generated if isinstance(generated,str) else str(generated.get("text",""))
                evaluation=evaluate_with_adapter(adapter, prompt, output, task["expected"], task["category"])
                reward=evaluation["reward"]
                record_attempt(task["id"],run_id,prompt,output,evaluation)
                with connect() as c:
                    updated_score=c.execute(
                        "SELECT AVG(score) FROM (SELECT score FROM task_attempts WHERE task_id=? ORDER BY created_at DESC LIMIT 20)",
                        (task["id"],)
                    ).fetchone()[0]
                task_recent_scores[task["id"]]=float(updated_score) if updated_score is not None else 50.0
                total_reward += reward
                successes += int(evaluation["score"] >= 75)
                samples.append({"prompt":prompt,"completion":output,"reward":reward,"task_id":task["id"]})
                if evaluation["score"] < float(cfg.get("reward_threshold","0.45"))*100:
                    save_mistake(task["id"],prompt,output,evaluation)
            except Exception as exc:
                save_mistake(task["id"],prompt,"",{"error":str(exc)},"generation_errors")
                raise
            if len(samples) >= batch_size or episode == episodes-1:
                # First learn from curated train-split examples when the adapter supports SFT.
                if supervised_rows and callable(getattr(adapter,"supervised_step",None)):
                    sup_batch=[]
                    for _ in range(min(batch_size,len(supervised_rows))):
                        item=supervised_rows[supervised_cursor % len(supervised_rows)]; supervised_cursor+=1
                        sup_batch.append({"prompt":item["name"],"completion":item["content"],"category":item["category"]})
                    sup_result=adapter.supervised_step(sup_batch)
                    sup_updates, sup_loss = validate_update_result(sup_result, "supervised_step")
                    updates += sup_updates
                    if sup_loss is not None: losses.append(sup_loss)
                result=adapter.train_step(samples)
                reward_updates, reward_loss = validate_update_result(result, "train_step")
                updates += reward_updates
                if reward_loss is not None: losses.append(reward_loss)
                samples=[]
            completed_episodes=episode+1
            with connect() as c:
                c.execute("UPDATE runs SET reward=?,success_rate=?,real_updates=?,avg_loss=?,episodes_completed=? WHERE id=?",
                          (total_reward/completed_episodes,successes/completed_episodes,updates,
                           sum(losses)/len(losses) if losses else 0,completed_episodes,run_id))
        # Flush a partial batch if the user stopped the run or its time budget expired.
        if samples:
            if supervised_rows and callable(getattr(adapter,"supervised_step",None)):
                sup_batch=[]
                for _ in range(min(batch_size,len(supervised_rows))):
                    item=supervised_rows[supervised_cursor % len(supervised_rows)]; supervised_cursor+=1
                    sup_batch.append({"prompt":item["name"],"completion":item["content"],"category":item["category"]})
                sup_result=adapter.supervised_step(sup_batch)
                sup_updates, sup_loss = validate_update_result(sup_result, "supervised_step")
                updates += sup_updates
                if sup_loss is not None: losses.append(sup_loss)
            result=adapter.train_step(samples)
            reward_updates, reward_loss = validate_update_result(result, "train_step")
            updates += reward_updates
            if reward_loss is not None: losses.append(reward_loss)
        final_checkpoint_id=None
        if updates>0:
            final_checkpoint_id=str(uuid.uuid4())
            final_checkpoint_path=str(CHECKPOINT_DIR/final_checkpoint_id)
            final_metadata=adapter.save_checkpoint(final_checkpoint_path) or {}
            if not checkpoint_has_artifacts(final_checkpoint_path):
                raise RuntimeError("Post-training checkpoint failed: adapter did not write any non-empty checkpoint files.")
            with connect() as c:
                c.execute("INSERT INTO checkpoints(id,run_id,label,kind,created_at,path,metadata_json) VALUES(?,?,?,?,?,?,?)",
                          (final_checkpoint_id,run_id,"Automatic checkpoint after training run","model_weights",
                           now(),final_checkpoint_path,json.dumps(final_metadata)))
                if feedback_ids_to_consume:
                    placeholders=",".join("?" for _ in feedback_ids_to_consume)
                    c.execute(f"UPDATE human_feedback SET consumed_at=? WHERE id IN ({placeholders}) AND consumed_at IS NULL",
                              [now(),*feedback_ids_to_consume])
        with connect() as c:
            state=c.execute("SELECT stop_requested FROM runs WHERE id=?",(run_id,)).fetchone()
            stopped=bool(state and state["stop_requested"])
            duration_hit=bool(deadline and time.monotonic()>=deadline)
            status="stopped" if (stopped or duration_hit) else "completed"
            note=("Stopped by user. " if stopped else "")+("Time budget reached. " if duration_hit else "")+f"Processed {completed_episodes} episodes. Training uses the configured adapter."
            if final_checkpoint_id:
                note+=f" Final model checkpoint: {final_checkpoint_id}."
            elif updates==0:
                note+=" No optimizer updates were reported; only the pre-training backup is available."
            c.execute("UPDATE runs SET status=?,finished_at=?,notes=?,real_updates=?,avg_loss=?,episodes_completed=?,human_feedback_examples=?,feedback_updates=? WHERE id=?",
                      (status,now(),note,updates,sum(losses)/len(losses) if losses else 0,
                       completed_episodes,feedback_examples,feedback_updates,run_id))
    except Exception as exc:
        with connect() as c:
            c.execute("UPDATE runs SET status='failed',finished_at=?,error=?,notes=? WHERE id=?",
                      (now(),f"{type(exc).__name__}: {exc}","No success is claimed when adapter training fails.",run_id))

@app.get("/",response_class=HTMLResponse)
def home():
    return (ROOT/"frontend"/"index.html").read_text(encoding="utf-8")

@app.get("/health")
def health(): return {"ok":True,"app":"NeuralForge RL Lab","version":"1.0.0"}

@app.get("/api/overview")
def overview():
    with connect() as c:
        return {"datasets":c.execute("SELECT COUNT(*) FROM datasets").fetchone()[0],
                "tasks":c.execute("SELECT COUNT(*) FROM tasks WHERE enabled=1").fetchone()[0],
                "mistakes":c.execute("SELECT COUNT(*) FROM mistakes").fetchone()[0],
                "verified":c.execute("""SELECT COUNT(*) FROM mistakes m WHERE COALESCE(
                  (SELECT e.action FROM mistake_events e WHERE e.mistake_id=m.id
                   ORDER BY e.created_at DESC,e.id DESC LIMIT 1),m.verification)='verified'""").fetchone()[0],
                "human_feedback":c.execute("SELECT COUNT(*) FROM human_feedback").fetchone()[0],
                "pending_feedback":c.execute("SELECT COUNT(*) FROM human_feedback WHERE consumed_at IS NULL").fetchone()[0],
                "runs":c.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
                "completed_runs":c.execute("SELECT COUNT(*) FROM runs WHERE status='completed'").fetchone()[0]}

@app.get("/api/datasets")
def list_datasets(split: str | None=None, category: str | None=None):
    query="SELECT * FROM datasets WHERE 1=1"; args=[]
    if split: query+=" AND split=?"; args.append(split)
    if category: query+=" AND category=?"; args.append(category)
    query+=" ORDER BY created_at DESC LIMIT 2000"
    with connect() as c: return [dict(x) for x in c.execute(query,args)]

@app.get("/api/datasets/stats")
def dataset_stats():
    with connect() as c:
        total=c.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        by_split={row["split"]:row["count"] for row in c.execute(
            "SELECT split,COUNT(*) AS count FROM datasets GROUP BY split")}
        by_category=[dict(row) for row in c.execute(
            "SELECT category,COUNT(*) AS count FROM datasets GROUP BY category ORDER BY count DESC,category")]
        lengths=c.execute("SELECT MIN(LENGTH(content)),AVG(LENGTH(content)),MAX(LENGTH(content)) FROM datasets").fetchone()
        duplicates=c.execute("""SELECT COUNT(*) FROM (
            SELECT content FROM datasets GROUP BY content HAVING COUNT(*)>1)""").fetchone()[0]
        empty=c.execute("SELECT COUNT(*) FROM datasets WHERE TRIM(content)=''").fetchone()[0]
    return {"total":total,"by_split":by_split,"by_category":by_category,
            "content_characters":{"min":lengths[0] or 0,"average":round(lengths[1] or 0,1),"max":lengths[2] or 0},
            "duplicate_content_groups":duplicates,"empty_examples":empty}


@app.get("/api/training/preview")
def training_preview():
    cfg=setting_map()
    with connect() as c:
        task_count=c.execute("SELECT COUNT(*) FROM tasks WHERE enabled=1").fetchone()[0]
        split_counts={row["split"]:row["count"] for row in c.execute(
            "SELECT split,COUNT(*) AS count FROM datasets GROUP BY split")}
        pending=c.execute("SELECT COUNT(*) FROM human_feedback WHERE consumed_at IS NULL").fetchone()[0]
        verified=c.execute("""SELECT COUNT(*) FROM mistakes m WHERE COALESCE(
            (SELECT e.action FROM mistake_events e WHERE e.mistake_id=m.id
             ORDER BY e.created_at DESC,e.id DESC LIMIT 1),m.verification)='verified'
            AND COALESCE(NULLIF((SELECT e.correction FROM mistake_events e WHERE e.mistake_id=m.id
             AND e.correction<>'' ORDER BY e.created_at DESC,e.id DESC LIMIT 1),''),m.correction)<>''""").fetchone()[0]
        active=c.execute("SELECT id,status FROM runs WHERE status IN ('queued','running') LIMIT 1").fetchone()
    warnings=[]
    if not task_count: warnings.append("Create or generate at least one enabled task.")
    if not split_counts.get("train",0): warnings.append("Import training examples in the train split.")
    if not (cfg.get("adapter_spec","").strip() or cfg.get("model_path","").strip()):
        warnings.append("Connect a model adapter before training.")
    if not pending: warnings.append("No human-rated attempts are waiting for reward replay.")
    if active: warnings.append("A training run is already "+active["status"]+".")
    return {"ready":not any(message.startswith(("Create","Import","Connect","A training")) for message in warnings),
            "enabled_tasks":task_count,"dataset_splits":split_counts,
            "pending_human_feedback":pending,"verified_corrections":verified,
            "adapter_configured":bool(cfg.get("adapter_spec","").strip() or cfg.get("model_path","").strip()),
            "active_run":dict(active) if active else None,"warnings":warnings,
            "note":"This is a configuration/data readiness check. It does not load the model or guarantee that training will succeed."}


# Safe text-based ingestion for common data, source-code, notebook and document formats.
DATASET_EXTENSIONS = {
    ".jsonl", ".json", ".csv", ".txt", ".md", ".rst", ".yaml", ".yml", ".toml", ".xml",
    ".html", ".htm", ".css", ".js", ".jsx", ".ts", ".tsx", ".py", ".ipynb", ".java",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".go", ".rs", ".php", ".rb", ".swift", ".kt",
    ".dart", ".scala", ".sql", ".sh", ".bash", ".zsh", ".bat", ".cmd", ".ps1", ".r",
    ".m", ".pl", ".lua", ".luau", ".vue", ".svelte", ".graphql", ".proto", ".pdf", ".docx", ".xlsx", ".pptx"
}
TEXT_EXTENSIONS = DATASET_EXTENSIONS - {".jsonl", ".json", ".csv", ".ipynb"}
DOCUMENT_CHUNK_EXTENSIONS = TEXT_EXTENSIONS - {".xlsx"}

def chunk_reference_text(text: str, max_chars: int = 3200) -> list[str]:
    """Split prose on paragraph boundaries so long documents fit model context."""
    paragraphs=[part.strip() for part in re.split(r"\n\s*\n", text or "") if part.strip()]
    chunks=[]; current=""
    for paragraph in paragraphs:
        if len(paragraph)>max_chars:
            if current:
                chunks.append(current); current=""
            remaining=paragraph
            while remaining:
                if len(remaining)<=max_chars:
                    chunks.append(remaining.strip()); break
                newline_boundary=remaining.rfind("\n",0,max_chars)
                space_boundary=remaining.rfind(" ",0,max_chars)
                boundary=newline_boundary if newline_boundary>=max_chars//2 else space_boundary
                if boundary<max_chars//2: boundary=max_chars
                chunks.append(remaining[:boundary].strip())
                remaining=remaining[boundary:].strip()
            continue
        candidate=(current+"\n\n"+paragraph).strip() if current else paragraph
        if len(candidate)>max_chars and current:
            chunks.append(current); current=paragraph
        else:
            current=candidate
    if current: chunks.append(current)
    return chunks or ([text.strip()] if (text or "").strip() else [])


def normalize_dataset_row(row: dict[str, Any], index: int, default_category: str = "general") -> dict[str, Any] | None:
    """Accept instruction, QA, chat-message and generic knowledge dataset schemas."""
    prompt_keys=("prompt","instruction","question","input","user","title","name")
    content_keys=("content","code","completion","output","response","answer","target","assistant","text")
    prompt=next((str(row.get(k) or "").strip() for k in prompt_keys
                 if row.get(k) is not None and str(row.get(k)).strip()), "")
    content=next((str(row.get(k) or "").strip() for k in content_keys
                  if row.get(k) is not None and str(row.get(k)).strip()), "")

    # OpenAI-style {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
    # and ShareGPT-style {"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}.
    conversation=row.get("messages") or row.get("conversations")
    if isinstance(conversation,list) and (not prompt or not content):
        turns=[]
        for message in conversation:
            if not isinstance(message,dict): continue
            role=str(message.get("role") or message.get("from") or "").lower()
            value=message.get("content",message.get("value",""))
            if isinstance(value,list):
                value=" ".join(str(part.get("text","")) if isinstance(part,dict) else str(part) for part in value)
            value=str(value or "").strip()
            if value: turns.append((role,value))
        assistant_roles={"assistant","gpt","bot","model"}
        user_roles={"user","human"}
        if turns:
            assistant_index=next((i for i in range(len(turns)-1,-1,-1) if turns[i][0] in assistant_roles),None)
            if assistant_index is not None:
                if not content: content=turns[assistant_index][1]
                if not prompt:
                    context=turns[:assistant_index]
                    prompt="\n".join(f"{role}: {value}" for role,value in context if role in user_roles or role in {"system","developer"})
            elif not prompt:
                prompt="\n".join(f"{role}: {value}" for role,value in turns if role in user_roles)

    if not prompt: prompt=f"Example {index+1}"
    if not content: return None
    excluded=set(prompt_keys+content_keys+("category","split","messages","conversations","metadata"))
    nested_metadata=row.get("metadata") if isinstance(row.get("metadata"),dict) else {}
    metadata={**nested_metadata,**{k:v for k,v in row.items() if k not in excluded}}
    return {"prompt":prompt[:300],"content":content,
            "category":str(row.get("category") or default_category)[:80],
            "split":str(row.get("split") or "").lower(),
            "metadata":metadata}


def prepared_dataset_rows(rows: list[dict[str, Any]], default_category: str, source_name: str,
                          digest: str | None = None, document_chunks: bool = False) -> list[tuple]:
    """Normalize imported schemas and prevent chunks of one document leaking across splits."""
    prepared=[]
    for index,row in enumerate(rows):
        if not isinstance(row,dict): continue
        item=normalize_dataset_row(row,index,default_category)
        if not item: continue
        split=item["split"]
        if split not in ("train","validation","test"):
            # A single document is a reference corpus, not three independent examples.
            split="train" if document_chunks else ("validation" if index%10==8 else "test" if index%10==9 else "train")
        chunks=chunk_reference_text(item["content"]) if document_chunks else [item["content"]]
        for chunk_index,chunk in enumerate(chunks,1):
            name=item["prompt"] if len(chunks)==1 else f"{item['prompt']} (part {chunk_index}/{len(chunks)})"
            metadata={"source_file":source_name,"row_metadata":item["metadata"]}
            if digest: metadata["sha256"]=digest
            if len(chunks)>1: metadata["chunk_index"]=chunk_index; metadata["chunk_count"]=len(chunks)
            prepared.append((str(uuid.uuid4()),name,item["category"],split,chunk,json.dumps(metadata,ensure_ascii=False),now()))
    return prepared


def parse_uploaded_material(filename: str, raw: bytes) -> list[dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError("PDF import requires pypdf. Install requirements.txt and restart.") from exc
        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for page_number, page in enumerate(reader.pages, 1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append({"prompt": f"{Path(filename).name} page {page_number}",
                              "content": page_text, "category": "document", "split": "train"})
        if not pages:
            raise ValueError("No selectable text was found in this PDF. Scanned-image OCR is not supported yet.")
        return pages
    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise ValueError("DOCX import requires python-docx. Install requirements.txt and restart.") from exc
        document = Document(io.BytesIO(raw))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        for table_index, table in enumerate(document.tables, 1):
            table_text = "\n[Table " + str(table_index) + "]\n"
            table_text += "\n".join(" | ".join(cell.text for cell in row.cells) for row in table.rows)
            paragraphs.append(table_text)
        content = "\n\n".join(paragraphs).strip()
        if not content:
            raise ValueError("No readable text was found in this DOCX file.")
        return [{"prompt": f"Reference material: {Path(filename).name}",
                 "content": content, "category": "document", "split": "train"}]
    if suffix == ".xlsx":
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            raise ValueError("XLSX import requires openpyxl. Install requirements.txt and restart.") from exc
        workbook=load_workbook(io.BytesIO(raw),read_only=True,data_only=True)
        rows=[]
        for sheet in workbook.worksheets:
            iterator=sheet.iter_rows(values_only=True)
            headers=next(iterator,None)
            if not headers: continue
            header_names=[str(value).strip() if value is not None else f"column_{i+1}"
                          for i,value in enumerate(headers)]
            for row_number,values in enumerate(iterator,2):
                if not any(value is not None and str(value).strip() for value in values): continue
                item={header_names[i]:value for i,value in enumerate(values) if i<len(header_names)}
                prompt_keys=("prompt","instruction","question","input","user","title","name")
                if not any(item.get(key) is not None and str(item.get(key)).strip() for key in prompt_keys):
                    item["prompt"]=f"{Path(filename).name} · {sheet.title} row {row_number}"
                content_keys=("content","code","completion","output","response","answer","target","assistant","text")
                if not any(item.get(key) is not None and str(item.get(key)).strip() for key in content_keys):
                    item["content"]=" | ".join(f"{key}: {value}" for key,value in item.items()
                                                if value is not None and str(value).strip())
                item.setdefault("category","spreadsheet")
                rows.append(item)
        workbook.close()
        if not rows: raise ValueError("No non-empty rows were found in this workbook.")
        return rows
    if suffix == ".pptx":
        try:
            from pptx import Presentation
        except ImportError as exc:
            raise ValueError("PPTX import requires python-pptx. Install requirements.txt and restart.") from exc
        presentation=Presentation(io.BytesIO(raw))
        slides=[]
        for slide_number,slide in enumerate(presentation.slides,1):
            parts=[]
            for shape in slide.shapes:
                if getattr(shape,"has_text_frame",False) and shape.text.strip():
                    parts.append(shape.text.strip())
                if getattr(shape,"has_table",False):
                    for row in shape.table.rows:
                        parts.append(" | ".join(cell.text for cell in row.cells))
            content="\n".join(parts).strip()
            if content:
                slides.append({"prompt":f"{Path(filename).name} slide {slide_number}",
                               "content":content,"category":"presentation","split":"train"})
        if not slides: raise ValueError("No readable slide text was found in this presentation.")
        return slides
    decoded = raw.decode("utf-8-sig")
    if suffix == ".jsonl":
        return [json.loads(line) for line in decoded.splitlines() if line.strip()]
    if suffix == ".json":
        obj = json.loads(decoded)
        return obj if isinstance(obj, list) else obj.get("examples", [obj] if isinstance(obj, dict) else [])
    if suffix == ".csv":
        return list(csv.DictReader(io.StringIO(decoded)))
    if suffix == ".ipynb":
        notebook = json.loads(decoded)
        cells = []
        for index, cell in enumerate(notebook.get("cells", []), 1):
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            if str(source).strip():
                cells.append({"prompt": f"{Path(filename).name} cell {index} ({cell.get('cell_type','code')})",
                              "content": str(source), "category": "jupyter-" + str(cell.get("cell_type","code"))})
        return cells
    if suffix in TEXT_EXTENSIONS:
        return [{"prompt": f"Reference material: {Path(filename).name}",
                 "content": decoded, "category": suffix.lstrip(".") or "general", "split":"train"}]
    raise ValueError("Unsupported or binary file type.")

@app.post("/api/datasets/import")
async def import_data(file: UploadFile=File(...), category: str="general"):
    if not file.filename or Path(file.filename).suffix.lower() not in DATASET_EXTENSIONS:
        raise HTTPException(400,"Unsupported file type. Upload supported datasets, source code, notebooks, text, or documentation files.")
    raw=await file.read()
    if len(raw)>15*1024*1024: raise HTTPException(413,"Dataset file exceeds the 15 MB import limit.")
    try:
        rows=parse_uploaded_material(file.filename, raw)
    except Exception as exc:
        raise HTTPException(400,f"Could not parse uploaded material: {type(exc).__name__}: {exc}")
    if not isinstance(rows,list): raise HTTPException(400,"Uploaded material did not produce examples.")
    is_document=Path(file.filename).suffix.lower() in DOCUMENT_CHUNK_EXTENSIONS
    digest=hashlib.sha256(raw).hexdigest()
    prepared=prepared_dataset_rows(rows,category,Path(file.filename).name,digest,document_chunks=is_document)
    count=0
    with connect() as c:
        for item in prepared:
            c.execute("INSERT INTO datasets(id,name,category,split,content,metadata_json,created_at) VALUES(?,?,?,?,?,?,?)",item)
            count+=1
    return {"added":count,"total_rows":len(rows),"splits":"train/validation/test"}

@app.post("/api/datasets/import-folder")
def import_inbox_folder():
    """Import supported text/data files dropped into data/inbox; never execute them."""
    allowed=DATASET_EXTENSIONS
    added=0; skipped=[]; scanned=0
    source_files=[*DATA_INBOX.rglob("*"),*DATA_EXAMPLES_DIR.rglob("*")]
    for path in sorted(source_files):
        if path.is_symlink() or not path.is_file() or path.name.lower()=="readme.md" or path.suffix.lower() not in allowed:
            continue
        scanned+=1
        raw=path.read_bytes()
        if len(raw)>15*1024*1024:
            skipped.append({"file":path.name,"reason":"over 15 MB"})
            continue
        digest=hashlib.sha256(raw).hexdigest()
        source_name=path.relative_to(DATA_DIR).as_posix()
        with connect() as c:
            if c.execute("SELECT 1 FROM datasets WHERE metadata_json LIKE ? LIMIT 1",("%"+digest+"%",)).fetchone():
                skipped.append({"file":path.name,"reason":"already imported"})
                continue
        try:
            rows=parse_uploaded_material(path.name, raw)
            if not isinstance(rows,list): raise ValueError("Expected a list of examples.")
        except Exception as exc:
            skipped.append({"file":path.name,"reason":f"parse error: {exc}"})
            continue
        for row in rows:
            if isinstance(row,dict) and str(row.get("prompt","")).startswith("Reference material: "):
                row["prompt"]="Reference file: "+path.name
        is_document=path.suffix.lower() in DOCUMENT_CHUNK_EXTENSIONS
        prepared=prepared_dataset_rows(rows,path.suffix.lower().lstrip("."),source_name,digest,document_chunks=is_document)
        with connect() as c:
            for item in prepared:
                c.execute("INSERT INTO datasets(id,name,category,split,content,metadata_json,created_at) VALUES(?,?,?,?,?,?,?)",item)
                added+=1
    return {"scanned_files":scanned,"added":added,"skipped":skipped,"folders":[str(DATA_INBOX),str(DATA_EXAMPLES_DIR)]}

@app.get("/api/datasets/export")
def export_datasets():
    with connect() as c: rows=[dict(x) for x in c.execute("SELECT * FROM datasets ORDER BY created_at")]
    def stream():
        for row in rows:
            try:
                metadata=json.loads(row.get("metadata_json") or "{}")
            except (TypeError,ValueError):
                metadata={}
            yield json.dumps({"prompt":row["name"],"category":row["category"],
                              "split":row["split"],"content":row["content"],
                              "metadata":metadata},ensure_ascii=False)+"\n"
    return StreamingResponse(stream(),media_type="application/x-ndjson",headers={"Content-Disposition":"attachment; filename=neuralforge-dataset.jsonl"})

@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    with connect() as c:
        cur = c.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))
        if not cur.rowcount:
            raise HTTPException(404, "Dataset example not found.")
    return {"deleted": True, "id": dataset_id}


@app.get("/api/tasks")
def list_tasks():
    with connect() as c:
        return [dict(x) for x in c.execute("""
          SELECT t.*,
            (SELECT COUNT(*) FROM task_attempts a WHERE a.task_id=t.id) AS attempt_count,
            (SELECT AVG(score) FROM task_attempts a WHERE a.task_id=t.id) AS average_score,
            (SELECT score FROM task_attempts a WHERE a.task_id=t.id ORDER BY created_at DESC LIMIT 1) AS latest_score
          FROM tasks t ORDER BY t.created_at DESC
        """)]

@app.get("/api/attempts")
def list_attempts(task_id: str | None = None, limit: int = 100):
    limit=max(1,min(limit,500))
    query="""SELECT a.*,t.title AS task_title,
      (SELECT COUNT(*) FROM human_feedback f WHERE f.attempt_id=a.id) AS feedback_count,
      (SELECT AVG(rating) FROM human_feedback f WHERE f.attempt_id=a.id) AS average_human_rating
      FROM task_attempts a LEFT JOIN tasks t ON t.id=a.task_id"""
    args=[]
    if task_id:
        query+=" WHERE a.task_id=?"; args.append(task_id)
    query+=" ORDER BY a.created_at DESC LIMIT ?"; args.append(limit)
    with connect() as c: rows=c.execute(query,args).fetchall()
    return [dict(row) for row in rows]


@app.post("/api/attempts/{attempt_id}/feedback")
def submit_attempt_feedback(attempt_id: str, payload: FeedbackIn):
    mistake_id=None
    with connect() as c:
        attempt=c.execute("SELECT id,task_id,prompt,output FROM task_attempts WHERE id=?",(attempt_id,)).fetchone()
        if not attempt:
            raise HTTPException(404,"Task attempt not found.")
        feedback_id=str(uuid.uuid4())
        reward=round((payload.rating-3)/2,4)
        c.execute("""INSERT INTO human_feedback
          (id,attempt_id,rating,reward,note,correction,evidence,created_at)
          VALUES(?,?,?,?,?,?,?,?)""",
          (feedback_id,attempt_id,payload.rating,reward,payload.note,
           payload.correction,payload.evidence,now()))
        if payload.correction.strip():
            mistake_id=str(uuid.uuid4())
            task=c.execute("SELECT category FROM tasks WHERE id=?",(attempt["task_id"],)).fetchone()
            category=task["category"] if task else "general"
            c.execute("""INSERT INTO mistakes
              (id,task_id,prompt,output,category,evidence,correction,verification,evaluator,created_at)
              VALUES(?,?,?,?,?,?,?,?,?,?)""",
              (mistake_id,attempt["task_id"],attempt["prompt"],attempt["output"],
               category,payload.evidence,payload.correction,"unverified",
               "human_feedback_candidate",now()))
    append_mistake_log("human_feedback_submitted",{
        "feedback_id":feedback_id,"attempt_id":attempt_id,"rating":payload.rating,
        "reward":reward,"note":payload.note,"correction":payload.correction,
        "evidence":payload.evidence,"mistake_id":mistake_id})
    if mistake_id:
        append_mistake_log("correction_candidate_created",{
            "mistake_id":mistake_id,"attempt_id":attempt_id,
            "verification":"unverified","note":"Human correction requires evidence review before supervised training."})
    return {"id":feedback_id,"attempt_id":attempt_id,"rating":payload.rating,
            "reward":reward,"queued_for_training":True,"mistake_id":mistake_id,
            "correction_requires_review":bool(mistake_id)}


@app.get("/api/feedback")
def list_feedback(pending_only: bool = False, limit: int = 200):
    limit=max(1,min(limit,500))
    query="""SELECT f.*,a.task_id,a.prompt,a.output
             FROM human_feedback f JOIN task_attempts a ON a.id=f.attempt_id"""
    if pending_only: query+=" WHERE f.consumed_at IS NULL"
    query+=" ORDER BY f.created_at DESC LIMIT ?"
    with connect() as c: return [dict(row) for row in c.execute(query,(limit,))]


@app.get("/api/tasks/performance")
def task_performance():
    with connect() as c:
        rows=c.execute("""
          SELECT t.id,t.title,t.category,COUNT(a.id) AS attempts,
            AVG(a.score) AS average_score,
            (SELECT a2.score FROM task_attempts a2 WHERE a2.task_id=t.id ORDER BY a2.created_at DESC LIMIT 1) AS latest_score,
            (SELECT a2.created_at FROM task_attempts a2 WHERE a2.task_id=t.id ORDER BY a2.created_at DESC LIMIT 1) AS last_attempt
          FROM tasks t LEFT JOIN task_attempts a ON a.task_id=t.id
          GROUP BY t.id ORDER BY attempts DESC,t.created_at DESC
        """).fetchall()
    return [dict(row) for row in rows]


@app.post("/api/tasks")
def add_task(t:TaskIn):
    tid=str(uuid.uuid4())
    with connect() as c:
        c.execute("INSERT INTO tasks(id,title,prompt,expected,difficulty,category,created_at) VALUES(?,?,?,?,?,?,?)",
                  (tid,t.title,t.prompt,t.expected,t.difficulty,t.category,now()))
    return {"id":tid}

@app.post("/api/tasks/generate-from-dataset")
def generate_tasks_from_dataset(limit: int = 30):
    limit=max(1,min(limit,200))
    with connect() as c:
        rows=c.execute("SELECT name,category,content FROM datasets WHERE split='train' ORDER BY created_at DESC LIMIT ?",(limit,)).fetchall()
        existing={r["title"] for r in c.execute("SELECT title FROM tasks")}
        created=0
        for row in rows:
            title=("Practice: "+row["name"])[:180]
            if title in existing:
                continue
            prompt=f"Demonstrate and explain this subject from the uploaded training material: {row['name']}. Produce a correct, useful answer in the format appropriate to the subject."
            c.execute("INSERT INTO tasks(id,title,prompt,expected,difficulty,category,created_at) VALUES(?,?,?,?,?,?,?)",
                      (str(uuid.uuid4()),title,prompt,row["content"],"easy",row["category"],now()))
            existing.add(title); created+=1
    return {"created":created,"available_examples":len(rows),"message":"Practice tasks generated from train-split examples only."}

@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: str):
    # Attempt and human-feedback history is retained as an audit record.
    with connect() as c:
        cur = c.execute("DELETE FROM tasks WHERE id=?", (task_id,))
        if not cur.rowcount:
            raise HTTPException(404, "Task not found.")
    return {"deleted": True, "id": task_id, "history_preserved": True}


@app.post("/api/tasks/{task_id}/try")
def try_task(task_id: str):
    with connect() as c:
        task=c.execute("SELECT * FROM tasks WHERE id=? AND enabled=1",(task_id,)).fetchone()
    if not task: raise HTTPException(404,"Enabled task not found.")
    try:
        adapter=get_adapter()
        if adapter is None: raise HTTPException(400,"Connect a model before testing a task.")
        prompt=build_prompt(dict(task))
        with MODEL_EXECUTION_LOCK:
            generated=adapter.generate(prompt,{"task_id":task_id,"difficulty":task["difficulty"],"category":task["category"],
                                               "max_new_tokens":int(setting_map().get("max_new_tokens","384"))})
        output=generated if isinstance(generated,str) else str(generated.get("text",""))
        evaluation=evaluate_with_adapter(adapter, prompt, output, task["expected"], task["category"])
        attempt_id=record_attempt(task_id,"",prompt,output,evaluation)
        mistake_id=None
        if evaluation["score"] < float(setting_map().get("reward_threshold","0.45"))*100:
            mistake_id=save_mistake(task_id,prompt,output,evaluation)
        return {"task_id":task_id,"attempt_id":attempt_id,"output":output,"evaluation":evaluation,"mistake_id":mistake_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500,f"Model task failed: {type(exc).__name__}: {exc}")


@app.patch("/api/tasks/{task_id}")
def toggle_task(task_id:str, enabled:bool):
    with connect() as c:
        cur=c.execute("UPDATE tasks SET enabled=? WHERE id=?",(int(enabled),task_id))
        if not cur.rowcount: raise HTTPException(404,"Task not found.")
    return {"id":task_id,"enabled":enabled}

@app.post("/api/evaluate")
def evaluate(payload:dict):
    return static_evaluate(str(payload.get("output","")),str(payload.get("expected","")))

@app.get("/api/mistakes")
def list_mistakes(verification:str | None=None):
    query="""SELECT m.id,m.task_id,m.prompt,m.output,m.category,
      COALESCE(NULLIF((SELECT e.evidence FROM mistake_events e WHERE e.mistake_id=m.id AND e.evidence<>'' ORDER BY e.created_at DESC,e.id DESC LIMIT 1),''),m.evidence) AS evidence,
      COALESCE(NULLIF((SELECT e.correction FROM mistake_events e WHERE e.mistake_id=m.id AND e.correction<>'' ORDER BY e.created_at DESC,e.id DESC LIMIT 1),''),m.correction) AS correction,
      m.evaluator,m.created_at,
      COALESCE((SELECT e.action FROM mistake_events e WHERE e.mistake_id=m.id
                ORDER BY e.created_at DESC,e.id DESC LIMIT 1),m.verification) AS verification
      FROM mistakes m"""
    args=[]
    if verification: query+=" WHERE COALESCE((SELECT e.action FROM mistake_events e WHERE e.mistake_id=m.id ORDER BY e.created_at DESC,e.id DESC LIMIT 1),m.verification)=?"; args.append(verification)
    query+=" ORDER BY m.created_at DESC LIMIT 1000"
    with connect() as c:return [dict(x) for x in c.execute(query,args)]

@app.post("/api/mistakes")
def add_mistake(m:MistakeIn):
    mid=str(uuid.uuid4())
    with connect() as c:
        c.execute("INSERT INTO mistakes(id,task_id,prompt,output,category,evidence,correction,verification,evaluator,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                  (mid,m.task_id,m.prompt,m.output,m.category,m.evidence,m.correction,"unverified","manual",now()))
    append_mistake_log("mistake_created",{"id":mid,"task_id":m.task_id,"prompt":m.prompt,"output":m.output,"category":m.category,"evidence":m.evidence,"correction":m.correction})
    return {"id":mid,"verification":"unverified","append_only":True}

@app.patch("/api/mistakes/{mid}")
def review_mistake(mid:str, review:ReviewIn):
    if review.verification not in ("verified","rejected","needs_review"):
        raise HTTPException(400,"Choose verified, rejected or needs_review.")
    with connect() as c:
        row=c.execute("""SELECT m.correction,m.evidence,
          COALESCE(NULLIF((SELECT e.correction FROM mistake_events e WHERE e.mistake_id=m.id AND e.correction<>'' ORDER BY e.created_at DESC,e.id DESC LIMIT 1),''),m.correction) AS current_correction,
          COALESCE(NULLIF((SELECT e.evidence FROM mistake_events e WHERE e.mistake_id=m.id AND e.evidence<>'' ORDER BY e.created_at DESC,e.id DESC LIMIT 1),''),m.evidence) AS current_evidence
          FROM mistakes m WHERE m.id=?""",(mid,)).fetchone()
        if not row: raise HTTPException(404,"Mistake not found.")
        correction=review.correction.strip() or row["current_correction"]
        evidence=review.evidence.strip() or row["current_evidence"]
        if review.verification=="verified" and (not correction or not evidence):
            raise HTTPException(400,"To verify a correction, provide both a correction and evidence (test result, error trace, or source reference).")
        # Strict append-only: original attempt is immutable; correction, evidence and status are new review events.
        event_id=str(uuid.uuid4())
        c.execute("INSERT INTO mistake_events(id,mistake_id,action,note,correction,evidence,created_at) VALUES(?,?,?,?,?,?,?)",
                  (event_id,mid,review.verification,review.note,correction,evidence,now()))
    append_mistake_log("review_event",{"event_id":event_id,"mistake_id":mid,"action":review.verification,"note":review.note,"correction":correction,"evidence":evidence})
    return {"id":mid,"verification":review.verification,"review_event_appended":True,"original_unchanged":True}

@app.get("/api/mistakes/export")
def export_mistakes():
    with connect() as c: rows=[dict(x) for x in c.execute("SELECT * FROM mistakes ORDER BY created_at")]
    def stream():
        for row in rows: yield json.dumps(row,ensure_ascii=False)+"\n"
    return StreamingResponse(stream(),media_type="application/x-ndjson",headers={"Content-Disposition":"attachment; filename=neuralforge-mistakes.jsonl"})

@app.get("/api/adapter/config")
def get_adapter_config():
    cfg=setting_map()
    return {"adapter_spec":cfg.get("adapter_spec",""),"model_path":cfg.get("model_path",""),
            "max_new_tokens":int(cfg.get("max_new_tokens","384")),"learning_rate":float(cfg.get("learning_rate","0.00001")),
            "batch_size":int(cfg.get("batch_size","2")),"reward_threshold":float(cfg.get("reward_threshold","0.45")),
            "use_lora":str(cfg.get("use_lora","true")).lower() not in {"0","false","no"},
            "lora_r":int(cfg.get("lora_r","8")),"lora_alpha":int(cfg.get("lora_alpha","16")),
            "lora_dropout":float(cfg.get("lora_dropout","0.05")),
            "lora_target_modules":cfg.get("lora_target_modules","")}

@app.put("/api/adapter/config")
def set_adapter_config(config:AdapterConfig):
    global _adapter_instance, _adapter_fingerprint
    values=config.model_dump()
    with RUN_SCHEDULER_LOCK:
        with connect() as c:
            active=c.execute("SELECT id,status FROM runs WHERE status IN ('queued','running') LIMIT 1").fetchone()
            if active:
                raise HTTPException(409,"Model configuration cannot change while a training run is active.")
            for k,v in values.items():
                c.execute("INSERT INTO adapter_settings(key,value,updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",
                          (k,str(v),now()))
        with _adapter_lock: _adapter_instance=None; _adapter_fingerprint=None
    return {"saved":True,"restart_required":False,"message":"Configuration saved. The adapter loads when a training run starts."}

@app.get("/api/adapter/status")
def adapter_status():
    cfg=setting_map(); spec=cfg.get("adapter_spec","").strip(); path=cfg.get("model_path","").strip()
    if not spec and not path:
        return {"configured":False,"connected":False,"can_train":False,"name":"No model connected",
                "message":"Set an adapter module or local Hugging Face model directory."}
    try:
        adapter=get_adapter()
        diagnostics=getattr(adapter,"diagnostics",None)
        details=diagnostics() if callable(diagnostics) else {}
        return {"configured":True,"connected":True,"can_train":True,
                "name":getattr(adapter,"name",type(adapter).__name__),
                "diagnostics":details,
                "message":"Adapter loaded. A training run will call generate and train_step."}
    except Exception as exc:
        return {"configured":True,"connected":False,"can_train":False,"error":str(exc)}

@app.get("/api/system/diagnostics")
def system_diagnostics():
    cfg = setting_map()
    with connect() as c:
        db_counts = {
            "datasets": c.execute("SELECT COUNT(*) FROM datasets").fetchone()[0],
            "tasks": c.execute("SELECT COUNT(*) FROM tasks").fetchone()[0],
            "mistakes": c.execute("SELECT COUNT(*) FROM mistakes").fetchone()[0],
            "runs": c.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
            "checkpoints": c.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0],
        }
    return {
        "ok": True, "python": __import__("sys").version.split()[0],
        "database": str(DB), "database_exists": DB.exists(), "counts": db_counts,
        "folders": {name: path.exists() for name, path in {
            "inbox": DATA_INBOX, "examples": DATA_EXAMPLES_DIR, "mistakes": MISTAKES_DIR,
            "checkpoints": CHECKPOINT_DIR}.items()},
        "adapter_configured": bool(cfg.get("adapter_spec", "").strip() or cfg.get("model_path", "").strip()),
        "adapter_spec": cfg.get("adapter_spec", ""),
        "model_path_configured": bool(cfg.get("model_path", "").strip()),
        "lora_enabled": str(cfg.get("use_lora","true")).lower() not in {"0","false","no"},
        "limits": {"upload_mb": 15, "max_run_hours": 24, "max_episodes": 1000000},
    }


@app.post("/api/adapter/test")
def test_adapter():
    try:
        adapter = get_adapter()
        if adapter is None:
            raise HTTPException(400, "No model is configured.")
        sample = "Explain what a variable is, using one short example."
        with MODEL_EXECUTION_LOCK:
            generated = adapter.generate(sample, {"category": "general", "max_new_tokens": 96, "diagnostic": True})
        output = generated if isinstance(generated, str) else str(generated.get("text", ""))
        if not output.strip():
            raise RuntimeError("The connected model returned an empty response.")
        evaluator = callable(getattr(adapter, "evaluate", None))
        diagnostics=getattr(adapter,"diagnostics",None)
        details=diagnostics() if callable(diagnostics) else {}
        return {"connected": True, "name": getattr(adapter, "name", type(adapter).__name__),
                "generation_ok": True, "sample_output": output[:1200],
                "model_diagnostics": details,
                "domain_evaluator": evaluator,
                "training_hooks": {"reward": callable(getattr(adapter, "train_step", None)),
                                   "supervised": callable(getattr(adapter, "supervised_step", None))},
                "note": "This test generates a sample only; it does not change model weights."}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Adapter test failed: {type(exc).__name__}: {exc}")


@app.post("/api/generate")
def generate_once(payload: dict):
    prompt = str(payload.get("prompt", "")).strip()
    category = str(payload.get("category", "general")).strip() or "general"
    if not prompt:
        raise HTTPException(400, "Prompt is required.")
    if len(prompt) > 20000:
        raise HTTPException(413, "Prompt exceeds the 20,000 character limit.")
    try:
        adapter = get_adapter()
        if adapter is None:
            raise HTTPException(400, "Connect a model before generating.")
        full_prompt = build_prompt({"prompt": prompt, "category": category})
        with MODEL_EXECUTION_LOCK:
            generated = adapter.generate(full_prompt, {"category": category,
                                                        "max_new_tokens": int(setting_map().get("max_new_tokens", "384"))})
        output = generated if isinstance(generated, str) else str(generated.get("text", ""))
        evaluation = evaluate_with_adapter(adapter, full_prompt, output, "", category)
        return {"output": output, "evaluation": evaluation, "model": getattr(adapter, "name", type(adapter).__name__)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Generation failed: {type(exc).__name__}: {exc}")


@app.post("/api/runs/start")
def start_run(request:RunIn, background:BackgroundTasks):
    rid=str(uuid.uuid4())
    with RUN_SCHEDULER_LOCK:
        with connect() as c:
            active=c.execute("SELECT id,status FROM runs WHERE status IN ('queued','running') LIMIT 1").fetchone()
            if active:
                raise HTTPException(409,f"Training run {active['id']} is already {active['status']}. Stop or finish it before starting another.")
            c.execute("INSERT INTO runs(id,status,episodes,reward,success_rate,real_updates,notes,created_at,duration_hours) VALUES(?,?,?,?,?,?,?,?,?)",
                      (rid,"queued",request.episodes,0,0,0,"Queued; imported train-split examples and reward feedback will be used when supported by the adapter.",now(),request.duration_hours))
    background.add_task(execute_training,rid,request.model_dump())
    return {"id":rid,"status":"queued","episodes":request.episodes,"duration_hours":request.duration_hours,"message":"Training run queued. Poll /api/runs for status."}

@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id:str):
    with connect() as c:
        row=c.execute("SELECT status FROM runs WHERE id=?",(run_id,)).fetchone()
        if not row: raise HTTPException(404,"Run not found.")
        if row["status"] not in ("queued","running"):
            return {"id":run_id,"status":row["status"],"stop_requested":False}
        c.execute("UPDATE runs SET stop_requested=1 WHERE id=?",(run_id,))
    return {"id":run_id,"stop_requested":True,"message":"Stop requested. The current training batch will finish before stopping."}

@app.get("/api/runs")
def list_runs():
    with connect() as c:return [dict(x) for x in c.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 200")]

@app.get("/api/runs/{run_id}")
def get_run(run_id:str):
    with connect() as c:
        row=c.execute("SELECT * FROM runs WHERE id=?",(run_id,)).fetchone()
        if not row: raise HTTPException(404,"Run not found.")
        return dict(row)

@app.get("/api/runs/{run_id}/report")
def run_report(run_id: str):
    with connect() as c:
        run=c.execute("SELECT * FROM runs WHERE id=?",(run_id,)).fetchone()
        if not run:
            raise HTTPException(404,"Run not found.")
        stats=c.execute("""SELECT COUNT(*) AS attempts,AVG(score) AS average_score,
            MIN(score) AS lowest_score,MAX(score) AS highest_score,
            AVG(reward) AS average_attempt_reward
            FROM task_attempts WHERE run_id=?""",(run_id,)).fetchone()
        attempts=c.execute("""SELECT a.id,a.task_id,t.title AS task_title,a.prompt,
            a.output,a.score,a.reward,a.evaluator,a.checks_json,a.created_at
            FROM task_attempts a LEFT JOIN tasks t ON t.id=a.task_id
            WHERE a.run_id=? ORDER BY a.created_at DESC LIMIT 100""",(run_id,)).fetchall()
        checkpoints=c.execute("""SELECT id,label,kind,created_at,metadata_json
            FROM checkpoints WHERE run_id=? ORDER BY created_at""",(run_id,)).fetchall()
    return {"run":dict(run),"attempt_stats":{
                "attempts":stats["attempts"],
                "average_score":round(stats["average_score"],2) if stats["average_score"] is not None else None,
                "lowest_score":stats["lowest_score"],"highest_score":stats["highest_score"],
                "average_attempt_reward":round(stats["average_attempt_reward"],4) if stats["average_attempt_reward"] is not None else None},
            "recent_attempts":[dict(row) for row in attempts],
            "checkpoints":[dict(row) for row in checkpoints],
            "note":"Attempt scores are only as reliable as the evaluator named on each attempt."}


@app.post("/api/checkpoints")
def save_checkpoint(payload:dict):
    label=str(payload.get("label","Manual checkpoint"))[:160]
    run_id=str(payload.get("run_id",""))
    cid=str(uuid.uuid4())
    try:
        adapter=get_adapter()
        if adapter is None: raise RuntimeError("No model adapter connected.")
        path=str(CHECKPOINT_DIR/f"{cid}")
        with MODEL_EXECUTION_LOCK:
            metadata=adapter.save_checkpoint(path) or {}
        if not checkpoint_has_artifacts(path):
            raise RuntimeError("Adapter returned without writing any non-empty checkpoint files.")
        kind="model_weights"
    except Exception as exc:
        raise HTTPException(400, f"Checkpoint save failed: {type(exc).__name__}: {exc}")
    with connect() as c:
        c.execute("INSERT INTO checkpoints(id,run_id,label,kind,created_at,path,metadata_json) VALUES(?,?,?,?,?,?,?)",
                  (cid,run_id,label,kind,now(),path,json.dumps(metadata)))
    return {"id":cid,"kind":kind,"metadata":metadata}

@app.get("/api/checkpoints")
def list_checkpoints():
    with connect() as c:return [dict(x) for x in c.execute("SELECT * FROM checkpoints ORDER BY created_at DESC")]

@app.post("/api/checkpoints/{checkpoint_id}/restore")
def restore_checkpoint(checkpoint_id:str):
    with connect() as c: row=c.execute("SELECT * FROM checkpoints WHERE id=?",(checkpoint_id,)).fetchone()
    if not row: raise HTTPException(404,"Checkpoint not found.")
    if not row["path"]: raise HTTPException(400,"This is metadata-only; no model weights were saved.")
    try:
        adapter=get_adapter()
        if adapter is None: raise RuntimeError("No model adapter connected.")
        with MODEL_EXECUTION_LOCK:
            result=adapter.load_checkpoint(row["path"])
        return {"restored":True,"result":result}
    except Exception as exc: raise HTTPException(400,str(exc))

@app.post("/api/benchmarks/run")
def run_benchmark(split: str="validation", limit: int=100):
    with MODEL_EXECUTION_LOCK:
        return _run_benchmark_locked(split,limit)


def _run_benchmark_locked(split: str="validation", limit: int=100):
    if split not in ("validation","test"):
        raise HTTPException(400,"Benchmark split must be validation or test.")
    limit=max(1,min(limit,500))
    try:
        adapter=get_adapter()
        if adapter is None: raise HTTPException(400,"Connect a model before running a benchmark.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400,str(exc))
    with connect() as c:
        rows=c.execute("SELECT name,category,content FROM datasets WHERE split=? ORDER BY created_at LIMIT ?",(split,limit)).fetchall()
    if not rows: raise HTTPException(400,f"No {split} examples are available.")
    details=[]; total=0.0; passed=0
    for row in rows:
        prompt=build_prompt({"prompt":f"Answer or solve this training example: {row['name']}","category":row["category"]})
        try:
            with MODEL_EXECUTION_LOCK:
                generated=adapter.generate(prompt,{"benchmark_split":split,"category":row["category"],
                                                   "max_new_tokens":int(setting_map().get("max_new_tokens","384"))})
            output=generated if isinstance(generated,str) else str(generated.get("text",""))
            evaluation=evaluate_with_adapter(adapter, prompt, output, row["content"], row["category"])
            total+=evaluation["score"]; passed+=int(evaluation["score"]>=75)
            details.append({"prompt":row["name"],"score":evaluation["score"],"expected_match":evaluation["expected_match"],
                            "output_preview":output[:1000],"checks":evaluation["checks"]})
        except Exception as exc:
            details.append({"prompt":row["name"],"error":f"{type(exc).__name__}: {exc}","score":0})
    count=len(details); avg=total/count; pass_rate=passed/count
    bid=str(uuid.uuid4()); model_name=getattr(adapter,"name",type(adapter).__name__)
    with connect() as c:
        c.execute("INSERT INTO benchmarks(id,split,examples,avg_score,pass_rate,details_json,created_at,model_name) VALUES(?,?,?,?,?,?,?,?)",
                  (bid,split,count,avg,pass_rate,json.dumps(details,ensure_ascii=False),now(),model_name))
    return {"id":bid,"split":split,"examples":count,"average_score":round(avg,1),
            "pass_rate":round(pass_rate*100,1),"model":model_name,"details":details,
            "warning":"Static heuristic only; this is not a Roblox runtime test."}

@app.get("/api/benchmarks")
def list_benchmarks():
    with connect() as c:
        rows=c.execute("SELECT * FROM benchmarks ORDER BY created_at DESC LIMIT 100").fetchall()
    return [dict(x) for x in rows]

@app.get("/api/metrics")
def metrics():
    with connect() as c:
        rows=[dict(x) for x in c.execute("SELECT id,status,episodes,reward,success_rate,real_updates,avg_loss,created_at,model_name FROM runs ORDER BY created_at")]
    return {"runs":rows,"latest":rows[-1] if rows else None}

@app.get("/api/docs")
def api_docs(): return {"openapi":"/openapi.json","swagger":"/docs","redoc":"/redoc"}

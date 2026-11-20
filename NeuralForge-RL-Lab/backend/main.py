"""NeuralForge RL Lab — local-first API and training orchestration.

Generated Luau is never executed by this server. Evaluation is static unless a
future, explicitly sandboxed evaluator is configured by the owner.
"""
from __future__ import annotations
import csv, importlib, io, json, os, sqlite3, threading, uuid
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
CHECKPOINT_DIR.mkdir(exist_ok=True)
app = FastAPI(title="NeuralForge RL Lab", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")
_adapter_lock = threading.RLock()
_adapter_instance = None
_adapter_fingerprint = None
_adapter_error = None

def now() -> str:
    return datetime.now(timezone.utc).isoformat()

def connect():
    c = sqlite3.connect(DB, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    return c

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
          created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS runs(
          id TEXT PRIMARY KEY,status TEXT NOT NULL,episodes INTEGER NOT NULL,reward REAL DEFAULT 0,
          success_rate REAL DEFAULT 0,real_updates INTEGER DEFAULT 0,notes TEXT DEFAULT '',
          created_at TEXT NOT NULL,started_at TEXT,finished_at TEXT,error TEXT,avg_loss REAL DEFAULT 0,
          model_name TEXT DEFAULT '');
        CREATE TABLE IF NOT EXISTS checkpoints(
          id TEXT PRIMARY KEY,run_id TEXT DEFAULT '',label TEXT NOT NULL,kind TEXT NOT NULL,
          created_at TEXT NOT NULL,path TEXT DEFAULT '',metadata_json TEXT DEFAULT '{}');
        CREATE TABLE IF NOT EXISTS adapter_settings(
          key TEXT PRIMARY KEY,value TEXT NOT NULL,updated_at TEXT NOT NULL);
        """)
        # Upgrade the early prototype database without destroying user data.
        migrations = {
          "datasets": {"metadata_json": "TEXT DEFAULT '{}'"},
          "tasks": {"category": "TEXT DEFAULT 'general'", "enabled": "INTEGER DEFAULT 1"},
          "runs": {"started_at": "TEXT", "finished_at": "TEXT", "error": "TEXT",
                   "avg_loss": "REAL DEFAULT 0", "model_name": "TEXT DEFAULT ''"},
          "checkpoints": {"path": "TEXT DEFAULT ''", "metadata_json": "TEXT DEFAULT '{}'"},
        }
        for table, wanted in migrations.items():
            existing = columns(c, table)
            for name, decl in wanted.items():
                if name not in existing:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
        if c.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 0:
            seed = [
              ("Luau local variable","syntax","local coins = 10","train"),
              ("Luau function","functions","local function add(a, b)\n    return a + b\nend","train"),
              ("Players service","services",'local Players = game:GetService("Players")',"validation"),
              ("Touched event","events","part.Touched:Connect(function(hit)\n    print(hit.Name)\nend)","test"),
            ]
            for name, cat, code, split in seed:
                c.execute("INSERT INTO datasets(id,name,category,split,content,created_at) VALUES(?,?,?,?,?,?)",
                          (str(uuid.uuid4()),name,cat,split,code,now()))
        if c.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0:
            c.execute("INSERT INTO tasks(id,title,prompt,expected,difficulty,category,created_at) VALUES(?,?,?,?,?,?,?)",
                      (str(uuid.uuid4()),"Create a health variable","Declare a local variable named health with value 100.",
                       "local health = 100","easy","syntax",now()))
        defaults = {"adapter_spec":"", "model_path":"", "max_new_tokens":"384",
                    "learning_rate":"0.00001", "batch_size":"2", "reward_threshold":"0.45"}
        for k,v in defaults.items():
            c.execute("INSERT OR IGNORE INTO adapter_settings(key,value,updated_at) VALUES(?,?,?)",(k,v,now()))

init_db()

class TaskIn(BaseModel):
    title: str = Field(min_length=1, max_length=180)
    prompt: str = Field(min_length=1, max_length=12000)
    expected: str = Field(default="", max_length=20000)
    difficulty: str = "easy"
    category: str = "general"

class MistakeIn(BaseModel):
    task_id: str = ""
    prompt: str
    output: str
    category: str = "failed_tasks"
    evidence: str = ""
    correction: str = ""

class ReviewIn(BaseModel):
    verification: str
    note: str = ""

class RunIn(BaseModel):
    episodes: int = Field(default=10, ge=1, le=5000)
    batch_size: int | None = Field(default=None, ge=1, le=32)

class AdapterConfig(BaseModel):
    adapter_spec: str = ""
    model_path: str = ""
    max_new_tokens: int = Field(default=384, ge=16, le=4096)
    learning_rate: float = Field(default=0.00001, gt=0, le=0.1)
    batch_size: int = Field(default=2, ge=1, le=32)
    reward_threshold: float = Field(default=0.45, ge=0, le=1)

def setting_map():
    with connect() as c:
        return {r["key"]:r["value"] for r in c.execute("SELECT key,value FROM adapter_settings")}

def get_adapter(force_reload=False):
    global _adapter_instance, _adapter_fingerprint, _adapter_error
    cfg = setting_map()
    spec = cfg.get("adapter_spec","").strip()
    model_path = cfg.get("model_path","").strip()
    fingerprint = (spec, model_path)
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
            try:
                adapter = factory({"model_path":model_path, **cfg})
            except TypeError:
                adapter = factory()
            for method in ("generate","train_step","save_checkpoint","load_checkpoint"):
                if not callable(getattr(adapter,method,None)):
                    raise TypeError(f"Adapter is missing required method: {method}")
            _adapter_instance, _adapter_fingerprint, _adapter_error = adapter, fingerprint, None
            return adapter
        except Exception as exc:
            _adapter_instance, _adapter_fingerprint, _adapter_error = None, fingerprint, f"{type(exc).__name__}: {exc}"
            raise RuntimeError(_adapter_error) from exc

def static_evaluate(output: str, expected: str = "") -> dict[str, Any]:
    text = (output or "").strip()
    if not text:
        return {"score":0.0,"reward":-1.0,"checks":{"non_empty":False},"expected_match":False,
                "warning":"Static heuristic only; generated Luau was not executed."}
    checks = {
      "non_empty": True,
      "luau_structure": any(x in text for x in ("local ","function","game:GetService","Connect(","return","Instance.new","task.wait")),
      "no_placeholder": not any(x in text.lower() for x in ("todo","demo output","insert code here")),
      "balanced_delimiters": all(text.count(a) >= text.count(b) for a,b in (("(",")"),("{","}"),("[","]"))),
    }
    expected_match = bool(expected.strip()) and expected.strip() in text
    raw = sum(checks.values()) / len(checks)
    if expected_match:
        raw = max(raw, 1.0)
    score = round(raw * 100, 1)
    return {"score":score,"reward":round((raw * 2)-1,4),"checks":checks,
            "expected_match":expected_match,
            "warning":"Static heuristic only. This score does not prove Roblox runtime correctness."}

def verified_examples(limit=6):
    with connect() as c:
        rows = c.execute("""
          SELECT m.prompt,m.correction FROM mistakes m
          WHERE COALESCE((SELECT e.action FROM mistake_events e WHERE e.mistake_id=m.id
                          ORDER BY e.created_at DESC,e.id DESC LIMIT 1),m.verification)='verified'
            AND m.correction<>''
          ORDER BY m.created_at DESC LIMIT ?
        """,(limit,)).fetchall()
    return [{"prompt":r["prompt"],"correction":r["correction"]} for r in rows]

def build_prompt(task):
    examples = verified_examples()
    if not examples:
        return task["prompt"] + "\n\nWrite valid Roblox Luau. Explain assumptions briefly and return code."
    block = "\n\nVerified examples from previous human-reviewed corrections:\n"
    for i,e in enumerate(examples,1):
        block += f"Example {i} task: {e['prompt']}\nExample {i} correction:\n{e['correction'][:1800]}\n"
    return task["prompt"] + block + "\nNow solve the new task. Do not copy an unrelated example."

def save_mistake(task_id,prompt,output,evaluation,category="failed_tasks"):
    mid=str(uuid.uuid4())
    with connect() as c:
        c.execute("INSERT INTO mistakes(id,task_id,prompt,output,category,evidence,correction,verification,evaluator,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                  (mid,task_id,prompt,output,category,json.dumps(evaluation), "", "unverified","static_heuristic",now()))
    return mid

def execute_training(run_id, request):
    started=now()
    with connect() as c:
        c.execute("UPDATE runs SET status='running',started_at=? WHERE id=?",(started,run_id))
        task_rows=c.execute("SELECT * FROM tasks WHERE enabled=1 ORDER BY created_at").fetchall()
    if not task_rows:
        with connect() as c:c.execute("UPDATE runs SET status='failed',error=?,finished_at=? WHERE id=?",("Create at least one enabled task.",now(),run_id))
        return
    try:
        adapter=get_adapter()
        if adapter is None:
            raise RuntimeError("No model is connected. Configure an adapter or a Hugging Face model path first.")
        cfg=setting_map()
        episodes=request["episodes"]
        batch_size=request.get("batch_size") or int(cfg.get("batch_size","2"))
        samples=[]; rewards=[]; total_reward=0.0; successes=0; updates=0; losses=[]
        for episode in range(episodes):
            task=task_rows[episode % len(task_rows)]
            prompt=build_prompt(dict(task))
            try:
                generated=adapter.generate(prompt, {"task_id":task["id"],"difficulty":task["difficulty"],"category":task["category"],
                                                   "max_new_tokens":int(cfg.get("max_new_tokens","384"))})
                output=generated if isinstance(generated,str) else str(generated.get("text",""))
                evaluation=static_evaluate(output,task["expected"])
                reward=evaluation["reward"]
                total_reward += reward
                successes += int(evaluation["score"] >= 75)
                samples.append({"prompt":prompt,"completion":output,"reward":reward,"task_id":task["id"]})
                rewards.append(reward)
                if evaluation["score"] < float(cfg.get("reward_threshold","0.45"))*100:
                    save_mistake(task["id"],prompt,output,evaluation)
            except Exception as exc:
                save_mistake(task["id"],prompt,"",{"error":str(exc)},"generation_errors")
                raise
            if len(samples) >= batch_size or episode == episodes-1:
                result=adapter.train_step(samples)
                if isinstance(result,dict):
                    updates += int(result.get("updates",1))
                    if result.get("loss") is not None: losses.append(float(result["loss"]))
                else:
                    updates += 1
                samples=[]; rewards=[]
            with connect() as c:
                c.execute("UPDATE runs SET reward=?,success_rate=?,real_updates=?,avg_loss=? WHERE id=?",
                          (total_reward/(episode+1),successes/(episode+1),updates,
                           sum(losses)/len(losses) if losses else 0,run_id))
        with connect() as c:
            c.execute("UPDATE runs SET status='completed',finished_at=?,notes=? WHERE id=?",
                      (now(),"Training completed using the configured adapter. Check adapter logs and evaluation metrics.",run_id))
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
                "verified":c.execute("SELECT COUNT(*) FROM mistakes WHERE verification='verified'").fetchone()[0],
                "runs":c.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
                "completed_runs":c.execute("SELECT COUNT(*) FROM runs WHERE status='completed'").fetchone()[0]}

@app.get("/api/datasets")
def list_datasets(split: str | None=None, category: str | None=None):
    query="SELECT * FROM datasets WHERE 1=1"; args=[]
    if split: query+=" AND split=?"; args.append(split)
    if category: query+=" AND category=?"; args.append(category)
    query+=" ORDER BY created_at DESC LIMIT 2000"
    with connect() as c: return [dict(x) for x in c.execute(query,args)]

@app.post("/api/datasets/import")
async def import_data(file: UploadFile=File(...), category: str="general"):
    if not file.filename or not file.filename.lower().endswith((".jsonl",".json",".csv")):
        raise HTTPException(400,"Upload a .jsonl, .json or .csv file.")
    raw=await file.read()
    if len(raw)>15*1024*1024: raise HTTPException(413,"Dataset file exceeds the 15 MB import limit.")
    try:
        decoded=raw.decode("utf-8-sig")
        if file.filename.lower().endswith(".jsonl"):
            rows=[json.loads(line) for line in decoded.splitlines() if line.strip()]
        elif file.filename.lower().endswith(".json"):
            data=json.loads(decoded); rows=data if isinstance(data,list) else data.get("examples",[])
        else:
            rows=list(csv.DictReader(io.StringIO(decoded)))
    except (UnicodeDecodeError,json.JSONDecodeError) as exc:
        raise HTTPException(400,f"Could not parse dataset: {exc}")
    if not isinstance(rows,list): raise HTTPException(400,"Dataset must contain a list of examples.")
    count=0
    with connect() as c:
        for i,row in enumerate(rows):
            if not isinstance(row,dict): continue
            code=str(row.get("content") or row.get("code") or row.get("completion") or row.get("output") or row.get("text") or "")
            prompt=str(row.get("prompt") or row.get("instruction") or row.get("title") or f"Example {i+1}")
            if not code.strip(): continue
            split=str(row.get("split") or ("validation" if i%10==8 else "test" if i%10==9 else "train")).lower()
            if split not in ("train","validation","test"): split="train"
            meta={k:v for k,v in row.items() if k not in ("content","code","completion","output","text","prompt","instruction","title")}
            c.execute("INSERT INTO datasets(id,name,category,split,content,metadata_json,created_at) VALUES(?,?,?,?,?,?,?)",
                      (str(uuid.uuid4()),prompt[:300],str(row.get("category") or category)[:80],split,code,json.dumps(meta,ensure_ascii=False),now()))
            count+=1
    return {"added":count,"total_rows":len(rows),"splits":"train/validation/test"}

@app.get("/api/datasets/export")
def export_datasets():
    with connect() as c: rows=[dict(x) for x in c.execute("SELECT * FROM datasets ORDER BY created_at")]
    def stream():
        for row in rows:
            yield json.dumps({"prompt":row["name"],"category":row["category"],"split":row["split"],"content":row["content"]},ensure_ascii=False)+"\n"
    return StreamingResponse(stream(),media_type="application/x-ndjson",headers={"Content-Disposition":"attachment; filename=neuralforge-dataset.jsonl"})

@app.get("/api/tasks")
def list_tasks():
    with connect() as c:return [dict(x) for x in c.execute("SELECT * FROM tasks ORDER BY created_at DESC")]

@app.post("/api/tasks")
def add_task(t:TaskIn):
    tid=str(uuid.uuid4())
    with connect() as c:
        c.execute("INSERT INTO tasks(id,title,prompt,expected,difficulty,category,created_at) VALUES(?,?,?,?,?,?,?)",
                  (tid,t.title,t.prompt,t.expected,t.difficulty,t.category,now()))
    return {"id":tid}

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
    query="""SELECT m.*,
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
    return {"id":mid,"verification":"unverified","append_only":True}

@app.patch("/api/mistakes/{mid}")
def review_mistake(mid:str, review:ReviewIn):
    if review.verification not in ("verified","rejected","needs_review"):
        raise HTTPException(400,"Choose verified, rejected or needs_review.")
    with connect() as c:
        row=c.execute("SELECT id FROM mistakes WHERE id=?",(mid,)).fetchone()
        if not row: raise HTTPException(404,"Mistake not found.")
        # Strict append-only: original attempt is immutable; status is derived from the latest review event.
        c.execute("INSERT INTO mistake_events(id,mistake_id,action,note,created_at) VALUES(?,?,?,?,?)",
                  (str(uuid.uuid4()),mid,review.verification,review.note,now()))
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
            "batch_size":int(cfg.get("batch_size","2")),"reward_threshold":float(cfg.get("reward_threshold","0.45"))}

@app.put("/api/adapter/config")
def set_adapter_config(config:AdapterConfig):
    global _adapter_instance, _adapter_fingerprint
    values=config.model_dump()
    with connect() as c:
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
        return {"configured":True,"connected":True,"can_train":True,
                "name":getattr(adapter,"name",type(adapter).__name__),
                "message":"Adapter loaded. A training run will call generate and train_step."}
    except Exception as exc:
        return {"configured":True,"connected":False,"can_train":False,"error":str(exc)}

@app.post("/api/runs/start")
def start_run(request:RunIn, background:BackgroundTasks):
    rid=str(uuid.uuid4())
    with connect() as c:
        c.execute("INSERT INTO runs(id,status,episodes,reward,success_rate,real_updates,notes,created_at) VALUES(?,?,?,?,?,?,?,?)",
                  (rid,"queued",request.episodes,0,0,0,"Queued; model weights are not changed until adapter train_step succeeds.",now()))
    background.add_task(execute_training,rid,request.model_dump())
    return {"id":rid,"status":"queued","episodes":request.episodes,"message":"Training run queued. Poll /api/runs for status."}

@app.get("/api/runs")
def list_runs():
    with connect() as c:return [dict(x) for x in c.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 200")]

@app.get("/api/runs/{run_id}")
def get_run(run_id:str):
    with connect() as c:
        row=c.execute("SELECT * FROM runs WHERE id=?",(run_id,)).fetchone()
        if not row: raise HTTPException(404,"Run not found.")
        return dict(row)

@app.post("/api/checkpoints")
def save_checkpoint(payload:dict):
    label=str(payload.get("label","Manual checkpoint"))[:160]
    run_id=str(payload.get("run_id",""))
    cid=str(uuid.uuid4())
    try:
        adapter=get_adapter()
        if adapter is None: raise RuntimeError("No model adapter connected.")
        path=str(CHECKPOINT_DIR/f"{cid}")
        metadata=adapter.save_checkpoint(path) or {}
        kind="model_weights"
    except Exception as exc:
        path=""; metadata={"error":str(exc)}; kind="metadata_only"
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
        result=adapter.load_checkpoint(row["path"])
        return {"restored":True,"result":result}
    except Exception as exc: raise HTTPException(400,str(exc))

@app.get("/api/metrics")
def metrics():
    with connect() as c:
        rows=[dict(x) for x in c.execute("SELECT id,status,episodes,reward,success_rate,real_updates,avg_loss,created_at,model_name FROM runs ORDER BY created_at")]
    return {"runs":rows,"latest":rows[-1] if rows else None}

@app.get("/api/docs")
def api_docs(): return {"openapi":"/openapi.json","swagger":"/docs","redoc":"/redoc"}

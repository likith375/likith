from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime, timezone
import sqlite3,json,csv,io,uuid

ROOT=Path(__file__).resolve().parents[1]
DB=ROOT/"neuralforge.sqlite3"
app=FastAPI(title="NeuralForge RL Lab",version="0.1.0")
app.mount("/static",StaticFiles(directory=ROOT/"frontend"),name="static")
def now(): return datetime.now(timezone.utc).isoformat()
def conn():
 c=sqlite3.connect(DB); c.row_factory=sqlite3.Row; return c
def init():
 with conn() as c:
  c.executescript("""
  CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY,name TEXT,category TEXT,split TEXT,content TEXT,created_at TEXT);
  CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,title TEXT,prompt TEXT,expected TEXT,difficulty TEXT,created_at TEXT);
  CREATE TABLE IF NOT EXISTS mistakes(id TEXT PRIMARY KEY,task_id TEXT,prompt TEXT,output TEXT,category TEXT,evidence TEXT,correction TEXT,verification TEXT,evaluator TEXT,created_at TEXT);
  CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,status TEXT,episodes INTEGER,reward REAL,success_rate REAL,real_updates INTEGER,notes TEXT,created_at TEXT);
  CREATE TABLE IF NOT EXISTS checkpoints(id TEXT PRIMARY KEY,run_id TEXT,label TEXT,kind TEXT,created_at TEXT);
  """)
  if c.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]==0:
   samples=[("Luau local variable","syntax","local coins = 10","train"),("Luau function","functions","local function add(a,b) return a+b end","train"),("Touched event","events","part.Touched:Connect(function(hit) print(hit.Name) end)","validation"),("Players service","services","local Players = game:GetService('Players')","test")]
   for n,cat,code,split in samples:c.execute("INSERT INTO datasets VALUES(?,?,?,?,?,?)",(str(uuid.uuid4()),n,cat,split,code,now()))
  if c.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]==0:
   c.execute("INSERT INTO tasks VALUES(?,?,?,?,?,?)",(str(uuid.uuid4()),"Create health variable","Declare local health with value 100.","local health = 100","easy",now()))
init()
class Task(BaseModel):
 title:str; prompt:str; expected:str=""; difficulty:str="easy"
class Mistake(BaseModel):
 task_id:str=""; prompt:str; output:str; category:str="failed_tasks"; evidence:str=""; correction:str=""
class Verify(BaseModel): verification:str
class Run(BaseModel): episodes:int=5
@app.get("/",response_class=HTMLResponse)
def home():return (ROOT/"frontend"/"index.html").read_text()
@app.get("/api/overview")
def overview():
 with conn() as c:return {"datasets":c.execute("SELECT COUNT(*) FROM datasets").fetchone()[0],"tasks":c.execute("SELECT COUNT(*) FROM tasks").fetchone()[0],"mistakes":c.execute("SELECT COUNT(*) FROM mistakes").fetchone()[0],"runs":c.execute("SELECT COUNT(*) FROM runs").fetchone()[0],"adapter":"Demo workflow simulator","training_ready":False}
@app.get("/api/datasets")
def datasets():
 with conn() as c:return [dict(x) for x in c.execute("SELECT * FROM datasets ORDER BY created_at DESC")]
@app.post("/api/datasets/import")
async def import_data(file:UploadFile=File(...),category:str="general"):
 raw=(await file.read()).decode("utf-8-sig")
 try:
  rows=[json.loads(x) for x in raw.splitlines() if x.strip()] if file.filename.lower().endswith(".jsonl") else list(csv.DictReader(io.StringIO(raw))) if file.filename.lower().endswith(".csv") else None
  if rows is None:raise HTTPException(400,"Upload JSONL or CSV.")
 except json.JSONDecodeError as e:raise HTTPException(400,str(e))
 count=0
 with conn() as c:
  for i,r in enumerate(rows):
   code=str(r.get("content") or r.get("code") or r.get("completion") or r.get("text") or "")
   if not code.strip():continue
   name=str(r.get("prompt") or r.get("instruction") or r.get("title") or f"Imported example {i+1}")
   split=str(r.get("split") or ("validation" if i%10==8 else "test" if i%10==9 else "train"))
   c.execute("INSERT INTO datasets VALUES(?,?,?,?,?,?)",(str(uuid.uuid4()),name,str(r.get("category") or category),split,code,now()));count+=1
 return {"added":count}
@app.get("/api/tasks")
def tasks():
 with conn() as c:return [dict(x) for x in c.execute("SELECT * FROM tasks ORDER BY created_at DESC")]
@app.post("/api/tasks")
def add_task(t:Task):
 tid=str(uuid.uuid4())
 with conn() as c:c.execute("INSERT INTO tasks VALUES(?,?,?,?,?,?)",(tid,t.title,t.prompt,t.expected,t.difficulty,now()))
 return {"id":tid}
@app.post("/api/evaluate")
def evaluate(p:dict):
 out=str(p.get("output",""));expected=str(p.get("expected",""))
 checks={"non_empty":bool(out.strip()),"luau_like":any(k in out for k in ("local ","function","game:GetService","Connect(","return","Instance.new")),"no_placeholder":"TODO" not in out and "DEMO OUTPUT" not in out}
 exact=bool(expected.strip()) and expected.strip() in out
 passed=sum(checks.values())+int(exact);total=len(checks)+int(bool(expected.strip()))
 return {"score":round(100*passed/total,1),"checks":checks,"expected_match":exact,"warning":"Static heuristic only; Luau is not executed."}
@app.get("/api/mistakes")
def mistakes():
 with conn() as c:return [dict(x) for x in c.execute("SELECT * FROM mistakes ORDER BY created_at DESC")]
@app.post("/api/mistakes")
def add_mistake(m:Mistake):
 mid=str(uuid.uuid4())
 with conn() as c:c.execute("INSERT INTO mistakes VALUES(?,?,?,?,?,?,?,?,?,?)",(mid,m.task_id,m.prompt,m.output,m.category,m.evidence,m.correction,"unverified","not_evaluated",now()))
 return {"id":mid,"verification":"unverified","append_only":True}
@app.patch("/api/mistakes/{mid}")
def verify(mid:str,v:Verify):
 if v.verification not in ("verified","rejected","needs_review"):raise HTTPException(400,"Invalid verification status")
 with conn() as c:
  old=c.execute("SELECT * FROM mistakes WHERE id=?",(mid,)).fetchone()
  if not old:raise HTTPException(404,"Not found")
  rid=str(uuid.uuid4())
  c.execute("INSERT INTO mistakes VALUES(?,?,?,?,?,?,?,?,?,?)",(rid,old["task_id"],old["prompt"],old["output"],old["category"],"Review event for "+mid,old["correction"],v.verification,"Human review",now()))
 return {"review_record":rid,"original_preserved":True}
@app.post("/api/runs/start")
def start_run(r:Run):
 n=max(1,min(r.episodes,1000));rid=str(uuid.uuid4())
 with conn() as c:
  c.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?,?)",(rid,"completed_demo",n,0.0,0.0,0,"Demo workflow only; no AI weights updated.",now()))
 return {"id":rid,"status":"completed_demo","episodes":n,"average_reward":0,"success_rate":0,"real_weight_updates":0,"message":"Demo run complete. Connect a compatible model trainer for actual learning."}
@app.get("/api/runs")
def runs():
 with conn() as c:return [dict(x) for x in c.execute("SELECT * FROM runs ORDER BY created_at DESC")]
@app.post("/api/checkpoints")
def save_checkpoint(p:dict):
 cid=str(uuid.uuid4())
 with conn() as c:c.execute("INSERT INTO checkpoints VALUES(?,?,?,?,?)",(cid,str(p.get("run_id","")),str(p.get("label","Manual checkpoint")),"metadata_only_no_model_weights",now()))
 return {"id":cid,"kind":"metadata_only_no_model_weights"}
@app.get("/api/checkpoints")
def checkpoints():
 with conn() as c:return [dict(x) for x in c.execute("SELECT * FROM checkpoints ORDER BY created_at DESC")]
@app.get("/api/adapter/status")
def adapter_status():return {"connected":False,"name":"Demo workflow simulator","can_train":False,"can_save_weights":False,"message":"Connect your Python adapter to enable genuine weight updates."}

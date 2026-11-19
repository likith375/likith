# NeuralForge RL Lab

A local-first training laboratory for connecting your own scratch-built Python AI. This repository contains the lab software, not an AI model.

## Features
- FastAPI backend and browser dashboard
- Import Luau datasets from JSONL or CSV
- Train/validation/test split metadata
- Task creation and deterministic static evaluation
- Append-only mistake records with verification status
- Transparent reward breakdown
- Demo mode for testing without a model
- Adapter interface for a user-supplied Python model
- Run history and checkpoint metadata

## Run locally
1. Install Python 3.10 or newer.
2. Install dependencies: pip install -r requirements.txt
3. Start: uvicorn backend.main:app --reload
4. Open http://127.0.0.1:8000

Demo mode is a workflow simulator. It does not update neural-network weights. Real training requires an adapter implementing train_step and checkpoint methods.

## Safety
Generated Luau is evaluated using static checks and task metadata only. The app does not execute arbitrary uploaded code. Mistake records are append-only; corrections remain unverified until reviewed.

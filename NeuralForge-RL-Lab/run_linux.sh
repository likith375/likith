#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.10 or newer is required."
  exit 1
fi
python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' || {
  echo "Python 3.10 or newer is required."
  exit 1
}

if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install -r requirements.txt
echo "Starting NeuralForge RL Lab at http://127.0.0.1:8000"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

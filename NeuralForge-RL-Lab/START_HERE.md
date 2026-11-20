# Start NeuralForge RL Lab

## Windows 10 / 11

1. Install Python 3.10 or newer from python.org and enable **Add Python to PATH**.
2. Download or clone this repository.
3. Open the `NeuralForge-RL-Lab` folder.
4. Double-click **`run_windows.bat`**.
5. Wait for the browser to open at http://127.0.0.1:8000.

The first launch creates a local virtual environment and installs the web-app dependencies. Keep the terminal window open while using the lab.

## Linux / macOS

Open a terminal in this folder and run:

```bash
chmod +x run_linux.sh
./run_linux.sh
```

Then open http://127.0.0.1:8000.

## GitHub Codespaces

1. Open the NeuralForge-RL-Lab folder in GitHub.
2. Choose **Code → Codespaces → Create codespace on main** if Codespaces is available.
3. Wait for the development container to install dependencies and start the server.
4. Open forwarded port 8000 named **NeuralForge RL Lab**.

## Important

NeuralForge is the training platform, not a pre-trained AI. It starts with example data and no connected model. Open **Model connection** in the dashboard to connect your own Python adapter or a local Hugging Face causal language model. The built-in Hugging Face option needs the extra dependencies in `requirements-ml.txt` and a compatible local model folder.

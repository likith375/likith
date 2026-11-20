@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher "py" was not found. Install Python 3.10 or newer and enable Add Python to PATH.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating NeuralForge virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 goto :failed
)

call ".venv\Scripts\activate.bat"
echo Installing or checking NeuralForge dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo Starting NeuralForge RL Lab...
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:8000"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
goto :eof

:failed
echo.
echo NeuralForge could not start. Read the error above.
pause
exit /b 1

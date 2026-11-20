@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    where python >nul 2>nul
    if errorlevel 1 goto :no_python
    set "PYTHON_CMD=python"
) else (
    set "PYTHON_CMD=py -3"
)

%PYTHON_CMD% -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
    echo Python 3.10 or newer is required.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating NeuralForge virtual environment...
    %PYTHON_CMD% -m venv .venv
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

:no_python
echo Python 3.10 or newer was not found. Install Python and enable Add Python to PATH.
pause
exit /b 1

:failed
echo.
echo NeuralForge could not start. Read the error above.
pause
exit /b 1

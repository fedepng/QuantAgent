@echo off
setlocal

cd /d "%~dp0"
set "PROJECT_PYTHON=.venv\Scripts\python.exe"

if not exist "%PROJECT_PYTHON%" (
    echo [QuantAgent] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 goto :error

    echo [QuantAgent] Installing dependencies...
    "%PROJECT_PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 goto :error
)

echo [QuantAgent] Starting at http://127.0.0.1:8000
"%PROJECT_PYTHON%" -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
exit /b %errorlevel%

:error
echo [QuantAgent] Failed to start.
exit /b 1

$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $project ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    python -m venv (Join-Path $project ".venv")
    & $python -m pip install -r (Join-Path $project "requirements.txt")
}
& $python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000


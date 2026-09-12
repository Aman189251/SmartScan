@echo off
REM Run the benchmark against every baseline scheduler.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\run_experiment.py %*
) else (
    python scripts\run_experiment.py %*
)
pause

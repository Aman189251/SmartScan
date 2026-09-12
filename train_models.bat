@echo off
REM Generate the historical dataset and train the offline models.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\train_models.py %*
) else (
    python scripts\train_models.py %*
)
pause

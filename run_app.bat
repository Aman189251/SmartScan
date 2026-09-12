@echo off
REM Launch the Smart Scan Strategy workstation.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" frontend\main.py %*
) else (
    python frontend\main.py %*
)
if errorlevel 1 pause

@echo off
cd /d "%~dp0"
start "STUDYARD browser" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8765"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" -m studyard
if errorlevel 1 pause

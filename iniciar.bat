@echo off
cd /d "%~dp0"
start "STUDYARD browser" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8765"
python -m studyard
if errorlevel 1 pause

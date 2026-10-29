@echo off
cd /d "%~dp0"
rem o resumo depende do Ollama; sem ele a API recusa conexao (WinError 10061)
tasklist /fi "imagename eq ollama.exe" | find /i "ollama.exe" >nul || (
  if exist "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
)
start "STUDYARD browser" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8765"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" -m studyard
if errorlevel 1 pause

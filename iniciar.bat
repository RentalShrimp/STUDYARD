@echo off
cd /d "%~dp0"
rem KV em q8_0 cabe na RTX 2060; sem isso o contexto grande vai para a RAM
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KV_CACHE_TYPE=q8_0
rem o resumo depende do Ollama; sem ele a API recusa conexao (WinError 10061)
tasklist /fi "imagename eq ollama.exe" | find /i "ollama.exe" >nul || (
  if exist "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
)
start "STUDYARD browser" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8765"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" -m studyard
if errorlevel 1 pause

from __future__ import annotations

import threading
from datetime import date
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from studyard.api_client import ApiClient
from studyard.capture import AudioCapture, CaptureError
from studyard.config import CONFIG_PATH, Config, load_config, record_errors
from studyard.engine import Recorder
from studyard.naming import day_dir, next_stem
from studyard.pending import process_pending_item, scan_pendings
from studyard.session import SessionFiles

STATIC = Path(__file__).resolve().parent / "static"
Source = Literal["mic", "system", "both"]


class StartBody(BaseModel):
    source: Source = "mic"
    save_audio: bool = False


class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.recording = False
        self.stop_flag = threading.Event()
        self.thread: threading.Thread | None = None
        self.recorder: Recorder | None = None
        self.capture = None
        self.message = ""
        self.folder = ""
        self.aula = ""
        self.resumo = ""
        self.last_text = ""
        self.status = "idle"
        self.offline = False
        self.source: Source = "mic"
        self.save_audio = False
        self.cfg: Config | None = None


def create_app(
    cfg_path: Path | None = None,
    api_factory=None,
    capture_factory=None,
) -> FastAPI:
    cfg_path = cfg_path or CONFIG_PATH
    make_api = api_factory or (lambda cfg: ApiClient(cfg))
    make_capture = capture_factory or (lambda: AudioCapture())
    state = AppState()
    app = FastAPI(title="STUDYARD")
    app.state.runtime = state
    app.state.cfg_path = cfg_path
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    def _load() -> Config:
        return load_config(cfg_path)

    def _status_payload() -> dict:
        cfg = state.cfg or _load()
        rec = state.recorder
        pendings = []
        try:
            pendings = [
                {
                    "stem": p["stem"],
                    "folder": str(p["folder"]),
                    "need": p["need"],
                }
                for p in scan_pendings(cfg.output_dir)
            ]
        except OSError:
            pass
        return {
            "state": rec.status if rec and state.recording else state.status,
            "recording": state.recording,
            "last_text": rec.last_text if rec else state.last_text,
            "folder": state.folder,
            "aula": state.aula,
            "resumo": state.resumo,
            "message": rec.message if rec and rec.message else state.message,
            "offline": rec.offline if rec else state.offline,
            "save_audio_default": cfg.save_audio,
            "pendings": pendings,
        }

    @app.get("/api/status")
    def status():
        with state.lock:
            return _status_payload()

    def _loop(capture, recorder: Recorder) -> None:
        try:
            while not state.stop_flag.is_set():
                pcm = capture.read_chunk(0.5)
                recorder.feed_pcm(pcm)
        except Exception as exc:
            with state.lock:
                state.message = str(exc)
                state.status = "error"
                if recorder:
                    recorder.status = "error"
                    recorder.message = str(exc)
        finally:
            try:
                capture.stop()
            except Exception:
                pass

    @app.post("/api/start")
    def start(body: StartBody):
        cfg = _load()
        errs = record_errors(cfg)
        if errs:
            return JSONResponse({"errors": errs}, status_code=400)
        with state.lock:
            if state.recording:
                return JSONResponse({"message": "sessão ativa"}, status_code=409)
            today = date.today()
            folder = day_dir(cfg.output_dir, today)
            folder.mkdir(parents=True, exist_ok=True)
            stem = next_stem(folder, today)
            session = SessionFiles(folder, stem, body.source, body.save_audio)
            session.create()
            api = make_api(cfg)
            recorder = Recorder(cfg, session, api)
            try:
                capture = make_capture()
                capture.start(body.source)
            except CaptureError as exc:
                return JSONResponse({"errors": [str(exc)]}, status_code=400)
            state.cfg = cfg
            state.recorder = recorder
            state.capture = capture
            state.recording = True
            state.stop_flag.clear()
            state.status = "recording"
            state.message = ""
            state.folder = str(folder)
            state.aula = str(session.aula_path)
            state.resumo = str(session.resumo_path)
            state.source = body.source
            state.save_audio = body.save_audio
            state.thread = threading.Thread(
                target=_loop, args=(capture, recorder), daemon=True
            )
            state.thread.start()
            return _status_payload()

    @app.post("/api/stop")
    def stop():
        with state.lock:
            if not state.recording:
                return JSONResponse({"message": "nenhuma sessão ativa"}, status_code=409)
            state.stop_flag.set()
            rec = state.recorder
            capture = state.capture
            thread = state.thread
        if capture is not None:
            try:
                capture.stop()
            except Exception:
                pass
        if thread is not None:
            thread.join(timeout=8)
        paths = rec.stop() if rec else {}
        with state.lock:
            state.recording = False
            state.capture = None
            state.thread = None
            if rec:
                state.status = rec.status
                state.message = rec.message
                state.last_text = rec.last_text
                state.offline = rec.offline
                state.folder = paths.get("folder", state.folder)
                state.aula = paths.get("aula", state.aula)
                state.resumo = paths.get("resumo", state.resumo)
            else:
                state.status = "idle"
        return _status_payload()

    @app.post("/api/process-pendings")
    def process_all():
        cfg = _load()
        errs = record_errors(cfg)
        if errs:
            return JSONResponse({"errors": errs}, status_code=400)
        with state.lock:
            if state.recording:
                return JSONResponse({"message": "sessão ativa"}, status_code=409)
        api = make_api(cfg)
        items = scan_pendings(cfg.output_dir)
        done = 0
        errors: list[str] = []
        for item in items:
            try:
                process_pending_item(item, api)
                done += 1
            except Exception as exc:
                errors.append(f"{item['stem']}: {exc}")
        with state.lock:
            state.message = f"{done} pendente(s) processado(s)" if done else state.message
            if errors:
                state.status = "error"
                state.message = "; ".join(errors)
            elif done:
                state.status = "saved"
        return {"processed": done, "errors": errors, **_status_payload()}

    return app


app = create_app()


def main() -> None:
    import uvicorn

    cfg = load_config()
    uvicorn.run(
        "studyard.app:app",
        host="127.0.0.1",
        port=cfg.port,
        reload=False,
    )


if __name__ == "__main__":
    main()

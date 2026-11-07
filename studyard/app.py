from __future__ import annotations

import ctypes
import sys
import threading
import time
from datetime import date
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from studyard.api_client import ApiClient, SummaryCancelled
from studyard.capture import AudioCapture, CaptureError, list_capture_devices
from studyard.config import CONFIG_PATH, Config, load_config, record_errors, summarize_errors
from studyard.engine import Recorder
from studyard.naming import day_dir, next_stem
from studyard.pending import process_pending_item, scan_pendings
from studyard.session import SessionFiles
from studyard.summarizer import DEFERRED_MESSAGE

STATIC = Path(__file__).resolve().parent / "static"
Source = Literal["mic", "system", "both"]
HEALTH_INTERVAL_S = 20


def free_ram_gb() -> float | None:
    """RAM livre em GB no Windows. Em outro sistema, ou se a leitura falhar, None."""
    if sys.platform != "win32":
        return None

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
        kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
        ok = kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    except (OSError, AttributeError):
        return None
    if not ok:
        return None
    return stat.ullAvailPhys / (1024**3)


class StartBody(BaseModel):
    source: Source = "mic"
    save_audio: bool = False
    mic_device: str | None = None
    system_device: str | None = None


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
        self.capture_error = ""
        self.summary_problem: str | None = None
        self.finishing = False
        self.cancel_summary = threading.Event()
        self.summary_thread: threading.Thread | None = None
        self.summary_started: float | None = None
        self.progress: dict | None = None


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

    def _alerts(pendings: list[dict]) -> list[dict]:
        alerts: list[dict] = []
        if state.recording:
            if state.capture_error:
                alerts.append(
                    {
                        "key": "capture",
                        "text": (
                            f"A gravação parou: {state.capture_error}. "
                            "Clique em Parar e grave de novo."
                        ),
                    }
                )
            if state.summary_problem:
                alerts.append(
                    {
                        "key": "summary",
                        "text": (
                            f"Resumo em risco: {state.summary_problem}. "
                            "A gravação continua; resolva antes de parar."
                        ),
                    }
                )
            if state.recorder:
                alerts.extend(state.recorder.alerts())
        else:
            if state.status in ("error", "offline") and state.message:
                alerts.append({"key": "failure", "text": state.message})
            if pendings and state.status != "summarizing":
                alerts.append(
                    {
                        "key": "pendings",
                        "text": (
                            f"{len(pendings)} aula(s) sem resumo — "
                            "clique em Processar pendentes."
                        ),
                    }
                )
        mem = _memory_alert()
        if mem:
            alerts.append(mem)
        return alerts

    def _memory_alert() -> dict | None:
        summarizing = state.status == "summarizing"
        rec = state.recorder
        if rec is not None and rec.status == "summarizing":
            summarizing = True
        if not summarizing:
            return None
        free = free_ram_gb()
        if free is None or free >= 3:
            return None
        shown = f"{free:.1f}".replace(".", ",")
        return {
            "key": "memory",
            "text": (
                f"Pouca memória livre ({shown} GB) — "
                "feche abas do navegador para o resumo não ficar lento."
            ),
        }

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
                # a sessão sendo finalizada não é pendência ainda
                if not (
                    (state.recording or state.finishing)
                    and str(p["aula_path"]) == state.aula
                )
            ]
        except OSError:
            pass
        if rec and state.recording:
            current = rec.status
            message = rec.message or state.message
        else:
            current = state.status
            message = state.message
        progress = None
        if state.progress is not None:
            progress = dict(state.progress)
            if state.summary_started is not None:
                progress["elapsed_s"] = max(
                    0, int(time.monotonic() - state.summary_started)
                )
        return {
            "state": current,
            "recording": state.recording,
            "finishing": state.finishing,
            "progress": progress,
            "last_text": rec.last_text if rec else state.last_text,
            "folder": state.folder,
            "aula": state.aula,
            "resumo": state.resumo,
            "message": message,
            "offline": rec.offline if rec else state.offline,
            "save_audio_default": cfg.save_audio,
            "pendings": pendings,
            "alerts": _alerts(pendings),
            "devices": getattr(state.capture, "labels", None) or [],
        }

    @app.get("/api/status")
    def status():
        with state.lock:
            return _status_payload()

    @app.get("/api/devices")
    def devices():
        try:
            return list_capture_devices()
        except Exception as exc:
            return JSONResponse(
                {"errors": [str(exc)], "mics": [], "loopbacks": []},
                status_code=500,
            )

    def _loop(capture, recorder: Recorder) -> None:
        try:
            while not state.stop_flag.is_set():
                pcm = capture.read_chunk(0.5)
                recorder.feed_pcm(pcm)
        except Exception as exc:
            with state.lock:
                state.capture_error = str(exc)
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

    def _monitor(api, recorder: Recorder) -> None:
        while True:
            problem = api.check_summary()
            with state.lock:
                if state.recorder is not recorder:
                    return
                state.summary_problem = problem
            if state.stop_flag.wait(HEALTH_INTERVAL_S):
                return

    @app.post("/api/start")
    def start(body: StartBody):
        cfg = _load()
        errs = record_errors(cfg)
        if errs:
            with state.lock:
                state.status = "error"
                state.message = " · ".join(errs)
            return JSONResponse({"errors": errs}, status_code=400)
        with state.lock:
            if state.finishing or state.status == "summarizing":
                return JSONResponse(
                    {"message": "finalização em andamento"}, status_code=409
                )
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
                capture.start(
                    body.source,
                    mic_device=body.mic_device,
                    system_device=body.system_device,
                )
            except CaptureError as exc:
                state.status = "error"
                state.message = str(exc)
                return JSONResponse({"errors": [str(exc)]}, status_code=400)
            state.cfg = cfg
            state.recorder = recorder
            state.capture = capture
            state.recording = True
            state.stop_flag.clear()
            state.capture_error = ""
            state.summary_problem = None
            state.finishing = False
            state.cancel_summary.clear()
            state.progress = None
            state.summary_started = None
            state.status = "recording"
            labels = getattr(capture, "labels", None) or []
            state.message = (
                "Capturando: " + "; ".join(labels) if labels else ""
            )
            state.folder = str(folder)
            state.aula = str(session.aula_path)
            state.resumo = str(session.resumo_path)
            state.source = body.source
            state.save_audio = body.save_audio
            state.thread = threading.Thread(
                target=_loop, args=(capture, recorder), daemon=True
            )
            state.thread.start()
            threading.Thread(target=_monitor, args=(api, recorder), daemon=True).start()
            return _status_payload()

    def _set_progress(info: dict) -> None:
        with state.lock:
            state.progress = {
                "part": int(info.get("part") or 1),
                "parts": int(info.get("parts") or 1),
                "tokens": int(info.get("tokens") or 0),
                "elapsed_s": int(info.get("elapsed_s") or 0),
            }

    def _busy() -> bool:
        alive = state.summary_thread is not None and state.summary_thread.is_alive()
        return state.finishing or state.status == "summarizing" or alive

    @app.post("/api/stop")
    def stop():
        with state.lock:
            if state.finishing or (
                state.summary_thread is not None and state.summary_thread.is_alive()
            ):
                return JSONResponse(
                    {"message": "finalização em andamento"}, status_code=409
                )
            if not state.recording:
                return JSONResponse({"message": "nenhuma sessão ativa"}, status_code=409)
            state.finishing = True
            state.cancel_summary.clear()
            state.stop_flag.set()
            rec = state.recorder
            capture = state.capture
            thread = state.thread
            if rec:
                rec.stopping = True
                rec.status = "stopping"
            state.message = "Finalizando a transcrição."
        if capture is not None:
            try:
                capture.stop()
            except Exception:
                pass
        if thread is not None:
            # espera o trecho em transcrição terminar, senão roda junto com o resumo
            thread.join(timeout=300)
        try:
            ready = rec.finish_audio() if rec else False
        except Exception as exc:
            with state.lock:
                state.recording = False
                state.finishing = False
                state.capture = None
                state.thread = None
                state.status = "error"
                state.message = str(exc)
                state.progress = None
                state.summary_started = None
                return _status_payload()
        with state.lock:
            state.capture = None
            state.thread = None
            state.recording = False
            if rec:
                state.last_text = rec.last_text
                state.offline = rec.offline
                state.folder = str(rec.session.folder)
                state.aula = str(rec.session.aula_path)
                state.resumo = str(rec.session.resumo_path)
            if not ready or rec is None:
                state.finishing = False
                state.status = rec.status if rec else "idle"
                state.message = rec.message if rec else state.message
                state.progress = None
                state.summary_started = None
                return _status_payload()
            state.status = "summarizing"
            state.message = rec.message
            state.summary_started = time.monotonic()
            state.progress = {"part": 1, "parts": 1, "tokens": 0, "elapsed_s": 0}

            def worker() -> None:
                try:
                    rec.summarize_pending(
                        cancel=state.cancel_summary,
                        on_progress=_set_progress,
                    )
                except SummaryCancelled:
                    pass
                except Exception as exc:
                    with state.lock:
                        state.status = "error"
                        state.message = str(exc)
                        state.offline = True
                        state.finishing = False
                        state.recording = False
                        state.progress = None
                        state.summary_started = None
                    return
                with state.lock:
                    state.status = rec.status
                    state.message = rec.message
                    state.offline = rec.offline
                    state.last_text = rec.last_text
                    state.resumo = str(rec.session.resumo_path)
                    state.finishing = False
                    state.recording = False
                    state.progress = None
                    state.summary_started = None

            state.summary_thread = threading.Thread(
                target=worker, name="studyard-summary", daemon=True
            )
            payload = _status_payload()
            state.summary_thread.start()
        return payload

    @app.post("/api/skip-summary")
    def skip_summary():
        with state.lock:
            if not _busy():
                return JSONResponse(
                    {"message": "nenhum resumo em andamento"}, status_code=409
                )
            state.cancel_summary.set()
            thread = state.summary_thread
        if thread is not None:
            thread.join(timeout=30)
        with state.lock:
            return _status_payload()

    @app.post("/api/process-pendings")
    def process_all():
        cfg = _load()
        errs = summarize_errors(cfg)
        if errs:
            return JSONResponse({"errors": errs}, status_code=400)
        with state.lock:
            if state.recording or _busy():
                return JSONResponse(
                    {"message": "finalização em andamento"}, status_code=409
                )
            state.finishing = True
            state.cancel_summary.clear()
            state.status = "summarizing"
            state.message = (
                "Gerando resumo dos pendentes — não feche o app nem desligue o PC até terminar."
            )
            state.summary_started = time.monotonic()
            state.progress = {"part": 1, "parts": 1, "tokens": 0, "elapsed_s": 0}
        try:
            items = scan_pendings(cfg.output_dir)
        except Exception as exc:
            with state.lock:
                state.finishing = False
                state.status = "error"
                state.message = str(exc)
                state.progress = None
                state.summary_started = None
                return _status_payload()
        api = make_api(cfg)

        def worker() -> None:
            done = 0
            errors: list[str] = []
            try:
                for item in items:
                    if state.cancel_summary.is_set():
                        raise SummaryCancelled()

                    def on_progress(info, stem=item["stem"]):
                        _set_progress(info)
                        with state.lock:
                            if state.status == "summarizing":
                                state.message = (
                                    f"{stem}: resumindo parte {info.get('part')} "
                                    f"de {info.get('parts')}"
                                )

                    try:
                        process_pending_item(
                            item,
                            api,
                            cancel=state.cancel_summary,
                            on_progress=on_progress,
                        )
                        done += 1
                    except SummaryCancelled:
                        raise
                    except Exception as exc:
                        errors.append(f"{item['stem']}: {exc}")
            except SummaryCancelled:
                with state.lock:
                    state.status = "deferred"
                    state.message = DEFERRED_MESSAGE
                    state.finishing = False
                    state.recording = False
                    state.progress = None
                    state.summary_started = None
                return
            with state.lock:
                state.finishing = False
                state.recording = False
                state.progress = None
                state.summary_started = None
                if errors:
                    state.status = "error"
                    state.message = "; ".join(errors)
                else:
                    state.status = "saved" if done else "idle"
                    state.message = f"{done} pendente(s) processado(s)"

        with state.lock:
            state.summary_thread = threading.Thread(
                target=worker, name="studyard-summary", daemon=True
            )
            payload = _status_payload()
            state.summary_thread.start()
        return payload

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

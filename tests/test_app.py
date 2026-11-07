import json
import threading
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from studyard.api_client import ApiError
from studyard.app import create_app
from studyard.capture import CaptureError
from studyard.wavutil import RATE


class ZeroCapture:
    def start(self, source, **kwargs):
        self.source = source
        self.kwargs = kwargs

    def read_chunk(self, seconds):
        return np.zeros(max(1, int(RATE * seconds)), dtype=np.float32)

    def stop(self):
        pass


class FakeApi:
    def transcribe(self, wav_bytes: bytes) -> str:
        return "fala"

    def summarize(self, transcript: str) -> str:
        return "resumo teste"

    def check_summary(self):
        return None


def _client(tmp_path: Path, key: str = "k") -> TestClient:
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "output_dir": str(tmp_path / "out"),
                "api_base_url": "https://example.invalid/v1",
                "api_key": key,
                "chunk_seconds": 60,
            }
        ),
        encoding="utf-8",
    )
    app = create_app(
        cfg_path=cfg,
        api_factory=lambda c: FakeApi(),
        capture_factory=lambda: ZeroCapture(),
    )
    return TestClient(app)


def test_index_serves_html(tmp_path: Path):
    client = _client(tmp_path)
    res = client.get("/")
    assert res.status_code == 200
    assert "Gravar" in res.text
    assert "Resumir depois" in res.text
    assert "api_key" not in res.text


def test_status_has_no_api_key(tmp_path: Path):
    client = _client(tmp_path)
    data = client.get("/api/status").json()
    assert "api_key" not in data
    assert data["finishing"] is False
    assert data["progress"] is None
    blob = json.dumps(data)
    assert "sk-" not in blob


def test_start_without_key_allowed(tmp_path: Path):
    client = _client(tmp_path, key="")
    res = client.post("/api/start", json={"source": "mic", "save_audio": False})
    assert res.status_code == 200
    assert res.json()["recording"] is True
    client.post("/api/stop")
    _settle(client)


def test_start_capture_error_stays_on_status(tmp_path: Path):
    class BoomCapture:
        def start(self, source, **kwargs):
            raise CaptureError("microfone WASAPI indisponível")

        def read_chunk(self, seconds):
            return np.zeros(1, dtype=np.float32)

        def stop(self):
            pass

    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "output_dir": str(tmp_path / "out"),
                "api_base_url": "",
                "api_key": "",
            }
        ),
        encoding="utf-8",
    )
    app = create_app(
        cfg_path=cfg,
        api_factory=lambda c: FakeApi(),
        capture_factory=lambda: BoomCapture(),
    )
    client = TestClient(app)
    res = client.post("/api/start", json={"source": "mic", "save_audio": False})
    assert res.status_code == 400
    status = client.get("/api/status").json()
    assert "microfone" in status["message"]
    assert status["state"] == "error"


def test_second_start_conflict(tmp_path: Path):
    client = _client(tmp_path)
    first = client.post("/api/start", json={"source": "mic", "save_audio": False})
    assert first.status_code == 200
    second = client.post("/api/start", json={"source": "mic", "save_audio": False})
    assert second.status_code == 409
    client.post("/api/stop")
    _settle(client)


def test_devices_endpoint(tmp_path: Path, monkeypatch):
    from studyard import app as appmod

    monkeypatch.setattr(
        appmod,
        "list_capture_devices",
        lambda: {
            "mics": [
                {
                    "name": "Microfone (Realtek HD Audio Mic input)",
                    "host": "Windows WDM-KS",
                    "kind": "analog",
                    "index": 24,
                }
            ],
            "loopbacks": [{"name": "Alto-falantes (AB13X USB Audio)", "kind": "loopback"}],
        },
    )
    client = _client(tmp_path)
    data = client.get("/api/devices").json()
    assert data["mics"][0]["kind"] == "analog"
    assert data["loopbacks"][0]["name"].startswith("Alto-falantes")


def test_start_forwards_device_choice(tmp_path: Path):
    captured = {}

    class RecCapture(ZeroCapture):
        def start(self, source, **kwargs):
            captured["source"] = source
            captured["kwargs"] = kwargs

    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "output_dir": str(tmp_path / "out"),
                "api_base_url": "https://example.invalid/v1",
                "api_key": "k",
                "chunk_seconds": 60,
            }
        ),
        encoding="utf-8",
    )
    app = create_app(
        cfg_path=cfg,
        api_factory=lambda c: FakeApi(),
        capture_factory=lambda: RecCapture(),
    )
    client = TestClient(app)
    res = client.post(
        "/api/start",
        json={
            "source": "both",
            "save_audio": False,
            "mic_device": "Mic input",
            "system_device": "AB13X",
        },
    )
    assert res.status_code == 200
    assert captured["source"] == "both"
    assert captured["kwargs"]["mic_device"] == "Mic input"
    assert captured["kwargs"]["system_device"] == "AB13X"
    client.post("/api/stop")
    _settle(client)


def _client_with(tmp_path: Path, api=None, capture=None) -> TestClient:
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "output_dir": str(tmp_path / "out"),
                "api_base_url": "https://example.invalid/v1",
                "api_key": "k",
                "chunk_seconds": 60,
            }
        ),
        encoding="utf-8",
    )
    app = create_app(
        cfg_path=cfg,
        api_factory=lambda c: api or FakeApi(),
        capture_factory=lambda: capture or ZeroCapture(),
    )
    return TestClient(app)


def _settle(client: TestClient) -> dict:
    last: dict = {}
    for _ in range(200):
        last = client.get("/api/status").json()
        if not last.get("finishing") and last.get("state") != "summarizing":
            return last
        time.sleep(0.02)
    raise AssertionError(f"não estabilizou: {last}")


def _wait_alert(client: TestClient, key: str) -> dict:
    for _ in range(100):
        for alert in client.get("/api/status").json()["alerts"]:
            if alert["key"] == key:
                return alert
        time.sleep(0.02)
    raise AssertionError(f"alerta {key} não apareceu")


def test_alert_while_recording_when_summary_server_down(tmp_path: Path):
    class DownApi(FakeApi):
        def check_summary(self):
            return "servidor de resumo fora do ar"

    client = _client_with(tmp_path, api=DownApi())
    client.post("/api/start", json={"source": "mic", "save_audio": False})
    alert = _wait_alert(client, "summary")
    assert "fora do ar" in alert["text"]
    assert client.get("/api/status").json()["recording"] is True
    client.post("/api/stop")
    _settle(client)


def test_alert_when_capture_dies_mid_recording(tmp_path: Path):
    class DyingCapture(ZeroCapture):
        def read_chunk(self, seconds):
            raise RuntimeError("dispositivo removido")

    client = _client_with(tmp_path, capture=DyingCapture())
    client.post("/api/start", json={"source": "mic", "save_audio": False})
    alert = _wait_alert(client, "capture")
    assert "dispositivo removido" in alert["text"]
    client.post("/api/stop")
    _settle(client)


def test_alert_for_failed_summary_and_pending_after_stop(tmp_path: Path):
    class FailingApi(FakeApi):
        def summarize(self, transcript: str) -> str:
            raise ApiError("servidor de resumo fora do ar")

    client = _client_with(tmp_path, api=FailingApi())
    client.post("/api/start", json={"source": "mic", "save_audio": False})
    data = client.post("/api/stop").json()
    assert data["state"] == "summarizing"
    assert data["finishing"] is True
    final = _settle(client)
    assert final["state"] == "error"
    assert {a["key"] for a in final["alerts"]} == {"failure", "pendings"}


def test_no_alerts_on_healthy_session(tmp_path: Path):
    client = _client_with(tmp_path)
    assert client.get("/api/status").json()["alerts"] == []
    client.post("/api/start", json={"source": "mic", "save_audio": False})
    assert client.get("/api/status").json()["alerts"] == []
    stopped = client.post("/api/stop").json()
    assert stopped["state"] == "summarizing"
    final = _settle(client)
    assert final["state"] == "saved"
    assert final["finishing"] is False
    assert final["alerts"] == []


class _HoldApi(FakeApi):
    def __init__(self):
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()

    def summarize(self, transcript, num_ctx=None, on_progress=None, cancel=None):
        self.calls += 1
        if on_progress:
            on_progress(312)
        self.started.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self.release.is_set():
                break
            if cancel is not None and cancel.is_set():
                break
            if cancel is not None:
                cancel.wait(0.05)
            else:
                self.release.wait(0.05)
        return "resumo teste"


def test_second_stop_is_conflict_and_summarizes_once(tmp_path: Path):
    api = _HoldApi()
    client = _client_with(tmp_path, api=api)
    client.post("/api/start", json={"source": "mic", "save_audio": False})
    first = client.post("/api/stop")
    assert first.status_code == 200
    assert first.json()["finishing"] is True
    assert api.started.wait(2)
    second = client.post("/api/stop")
    assert second.status_code == 409
    assert second.json()["message"] == "finalização em andamento"
    blocked = client.post("/api/start", json={"source": "mic", "save_audio": False})
    assert blocked.status_code == 409
    assert api.calls == 1
    api.release.set()
    final = _settle(client)
    assert api.calls == 1
    assert final["finishing"] is False
    assert final["state"] == "saved"


def test_skip_summary_keeps_pending_and_releases_app(tmp_path: Path):
    api = _HoldApi()
    client = _client_with(tmp_path, api=api)
    client.post("/api/start", json={"source": "mic", "save_audio": False})
    stopped = client.post("/api/stop").json()
    assert stopped["state"] == "summarizing"
    assert api.started.wait(2)
    skipped = client.post("/api/skip-summary")
    assert skipped.status_code == 200
    body = skipped.json()
    assert body["recording"] is False
    assert body["finishing"] is False
    assert body["message"] == "Resumo adiado — use Processar pendentes"
    pending = Path(body["aula"]).with_suffix(".pending.json")
    assert pending.exists()
    assert "summarize" in pending.read_text(encoding="utf-8")
    assert not Path(body["resumo"]).exists()
    assert api.calls == 1


def test_progress_is_in_status_while_summarizing(tmp_path: Path):
    api = _HoldApi()
    client = _client_with(tmp_path, api=api)
    client.post("/api/start", json={"source": "mic", "save_audio": False})
    client.post("/api/stop")
    assert api.started.wait(2)
    data = client.get("/api/status").json()
    assert data["state"] == "summarizing"
    assert data["progress"]["part"] == 1
    assert data["progress"]["parts"] >= 1
    assert data["progress"]["tokens"] == 312
    first = data["progress"]["elapsed_s"]
    time.sleep(1.1)
    second = client.get("/api/status").json()["progress"]["elapsed_s"]
    assert second >= first + 1
    api.release.set()
    _settle(client)


def test_process_pendings_runs_in_background_and_rejects_a_second_one(tmp_path: Path):
    day = tmp_path / "out" / "2026-08-26"
    day.mkdir(parents=True)
    (day / "2026-08-26_aula.pending.json").write_text(
        json.dumps({"need": ["summarize"], "save_audio": False}),
        encoding="utf-8",
    )
    api = _HoldApi()
    client = _client_with(tmp_path, api=api)
    first = client.post("/api/process-pendings")
    assert first.status_code == 200
    assert first.json()["state"] == "summarizing"
    assert api.started.wait(2)
    second = client.post("/api/process-pendings")
    assert second.status_code == 409
    assert second.json()["message"] == "finalização em andamento"
    again = client.post("/api/stop")
    assert again.status_code == 409
    api.release.set()
    final = _settle(client)
    assert final["state"] == "saved"
    assert api.calls == 1


def test_low_memory_alert_only_while_summarizing(tmp_path: Path, monkeypatch):
    from studyard import app as appmod

    client = _client(tmp_path)
    runtime = client.app.state.runtime
    monkeypatch.setattr(appmod, "free_ram_gb", lambda: 2.7)
    idle = client.get("/api/status").json()
    assert all(a["key"] != "memory" for a in idle["alerts"])
    runtime.status = "summarizing"
    low = client.get("/api/status").json()
    texts = [a["text"] for a in low["alerts"] if a["key"] == "memory"]
    assert texts == [
        "Pouca memória livre (2,7 GB) — feche abas do navegador para o resumo não ficar lento."
    ]
    monkeypatch.setattr(appmod, "free_ram_gb", lambda: 3.0)
    at_limit = client.get("/api/status").json()
    assert all(a["key"] != "memory" for a in at_limit["alerts"])
    monkeypatch.setattr(appmod, "free_ram_gb", lambda: None)
    unknown = client.get("/api/status").json()
    assert all(a["key"] != "memory" for a in unknown["alerts"])


def test_free_ram_gb_reads_windows_or_returns_none(monkeypatch):
    from studyard.app import free_ram_gb

    if __import__("sys").platform == "win32":
        value = free_ram_gb()
        assert isinstance(value, float)
        assert value > 0
    monkeypatch.setattr("studyard.app.sys.platform", "linux")
    assert free_ram_gb() is None

import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

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
    assert "api_key" not in res.text


def test_status_has_no_api_key(tmp_path: Path):
    client = _client(tmp_path)
    data = client.get("/api/status").json()
    assert "api_key" not in data
    blob = json.dumps(data)
    assert "sk-" not in blob


def test_start_without_key_allowed(tmp_path: Path):
    client = _client(tmp_path, key="")
    res = client.post("/api/start", json={"source": "mic", "save_audio": False})
    assert res.status_code == 200
    assert res.json()["recording"] is True
    client.post("/api/stop")


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

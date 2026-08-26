import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from studyard.app import create_app
from studyard.capture import CaptureError
from studyard.wavutil import RATE


class ZeroCapture:
    def start(self, source):
        self.source = source

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
        def start(self, source):
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

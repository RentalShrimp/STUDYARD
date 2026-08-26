import json
from pathlib import Path

import numpy as np

from studyard.api_client import ApiError
from studyard.pending import process_pending_item, scan_pendings
from studyard.session import SessionFiles, fail_marker
from studyard.wavutil import write_wav


class FakeApi:
    def __init__(self):
        self.transcribe_calls = 0
        self.summarize_calls = 0
        self.fail_summarize = False

    def transcribe(self, wav_bytes: bytes) -> str:
        self.transcribe_calls += 1
        return "texto do wav"

    def summarize(self, transcript: str) -> str:
        self.summarize_calls += 1
        if self.fail_summarize:
            raise ApiError("sum")
        return "resumo pendente"


def test_existing_pending_json_listed(tmp_path: Path):
    day = tmp_path / "2026-08-26"
    day.mkdir()
    (day / "2026-08-26_aula.pending.json").write_text(
        json.dumps({"need": ["summarize"], "save_audio": False}),
        encoding="utf-8",
    )
    items = scan_pendings(tmp_path)
    assert len(items) == 1
    assert items[0]["need"] == ["summarize"]
    assert items[0]["stem"] == "2026-08-26_aula"


def test_orphan_wav_empty_aula_creates_transcribe_pending(tmp_path: Path):
    day = tmp_path / "2026-08-26"
    s = SessionFiles(day, "2026-08-26_aula", "mic", save_audio=False)
    s.create()
    write_wav(s.wav_path, np.zeros(1600, dtype=np.float32))
    items = scan_pendings(tmp_path)
    assert len(items) == 1
    assert "transcribe" in items[0]["need"]
    assert "summarize" in items[0]["need"]
    assert s.pending_path.exists()


def test_orphan_wav_complete_aula_summarize_only(tmp_path: Path):
    day = tmp_path / "2026-08-26"
    s = SessionFiles(day, "2026-08-26_aula", "mic", save_audio=False)
    s.create()
    s.append_transcript("aula completa sem buracos")
    write_wav(s.wav_path, np.zeros(1600, dtype=np.float32))
    items = scan_pendings(tmp_path)
    assert items[0]["need"] == ["summarize"]


def test_wav_with_marker_needs_transcribe(tmp_path: Path):
    day = tmp_path / "2026-08-26"
    s = SessionFiles(day, "2026-08-26_aula", "mic", save_audio=False)
    s.create()
    s.append_transcript(fail_marker(10))
    write_wav(s.wav_path, np.zeros(1600, dtype=np.float32))
    items = scan_pendings(tmp_path)
    assert "transcribe" in items[0]["need"]


def test_process_pending_transcribe_and_summarize(tmp_path: Path):
    day = tmp_path / "2026-08-26"
    s = SessionFiles(day, "2026-08-26_aula", "mic", save_audio=False)
    s.create()
    s.append_transcript(fail_marker(3))
    s.append_pcm(np.zeros(1600, dtype=np.float32))
    s.flush_wav()
    s.write_pending(["transcribe", "summarize"])
    items = scan_pendings(tmp_path)
    api = FakeApi()
    process_pending_item(items[0], api, transcribe_wav=lambda p: api.transcribe(p.read_bytes()))
    assert "texto do wav" in s.read_transcript_body()
    assert fail_marker(3) not in s.read_transcript_body()
    assert s.resumo_path.exists()
    assert not s.pending_path.exists()
    assert not s.wav_path.exists()

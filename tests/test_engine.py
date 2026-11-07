import json
from pathlib import Path

import numpy as np

from studyard.api_client import ApiError
from studyard.config import Config
from studyard.engine import Recorder
from studyard.session import SessionFiles, fail_marker
from studyard.wavutil import RATE, pcm_to_wav_bytes


class FakeApi:
    def __init__(self):
        self.transcribe_calls = 0
        self.fail_next = 0
        self.summarize_ok = True
        self.full_wav_text = "transcrição completa do wav"

    def transcribe(self, wav_bytes: bytes) -> str:
        self.transcribe_calls += 1
        if self.fail_next:
            self.fail_next -= 1
            raise ApiError("down")
        return f"chunk-{self.transcribe_calls}"

    def summarize(self, transcript: str) -> str:
        if not self.summarize_ok:
            raise ApiError("sum")
        return "resumo ok"


def _asr(api: FakeApi):
    def run(pcm):
        return api.transcribe(pcm_to_wav_bytes(np.asarray(pcm)))

    return run


def _cfg(tmp_path: Path, **kw) -> Config:
    base = dict(
        output_dir=tmp_path,
        api_base_url="https://example.invalid/v1",
        api_key="k",
        transcription_model="whisper-1",
        summary_model="gpt-4o-mini",
        language="pt",
        chunk_seconds=1,
        save_audio=False,
        port=8765,
    )
    base.update(kw)
    return Config(**base)


def _session(tmp_path: Path, save_audio: bool = False) -> SessionFiles:
    s = SessionFiles(tmp_path, "2026-08-26_aula", "mic", save_audio=save_audio)
    s.create()
    return s


def test_three_failures_write_marker_and_continue(tmp_path: Path):
    api = FakeApi()
    api.fail_next = 3
    rec = Recorder(_cfg(tmp_path), _session(tmp_path), api, asr=_asr(api))
    rec.process_chunk(np.zeros(RATE, dtype=np.float32), elapsed_s=125)
    body = rec.session.read_transcript_body()
    assert fail_marker(125) in body
    assert rec.had_chunk_failure is True
    rec.process_chunk(np.zeros(RATE, dtype=np.float32), elapsed_s=126)
    assert "chunk-" in rec.session.read_transcript_body()


def test_stop_success_writes_resumo_deletes_wav(tmp_path: Path):
    api = FakeApi()
    s = _session(tmp_path, save_audio=False)
    rec = Recorder(_cfg(tmp_path), s, api, asr=_asr(api))
    rec.process_chunk(np.zeros(RATE, dtype=np.float32), elapsed_s=1)
    rec.stop()
    assert s.resumo_path.exists()
    assert "resumo ok" in s.resumo_path.read_text(encoding="utf-8")
    assert not s.wav_path.exists()


def test_stop_offline_writes_pending_keeps_wav(tmp_path: Path):
    api = FakeApi()
    api.fail_next = 100
    s = _session(tmp_path)
    rec = Recorder(_cfg(tmp_path), s, api, asr=_asr(api))
    rec.process_chunk(np.zeros(RATE, dtype=np.float32), elapsed_s=1)
    rec.stop()
    assert s.pending_path.exists()
    assert s.wav_path.exists()
    assert not s.resumo_path.exists()


def test_stop_after_failures_retranscribes_wav(tmp_path: Path):
    class RetranscribeApi(FakeApi):
        def transcribe(self, wav_bytes: bytes) -> str:
            self.transcribe_calls += 1
            if self.transcribe_calls <= 3:
                raise ApiError("down")
            return self.full_wav_text

    api = RetranscribeApi()
    s = _session(tmp_path)
    rec = Recorder(_cfg(tmp_path), s, api, asr=_asr(api))
    rec.process_chunk(np.zeros(RATE, dtype=np.float32), elapsed_s=1)
    rec.stop()
    assert rec.session.read_transcript_body().strip() == api.full_wav_text
    assert s.resumo_path.exists()


def test_summarize_fail_keeps_aula_pending_summarize(tmp_path: Path):
    api = FakeApi()
    api.summarize_ok = False
    s = _session(tmp_path)
    rec = Recorder(_cfg(tmp_path), s, api, asr=_asr(api))
    rec.process_chunk(np.zeros(RATE, dtype=np.float32), elapsed_s=1)
    rec.stop()
    assert "chunk-" in s.read_transcript_body()
    assert s.pending_path.exists()
    data = s.pending_path.read_text(encoding="utf-8")
    assert "summarize" in data
    assert "transcribe" not in data
    assert s.wav_path.exists()


def test_pending_exists_while_summarizing(tmp_path: Path):
    class WatchApi(FakeApi):
        def summarize(self, transcript: str) -> str:
            self.pending_during = s.pending_path.exists()
            self.status_during = rec.status
            return "resumo ok"

    api = WatchApi()
    s = _session(tmp_path)
    rec = Recorder(_cfg(tmp_path), s, api, asr=_asr(api))
    rec.process_chunk(np.zeros(RATE, dtype=np.float32), elapsed_s=1)
    rec.stop()
    assert api.pending_during is True
    assert api.status_during == "summarizing"
    assert not s.pending_path.exists()


def test_chunk_finished_after_stop_does_not_report_recording(tmp_path: Path):
    api = FakeApi()
    rec = Recorder(_cfg(tmp_path), _session(tmp_path), api, asr=_asr(api))
    rec.stopping = True
    rec.process_chunk(np.zeros(RATE, dtype=np.float32), elapsed_s=1)
    assert rec.status == "stopping"


def test_alerts_report_failed_chunk_while_recording(tmp_path: Path):
    api = FakeApi()
    api.fail_next = 3
    rec = Recorder(_cfg(tmp_path), _session(tmp_path), api, asr=_asr(api))
    rec.process_chunk(np.zeros(RATE, dtype=np.float32), 0)
    # o alerta continua depois que a transcrição volta: o trecho ainda está perdido
    rec.process_chunk(np.zeros(RATE, dtype=np.float32), 1)
    alerts = rec.alerts()
    assert [a["key"] for a in alerts] == ["transcription"]
    assert "1 trecho" in alerts[0]["text"]
    assert "down" in alerts[0]["text"]


def test_alerts_report_silence_and_clear_on_speech(tmp_path: Path):
    now = [0.0]
    api = FakeApi()
    rec = Recorder(
        _cfg(tmp_path), _session(tmp_path), api, clock=lambda: now[0], asr=_asr(api)
    )
    now[0] = 119.0
    assert rec.alerts() == []
    now[0] = 120.0
    assert [a["key"] for a in rec.alerts()] == ["silence"]
    rec.process_chunk(np.zeros(RATE, dtype=np.float32), 120)
    assert rec.alerts() == []


def test_long_transcript_checkpoint_survives_part_failure(tmp_path: Path):
    class PartsApi(FakeApi):
        def __init__(self):
            super().__init__()
            self.n = 0

        def summarize(self, transcript, **kwargs):
            self.n += 1
            if self.n == 1:
                return "parcial-1"
            raise ApiError("caiu na segunda parte")

    text = "\n".join(["conteudo da aula sobre filas."] * 800)
    assert len(text) > 12000
    api = PartsApi()
    s = _session(tmp_path)
    s.append_transcript(text)
    rec = Recorder(_cfg(tmp_path), s, api, asr=_asr(FakeApi()))
    rec.stop()
    path = tmp_path / "2026-08-26_aula.resumo-partes.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["parts"] == ["parcial-1"]
    assert s.pending_path.exists()
    assert not s.resumo_path.exists()


def test_message_warns_not_to_shut_down_while_summarizing(tmp_path: Path):
    seen = []
    api = FakeApi()
    rec = Recorder(_cfg(tmp_path), _session(tmp_path), api, asr=_asr(api))
    original = api.summarize

    def spy(text):
        seen.append(rec.message)
        return original(text)

    api.summarize = spy
    rec.stop()
    assert "desligue o PC" in seen[0]

from pathlib import Path

from studyard.session import SessionFiles, fail_marker


def test_append_keeps_order(tmp_path: Path):
    s = SessionFiles(tmp_path, "2026-08-26_aula", "mic", save_audio=False)
    s.create()
    s.append_transcript("um")
    s.append_transcript("dois")
    body = s.read_transcript_body()
    assert body.index("um") < body.index("dois")
    assert s.aula_path.read_text(encoding="utf-8").count("um") == 1


def test_fail_marker_format():
    assert fail_marker(125) == "[trecho ~02:05 não transcrito]"


def test_pending_roundtrip(tmp_path: Path):
    s = SessionFiles(tmp_path, "2026-08-26_aula", "system", save_audio=True)
    s.create()
    s.write_pending(["transcribe", "summarize"])
    data = s.pending_path.read_text(encoding="utf-8")
    assert "transcribe" in data
    assert '"save_audio":true' in data.replace(" ", "")
    s.clear_pending()
    assert not s.pending_path.exists()


def test_finalize_deletes_wav_when_save_false(tmp_path: Path):
    import numpy as np

    s = SessionFiles(tmp_path, "2026-08-26_aula", "mic", save_audio=False)
    s.create()
    s.append_pcm(np.zeros(1600, dtype=np.float32))
    s.flush_wav()
    assert s.wav_path.exists()
    s.finalize_audio(success=True)
    assert not s.wav_path.exists()


def test_finalize_keeps_wav_on_failure(tmp_path: Path):
    import numpy as np

    s = SessionFiles(tmp_path, "2026-08-26_aula", "mic", save_audio=False)
    s.create()
    s.append_pcm(np.zeros(1600, dtype=np.float32))
    s.flush_wav()
    s.finalize_audio(success=False)
    assert s.wav_path.exists()

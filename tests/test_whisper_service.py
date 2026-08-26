import numpy as np

from studyard.whisper_service import transcribe_float32, transcribe_pcm
from studyard.wavutil import RATE, write_wav


def test_stub_returns_text_for_enough_pcm(monkeypatch):
    monkeypatch.setenv("STUDYARD_AUDIO_STUB", "1")
    pcm = (np.zeros(1600, dtype=np.int16)).tobytes()
    assert transcribe_pcm(pcm) == "texto da aula"


def test_stub_empty_for_short_pcm(monkeypatch):
    monkeypatch.setenv("STUDYARD_AUDIO_STUB", "1")
    assert transcribe_pcm(b"\x00\x00") == ""


def test_float32_wrapper_uses_stub(monkeypatch):
    monkeypatch.setenv("STUDYARD_AUDIO_STUB", "1")
    audio = np.zeros(RATE, dtype=np.float32)
    assert transcribe_float32(audio) == "texto da aula"


def test_wav_file_stub(monkeypatch, tmp_path):
    monkeypatch.setenv("STUDYARD_AUDIO_STUB", "1")
    from studyard.whisper_service import transcribe_wav_path

    path = tmp_path / "a.wav"
    write_wav(path, np.zeros(RATE, dtype=np.float32))
    assert transcribe_wav_path(path) == "texto da aula"

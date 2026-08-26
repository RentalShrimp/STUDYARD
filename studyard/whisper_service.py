"""PCM transcription via faster-whisper (stubbed in tests)."""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np

from studyard.wavutil import RATE, float_to_int16


class TranscribeError(Exception):
    pass


def transcribe_pcm(
    pcm: bytes,
    sample_rate: int = 16000,
    language: str = "pt",
    model_size: str = "base",
) -> str:
    if os.environ.get("STUDYARD_AUDIO_STUB") == "1":
        return "texto da aula" if len(pcm) >= 1600 else ""

    try:
        from faster_whisper import WhisperModel

        cache = getattr(transcribe_pcm, "_models", {})
        if model_size not in cache:
            cache[model_size] = WhisperModel(
                model_size, device="cpu", compute_type="int8"
            )
            transcribe_pcm._models = cache
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = cache[model_size].transcribe(audio, language=language)
        return " ".join(s.text.strip() for s in segments).strip()
    except TranscribeError:
        raise
    except Exception:
        raise TranscribeError("whisper model unavailable") from None


def transcribe_float32(
    pcm: np.ndarray,
    sample_rate: int = RATE,
    language: str = "pt",
    model_size: str = "base",
) -> str:
    raw = float_to_int16(np.asarray(pcm, dtype=np.float32).ravel())
    return transcribe_pcm(raw, sample_rate=sample_rate, language=language, model_size=model_size)


def transcribe_wav_path(
    path: Path,
    language: str = "pt",
    model_size: str = "base",
) -> str:
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        width = wf.getsampwidth()
        channels = wf.getnchannels()
    if width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        raise TranscribeError("WAV precisa ser PCM 16-bit")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return transcribe_float32(audio, sample_rate=rate, language=language, model_size=model_size)

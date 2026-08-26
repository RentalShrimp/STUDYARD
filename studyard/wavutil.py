from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np

RATE = 16000


def float_to_int16(pcm: np.ndarray) -> bytes:
    clipped = np.clip(pcm, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def write_wav(path: Path, pcm: np.ndarray, rate: int = RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(float_to_int16(pcm))


def pcm_to_wav_bytes(pcm: np.ndarray, rate: int = RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(float_to_int16(pcm))
    return buf.getvalue()

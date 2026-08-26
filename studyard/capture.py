from __future__ import annotations

from typing import Literal

import numpy as np

Source = Literal["mic", "system", "both"]


class CaptureError(Exception):
    pass


def mix_mono(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = min(len(a), len(b))
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    return np.clip((a[:n] + b[:n]) * 0.5, -1.0, 1.0).astype(np.float32)


def _wasapi_hostapi_index() -> int:
    import sounddevice as sd

    for i, host in enumerate(sd.query_hostapis()):
        if "WASAPI" in str(host.get("name", "")).upper():
            return i
    raise CaptureError("WASAPI não encontrado neste Windows")


def _default_wasapi_devices() -> tuple[int | None, int | None]:
    import sounddevice as sd

    host = _wasapi_hostapi_index()
    info = sd.query_hostapis(host)
    return info.get("default_input_device"), info.get("default_output_device")


class AudioCapture:
    def __init__(self, rate: int = 16000):
        self.rate = rate
        self._source: Source | None = None
        self._streams: list = []

    def start(self, source: Source) -> None:
        import sounddevice as sd

        self.stop()
        self._source = source
        mic_dev, out_dev = _default_wasapi_devices()
        try:
            if source in ("mic", "both"):
                if mic_dev is None or mic_dev < 0:
                    raise CaptureError("microfone WASAPI indisponível")
                stream = sd.InputStream(
                    device=mic_dev,
                    channels=1,
                    samplerate=self.rate,
                    dtype="float32",
                )
                stream.start()
                self._streams.append(("mic", stream))
            if source in ("system", "both"):
                if out_dev is None or out_dev < 0:
                    raise CaptureError("dispositivo de saída WASAPI indisponível")
                extra = sd.WasapiSettings(loopback=True)
                stream = sd.InputStream(
                    device=out_dev,
                    channels=1,
                    samplerate=self.rate,
                    dtype="float32",
                    extra_settings=extra,
                )
                stream.start()
                self._streams.append(("system", stream))
        except CaptureError:
            self.stop()
            raise
        except Exception as exc:
            self.stop()
            raise CaptureError(f"falha ao abrir áudio: {exc}") from exc
        if not self._streams:
            raise CaptureError("nenhuma fonte de áudio aberta")

    def read_chunk(self, seconds: float) -> np.ndarray:
        frames = max(1, int(self.rate * seconds))
        parts: dict[str, np.ndarray] = {}
        for name, stream in self._streams:
            data, overflowed = stream.read(frames)
            _ = overflowed
            parts[name] = np.asarray(data, dtype=np.float32).reshape(-1)
        if "mic" in parts and "system" in parts:
            return mix_mono(parts["mic"], parts["system"])
        if "mic" in parts:
            return parts["mic"]
        return parts["system"]

    def stop(self) -> None:
        for _, stream in self._streams:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self._streams = []
        self._source = None

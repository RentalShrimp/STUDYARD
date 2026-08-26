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


def resample_mono(pcm: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    pcm = np.asarray(pcm, dtype=np.float32).ravel()
    if from_rate <= 0 or to_rate <= 0:
        raise ValueError("sample rate inválido")
    if from_rate == to_rate or pcm.size == 0:
        return pcm.copy()
    n_out = max(1, int(round(pcm.size * to_rate / from_rate)))
    if pcm.size == 1:
        return np.full(n_out, pcm[0], dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, pcm.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, pcm).astype(np.float32)


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


def _device_rate(device: int) -> int:
    import sounddevice as sd

    info = sd.query_devices(device)
    rate = int(round(float(info.get("default_samplerate") or 48000)))
    return rate if rate > 0 else 48000


def _wasapi_extra(*, loopback: bool = False):
    import sounddevice as sd

    try:
        return sd.WasapiSettings(exclusive=False, auto_convert=True, loopback=loopback)
    except TypeError:
        extra = sd.WasapiSettings(exclusive=False, auto_convert=True)
        if loopback:
            flag = getattr(sd._lib, "paWinWasapiLoopback", 0)
            if flag:
                extra._streaminfo.flags |= flag
        return extra


def _open_input(device: int, *, loopback: bool = False):
    import sounddevice as sd

    rate = _device_rate(device)
    stream = sd.InputStream(
        device=device,
        channels=1,
        samplerate=rate,
        dtype="float32",
        extra_settings=_wasapi_extra(loopback=loopback),
    )
    stream.start()
    native = int(round(float(getattr(stream, "samplerate", rate) or rate)))
    return stream, native


class AudioCapture:
    def __init__(self, rate: int = 16000):
        self.rate = rate
        self._source: Source | None = None
        self._streams: list[tuple[str, object, int]] = []

    def start(self, source: Source) -> None:
        self.stop()
        self._source = source
        mic_dev, out_dev = _default_wasapi_devices()
        try:
            if source in ("mic", "both"):
                if mic_dev is None or mic_dev < 0:
                    raise CaptureError("microfone WASAPI indisponível")
                stream, native = _open_input(mic_dev, loopback=False)
                self._streams.append(("mic", stream, native))
            if source in ("system", "both"):
                if out_dev is None or out_dev < 0:
                    raise CaptureError("dispositivo de saída WASAPI indisponível")
                stream, native = _open_input(out_dev, loopback=True)
                self._streams.append(("system", stream, native))
        except CaptureError:
            self.stop()
            raise
        except Exception as exc:
            self.stop()
            raise CaptureError(f"falha ao abrir áudio: {exc}") from exc
        if not self._streams:
            raise CaptureError("nenhuma fonte de áudio aberta")

    def read_chunk(self, seconds: float) -> np.ndarray:
        parts: dict[str, np.ndarray] = {}
        for name, stream, native in self._streams:
            frames = max(1, int(native * seconds))
            data, _overflowed = stream.read(frames)
            pcm = np.asarray(data, dtype=np.float32).reshape(-1)
            parts[name] = resample_mono(pcm, native, self.rate)
        if "mic" in parts and "system" in parts:
            return mix_mono(parts["mic"], parts["system"])
        if "mic" in parts:
            return parts["mic"]
        return parts["system"]

    def stop(self) -> None:
        for item in self._streams:
            stream = item[1]
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self._streams = []
        self._source = None

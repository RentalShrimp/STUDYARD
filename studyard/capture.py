from __future__ import annotations

from typing import Literal

import numpy as np

Source = Literal["mic", "system", "both"]

LOOPBACK_RATE = 48000


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


def to_mono(frames: np.ndarray) -> np.ndarray:
    data = np.asarray(frames, dtype=np.float32)
    if data.size == 0:
        return np.zeros(0, dtype=np.float32)
    if data.ndim == 1:
        return data
    if data.ndim == 2:
        if data.shape[1] <= 1:
            return data.reshape(-1)
        return data.mean(axis=1).astype(np.float32)
    raise ValueError("pcm inválido")


def channel_count_from_info(info: dict, *, loopback: bool) -> int:
    if loopback:
        n = int(info.get("max_output_channels") or 0)
        return n if n > 0 else 2
    n = int(info.get("max_input_channels") or 0)
    return n if n > 0 else 1


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


def _wasapi_extra():
    import sounddevice as sd

    return sd.WasapiSettings(exclusive=False, auto_convert=True)


def _open_input(device: int):
    import sounddevice as sd

    rate = _device_rate(device)
    info = sd.query_devices(device)
    native_ch = channel_count_from_info(info, loopback=False)
    last: Exception | None = None
    seen: set[int] = set()
    for ch in (native_ch, 2, 1):
        if ch <= 0 or ch in seen:
            continue
        seen.add(ch)
        try:
            stream = sd.InputStream(
                device=device,
                channels=ch,
                samplerate=rate,
                dtype="float32",
                extra_settings=_wasapi_extra(),
            )
            stream.start()
            native_rate = getattr(stream, "samplerate", rate) or rate
            native = int(round(float(native_rate)))
            return stream, native
        except Exception as exc:
            last = exc
    raise last if last else CaptureError("falha ao abrir áudio")


class _LoopbackReader:
    def __init__(self, recorder_cm, recorder):
        self._cm = recorder_cm
        self._rec = recorder

    def read(self, frames: int):
        data = self._rec.record(numframes=max(1, int(frames)))
        return data, False

    def stop(self) -> None:
        return None

    def close(self) -> None:
        try:
            self._cm.__exit__(None, None, None)
        except Exception:
            pass


def _open_loopback():
    try:
        import soundcard as sc
    except ImportError as exc:
        raise CaptureError(
            "áudio do sistema indisponível — instale o pacote soundcard"
        ) from exc
    speaker = sc.default_speaker()
    if speaker is None:
        raise CaptureError("dispositivo de saída WASAPI indisponível")
    try:
        mic = sc.get_microphone(id=speaker.id, include_loopback=True)
    except Exception as exc:
        raise CaptureError("áudio do sistema (loopback) indisponível") from exc
    if not getattr(mic, "isloopback", False):
        raise CaptureError("áudio do sistema (loopback) indisponível")
    nch = int(getattr(mic, "channels", 0) or 0)
    if nch < 2:
        nch = 2
    rec_cm = mic.recorder(samplerate=LOOPBACK_RATE, channels=nch)
    recorder = rec_cm.__enter__()
    return _LoopbackReader(rec_cm, recorder), LOOPBACK_RATE


class AudioCapture:
    def __init__(self, rate: int = 16000):
        self.rate = rate
        self._source: Source | None = None
        self._streams: list[tuple[str, object, int]] = []

    def start(self, source: Source) -> None:
        self.stop()
        self._source = source
        mic_dev, _out_dev = _default_wasapi_devices()
        try:
            if source in ("mic", "both"):
                if mic_dev is None or mic_dev < 0:
                    raise CaptureError("microfone WASAPI indisponível")
                stream, native = _open_input(mic_dev)
                self._streams.append(("mic", stream, native))
            if source in ("system", "both"):
                stream, native = _open_loopback()
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
            pcm = to_mono(data)
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

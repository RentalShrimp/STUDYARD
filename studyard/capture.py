from __future__ import annotations

import threading
import time
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


def sum_mono(parts: list[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(p, dtype=np.float32).ravel() for p in parts]
    arrays = [a for a in arrays if a.size]
    if not arrays:
        return np.zeros(0, dtype=np.float32)
    n = min(a.size for a in arrays)
    out = np.zeros(n, dtype=np.float32)
    for a in arrays:
        out += a[:n]
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _rms(pcm: np.ndarray) -> float:
    data = np.asarray(pcm, dtype=np.float32).ravel()
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(data * data)))


def combine_capture_parts(parts: dict[str, np.ndarray]) -> np.ndarray:
    mic = [v for k, v in parts.items() if str(k).startswith("mic")]
    sys = [v for k, v in parts.items() if str(k).startswith("system")]
    mic_m = sum_mono(mic) if mic else None
    sys_m = sum_mono(sys) if sys else None
    if mic_m is not None and sys_m is not None:
        if _rms(mic_m) < 1e-4:
            return sys_m
        if _rms(sys_m) < 1e-4:
            return mic_m
        return mix_mono(mic_m, sys_m)
    if mic_m is not None:
        return mic_m
    if sys_m is not None:
        return sys_m
    return np.zeros(0, dtype=np.float32)


def normalize_peak(
    pcm: np.ndarray, target: float = 0.4, max_gain: float = 20.0
) -> np.ndarray:
    data = np.asarray(pcm, dtype=np.float32).ravel()
    if data.size == 0:
        return data.copy()
    peak = float(np.max(np.abs(data)))
    if peak < 1e-6:
        return data.copy()
    gain = min(float(max_gain), float(target) / peak)
    if gain <= 1.0:
        return data.copy()
    return np.clip(data * gain, -1.0, 1.0).astype(np.float32)


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


def is_mic_array(name: str) -> bool:
    n = str(name).casefold()
    return "mic array" in n or "grupo de microfones" in n


def is_analog_jack(name: str) -> bool:
    n = str(name).casefold()
    if is_mic_array(n):
        return False
    return "mic input" in n


def _host_kind(hostapi_name: str) -> str:
    u = str(hostapi_name).upper()
    if "WASAPI" in u:
        return "wasapi"
    if "WDM-KS" in u or "WDMKS" in u:
        return "wdm-ks"
    return "other"


def select_mic_device_indices(devices: list[dict]) -> list[int]:
    wasapi: list[int] = []
    analog: list[int] = []
    array: list[int] = []
    for d in devices:
        if int(d.get("max_input_channels") or 0) <= 0:
            continue
        kind = _host_kind(str(d.get("hostapi_name") or ""))
        name = str(d.get("name") or "")
        idx = int(d["index"])
        if kind == "wasapi":
            if is_mic_array(name):
                continue
            wasapi.append(idx)
        elif kind == "wdm-ks":
            if is_analog_jack(name):
                analog.append(idx)
            elif is_mic_array(name):
                array.append(idx)
    return wasapi + analog + array


def _wanted(name: str, wanted: str | None) -> bool:
    if not wanted or wanted.strip().casefold() in ("all", "*", "todos"):
        return True
    return wanted.casefold() in str(name).casefold()


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


def _enumerate_inputs() -> list[dict]:
    import sounddevice as sd

    hostapis = list(sd.query_hostapis())
    out: list[dict] = []
    for i, d in enumerate(sd.query_devices()):
        host_i = int(d.get("hostapi") or 0)
        host_name = ""
        if host_i < len(hostapis):
            host_name = str(hostapis[host_i].get("name") or "")
        out.append(
            {
                "index": i,
                "name": str(d.get("name") or ""),
                "hostapi_name": host_name,
                "max_input_channels": int(d.get("max_input_channels") or 0),
            }
        )
    return out


def list_capture_devices() -> dict:
    mics = []
    devices = _enumerate_inputs()
    by_idx = {d["index"]: d for d in devices}
    for idx in select_mic_device_indices(devices):
        d = by_idx[idx]
        if is_analog_jack(d["name"]):
            kind = "analog"
        elif is_mic_array(d["name"]):
            kind = "array"
        else:
            kind = "mic"
        mics.append(
            {
                "name": d["name"],
                "host": d["hostapi_name"],
                "kind": kind,
                "index": idx,
            }
        )
    loopbacks = []
    try:
        import soundcard as sc

        for mic in sc.all_microphones(include_loopback=True):
            if getattr(mic, "isloopback", False):
                loopbacks.append({"name": mic.name, "kind": "loopback"})
    except Exception:
        pass
    return {"mics": mics, "loopbacks": loopbacks}


def _mic_open_specs(wanted: str | None) -> list[dict]:
    devices = _enumerate_inputs()
    by_idx = {d["index"]: d for d in devices}
    picked = [
        by_idx[i] for i in select_mic_device_indices(devices) if i in by_idx
    ]
    picked = [d for d in picked if _wanted(d["name"], wanted)]
    if picked:
        return picked
    if wanted and wanted.strip().casefold() not in ("all", "*", "todos"):
        for d in devices:
            if int(d["max_input_channels"]) > 0 and _wanted(d["name"], wanted):
                return [d]
    return picked


class _PcmRing:
    def __init__(self, max_samples: int = 48000 * 5):
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._buf = np.zeros(0, dtype=np.float32)
        self._max = max_samples

    def push(self, pcm: np.ndarray) -> None:
        chunk = np.asarray(pcm, dtype=np.float32).ravel()
        if chunk.size == 0:
            return
        with self._cv:
            self._buf = np.concatenate([self._buf, chunk])
            if self._buf.size > self._max:
                self._buf = self._buf[-self._max:]
            self._cv.notify_all()

    def read(self, frames: int, timeout: float) -> np.ndarray:
        frames = max(1, int(frames))
        deadline = time.monotonic() + max(0.05, float(timeout))
        with self._cv:
            while self._buf.size < frames:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cv.wait(timeout=remaining)
            n = min(frames, int(self._buf.size))
            out = self._buf[:n].copy()
            self._buf = self._buf[n:]
        if out.size < frames:
            out = np.concatenate(
                [out, np.zeros(frames - out.size, dtype=np.float32)]
            )
        return out.astype(np.float32)


class _BufferedInput:
    def __init__(self, stream, ring: _PcmRing, rate: int):
        self._stream = stream
        self._ring = ring
        self.rate = rate

    def read(self, frames: int):
        timeout = (frames / max(1, int(self.rate))) + 0.4
        return self._ring.read(frames, timeout), False

    def stop(self) -> None:
        try:
            self._stream.stop()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._stream.close()
        except Exception:
            pass


def _open_input_buffered(device: int):
    import sounddevice as sd

    rate = _device_rate(device)
    info = sd.query_devices(device)
    host = str(sd.query_hostapis(int(info["hostapi"])).get("name") or "")
    native_ch = channel_count_from_info(info, loopback=False)
    extra = _wasapi_extra() if "WASAPI" in host.upper() else None
    last: Exception | None = None
    seen: set[int] = set()
    for ch in (native_ch, 2, 1):
        if ch <= 0 or ch in seen:
            continue
        seen.add(ch)
        ring = _PcmRing()

        def cb(indata, frames, time_info, status, _ring=ring):
            _ring.push(to_mono(indata))

        try:
            kw = {
                "device": device,
                "channels": ch,
                "samplerate": rate,
                "dtype": "float32",
                "callback": cb,
                "blocksize": 512,
            }
            if extra is not None:
                kw["extra_settings"] = extra
            stream = sd.InputStream(**kw)
            stream.start()
            native_rate = getattr(stream, "samplerate", rate) or rate
            native = int(round(float(native_rate)))
            return _BufferedInput(stream, ring, native), native
        except Exception as exc:
            last = exc
    raise last if last else CaptureError("falha ao abrir áudio")


class _LoopbackReader:
    def __init__(self, recorder_cm, recorder, rate: int):
        self._cm = recorder_cm
        self._rec = recorder
        self._ring = _PcmRing()
        self._alive = True
        self.rate = rate
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        while self._alive:
            try:
                data = self._rec.record(numframes=1024)
                self._ring.push(to_mono(data))
            except Exception:
                break

    def read(self, frames: int):
        timeout = (frames / max(1, int(self.rate))) + 0.4
        return self._ring.read(frames, timeout), False

    def stop(self) -> None:
        self._alive = False

    def close(self) -> None:
        self._alive = False
        try:
            self._cm.__exit__(None, None, None)
        except Exception:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


def _open_loopbacks(wanted: str | None):
    try:
        import soundcard as sc
    except ImportError as exc:
        raise CaptureError(
            "áudio do sistema indisponível — instale o pacote soundcard"
        ) from exc
    mics = [
        m
        for m in sc.all_microphones(include_loopback=True)
        if getattr(m, "isloopback", False)
    ]
    if wanted and wanted.strip().casefold() not in ("all", "*", "todos"):
        mics = [m for m in mics if _wanted(m.name, wanted)]
    if not mics:
        speaker = sc.default_speaker()
        if speaker is None:
            raise CaptureError("dispositivo de saída WASAPI indisponível")
        try:
            mic = sc.get_microphone(id=speaker.id, include_loopback=True)
        except Exception as exc:
            raise CaptureError(
                "áudio do sistema (loopback) indisponível"
            ) from exc
        if not getattr(mic, "isloopback", False):
            raise CaptureError("áudio do sistema (loopback) indisponível")
        mics = [mic]
    opened = []
    errors: list[str] = []
    for mic in mics:
        nch = int(getattr(mic, "channels", 0) or 0)
        if nch < 2:
            nch = 2
        try:
            rec_cm = mic.recorder(samplerate=LOOPBACK_RATE, channels=nch)
            recorder = rec_cm.__enter__()
            reader = _LoopbackReader(rec_cm, recorder, LOOPBACK_RATE)
            opened.append((reader, LOOPBACK_RATE, mic.name))
        except Exception as exc:
            errors.append(str(exc))
    if not opened:
        raise CaptureError(
            errors[0] if errors else "áudio do sistema (loopback) indisponível"
        )
    return opened


class AudioCapture:
    def __init__(self, rate: int = 16000):
        self.rate = rate
        self._source: Source | None = None
        self._streams: list[tuple[str, object, int]] = []
        self.labels: list[str] = []

    def start(
        self,
        source: Source,
        mic_device: str | None = None,
        system_device: str | None = None,
    ) -> None:
        self.stop()
        self._source = source
        errors: list[str] = []
        try:
            if source in ("mic", "both"):
                specs = _mic_open_specs(mic_device)
                if not specs:
                    mic_dev, _out_dev = _default_wasapi_devices()
                    if mic_dev is None or mic_dev < 0:
                        raise CaptureError("microfone WASAPI indisponível")
                    specs = [
                        {"index": int(mic_dev), "name": "microfone WASAPI"}
                    ]
                for spec in specs:
                    try:
                        stream, native = _open_input_buffered(int(spec["index"]))
                        self._streams.append(
                            (f"mic:{spec['index']}", stream, native)
                        )
                        self.labels.append(str(spec["name"]))
                    except Exception as exc:
                        errors.append(f"{spec['name']}: {exc}")
            if source in ("system", "both"):
                try:
                    for reader, native, label in _open_loopbacks(
                        system_device
                    ):
                        self._streams.append(
                            (f"system:{label}", reader, native)
                        )
                        self.labels.append(f"sistema:{label}")
                except CaptureError as exc:
                    errors.append(str(exc))
                except Exception as exc:
                    errors.append(str(exc))
        except CaptureError:
            self.stop()
            raise
        except Exception as exc:
            self.stop()
            raise CaptureError(f"falha ao abrir áudio: {exc}") from exc
        if not self._streams:
            self.stop()
            raise CaptureError(
                errors[0] if errors else "nenhuma fonte de áudio aberta"
            )

    def read_chunk(self, seconds: float) -> np.ndarray:
        parts: dict[str, np.ndarray] = {}
        for name, stream, native in self._streams:
            frames = max(1, int(native * seconds))
            data, _overflowed = stream.read(frames)
            pcm = to_mono(data)
            parts[name] = resample_mono(pcm, native, self.rate)
        mixed = combine_capture_parts(parts)
        return normalize_peak(mixed)

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
        self.labels = []

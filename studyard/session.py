from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from studyard.naming import resumo_stem
from studyard.wavutil import write_wav

HEADER_END = "## Transcrição\n\n"


def fail_marker(elapsed_seconds: int) -> str:
    mm, ss = divmod(int(elapsed_seconds), 60)
    return f"[trecho ~{mm:02d}:{ss:02d} não transcrito]"


def resumo_path_for(folder: Path, aula_stem: str) -> Path:
    return folder / f"{resumo_stem(aula_stem)}.md"


class SessionFiles:
    def __init__(self, folder: Path, stem: str, source: str, save_audio: bool):
        self.folder = folder
        self.stem = stem
        self.source = source
        self.save_audio = save_audio
        self._pcm = np.zeros(0, dtype=np.float32)

    @property
    def aula_path(self) -> Path:
        return self.folder / f"{self.stem}.md"

    @property
    def resumo_path(self) -> Path:
        return resumo_path_for(self.folder, self.stem)

    @property
    def wav_path(self) -> Path:
        return self.folder / f"{self.stem}.wav"

    @property
    def pending_path(self) -> Path:
        return self.folder / f"{self.stem}.pending.json"

    def create(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        day = self.stem[:10]
        header = (
            f"# Aula\n\n"
            f"- Data: {day}\n"
            f"- Fonte: {self.source}\n\n"
            f"{HEADER_END}"
        )
        self.aula_path.write_text(header, encoding="utf-8")

    def append_transcript(self, text: str) -> None:
        chunk = text if text.endswith("\n") else text + "\n"
        with self.aula_path.open("a", encoding="utf-8") as fh:
            fh.write(chunk)

    def replace_body(self, text: str) -> None:
        raw = self.aula_path.read_text(encoding="utf-8")
        idx = raw.find(HEADER_END)
        if idx < 0:
            raise ValueError("cabeçalho da aula ausente")
        prefix = raw[: idx + len(HEADER_END)]
        body = text if text.endswith("\n") else text + "\n"
        self.aula_path.write_text(prefix + body, encoding="utf-8")

    def read_transcript_body(self) -> str:
        raw = self.aula_path.read_text(encoding="utf-8")
        idx = raw.find(HEADER_END)
        if idx < 0:
            return raw
        return raw[idx + len(HEADER_END) :]

    def write_summary(self, text: str) -> None:
        body = text if text.endswith("\n") else text + "\n"
        self.resumo_path.write_text(body, encoding="utf-8")

    def write_pending(self, need: list[str]) -> None:
        payload = {"need": need, "save_audio": self.save_audio}
        self.pending_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def clear_pending(self) -> None:
        if self.pending_path.exists():
            self.pending_path.unlink()

    def append_pcm(self, pcm: np.ndarray) -> None:
        if pcm.size == 0:
            return
        self._pcm = np.concatenate([self._pcm, np.asarray(pcm, dtype=np.float32).ravel()])

    def flush_wav(self) -> None:
        write_wav(self.wav_path, self._pcm)

    def wav_bytes(self) -> bytes:
        from studyard.wavutil import pcm_to_wav_bytes

        return pcm_to_wav_bytes(self._pcm)

    def finalize_audio(self, success: bool) -> None:
        if success and not self.save_audio:
            if self.wav_path.exists():
                self.wav_path.unlink()
            return
        if self._pcm.size:
            self.flush_wav()

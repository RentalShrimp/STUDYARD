from __future__ import annotations

import time

import numpy as np

from studyard.api_client import ApiError
from studyard.config import Config
from studyard.session import SessionFiles, fail_marker
from studyard.wavutil import RATE, pcm_to_wav_bytes


class Recorder:
    def __init__(
        self,
        cfg: Config,
        session: SessionFiles,
        api,
        clock=time.monotonic,
    ):
        self.cfg = cfg
        self.session = session
        self.api = api
        self.clock = clock
        self.had_chunk_failure = False
        self.status = "recording"
        self.last_text = ""
        self.offline = False
        self.message = ""
        self._buf = np.zeros(0, dtype=np.float32)
        self._started = clock()
        self._chunk_samples = int(cfg.chunk_seconds * RATE)

    def _elapsed_s(self) -> int:
        return int(self.clock() - self._started)

    def feed_pcm(self, pcm: np.ndarray) -> None:
        if pcm.size == 0:
            return
        chunk = np.asarray(pcm, dtype=np.float32).ravel()
        self._buf = np.concatenate([self._buf, chunk])
        self.session.append_pcm(chunk)
        while len(self._buf) >= self._chunk_samples:
            piece = self._buf[: self._chunk_samples]
            self._buf = self._buf[self._chunk_samples :]
            self.process_chunk(piece, self._elapsed_s(), already_appended=True)

    def process_chunk(
        self,
        pcm: np.ndarray,
        elapsed_s: int,
        already_appended: bool = False,
    ) -> None:
        if not already_appended:
            self.session.append_pcm(pcm)
        self.status = "transcribing"
        wav_bytes = pcm_to_wav_bytes(pcm)
        last_err: ApiError | None = None
        for attempt in range(3):
            try:
                text = self.api.transcribe(wav_bytes)
                self.last_text = text
                self.session.append_transcript(text)
                self.offline = False
                self.status = "recording"
                return
            except ApiError as exc:
                last_err = exc
        self.had_chunk_failure = True
        self.offline = True
        self.status = "offline"
        self.message = str(last_err) if last_err else "API indisponível"
        self.session.append_transcript(fail_marker(elapsed_s))

    def stop(self) -> dict:
        if self._buf.size:
            leftover = self._buf
            self._buf = np.zeros(0, dtype=np.float32)
            self.process_chunk(leftover, self._elapsed_s(), already_appended=True)
        self.session.flush_wav()

        if self.had_chunk_failure:
            return self._recover_or_pending()
        return self._summarize_or_pending(need_transcribe=False)

    def _recover_or_pending(self) -> dict:
        try:
            self.status = "transcribing"
            text = self.api.transcribe(self.session.wav_bytes())
            self.session.replace_body(text)
            self.last_text = text
            self.had_chunk_failure = False
            return self._summarize_or_pending(need_transcribe=False)
        except ApiError as exc:
            self.offline = True
            self.status = "offline"
            self.message = str(exc)
            self.session.write_pending(["transcribe", "summarize"])
            self.session.finalize_audio(success=False)
            return self._paths()

    def _summarize_or_pending(self, need_transcribe: bool) -> dict:
        try:
            self.status = "summarizing"
            summary = self.api.summarize(self.session.read_transcript_body())
            self.session.write_summary(summary)
            self.session.clear_pending()
            self.session.finalize_audio(success=True)
            self.status = "saved"
            self.offline = False
            self.message = f"salvo em {self.session.folder}"
            return self._paths()
        except ApiError as exc:
            self.offline = True
            self.status = "error"
            self.message = str(exc)
            need = ["transcribe", "summarize"] if need_transcribe else ["summarize"]
            self.session.write_pending(need)
            self.session.finalize_audio(success=False)
            return self._paths()

    def _paths(self) -> dict:
        return {
            "folder": str(self.session.folder),
            "aula": str(self.session.aula_path),
            "resumo": str(self.session.resumo_path),
            "wav": str(self.session.wav_path),
        }

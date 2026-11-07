from __future__ import annotations

import time

import numpy as np

from studyard.api_client import ApiError, SummaryCancelled
from studyard.config import Config
from studyard.session import SessionFiles, fail_marker
from studyard.summarizer import (
    DEFERRED_MESSAGE,
    checkpoint_path,
    clear_checkpoint,
    summarize_text,
)
from studyard.wavutil import RATE
from studyard.whisper_service import TranscribeError, transcribe_float32


# sem fala transcrita por esse tempo, a captura pode estar muda
SILENCE_ALERT_S = 120


class Recorder:
    def __init__(
        self,
        cfg: Config,
        session: SessionFiles,
        api,
        clock=time.monotonic,
        asr=None,
    ):
        self.cfg = cfg
        self.session = session
        self.api = api
        self.asr = asr
        self.clock = clock
        self.had_chunk_failure = False
        self.failed_chunks = 0
        self.status = "recording"
        self.stopping = False
        self.last_text = ""
        self.offline = False
        self.message = ""
        self._buf = np.zeros(0, dtype=np.float32)
        self._started = clock()
        self._last_text_at = self._started
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

    def _transcribe(self, pcm: np.ndarray) -> str:
        if self.asr is not None:
            return self.asr(pcm)
        return transcribe_float32(
            pcm,
            language=self.cfg.language,
            model_size=self.cfg.whisper_model,
        )

    def process_chunk(
        self,
        pcm: np.ndarray,
        elapsed_s: int,
        already_appended: bool = False,
    ) -> None:
        if not already_appended:
            self.session.append_pcm(pcm)
        self.status = "transcribing"
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                text = self._transcribe(pcm)
                if text:
                    self.last_text = text
                    self._last_text_at = self.clock()
                    self.session.append_transcript(text)
                self.offline = False
                self.status = "stopping" if self.stopping else "recording"
                return
            except (ApiError, TranscribeError) as exc:
                last_err = exc
        self.had_chunk_failure = True
        self.failed_chunks += 1
        self.offline = True
        self.status = "offline"
        self.message = str(last_err) if last_err else "transcrição indisponível"
        self.session.append_transcript(fail_marker(elapsed_s))

    def alerts(self) -> list[dict]:
        if self.stopping:
            return []
        out: list[dict] = []
        if self.failed_chunks:
            out.append(
                {
                    "key": "transcription",
                    "text": (
                        f"{self.failed_chunks} trecho(s) não transcrito(s): {self.message}. "
                        "O áudio continua guardado e será transcrito de novo ao parar."
                    ),
                }
            )
        silent_s = self.clock() - self._last_text_at
        if silent_s >= SILENCE_ALERT_S:
            out.append(
                {
                    "key": "silence",
                    "text": (
                        f"Nenhuma fala transcrita há {int(silent_s // 60)} min. "
                        "Confira a fonte de áudio (ou é só silêncio na aula)."
                    ),
                }
            )
        return out

    def stop(self) -> dict:
        """Caminho síncrono: áudio, pending e resumo. A API resume em outra thread."""
        if not self.finish_audio():
            return self._paths()
        try:
            return self.summarize_pending()
        except SummaryCancelled:
            return self._paths()

    def finish_audio(self) -> bool:
        """Último trecho, WAV e .pending.json. True se o resumo pode começar."""
        self.stopping = True
        self.status = "stopping"
        if self._buf.size:
            leftover = self._buf
            self._buf = np.zeros(0, dtype=np.float32)
            self.process_chunk(leftover, self._elapsed_s(), already_appended=True)
        self.session.flush_wav()
        if self.had_chunk_failure:
            try:
                self.status = "transcribing"
                text = self._transcribe(self.session._pcm)
                self.session.replace_body(text)
                self.last_text = text
                self.had_chunk_failure = False
            except (ApiError, TranscribeError) as exc:
                self.offline = True
                self.status = "offline"
                self.message = str(exc)
                self.session.write_pending(["transcribe", "summarize"])
                self.session.finalize_audio(success=False)
                return False
        self.status = "summarizing"
        self.message = "Gerando resumo — não feche o app nem desligue o PC até terminar."
        # se o app fechar durante o resumo, "Processar pendentes" retoma
        self.session.write_pending(["summarize"])
        return True

    def summarize_pending(self, cancel=None, on_progress=None) -> dict:
        checkpoint = checkpoint_path(self.session.folder, self.session.stem)
        try:
            summary = summarize_text(
                self.api,
                self.session.read_transcript_body(),
                checkpoint=checkpoint,
                cancel=cancel,
                on_progress=on_progress,
            )
            self.session.write_summary(summary)
            self.session.clear_pending()
            clear_checkpoint(checkpoint)
            self.session.finalize_audio(success=True)
            self.status = "saved"
            self.offline = False
            self.message = f"salvo em {self.session.folder}"
            return self._paths()
        except SummaryCancelled:
            self.status = "deferred"
            self.message = DEFERRED_MESSAGE
            raise
        except ApiError as exc:
            self.offline = True
            self.status = "error"
            self.message = str(exc)
            self.session.finalize_audio(success=False)
            return self._paths()

    def _paths(self) -> dict:
        return {
            "folder": str(self.session.folder),
            "aula": str(self.session.aula_path),
            "resumo": str(self.session.resumo_path),
            "wav": str(self.session.wav_path),
        }

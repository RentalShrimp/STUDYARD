from __future__ import annotations

import httpx

from studyard.config import Config

SYSTEM_PROMPT = (
    "Você resume aulas de mestrado em Markdown em português. "
    "Extraia tópicos, definições, exemplos e ênfases do professor. "
    "Não invente conteúdo que não esteja na transcrição. "
    "Ignore linhas do tipo [trecho ~mm:ss não transcrito]."
)


class ApiError(Exception):
    pass


class ApiClient:
    def __init__(self, cfg: Config, http: httpx.Client | None = None):
        self.cfg = cfg
        self._owns = http is None
        self.http = http or httpx.Client(timeout=httpx.Timeout(180.0))

    def close(self) -> None:
        if self._owns:
            self.http.close()

    def transcribe(self, wav_bytes: bytes) -> str:
        url = f"{self.cfg.api_base_url}/v1/audio/transcriptions"
        files = {"file": ("chunk.wav", wav_bytes, "audio/wav")}
        data = {
            "model": self.cfg.transcription_model,
            "language": self.cfg.language,
        }
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        try:
            resp = self.http.post(url, files=files, data=data, headers=headers, timeout=120.0)
        except httpx.HTTPError as exc:
            raise ApiError(str(exc)) from exc
        if resp.status_code >= 400:
            raise ApiError(f"transcrição HTTP {resp.status_code}: {resp.text[:300]}")
        payload = resp.json()
        text = payload.get("text")
        if not isinstance(text, str):
            raise ApiError("resposta de transcrição sem campo text")
        return text

    def summarize(self, transcript: str) -> str:
        url = f"{self.cfg.api_base_url}/v1/chat/completions"
        body = {
            "model": self.cfg.summary_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = self.http.post(url, json=body, headers=headers, timeout=180.0)
        except httpx.HTTPError as exc:
            raise ApiError(str(exc)) from exc
        if resp.status_code >= 400:
            raise ApiError(f"resumo HTTP {resp.status_code}: {resp.text[:300]}")
        payload = resp.json()
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ApiError("resposta de resumo inválida") from exc
        if not isinstance(content, str):
            raise ApiError("resposta de resumo sem texto")
        return content

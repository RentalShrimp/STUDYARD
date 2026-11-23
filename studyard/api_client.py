from __future__ import annotations

import json
import math
import threading

import httpx

from studyard.config import Config

SYSTEM_PROMPT = (
    "Você resume aulas de mestrado em Markdown em português. "
    "Extraia tópicos, definições, exemplos e ênfases do professor. "
    "Não invente conteúdo que não esteja na transcrição. "
    "Ignore linhas do tipo [trecho ~mm:ss não transcrito]."
)

# leitura longa entre tokens; o total deixa de ser um teto de 1800 s
STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=900.0, write=60.0, pool=10.0)


class ApiError(Exception):
    pass


class SummaryCancelled(Exception):
    """O usuário adiou o resumo; as partes já gravadas ficam no checkpoint."""


def summary_num_ctx_for(text: str, ceiling: int) -> int:
    """Contexto do Ollama pelo tamanho do texto.

    `ceiling` (cfg.summary_num_ctx) é o teto, não o valor fixo. O resultado
    sobe em degraus de 4096 e fica entre 8192 e esse teto.
    """
    if ceiling <= 0:
        return 0
    estimate = math.ceil(len(text) / 3) + 1024 + 2048
    rounded = math.ceil(estimate / 4096) * 4096
    if ceiling < 8192:
        return min(rounded, ceiling)
    return min(max(rounded, 8192), ceiling)


def openai_url(base_url: str, resource: str) -> str:
    """Join an OpenAI-compatible base with a resource (e.g. chat/completions).

    `api_base_url` may already end with `/v1` (documented convention) or not.
    """
    base = base_url.rstrip("/")
    resource = resource.lstrip("/")
    if resource.startswith("v1/"):
        resource = resource[3:]
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}/{resource}"


def ollama_url(base_url: str, resource: str) -> str:
    """Native Ollama endpoint (e.g. api/chat) from a base that may end with `/v1`."""
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return f"{base}/{resource.lstrip('/')}"


class ApiClient:
    def __init__(self, cfg: Config, http: httpx.Client | None = None):
        self.cfg = cfg
        self._owns = http is None
        self.http = http or httpx.Client(timeout=httpx.Timeout(180.0))

    def close(self) -> None:
        if self._owns:
            self.http.close()

    def transcribe(self, wav_bytes: bytes) -> str:
        url = openai_url(self.cfg.api_base_url, "audio/transcriptions")
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

    def check_summary(self) -> str | None:
        """Confere se o resumo vai funcionar; devolve o problema ou None."""
        native = self.cfg.summary_num_ctx > 0
        if native:
            url = ollama_url(self.cfg.api_base_url, "api/tags")
        else:
            url = openai_url(self.cfg.api_base_url, "models")
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        try:
            resp = self.http.get(url, headers=headers, timeout=5.0)
        except httpx.HTTPError:
            return f"servidor de resumo fora do ar em {self.cfg.api_base_url} — abra o Ollama"
        if resp.status_code >= 400:
            return f"servidor de resumo respondeu HTTP {resp.status_code}"
        if native:
            model = self.cfg.summary_model
            try:
                names = {m.get("name") for m in resp.json()["models"]}
            except (ValueError, KeyError, TypeError, AttributeError):
                return None
            if model not in names and f"{model}:latest" not in names:
                return f"modelo {model} não está instalado no Ollama"
        return None

    def summarize(
        self,
        transcript: str,
        num_ctx: int | None = None,
        on_progress=None,
        cancel: threading.Event | None = None,
    ) -> str:
        if cancel is not None and cancel.is_set():
            raise SummaryCancelled()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]
        if num_ctx is None:
            native = self.cfg.summary_num_ctx > 0
            ctx = summary_num_ctx_for(transcript, self.cfg.summary_num_ctx) if native else 0
        else:
            native = num_ctx > 0
            ctx = num_ctx
        if native:
            return self._summarize_native(messages, ctx, on_progress, cancel)
        return self._summarize_openai(messages, cancel)

    def _summarize_openai(self, messages: list, cancel: threading.Event | None) -> str:
        url = openai_url(self.cfg.api_base_url, "chat/completions")
        body = {"model": self.cfg.summary_model, "messages": messages}
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = self.http.post(url, json=body, headers=headers, timeout=1800.0)
        except httpx.ConnectError as exc:
            raise ApiError(
                f"servidor de resumo fora do ar em {self.cfg.api_base_url} — "
                "abra o Ollama e use 'processar pendentes'"
            ) from exc
        except httpx.HTTPError as exc:
            if cancel is not None and cancel.is_set():
                raise SummaryCancelled() from exc
            raise ApiError(str(exc)) from exc
        return _content_from_payload(resp, native=False)

    def _summarize_native(self, messages, num_ctx: int, on_progress, cancel) -> str:
        url = ollama_url(self.cfg.api_base_url, "api/chat")
        body = {
            "model": self.cfg.summary_model,
            "messages": messages,
            "stream": True,
            "options": {"num_ctx": num_ctx},
        }
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        try:
            with self.http.stream(
                "POST", url, json=body, headers=headers, timeout=STREAM_TIMEOUT
            ) as resp:
                if resp.status_code >= 400:
                    raw = resp.read().decode("utf-8", errors="replace")
                    raise ApiError(f"resumo HTTP {resp.status_code}: {raw[:300]}")
                return _read_ndjson(resp, on_progress, cancel)
        except SummaryCancelled:
            raise
        except ApiError:
            raise
        except httpx.ConnectError as exc:
            raise ApiError(
                f"servidor de resumo fora do ar em {self.cfg.api_base_url} — "
                "abra o Ollama e use 'processar pendentes'"
            ) from exc
        except (httpx.HTTPError, httpx.StreamError) as exc:
            if cancel is not None and cancel.is_set():
                raise SummaryCancelled() from exc
            raise ApiError(str(exc)) from exc


def _content_from_payload(resp: httpx.Response, native: bool) -> str:
    if resp.status_code >= 400:
        raise ApiError(f"resumo HTTP {resp.status_code}: {resp.text[:300]}")
    try:
        payload = resp.json()
        if native:
            content = payload["message"]["content"]
        else:
            content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ApiError("resposta de resumo inválida") from exc
    if not isinstance(content, str):
        raise ApiError("resposta de resumo sem texto")
    return content


def _read_ndjson(resp: httpx.Response, on_progress, cancel: threading.Event | None) -> str:
    """Junta o NDJSON do Ollama e avisa a cada token. Fechar a resposta cancela."""
    chunks: list[str] = []
    tokens = 0
    finished = threading.Event()

    def watch() -> None:
        while not finished.is_set():
            if cancel is not None and cancel.is_set():
                try:
                    resp.close()
                except Exception:
                    pass
                return
            finished.wait(0.2)

    if cancel is not None:
        threading.Thread(target=watch, daemon=True).start()
    try:
        for line in resp.iter_lines():
            if cancel is not None and cancel.is_set():
                try:
                    resp.close()
                except Exception:
                    pass
                raise SummaryCancelled()
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            line = (line or "").strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ApiError("resposta de resumo inválida") from exc
            if not isinstance(payload, dict):
                raise ApiError("resposta de resumo inválida")
            if payload.get("error"):
                raise ApiError(str(payload["error"])[:300])
            message = payload.get("message") or {}
            content = message.get("content") if isinstance(message, dict) else ""
            if content:
                chunks.append(content)
                tokens += 1
                if on_progress is not None:
                    on_progress(tokens)
            if payload.get("done") and isinstance(payload.get("eval_count"), int):
                tokens = payload["eval_count"]
                if on_progress is not None:
                    on_progress(tokens)
    finally:
        finished.set()
    text = "".join(chunks)
    if not text.strip():
        raise ApiError("resposta de resumo sem texto")
    return text

from pathlib import Path

import httpx

from studyard.api_client import ApiClient, ApiError, openai_url
from studyard.config import Config


def _cfg(**kw):
    base = dict(
        output_dir=Path("."),
        api_base_url="https://example.invalid/v1",
        api_key="k",
        transcription_model="whisper-1",
        summary_model="gpt-4o-mini",
        language="pt",
        chunk_seconds=25,
        save_audio=False,
        port=8765,
    )
    base.update(kw)
    return Config(**base)


def test_openai_url_does_not_double_v1():
    assert openai_url("http://127.0.0.1:11434/v1", "chat/completions") == (
        "http://127.0.0.1:11434/v1/chat/completions"
    )
    assert openai_url("http://127.0.0.1:11434", "chat/completions") == (
        "http://127.0.0.1:11434/v1/chat/completions"
    )
    assert openai_url("https://api.openai.com/v1/", "v1/chat/completions") == (
        "https://api.openai.com/v1/chat/completions"
    )


def test_transcribe_posts_multipart():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer k"
        return httpx.Response(200, json={"text": "olá aula"})

    client = ApiClient(_cfg(), http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert client.transcribe(b"RIFF") == "olá aula"


def test_summarize_sends_transcript():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "# Resumo\n\nok"}}]},
        )

    client = ApiClient(_cfg(), http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert "ok" in client.summarize("corpo da aula")


def test_http_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="nope")

    client = ApiClient(_cfg(), http=httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        client.transcribe(b"x")
        raise AssertionError("expected ApiError")
    except ApiError:
        pass

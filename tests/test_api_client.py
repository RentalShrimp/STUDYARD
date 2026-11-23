import json
import threading
from pathlib import Path

import httpx

from studyard.api_client import (
    STREAM_TIMEOUT,
    ApiClient,
    ApiError,
    SummaryCancelled,
    openai_url,
    summary_num_ctx_for,
)
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
        body = json.loads(request.content)
        assert "options" not in body
        assert body.get("stream") is not True
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


def _ndjson(*objs) -> str:
    return "\n".join(json.dumps(obj) for obj in objs)


def test_summarize_with_num_ctx_streams_and_sizes_context():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["options"] == {"num_ctx": 8192}
        return httpx.Response(
            200,
            text=_ndjson(
                {"message": {"role": "assistant", "content": "# Resumo"}, "done": False},
                {"message": {"role": "assistant", "content": "\n\nok"}, "done": False},
                {"done": True, "eval_count": 2},
            ),
        )

    cfg = _cfg(api_base_url="http://127.0.0.1:11434/v1", summary_num_ctx=32768)
    client = ApiClient(cfg, http=httpx.Client(transport=httpx.MockTransport(handler)))
    text = client.summarize("corpo da aula", on_progress=seen.append)
    assert text == "# Resumo\n\nok"
    assert seen == [1, 2, 2]


def test_explicit_num_ctx_is_not_recomputed():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["options"] == {"num_ctx": 12288}
        assert body["stream"] is True
        return httpx.Response(
            200,
            text=_ndjson(
                {"message": {"content": "ok"}, "done": False},
                {"done": True, "eval_count": 1},
            ),
        )

    cfg = _cfg(api_base_url="http://127.0.0.1:11434/v1", summary_num_ctx=32768)
    client = ApiClient(cfg, http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert client.summarize("curto", num_ctx=12288) == "ok"


def test_native_stream_timeout_is_per_read():
    class Spy(httpx.Client):
        def stream(self, method, url, **kwargs):
            self.used = kwargs.get("timeout")
            return super().stream(method, url, **kwargs)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=_ndjson({"message": {"content": "ok"}, "done": True, "eval_count": 1}),
        )

    cfg = _cfg(api_base_url="http://127.0.0.1:11434/v1", summary_num_ctx=8192)
    http = Spy(transport=httpx.MockTransport(handler))
    ApiClient(cfg, http=http).summarize("aula")
    assert http.used.read == STREAM_TIMEOUT.read == 900


def test_stream_cancel_stops_before_the_end():
    def handler(request: httpx.Request) -> httpx.Response:
        lines = [
            {"message": {"content": "A"}, "done": False},
            {"message": {"content": "B"}, "done": False},
            {"message": {"content": "C"}, "done": False},
            {"done": True, "eval_count": 3},
        ]
        return httpx.Response(200, text=_ndjson(*lines))

    cancel = threading.Event()

    def on_progress(n):
        if n >= 1:
            cancel.set()

    cfg = _cfg(api_base_url="http://127.0.0.1:11434/v1", summary_num_ctx=8192)
    client = ApiClient(cfg, http=httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        client.summarize("texto", cancel=cancel, on_progress=on_progress)
        raise AssertionError("esperava cancelamento")
    except SummaryCancelled:
        pass


def test_summary_num_ctx_bounds_and_rounding():
    assert summary_num_ctx_for("", 32768) == 8192
    assert summary_num_ctx_for("a" * 15360, 32768) == 8192
    assert summary_num_ctx_for("a" * 15361, 32768) == 12288
    assert summary_num_ctx_for("a" * 27648, 32768) == 12288
    assert summary_num_ctx_for("a" * 27649, 32768) == 16384
    assert summary_num_ctx_for("a" * 500000, 16384) == 16384
    assert summary_num_ctx_for("a" * 15361, 10000) == 10000
    assert summary_num_ctx_for("a" * 10, 8192) == 8192
    assert summary_num_ctx_for("a" * 500000, 4096) == 4096
    assert summary_num_ctx_for("qualquer", 0) == 0


def _check(handler, **kw):
    client = ApiClient(_cfg(**kw), http=httpx.Client(transport=httpx.MockTransport(handler)))
    return client.check_summary()


def test_check_summary_ok_when_model_installed():
    def handler(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})

    assert _check(handler, summary_num_ctx=32768, summary_model="llama3.1:8b") is None


def test_check_summary_reports_missing_model():
    def handler(request):
        return httpx.Response(200, json={"models": [{"name": "outro:8b"}]})

    problem = _check(handler, summary_num_ctx=32768, summary_model="llama3.1:8b")
    assert "llama3.1:8b" in problem


def test_check_summary_reports_server_down():
    def handler(request):
        raise httpx.ConnectError("[WinError 10061]", request=request)

    problem = _check(handler)
    assert "fora do ar" in problem
    assert "10061" not in problem


def test_summarize_connect_error_is_readable():
    def handler(request):
        raise httpx.ConnectError("[WinError 10061]", request=request)

    client = ApiClient(_cfg(), http=httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        client.summarize("texto")
    except ApiError as exc:
        assert "fora do ar" in str(exc)
    else:
        raise AssertionError("esperava ApiError")

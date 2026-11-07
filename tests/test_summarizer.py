import json
import threading
from pathlib import Path

from studyard.api_client import ApiError, SummaryCancelled
from studyard.summarizer import split_blocks, summarize_text


def _lecture() -> str:
    return "\n".join(f"linha {i:02d} sobre a aula de trafego." for i in range(8))


class ScriptApi:
    def __init__(self, results):
        self.results = list(results)
        self.calls: list[str] = []

    def summarize(self, transcript, num_ctx=None, on_progress=None, cancel=None):
        self.calls.append(transcript)
        if cancel is not None and cancel.is_set():
            raise SummaryCancelled()
        if not self.results:
            raise AssertionError(transcript[:80])
        item = self.results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_short_text_stays_one_block():
    assert split_blocks("olá aula") == ["olá aula"]
    assert split_blocks("   ") == []
    assert split_blocks("a" * 12000) == ["a" * 12000]
    assert [len(part) for part in split_blocks("a" * 12001)] == [12000, 1]


def test_split_blocks_on_newline_and_sentence():
    lines = [f"linha {i:02d} abc" for i in range(12)]
    text = "\n".join(lines)
    blocks = split_blocks(text, limit=80)
    assert len(blocks) > 1
    assert all(len(block) <= 80 for block in blocks)
    assert [line for block in blocks for line in block.split("\n") if line] == lines

    sentence = "Frase curta de exemplo. "
    prose = (sentence * 12).strip()
    parts = split_blocks(prose, limit=90)
    assert len(parts) > 1
    assert all(len(part) <= 90 for part in parts)
    assert " ".join(parts) == prose

    raw = "a" * 250
    hard = split_blocks(raw, limit=100)
    assert "".join(hard) == raw
    assert [len(part) for part in hard] == [100, 100, 50]


def test_short_transcript_is_one_call():
    api = ScriptApi(["resumo unico"])
    assert summarize_text(api, "aula curta sobre patentes") == "resumo unico"
    assert api.calls == ["aula curta sobre patentes"]


def test_checkpoint_resumes_from_next_part(tmp_path: Path):
    text = _lecture()
    n = len(split_blocks(text, 40))
    assert n >= 2
    path = tmp_path / "aula.resumo-partes.json"
    first = ScriptApi(["parcial-1", ApiError("boom")])
    try:
        summarize_text(first, text, checkpoint=path, block_chars=40)
        raise AssertionError("esperava falha na segunda parte")
    except ApiError:
        pass
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["parts"] == ["parcial-1"]
    assert first.calls[0].startswith("Parte 1 de")
    assert first.calls[1].startswith("Parte 2 de")

    second = ScriptApi(["parcial"] * (n - 1) + ["FINAL"])
    assert summarize_text(second, text, checkpoint=path, block_chars=40) == "FINAL"
    assert second.calls[0].startswith("Parte 2 de")
    assert all(not call.startswith("Parte 1 de") for call in second.calls)
    assert second.calls[-1].startswith("Junte")
    assert "parcial-1" in second.calls[-1]


def test_checkpoint_restarts_when_hash_changes(tmp_path: Path):
    text = _lecture()
    n = len(split_blocks(text, 40))
    path = tmp_path / "aula.resumo-partes.json"
    path.write_text(
        json.dumps({"transcript_hash": "outro", "parts": ["velho"]}),
        encoding="utf-8",
    )
    api = ScriptApi(["p"] * n + ["FINAL"])
    assert summarize_text(api, text, checkpoint=path, block_chars=40) == "FINAL"
    assert api.calls[0].startswith("Parte 1 de")
    parts = json.loads(path.read_text(encoding="utf-8"))["parts"]
    assert "velho" not in parts
    assert parts[0] == "p"


def test_cancel_keeps_finished_parts(tmp_path: Path):
    text = _lecture()
    path = tmp_path / "aula.resumo-partes.json"
    calls = []

    class Api:
        def summarize(self, transcript, num_ctx=None, on_progress=None, cancel=None):
            calls.append(transcript)
            if len(calls) >= 2:
                if cancel is not None:
                    cancel.set()
                return "nao gravar"
            return "p1"

    try:
        summarize_text(Api(), text, checkpoint=path, cancel=threading.Event(), block_chars=40)
        raise AssertionError("esperava cancelamento")
    except SummaryCancelled:
        pass
    assert json.loads(path.read_text(encoding="utf-8"))["parts"] == ["p1"]
    assert len(calls) == 2


def test_cancel_before_start_does_not_call_the_model(tmp_path: Path):
    cancel = threading.Event()
    cancel.set()
    api = ScriptApi(["nao"])
    try:
        summarize_text(api, "texto curto", cancel=cancel)
        raise AssertionError("esperava cancelamento")
    except SummaryCancelled:
        pass
    assert api.calls == []

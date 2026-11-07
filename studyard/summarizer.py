"""Resumo em partes, com checkpoint para retomar depois de uma falha."""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from pathlib import Path

from studyard.api_client import SummaryCancelled

BLOCK_CHARS = 12000
DEFERRED_MESSAGE = "Resumo adiado — use Processar pendentes"


def transcript_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def checkpoint_path(folder: Path, stem: str) -> Path:
    return folder / f"{stem}.resumo-partes.json"


def clear_checkpoint(path: Path) -> None:
    if path.exists():
        path.unlink()


def split_blocks(text: str, limit: int = BLOCK_CHARS) -> list[str]:
    """Corta em blocos de até `limit` caracteres, de preferência em linha ou frase."""
    body = text.strip()
    if not body:
        return []
    if limit < 1:
        limit = 1
    if len(body) <= limit:
        return [body]
    blocks: list[str] = []
    rest = body
    while rest:
        if len(rest) <= limit:
            blocks.append(rest)
            break
        cut = _cut_at(rest, limit)
        if cut <= 0:
            cut = min(limit, len(rest))
        piece = rest[:cut].strip()
        nxt = rest[cut:].strip()
        if piece:
            blocks.append(piece)
        if not nxt or nxt == rest:
            break
        rest = nxt
    return blocks


def _cut_at(text: str, limit: int) -> int:
    window = text[:limit]
    newline = window.rfind("\n")
    if newline > 0:
        return newline + 1
    best = -1
    for mark in (". ", "! ", "? "):
        idx = window.rfind(mark)
        if idx >= 0:
            best = max(best, idx + len(mark))
    if best > 0:
        return best
    return limit


def summarize_text(
    api,
    transcript: str,
    checkpoint: Path | None = None,
    cancel=None,
    on_progress=None,
    block_chars: int = BLOCK_CHARS,
) -> str:
    """Resume o texto. Acima de um bloco, grava cada parte no checkpoint."""
    digest = transcript_hash(transcript)
    blocks = split_blocks(transcript, block_chars)
    if not blocks:
        blocks = [""]
    started = time.monotonic()

    def emit(part: int, parts: int, tokens: int) -> None:
        if on_progress is None:
            return
        on_progress(
            {
                "part": part,
                "parts": parts,
                "tokens": int(tokens),
                "elapsed_s": int(time.monotonic() - started),
            }
        )

    if len(blocks) == 1:
        emit(1, 1, 0)
        return _call(api, blocks[0], cancel=cancel, on_tokens=lambda n: emit(1, 1, n))

    done = _load_parts(checkpoint, digest)
    if len(done) > len(blocks):
        done = []
    total = len(blocks) + 1
    for index, block in enumerate(blocks):
        if index < len(done):
            continue
        part_no = index + 1
        if cancel is not None and cancel.is_set():
            raise SummaryCancelled()
        emit(part_no, total, 0)
        text = _call(
            api,
            _part_prompt(part_no, len(blocks), block),
            cancel=cancel,
            on_tokens=lambda n, part=part_no: emit(part, total, n),
        )
        done.append(text)
        if checkpoint is not None:
            _write_checkpoint(checkpoint, digest, done)
        if cancel is not None and cancel.is_set():
            raise SummaryCancelled()

    if cancel is not None and cancel.is_set():
        raise SummaryCancelled()
    emit(total, total, 0)
    return _call(
        api,
        _final_prompt(done),
        cancel=cancel,
        on_tokens=lambda n: emit(total, total, n),
    )


def _part_prompt(i: int, n: int, block: str) -> str:
    return (
        f"Parte {i} de {n} da transcrição de uma aula.\n"
        "Resuma só esta parte em Markdown, em português. "
        "Não invente conteúdo que não esteja no trecho. "
        "Ignore linhas do tipo [trecho ~mm:ss não transcrito].\n\n"
        f"{block}"
    )


def _final_prompt(parts: list[str]) -> str:
    # sem "Parte N" na entrada: com títulos numerados o modelo só empilha as partes
    body = "\n\n---\n\n".join(parts)
    return (
        "Junte os resumos parciais abaixo, em ordem cronológica, num único resumo "
        "da aula em Markdown, em português. Organize por assunto, não por parte: "
        "agrupe o que aparece em mais de um trecho e não use títulos como "
        "\"Parte 1\". Destaque tópicos, definições, exemplos e ênfases do professor. "
        "Não invente conteúdo. "
        "Ignore marcadores do tipo [trecho ~mm:ss não transcrito].\n\n"
        f"{body}"
    )


def _load_parts(path: Path | None, digest: str) -> list[str]:
    if path is None or not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if data.get("transcript_hash") != digest:
        clear_checkpoint(path)
        return []
    parts = data.get("parts")
    if not isinstance(parts, list) or any(not isinstance(part, str) for part in parts):
        return []
    return list(parts)


def _write_checkpoint(path: Path, digest: str, parts: list[str]) -> None:
    payload = {"transcript_hash": digest, "parts": parts}
    raw = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(raw, encoding="utf-8")
    tmp.replace(path)


def _call(api, text: str, *, cancel, on_tokens) -> str:
    if cancel is not None and cancel.is_set():
        raise SummaryCancelled()
    fn = api.summarize
    kwargs = _accepted_kwargs(fn, num_ctx=None, on_progress=on_tokens, cancel=cancel)
    try:
        result = fn(text, **kwargs) if kwargs else fn(text)
    except SummaryCancelled:
        raise
    if cancel is not None and cancel.is_set():
        raise SummaryCancelled()
    if not isinstance(result, str):
        raise TypeError("resumo sem texto")
    return result


def _accepted_kwargs(fn, **candidates):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return candidates
    return {key: value for key, value in candidates.items() if key in params}

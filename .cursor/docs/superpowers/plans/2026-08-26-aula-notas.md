# STUDYARD Lecture Notes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Local Windows app that records a class (mic and/or system audio), appends a faithful transcript to `aaaa-mm-dd_aula.md` live, writes `aaaa-mm-dd_resumo.md` on stop, and recovers via WAV if the API drops.

**Architecture:** One Python process (FastAPI + Uvicorn on `127.0.0.1`) serves a static UI and owns capture, chunked OpenAI-compatible transcription, session files, and pending recovery. The browser never sees `api_key`. `config.json` at the repo root is re-read on Record and on process-pendings.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, httpx, NumPy, sounddevice (WASAPI), pytest.

## Global Constraints

- Bind **only** `127.0.0.1` (never `0.0.0.0`); default port `8765`.
- Config path: `D:\IA\STUDYARD\config.json`; example committed as `config.example.json`; never commit secrets.
- Default `output_dir`: `D:\IA\STUDYARD\transcricao`; day folder `{output_dir}\aaaa-mm-dd\`.
- Filenames: `{date}_aula.md` / `{date}_resumo.md`; next same-day session `{date}_aula-2.md` (no `-1`).
- Stem is taken if `*_aula.md` **or** `*_aula.wav` exists (pending session occupies the stem).
- Capture: WASAPI, mono 16 kHz; source `mic` | `system` | `both` mixed to one stream.
- Always write a session WAV while recording; delete it after success only if session `save_audio` is false.
- Live chunks: `chunk_seconds` default `25`; append to aula.md; 1 try + 2 retries then `[trecho ~mm:ss não transcrito]`.
- Transcription: `POST {api_base_url}/v1/audio/transcriptions`. Summary: `POST {api_base_url}/v1/chat/completions`.
- One active recording session at a time.
- UI copy in Portuguese; no API key editor in the page.
- Tests mock capture and HTTP; do not hit a real API in pytest.

## File map

| Path | Responsibility |
| --- | --- |
| `pyproject.toml` | Package `studyard`, deps, pytest |
| `.gitignore` | `config.json`, `__pycache__`, `.venv`, `transcricao/` |
| `config.example.json` | All keys, empty `api_key` |
| `studyard/config.py` | Load + validate + `record_errors()` |
| `studyard/naming.py` | Day folder + `next_stem()` |
| `studyard/session.py` | Markdown header/append/replace, pending JSON, WAV finalize |
| `studyard/wavutil.py` | PCM float32 mono 16 kHz → WAV bytes/file |
| `studyard/api_client.py` | Transcribe + summarize with retries helper used by engine |
| `studyard/capture.py` | WASAPI mic / loopback / mix |
| `studyard/pending.py` | Scan `output_dir` for `.pending.json` and orphan WAVs |
| `studyard/engine.py` | Record loop, chunk fail markers, stop/resumo/pending |
| `studyard/app.py` | FastAPI: static UI, `/api/status`, start/stop/pendings |
| `studyard/static/index.html` | Controls + status + folder path |
| `studyard/static/app.js` | Poll status, Portuguese labels |
| `studyard/static/style.css` | Minimal readable layout |
| `tests/conftest.py` | tmp_path config fixture |
| `tests/test_config.py` | Defaults + refuse-to-record |
| `tests/test_naming.py` | First stem and `-2`/`-3` |
| `tests/test_session.py` | Append order, pending, wav keep/delete |
| `tests/test_api_client.py` | HTTP contract with httpx mock |
| `tests/test_engine.py` | Chunk retries, pending on API down, retranscribe on stop |
| `tests/test_pending.py` | Crash scan: wav without resumo |
| `tests/test_app.py` | Second start ignored; empty key rejected |

---

### Task 1: Scaffold, gitignore, config load

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `config.example.json`
- Create: `studyard/__init__.py`
- Create: `studyard/config.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Config` dataclass; `load_config(path: Path) -> Config`; `record_errors(cfg: Config) -> list[str]` (empty list means Record is allowed); `CONFIG_PATH` = repo root `config.json`; `REPO_ROOT` = parent of `studyard/`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path

from studyard.config import load_config, record_errors


def test_defaults_when_file_has_only_api(tmp_path: Path):
    p = tmp_path / "config.json"
    p.write_text(
        '{"api_base_url": "https://api.openai.com/v1", "api_key": "sk-test"}',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.output_dir == Path(r"D:\IA\STUDYARD\transcricao")
    assert cfg.transcription_model == "whisper-1"
    assert cfg.summary_model == "gpt-4o-mini"
    assert cfg.language == "pt"
    assert cfg.chunk_seconds == 25
    assert cfg.save_audio is False
    assert cfg.port == 8765
    assert record_errors(cfg) == []


def test_record_errors_on_empty_key(tmp_path: Path):
    p = tmp_path / "config.json"
    p.write_text('{"api_base_url": "https://x", "api_key": ""}', encoding="utf-8")
    cfg = load_config(p)
    errs = record_errors(cfg)
    assert any("api_key" in e for e in errs)


def test_record_errors_on_empty_base_url(tmp_path: Path):
    p = tmp_path / "config.json"
    p.write_text('{"api_base_url": "", "api_key": "k"}', encoding="utf-8")
    cfg = load_config(p)
    errs = record_errors(cfg)
    assert any("api_base_url" in e for e in errs)


def test_strips_trailing_slash_on_base_url(tmp_path: Path):
    p = tmp_path / "config.json"
    p.write_text(
        '{"api_base_url": "https://api.openai.com/v1/", "api_key": "k"}',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.api_base_url == "https://api.openai.com/v1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`

Expected: FAIL with `ModuleNotFoundError: studyard.config` (or collection error).

- [ ] **Step 3: Write minimal implementation**

`pyproject.toml`:

```toml
[project]
name = "studyard"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.32",
  "httpx>=0.27",
  "numpy>=2.0",
  "sounddevice>=0.5",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
studyard = "studyard.app:main"

[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["studyard*"]

[tool.setuptools.package-data]
studyard = ["static/*"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

`.gitignore`:

```
config.json
.venv/
__pycache__/
*.pyc
.pytest_cache/
transcricao/
*.egg-info/
```

`config.example.json`:

```json
{
  "output_dir": "D:\\IA\\STUDYARD\\transcricao",
  "api_base_url": "https://api.openai.com/v1",
  "api_key": "",
  "transcription_model": "whisper-1",
  "summary_model": "gpt-4o-mini",
  "language": "pt",
  "chunk_seconds": 25,
  "save_audio": false,
  "port": 8765
}
```

`studyard/config.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.json"
DEFAULT_OUTPUT = Path(r"D:\IA\STUDYARD\transcricao")


@dataclass(frozen=True)
class Config:
    output_dir: Path
    api_base_url: str
    api_key: str
    transcription_model: str
    summary_model: str
    language: str
    chunk_seconds: int
    save_audio: bool
    port: int


def load_config(path: Path | None = None) -> Config:
    path = path or CONFIG_PATH
    raw: dict = {}
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
    base = str(raw.get("api_base_url", "")).rstrip("/")
    out = Path(raw.get("output_dir") or DEFAULT_OUTPUT)
    return Config(
        output_dir=out,
        api_base_url=base,
        api_key=str(raw.get("api_key", "")),
        transcription_model=str(raw.get("transcription_model", "whisper-1")),
        summary_model=str(raw.get("summary_model", "gpt-4o-mini")),
        language=str(raw.get("language", "pt")),
        chunk_seconds=int(raw.get("chunk_seconds", 25)),
        save_audio=bool(raw.get("save_audio", False)),
        port=int(raw.get("port", 8765)),
    )


def record_errors(cfg: Config) -> list[str]:
    errs: list[str] = []
    if not cfg.api_base_url:
        errs.append("api_base_url vazio — ajuste config.json")
    if not cfg.api_key:
        errs.append("api_key vazio — ajuste config.json")
    try:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        errs.append(f"output_dir inacessível: {exc}")
    return errs
```

Empty `studyard/__init__.py`. `tests/conftest.py` can be empty or omit.

- [ ] **Step 4: Run tests and make sure they pass**

Run: `python -m pytest tests/test_config.py -v`

Expected: PASS (4 tests). Create `.venv`, `pip install -e ".[dev]"` first if needed.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore config.example.json studyard/__init__.py studyard/config.py tests/test_config.py
git commit -m "feat: load and validate config.json for recording"
```

---

### Task 2: Day folder and session stem

**Files:**
- Create: `studyard/naming.py`
- Create: `tests/test_naming.py`

**Interfaces:**
- Consumes: none
- Produces: `day_dir(output_dir: Path, day: date) -> Path` returns `output_dir / day.isoformat()`; `next_stem(day_folder: Path, day: date) -> str` returns `"YYYY-MM-DD_aula"` or `"YYYY-MM-DD_aula-N"` for smallest N≥2 whose `{stem}.md` (`*_aula.md`) or `{stem}.wav` does not exist. Check files named `{iso}_aula.md`, `{iso}_aula.wav`, `{iso}_aula-2.md`, etc.

- [ ] **Step 1: Write the failing test**

```python
from datetime import date
from pathlib import Path

from studyard.naming import day_dir, next_stem


def test_day_dir(tmp_path: Path):
    d = date(2026, 8, 26)
    assert day_dir(tmp_path, d) == tmp_path / "2026-08-26"


def test_first_stem_when_empty(tmp_path: Path):
    folder = tmp_path / "2026-08-26"
    folder.mkdir()
    assert next_stem(folder, date(2026, 8, 26)) == "2026-08-26_aula"


def test_second_stem_after_aula_md(tmp_path: Path):
    folder = tmp_path / "2026-08-26"
    folder.mkdir()
    (folder / "2026-08-26_aula.md").write_text("x", encoding="utf-8")
    assert next_stem(folder, date(2026, 8, 26)) == "2026-08-26_aula-2"


def test_pending_wav_occupies_stem(tmp_path: Path):
    folder = tmp_path / "2026-08-26"
    folder.mkdir()
    (folder / "2026-08-26_aula.wav").write_bytes(b"RIFF")
    assert next_stem(folder, date(2026, 8, 26)) == "2026-08-26_aula-2"


def test_third_after_two_and_hole_not_filled(tmp_path: Path):
    folder = tmp_path / "2026-08-26"
    folder.mkdir()
    (folder / "2026-08-26_aula.md").write_text("a", encoding="utf-8")
    (folder / "2026-08-26_aula-2.md").write_text("b", encoding="utf-8")
    assert next_stem(folder, date(2026, 8, 26)) == "2026-08-26_aula-3"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_naming.py -v`

Expected: FAIL `ModuleNotFoundError: studyard.naming`

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from datetime import date
from pathlib import Path


def day_dir(output_dir: Path, day: date) -> Path:
    return output_dir / day.isoformat()


def _occupied(folder: Path, stem: str) -> bool:
    return (folder / f"{stem}.md").exists() or (folder / f"{stem}.wav").exists()


def next_stem(day_folder: Path, day: date) -> str:
    iso = day.isoformat()
    first = f"{iso}_aula"
    if not _occupied(day_folder, first):
        return first
    n = 2
    while _occupied(day_folder, f"{iso}_aula-{n}"):
        n += 1
    return f"{iso}_aula-{n}"
```

Note: aula file is `{stem}.md` where stem already includes `_aula`. So files are `2026-08-26_aula.md` not `2026-08-26_aula_aula.md`. WAV is `{stem}.wav`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_naming.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add studyard/naming.py tests/test_naming.py
git commit -m "feat: allocate dated aula stems without collisions"
```

---

### Task 3: Session files (markdown, pending, wav keep/delete)

**Files:**
- Create: `studyard/wavutil.py`
- Create: `studyard/session.py`
- Create: `tests/test_session.py`

**Interfaces:**
- Consumes: `next_stem` not required inside Session; caller passes `stem`
- Produces: `FAIL_MARKER = "[trecho ~{mm}:{ss} não transcrito]"` via `fail_marker(elapsed_seconds: int) -> str`; class `SessionFiles` with:
  - `__init__(self, folder: Path, stem: str, source: str, save_audio: bool)`
  - `aula_path`, `resumo_path`, `wav_path`, `pending_path` properties (`Path`)
  - `create(self) -> None` — mkdir folder, write header (`# Aula\n\n- Data: ...\n- Fonte: ...\n\n## Transcrição\n\n`)
  - `append_transcript(self, text: str) -> None` — append text + newline, never rewrite whole file
  - `replace_body(self, text: str) -> None` — keep header up to and including `## Transcrição\n\n`, replace after
  - `read_transcript_body(self) -> str`
  - `write_summary(self, text: str) -> None`
  - `write_pending(self, need: list[str]) -> None` — JSON `{"need": [...], "save_audio": bool}`
  - `clear_pending(self) -> None`
  - `append_pcm(self, pcm: "np.ndarray") -> None` — grow WAV (or buffer then write); simplest: keep `list` of pcm and `flush_wav()`
  - `finalize_audio(self, success: bool) -> None` — if success and not save_audio, delete wav; if not success, keep wav

Use numpy float32 -1..1, 16000 Hz, `wavutil.write_wav(path, pcm)` / `pcm_to_wav_bytes(pcm)`.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from studyard.session import SessionFiles, fail_marker


def test_append_keeps_order(tmp_path: Path):
    s = SessionFiles(tmp_path, "2026-08-26_aula", "mic", save_audio=False)
    s.create()
    s.append_transcript("um")
    s.append_transcript("dois")
    body = s.read_transcript_body()
    assert body.index("um") < body.index("dois")
    assert s.aula_path.read_text(encoding="utf-8").count("um") == 1


def test_fail_marker_format():
    assert fail_marker(125) == "[trecho ~02:05 não transcrito]"


def test_pending_roundtrip(tmp_path: Path):
    s = SessionFiles(tmp_path, "2026-08-26_aula", "system", save_audio=True)
    s.create()
    s.write_pending(["transcribe", "summarize"])
    data = s.pending_path.read_text(encoding="utf-8")
    assert "transcribe" in data
    assert "true" in data.lower() or '"save_audio": true' in data.replace(" ", "")
    s.clear_pending()
    assert not s.pending_path.exists()


def test_finalize_deletes_wav_when_save_false(tmp_path: Path, monkeypatch):
    import numpy as np
    from studyard import session as session_mod

    s = SessionFiles(tmp_path, "2026-08-26_aula", "mic", save_audio=False)
    s.create()
    s.append_pcm(np.zeros(1600, dtype=np.float32))
    s.flush_wav()
    assert s.wav_path.exists()
    s.finalize_audio(success=True)
    assert not s.wav_path.exists()


def test_finalize_keeps_wav_on_failure(tmp_path: Path):
    import numpy as np

    s = SessionFiles(tmp_path, "2026-08-26_aula", "mic", save_audio=False)
    s.create()
    s.append_pcm(np.zeros(1600, dtype=np.float32))
    s.flush_wav()
    s.finalize_audio(success=False)
    assert s.wav_path.exists()
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_session.py -v`

Expected: FAIL missing `studyard.session`

- [ ] **Step 3: Implement `wavutil.py` and `session.py`**

`studyard/wavutil.py`:

```python
from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np

RATE = 16000


def float_to_int16(pcm: np.ndarray) -> bytes:
    clipped = np.clip(pcm, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def write_wav(path: Path, pcm: np.ndarray, rate: int = RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(float_to_int16(pcm))


def pcm_to_wav_bytes(pcm: np.ndarray, rate: int = RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(float_to_int16(pcm))
    return buf.getvalue()
```

`studyard/session.py` — implement the interface above. Header marker `## Transcrição`. `fail_marker`: `mm, ss = divmod(int(elapsed_seconds), 60); return f"[trecho ~{mm:02d}:{ss:02d} não transcrito]"`. `append_pcm` concatenates numpy arrays on `self._pcm`. `flush_wav` calls `write_wav`. `read_transcript_body` splits on `## Transcrição\n\n`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_session.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add studyard/wavutil.py studyard/session.py tests/test_session.py
git commit -m "feat: persist aula markdown, pending json, and session wav"
```

---

### Task 4: OpenAI-compatible API client

**Files:**
- Create: `studyard/api_client.py`
- Create: `tests/test_api_client.py`

**Interfaces:**
- Consumes: `Config` fields `api_base_url`, `api_key`, models, language
- Produces: `class ApiError(Exception)`; `class ApiClient:`
  - `__init__(self, cfg: Config, http: httpx.Client | None = None)`
  - `transcribe(self, wav_bytes: bytes) -> str` — POST `{base}/v1/audio/transcriptions` multipart file `file` filename `chunk.wav`, fields `model`, `language`; parse JSON `text`
  - `summarize(self, transcript: str) -> str` — POST `{base}/v1/chat/completions` JSON messages: system prompt exactly:

```
Você resume aulas de mestrado em Markdown em português. Extraia tópicos, definições, exemplos e ênfases do professor. Não invente conteúdo que não esteja na transcrição. Ignore linhas do tipo [trecho ~mm:ss não transcrito].
```

user = transcript. `model` = `cfg.summary_model`. Return `choices[0].message.content`.

Do **not** implement chunk retries here (engine does that). Client raises `ApiError` on HTTP error or timeout.

- [ ] **Step 1: Failing test** using `httpx.MockTransport`

```python
import httpx

from studyard.api_client import ApiClient, ApiError
from studyard.config import Config
from pathlib import Path


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


def test_transcribe_posts_multipart():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/v1/audio/transcriptions")
        assert request.headers["authorization"] == "Bearer k"
        return httpx.Response(200, json={"text": "olá aula"})

    client = ApiClient(_cfg(), http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert client.transcribe(b"RIFF") == "olá aula"


def test_summarize_sends_transcript():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/v1/chat/completions")
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
        assert False, "expected ApiError"
    except ApiError:
        pass
```

- [ ] **Step 2: Run — expect fail**

Run: `python -m pytest tests/test_api_client.py -v`

- [ ] **Step 3: Implement `ApiClient`** with timeout 120s on transcribe, 180s on summarize.

- [ ] **Step 4: pytest pass**

- [ ] **Step 5: Commit** `feat: call OpenAI-compatible transcribe and chat endpoints`

---

### Task 5: Engine (chunk loop, retries, stop, pending)

**Files:**
- Create: `studyard/engine.py`
- Create: `tests/test_engine.py`

**Interfaces:**
- Consumes: `SessionFiles`, `ApiClient`, `pcm_to_wav_bytes`, `fail_marker`, `Config`
- Produces: `class Recorder:`
  - `__init__(self, cfg: Config, session: SessionFiles, api: ApiClient, clock=time.monotonic)`
  - `feed_pcm(self, pcm: np.ndarray) -> None` — accumulate; when samples ≥ chunk_seconds * 16000, process one chunk
  - `process_chunk(self, pcm: np.ndarray, elapsed_s: int) -> None` — try transcribe up to 3 attempts (2 retries); on fail append `fail_marker(elapsed_s)` and set `self.had_chunk_failure = True`; on success `append_transcript`
  - `stop(self) -> dict` — flush remainder as last chunk; then:
    - if not `had_chunk_failure`: `summarize(read_transcript_body())` → `write_summary`; `finalize_audio(success=True)`; `clear_pending`
    - if `had_chunk_failure`: try `transcribe(full wav bytes)` `replace_body` then summarize; if API fails: `write_pending(["transcribe","summarize"])` `finalize_audio(success=False)`
  - `self.status: str` in `{"idle","recording","transcribing","summarizing","offline","error","saved"}`
  - `self.last_text: str`
  - `self.offline: bool`

For tests, do **not** use real threads. Call `feed_pcm` / `process_chunk` / `stop` directly. Fake API:

```python
class FakeApi:
    def __init__(self):
        self.transcribe_calls = 0
        self.fail_next = 0
        self.summarize_ok = True
    def transcribe(self, wav_bytes: bytes) -> str:
        self.transcribe_calls += 1
        if self.fail_next:
            self.fail_next -= 1
            raise ApiError("down")
        return f"chunk-{self.transcribe_calls}"
    def summarize(self, transcript: str) -> str:
        if not self.summarize_ok:
            raise ApiError("sum")
        return "resumo ok"
```

Tests:

1. Three transcribe failures → marker in aula.md, `had_chunk_failure`, session continues (call process_chunk again successfully).
2. All chunks ok → stop writes resumo, deletes wav (`save_audio=False`).
3. Chunks fail, stop with API still down → pending json + wav kept.
4. Chunks fail, stop with API up → full wav transcribe replaces body, resumo written.
5. Summarize fails after good chunks → aula.md intact, pending `["summarize"]`, wav kept.

`process_chunk` retries: `for attempt in range(3): try: ... break except ApiError: if attempt==2: append marker`.

- [ ] **Step 1–5:** TDD as above. Commit `feat: chunk transcription engine with pending recovery`

---

### Task 6: Pending scan after crash

**Files:**
- Create: `studyard/pending.py`
- Create: `tests/test_pending.py`

**Interfaces:**
- Consumes: `SessionFiles` paths convention, `FAIL_MARKER` substring `não transcrito`
- Produces: `scan_pendings(output_dir: Path) -> list[dict]` each dict: `stem`, `folder`, `need`, `save_audio`, `aula_path`, `wav_path`
  - Include every `*.pending.json` under `output_dir/*/`
  - For every `*_aula*.wav` (files matching `*_aula.md` stem: name endswith `_aula.wav` or `_aula-N.wav`) without sibling `{stem.replace}` wait: wav name is `{stem}.wav` where stem is `DATE_aula` or `DATE_aula-2`. Sibling resumo is `{stem.replace("_aula", "_resumo", 1)}` **wrong** because `2026-08-26_aula-2` → resumo `2026-08-26_resumo-2`.

Resumo path rule (lock this): stem `2026-08-26_aula` → `2026-08-26_resumo.md`; stem `2026-08-26_aula-2` → `2026-08-26_resumo-2.md`. Helper `resumo_stem(aula_stem: str) -> str`: replace the last `_aula` with `_resumo` only at the `_aula` token: if `stem.endswith("_aula")`: `stem[:-5] + "_resumo"`; if `"_aula-" in stem`: `stem.replace("_aula-", "_resumo-", 1)`.

If wav exists and corresponding resumo.md missing and no pending.json yet: create pending.json. `need` = `["transcribe","summarize"]` if aula.md missing, body empty, or contains `não transcrito`; else `["summarize"]`. `save_audio` default false when synthesizing pending.

`process_pending_item(item, api: ApiClient, save_audio_fallback: bool)`: load SessionFiles from folder/stem; if transcribe in need: transcribe full wav, replace_body; always then summarize; clear pending; finalize_audio based on pending save_audio.

- [ ] Tests: orphan wav + empty aula → pending transcribe; wav + complete aula no marker no resumo → summarize only; existing pending.json listed as-is.

- [ ] Commit `feat: scan and process leftover lecture pendings`

---

### Task 7: Capture (WASAPI)

**Files:**
- Create: `studyard/capture.py`

**Interfaces:**
- Consumes: numpy, sounddevice
- Produces: `list_devices() -> dict` with `mic` and `system` labels; `class AudioCapture:` `start(source: Literal["mic","system","both"])`, `read_chunk(seconds: float) -> np.ndarray`, `stop()`. Mix both by adding and clipping. Sample rate 16000 mono.

No unit test hitting hardware. Optional test: mix function `mix_mono(a, b)` if extracted.

```python
def mix_mono(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = min(len(a), len(b))
    return np.clip((a[:n] + b[:n]) * 0.5, -1.0, 1.0)
```

WASAPI loopback via `sounddevice.WasapiSettings(loopback=True)` on the default output device. If loopback unavailable, `start("system"|"both")` raises `CaptureError` with Portuguese message.

- [ ] **Step:** implement + tiny test for `mix_mono` only. Commit `feat: capture WASAPI mic and system loopback`

---

### Task 8: FastAPI app + UI

**Files:**
- Create: `studyard/app.py`
- Create: `studyard/static/index.html`
- Create: `studyard/static/app.js`
- Create: `studyard/static/style.css`
- Create: `tests/test_app.py`

**Interfaces:**
- `create_app(cfg_path: Path | None = None, api_factory=None, capture_factory=None) -> FastAPI` for tests inject fakes.
- `GET /` static index
- `GET /api/status` JSON: `state`, `last_text`, `folder`, `aula`, `resumo`, `message`, `save_audio_default`, `pendings` (from scan), `recording` bool
- `POST /api/start` JSON `{source, save_audio}` — `load_config()` fresh; if `record_errors`: 400 `{errors}`; if already recording: 409 `{message: "sessão ativa"}`; else start background thread: capture + feed engine
- `POST /api/stop` — stop capture, `engine.stop()`, return paths
- `POST /api/process-pendings` — `load_config()` fresh; process all scan_pendings

Background thread: loop `read_chunk(0.5)` until stop flag; `engine.feed_pcm`. Keep `AppState` singleton on `app.state`.

UI: source radios mic/sistema/ambos; checkbox salvar áudio (unchecked; on load set from status.save_audio_default); Gravar / Parar; status text; pasta path; lista pendentes; botão Processar pendentes. Poll `/api/status` every 1s. **Never** display api_key.

`main()`: `uvicorn.run(app, host="127.0.0.1", port=cfg.port)`.

Tests with `TestClient`: start without key → 400; start twice → 409 on second (inject dummy capture that returns zeros).

- [ ] Commit `feat: localhost UI to record, stop, and process pendings`

---

### Task 9: README how to run

**Files:**
- Create: `README.md`

Portuguese: copy `config.example.json` → `config.json`, fill key, `python -m venv .venv`, `pip install -e ".[dev]"`, `studyard` or `python -m studyard.app`, open `http://127.0.0.1:8765`. Manual test checklist from spec (mic / Zoom / wifi off).

- [ ] Commit `docs: explain how to run STUDYARD locally`

---

## Self-review (spec coverage)

| Spec item | Task |
| --- | --- |
| localhost FastAPI, key not in browser | 8 |
| config.json fields + re-read on Gravar/pendentes | 1, 8 |
| day folder + stems -2 | 2 |
| live append aula.md | 3, 5 |
| resumo on stop | 5 |
| always temp WAV; delete if save_audio false | 3, 5 |
| chunk retries + marker | 5 |
| pending + process from full WAV | 5, 6 |
| crash scan orphan wav | 6 |
| WASAPI mic/system/both | 7 |
| OpenAI-compatible URLs | 4 |
| UI status + folder + pendings | 8 |
| tests listed in spec | 1–6, 8 |

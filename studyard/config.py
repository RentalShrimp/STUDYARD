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

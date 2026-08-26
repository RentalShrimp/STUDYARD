import json
from pathlib import Path

from studyard.config import load_config, record_errors, summarize_errors


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
    assert cfg.whisper_model == "base"
    assert cfg.port == 8765
    assert record_errors(cfg) == []


def test_record_errors_allows_empty_key(tmp_path: Path):
    p = tmp_path / "config.json"
    p.write_text(
        json.dumps(
            {"api_base_url": "", "api_key": "", "output_dir": str(tmp_path / "out")}
        ),
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert record_errors(cfg) == []


def test_summarize_errors_on_empty_key(tmp_path: Path):
    p = tmp_path / "config.json"
    p.write_text('{"api_base_url": "https://x", "api_key": ""}', encoding="utf-8")
    cfg = load_config(p)
    errs = summarize_errors(cfg)
    assert any("api_key" in e for e in errs)


def test_summarize_errors_on_empty_base_url(tmp_path: Path):
    p = tmp_path / "config.json"
    p.write_text('{"api_base_url": "", "api_key": "k"}', encoding="utf-8")
    cfg = load_config(p)
    errs = summarize_errors(cfg)
    assert any("api_base_url" in e for e in errs)


def test_strips_trailing_slash_on_base_url(tmp_path: Path):
    p = tmp_path / "config.json"
    p.write_text(
        '{"api_base_url": "https://api.openai.com/v1/", "api_key": "k"}',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.api_base_url == "https://api.openai.com/v1"

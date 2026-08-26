from __future__ import annotations

import json
from pathlib import Path

from studyard.naming import resumo_stem
from studyard.session import HEADER_END, SessionFiles
from studyard.whisper_service import transcribe_wav_path


def _stem_from_pending(name: str) -> str | None:
    if name.endswith(".pending.json"):
        return name[: -len(".pending.json")]
    return None


def _aula_stem_from_wav(name: str) -> str | None:
    if not name.endswith(".wav"):
        return None
    stem = name[: -len(".wav")]
    if stem.endswith("_aula") or "_aula-" in stem:
        return stem
    return None


def _need_for_orphan(aula_path: Path) -> list[str]:
    if not aula_path.exists():
        return ["transcribe", "summarize"]
    text = aula_path.read_text(encoding="utf-8")
    idx = text.find(HEADER_END)
    body = text[idx + len(HEADER_END) :] if idx >= 0 else text
    if not body.strip() or "não transcrito" in body:
        return ["transcribe", "summarize"]
    return ["summarize"]


def _item(
    folder: Path,
    stem: str,
    need: list[str],
    save_audio: bool,
) -> dict:
    return {
        "stem": stem,
        "folder": folder,
        "need": need,
        "save_audio": save_audio,
        "aula_path": folder / f"{stem}.md",
        "wav_path": folder / f"{stem}.wav",
        "resumo_path": folder / f"{resumo_stem(stem)}.md",
        "pending_path": folder / f"{stem}.pending.json",
    }


def scan_pendings(output_dir: Path) -> list[dict]:
    items: list[dict] = []
    seen: set[tuple[Path, str]] = set()
    if not output_dir.exists():
        return items
    for day_folder in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        for pending in sorted(day_folder.glob("*.pending.json")):
            stem = _stem_from_pending(pending.name)
            if not stem:
                continue
            raw = json.loads(pending.read_text(encoding="utf-8"))
            need = list(raw.get("need") or ["summarize"])
            save_audio = bool(raw.get("save_audio", False))
            items.append(_item(day_folder, stem, need, save_audio))
            seen.add((day_folder, stem))
        for wav in sorted(day_folder.glob("*.wav")):
            stem = _aula_stem_from_wav(wav.name)
            if not stem or (day_folder, stem) in seen:
                continue
            resumo = day_folder / f"{resumo_stem(stem)}.md"
            if resumo.exists():
                continue
            s = SessionFiles(day_folder, stem, source="mic", save_audio=False)
            if not s.aula_path.exists():
                s.create()
            need = _need_for_orphan(s.aula_path)
            s.save_audio = False
            s.write_pending(need)
            items.append(_item(day_folder, stem, need, False))
            seen.add((day_folder, stem))
    return items


def process_pending_item(item: dict, api, transcribe_wav=None) -> None:
    folder: Path = item["folder"]
    stem: str = item["stem"]
    save_audio = bool(item["save_audio"])
    s = SessionFiles(folder, stem, source="mic", save_audio=save_audio)
    if not s.aula_path.exists():
        s.create()
    if "transcribe" in item["need"]:
        wav_path: Path = item["wav_path"]
        asr = transcribe_wav or transcribe_wav_path
        text = asr(wav_path)
        s.replace_body(text)
    summary = api.summarize(s.read_transcript_body())
    s.write_summary(summary)
    s.clear_pending()
    s.finalize_audio(success=True)

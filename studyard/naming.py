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

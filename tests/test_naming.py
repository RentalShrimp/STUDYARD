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

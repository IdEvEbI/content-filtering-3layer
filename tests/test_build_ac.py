from pathlib import Path
from scripts.build_ac import load_words


def test_load_words_dedupe(tmp_path: Path):
    sample = tmp_path / "raw.txt"
    sample.write_text("\n".join(["abc", "abc", "xyz"]), encoding="utf‑8")
    words = load_words(sample)
    assert words == ["abc", "xyz"]

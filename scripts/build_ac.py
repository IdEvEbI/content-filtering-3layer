"""Clean and sort raw sensitive-word list.

Usage:
    python scripts/build_ac.py --in data/sensitive_raw.txt --out data/sensitive.txt
"""

import argparse
from pathlib import Path


def load_words(path: Path) -> list[str]:
    """Load raw words, strip whitespace, deduplicate while preserving order."""
    words: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf‑8") as f:
        for line in f:
            w = line.strip()
            if w and w not in seen:
                words.append(w)
                seen.add(w)
    return words


def save_words(words: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(words), encoding="utf‑8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True,
                        help="raw txt input path")
    parser.add_argument("--out", dest="out_path", required=True,
                        help="clean txt output path")
    args = parser.parse_args()

    raw_path = Path(args.in_path)
    out_path = Path(args.out_path)

    words = load_words(raw_path)
    save_words(words, out_path)
    print(f"Saved {len(words)} unique words ➜ {out_path}")


if __name__ == "__main__":
    main()

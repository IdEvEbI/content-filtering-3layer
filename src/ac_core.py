"""AC automaton wrapper for sensitive‑word matching."""
from pathlib import Path
import ahocorasick
from typing import List, Tuple


class SensitiveMatcher:
    """Wrap pyahocorasick.Automaton with a friendly API."""

    def __init__(self, dict_path: str | Path):
        self._automaton = ahocorasick.Automaton()
        self.load_dict(dict_path)

    def load_dict(self, dict_path: str | Path) -> None:
        """Load word list (one per line)."""
        for idx, word in enumerate(
            Path(dict_path).read_text(encoding="utf-8").splitlines()
        ):
            if word:
                # value=(idx, word) 保留索引，方便后续排序或调试
                self._automaton.add_word(word, (idx, word))
        self._automaton.make_automaton()

    def find(self, text: str) -> List[Tuple[str, int, int]]:
        """Return (word, start, end) list. End is exclusive."""
        return [
            (word, end - len(word) + 1, end + 1)
            for end, (_, word) in self._automaton.iter(text)
        ]

    # 可选：判断是否命中
    def has_match(self, text: str) -> bool:
        return any(True for _ in self._automaton.iter(text))

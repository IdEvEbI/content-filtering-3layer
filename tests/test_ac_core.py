from src.ac_core import SensitiveMatcher
from pathlib import Path


def test_match_and_positions(tmp_path: Path):
    # 准备临时词库
    dict_path = tmp_path / "dict.txt"
    dict_path.write_text("\n".join(["abc", "敏感"]), encoding="utf-8")

    matcher = SensitiveMatcher(dict_path)
    text = "这是abc和敏感词的混合abc"
    matches = matcher.find(text)

    # 预期两次 abc + 一次 敏感
    assert matches == [
        ("abc", 2, 5),
        ("敏感", 6, 8),
        ("abc", 12, 15),
    ]


def test_no_match(tmp_path: Path):
    dict_path = tmp_path / "dict.txt"
    dict_path.write_text("xyz", encoding="utf-8")
    matcher = SensitiveMatcher(dict_path)

    assert matcher.find("no hit") == []
    assert matcher.has_match("no hit") is False

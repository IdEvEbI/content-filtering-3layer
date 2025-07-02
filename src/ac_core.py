"""
AC自动机敏感词匹配封装

本模块用于高效检测文本中的敏感词，支持词典加载、批量查找、跳过base64编码内容和URL内容等功能。
"""
from pathlib import Path
import ahocorasick
from typing import List, Tuple
import re


class SensitiveMatcher:
    """
    敏感词检测器，基于 pyahocorasick.Automaton 封装
    支持加载词典、批量查找、跳过base64编码内容和URL内容
    """

    def __init__(self, dict_path: str | Path):
        """
        初始化敏感词检测器，加载词典
        Args:
            dict_path: 敏感词词典文件路径
        """
        self._automaton = ahocorasick.Automaton()
        self.load_dict(dict_path)

    def load_dict(self, dict_path: str | Path) -> None:
        """
        加载敏感词词典（每行一个词）
        Args:
            dict_path: 敏感词词典文件路径
        """
        for idx, word in enumerate(
            Path(dict_path).read_text(encoding="utf-8").splitlines()
        ):
            if word:
                # value=(idx, word) 保留索引，方便后续排序或调试
                self._automaton.add_word(word, (idx, word))
        self._automaton.make_automaton()

    def _is_base64_content(self, text: str, start: int, end: int) -> bool:
        """
        检测指定位置的内容是否为 base64 编码
        通过字符集比例和长度判断，避免误报
        Args:
            text: 完整文本
            start: 匹配开始位置
            end: 匹配结束位置
        Returns:
            bool: 是否为 base64 编码内容
        """
        check_start = max(0, start - 50)
        check_end = min(len(text), end + 50)
        check_text = text[check_start:check_end]
        base64_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
        text_chars = set(check_text)
        base64_ratio = len(text_chars.intersection(base64_chars)) / len(text_chars) if text_chars else 0
        return base64_ratio > 0.8 and len(check_text) > 20

    def _is_url_content(self, text: str, start: int, end: int) -> bool:
        """
        检测指定位置的内容是否在 URL 中
        通过正则表达式匹配常见的 URL 模式
        Args:
            text: 完整文本
            start: 匹配开始位置
            end: 匹配结束位置
        Returns:
            bool: 是否在 URL 中
        """
        # 向前后各检查100个字符，扩大检测范围以找到完整的URL
        check_start = max(0, start - 100)
        check_end = min(len(text), end + 100)
        check_text = text[check_start:check_end]

        # URL 正则表达式模式，匹配常见的 URL 格式
        url_patterns = [
            r'https?://[^\s<>"{}|\\^`\[\]]+',  # http/https URL
            r'www\.[^\s<>"{}|\\^`\[\]]+',     # www 开头的 URL
            r'ftp://[^\s<>"{}|\\^`\[\]]+',    # ftp URL
            r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # 域名格式
        ]

        for pattern in url_patterns:
            for match in re.finditer(pattern, check_text):
                # 检查敏感词匹配位置是否在 URL 范围内
                url_start = check_start + match.start()
                url_end = check_start + match.end()
                if url_start <= start and end <= url_end:
                    return True

        return False

    def find(self, text: str) -> List[Tuple[str, int, int]]:
        """
        查找文本中的所有敏感词，返回(词, 起始, 结束)元组列表
        会自动跳过base64编码区域和URL中的命中
        Args:
            text: 待检测文本
        Returns:
            List[Tuple[str, int, int]]: (敏感词, 起始位置, 结束位置)列表，结束位置为开区间
        """
        results = []
        for end, (_, word) in self._automaton.iter(text):
            start = end - len(word) + 1
            # 跳过base64编码内容和URL中的匹配
            if self._is_base64_content(text, start, end + 1) or self._is_url_content(text, start, end + 1):
                continue
            results.append((word, start, end + 1))
        return results

    def has_match(self, text: str) -> bool:
        """
        判断文本中是否存在敏感词（自动跳过base64区域和URL）
        Args:
            text: 待检测文本
        Returns:
            bool: 是否存在敏感词
        """
        for end, (_, word) in self._automaton.iter(text):
            start = end - len(word) + 1
            # 跳过base64编码内容和URL中的匹配
            if self._is_base64_content(text, start, end + 1) or self._is_url_content(text, start, end + 1):
                continue
            return True
        return False

# Phase A 步骤 2 — AC 引擎封装

> 本笔记讲解 **Phase A · Step 2**：封装敏感词 **AC 自动机核心库**，提供统一的匹配接口，后续 FastAPI 服务将直接依赖此模块。

---

## 1 实操步骤

> 分为四块：**依赖安装 → 核心代码 → 单元测试 → 运行与提交**。

### 1‑1 依赖安装

**目的**：引入高性能 AC 自动机实现；本阶段使用 `pyahocorasick`，后续可无缝替换 Hyperscan。

```bash
# 进入虚拟环境后
pip install pyahocorasick==2.0.0  # 纯 Python 绑定，跨平台易部署

# 将运行依赖写入锁定文件，保证 CI 环境一致
echo "pyahocorasick==2.0.0" >> requirements.txt
```

---

### 1‑2 核心代码（`src/ac_core.py`）

#### 1‑2‑1 创建文件

**目的**：封装加载词库及匹配逻辑，向外暴露简单易用 API。

**操作**：

1. 在 `src` 目录 **新建** `ac_core.py`，输入以下内容并保存：

   ```python
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
   
   ```

> **性能提示**：`make_automaton()` 代价固定，建议在服务启动时加载一次并长期复用。

---

### 1‑3 单元测试（`tests/test_ac_core.py`）

#### 1‑3‑1 创建测试文件

**目的**：验证词典加载、匹配位置、无命中返回空列表。

```python
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

```

---

### 1‑4 运行与提交

#### 1‑4‑1 本地验证

```bash
pytest -q
flake8
```

全部通过说明核心逻辑可靠。

#### 1‑4‑2 提交 PR

```bash
git switch -c ac-core
git add .
git commit -m "feat: add ac automaton core module"
git push -u origin ac-core
# GitHub: 创建 PR ➜ base: dev ← compare: ac-core
# CI 全绿 ➜ Merge ➜ Delete branch
# 本地: git switch dev && git branch -d ac-core
```

---

## 2 附加说明

* **pyahocorasick**：C 扩展 + 双数组 trie，在中小规模词库下性能足够；后续可并行化或替换 Hyperscan。
* **API 设计**：`find` 返回 **end-exclusive** 区间，与 Python 切片一致。
* **内存开销**：约 2–3 × 词库大小，加载一次后复用；多进程时可通过 `fork` 共享。

---

## 总结

1. 封装 `SensitiveMatcher`，实现词库加载与高效匹配。
2. 单元测试覆盖命中、位置信息、空结果三场景。
3. 通过 PR 合并至 `dev`，核心库可供后续 FastAPI 服务直接调用。

> **一句话总结**：统一、轻量的 AC 匹配库，为实时敏感词拦截打下性能基础。

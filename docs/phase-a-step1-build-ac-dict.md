# Phase A 步骤 1 — 词库清洗脚本

> 本笔记讲解 **Phase A · Step 1**：编写词库清洗脚本，将原始敏感词列表「**去重且保留原有顺序**」后生成标准词库文件，供后续 AC 自动机服务直接加载。

---

## 1 实操步骤

> 分为四块：**脚本开发 → 环境配置 → 单元测试 → 运行与提交**。每一步先写 *目的*，再写 *操作*。

### 1‑1 脚本开发（`scripts/build_ac.py`）

#### 1‑1‑1 创建脚本文件

**目的**：实现敏感词去重、保留原顺序并输出标准词库。

**操作**：

1. 在 `scripts` 目录 **新建** `build_ac.py` 文件。
2. 输入以下内容并保存：

   ```python
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
   ```

#### 1‑1‑2 将 `scripts` 设为可导入包

**目的**：方便测试与其他模块通过 `from scripts.build_ac import load_words` 导入函数。

```bash
touch scripts/__init__.py
```

#### 1‑1‑3 准备示例原始词库

**目的**：提供最小可跑通脚本的输入文件。

**操作**：

1. 在 `data` 目录 **新建** `sensitive_raw.txt`，填入示例：

   ```text
   敏感词A
   敏感词B
   敏感词A  # 重复
   敏感词C
   ```

---

### 1‑2 环境配置（Flake8 & pytest）

#### 1‑2‑1 创建 Flake8 配置

**目的**：自定义静态检查，忽略虚拟环境等目录，并放宽行宽。

```bash
cat > .flake8 <<'FLAKE'
[flake8]
exclude = .venv,.git,__pycache__,.pytest_cache,.mypy_cache,build,dist,*.egg-info
max-line-length = 120
FLAKE
```

#### 1‑2‑2 创建 pytest 配置

**目的**：简化测试输出并确保项目根路径被添加到 `PYTHONPATH`。

```bash
cat > pytest.ini <<'PY'
[pytest]
pythonpath = .
addopts = -v --tb=short
PY
```

---

### 1‑3 单元测试（`tests/test_build_ac.py`）

#### 1‑3‑1 创建测试文件

**目的**：验证脚本的去重与顺序保留逻辑。

**操作**：

1. 在 `tests` 目录 **新建** `test_build_ac.py`，输入以下内容：

   ```python
   from pathlib import Path
   from scripts.build_ac import load_words
   
   
   def test_load_words_dedupe(tmp_path: Path):
       sample = tmp_path / "raw.txt"
       sample.write_text("\n".join(["abc", "abc", "xyz"]), encoding="utf‑8")
       words = load_words(sample)
       assert words == ["abc", "xyz"]
   ```

---

### 1‑4 运行脚本并提交 PR

#### 1‑4‑1 生成清洗后词库

```bash
python scripts/build_ac.py --in data/sensitive_raw.txt --out data/sensitive.txt
```

预期输出：

```text
Saved 3 unique words ➜ data/sensitive.txt
```

#### 1‑4‑2 运行测试与静态检查

```bash
pytest -q
flake8
```

全部通过即表示脚本与测试就绪。

#### 1‑4‑3 提交并推送功能分支

```bash
git switch -c build-ac-dict
git add .
git commit -m "feat: add sensitive-word dedup script"
git push -u origin build-ac-dict
# GitHub: 创建 PR ➜ base: dev ← compare: build-ac-dict
# 等 CI 绿灯 ➜ Merge ➜ Delete branch
# 本地: git switch dev && git branch -d build-ac-dict
```

---

## 2 附加说明

* **为何保留原有顺序？** 便于人工比对与版本差异审计；若将来需要最长优先匹配，可再排序。
* **顺序去重实现**：`list` 存储出现顺序 + `set` 判重，时间复杂度 O(n)。
* **热更新**：脚本输出文件名固定，AC 服务可通过热加载替换词库。

---

## 总结

1. `build_ac.py` 去重并保留顺序，输出标准词库。
2. 新增 `scripts/__init__.py`、`.flake8`、`pytest.ini`，确保可导入与代码规范。
3. 单元测试覆盖核心逻辑，CI 静态检查 + 测试通过。
4. PR 合并至 `dev`，远程与本地功能分支已清理。

> **一句话总结**：去重且可追溯的词库，为后续 AC 高效匹配奠定基础。

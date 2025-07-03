# Phase A 步骤 4 — 数据库批量扫描脚本

> 本笔记讲解 **Phase A · Step 4**：编写批处理脚本，扫描 `dzx_forum_post` 表，提取含敏感词的文本片段，为后续标注与模型训练做数据准备。本步骤实现了高效的大规模数据处理能力，支持400万+记录的快速扫描。

---

## 1 实操步骤

> 四块内容：**依赖安装 → 脚本代码 → 字段校验测试 → 运行与提交**。每一步先写 *目的*，再写 *操作*。本步骤重点关注大规模数据处理的性能和稳定性。

### 1‑1 依赖安装

```bash
pip install mysql-connector-python
pip install tqdm
pip install python-dotenv      # 从 .env 读取 DB_DSN 等配置

# 锁定依赖
pip freeze | grep -E "(mysql-connector-python|tqdm)" >> requirements.txt

# 手动更新 requirements-dev.txt 增加 python-dotenv==1.1.1
```

> **mysql-connector-python**：官方 MySQL 驱动，提供稳定的数据库连接
> **tqdm**：进度条显示，实时监控扫描进度
> **python-dotenv**：在项目根目录读取 **`.env`**，避免把账号密码硬编码进脚本或仓库，提高安全性

实例 `.env.example` 内容如下：

```ini
# 数据库连接配置
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=password
DB_NAME=db_name
DB_CHARSET=utf8mb4 
```

---

### 1‑2 更新 AC 自动机敏感词匹配核心

#### 1‑2‑1 修改 `src/ac_core.py`

```python
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

```

#### 1‑2‑2 文件职责和主要处理逻辑

**文件职责和作用：**

- 提供基于 AC 自动机的高效敏感词检测功能，时间复杂度 O(n+m)
- 支持词典加载、批量查找、智能过滤等功能
- 作为整个内容过滤系统的核心算法实现，为后续处理提供基础

**主要处理逻辑：**

- 使用 `pyahocorasick` 库实现 AC 自动机算法
- 支持从文件加载敏感词词典（每行一个词）
- 自动跳过 **base64** 编码内容中的敏感词匹配
- 自动跳过 **URL** 内容中的敏感词匹配
- 提供 `find()` 方法返回所有匹配结果
- 提供 `has_match()` 方法快速判断是否存在敏感词

### 1‑3 批量扫描脚本

#### 1‑3‑1 创建文件 `scripts/batch_scan.py`

```python
"""
批量扫描论坛帖子，提取敏感词上下文

此脚本用于扫描数据库中的论坛帖子，检测敏感词并提取相关上下文信息。
支持分批处理大量数据，避免内存溢出。
"""

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator

from mysql.connector import connect
from tqdm import tqdm
from dotenv import load_dotenv

try:
    from src.ac_core import SensitiveMatcher
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.ac_core import SensitiveMatcher


def get_db_config() -> dict[str, str | int]:
    """从环境变量获取数据库连接配置"""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '3306')),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'itheimabbs201408'),
        'charset': os.getenv('DB_CHARSET', 'utf8mb4')
    }


@contextmanager
def get_database_connection():
    """数据库连接上下文管理器"""
    conn = None
    try:
        conn = connect(**get_db_config())
        yield conn
    finally:
        if conn:
            conn.close()


def get_total_records(cursor, limit: int | None = None) -> int:
    """获取总记录数"""
    if limit:
        return limit

    cursor.execute("SELECT COUNT(*) FROM dzx_forum_post")
    result = cursor.fetchone()

    if not result:
        return 0

    # 处理不同的返回格式
    if isinstance(result, dict):
        return list(result.values())[0]
    return result[0]


def iter_rows(cursor, chunk_size: int, limit: int | None = None) -> Iterator[dict[str, str]]:
    """分批迭代数据库行数据"""
    sql = "SELECT pid, author, authorid, subject, message, useip FROM dzx_forum_post"
    if limit:
        sql += f" LIMIT {limit}"

    cursor.execute(sql)
    columns = [col[0] for col in cursor.description]

    while rows := cursor.fetchmany(chunk_size):
        for row in rows:
            # 确保返回字典格式
            if not isinstance(row, dict):
                row = dict(zip(columns, row))
            yield row


class SimpleOutputFile:
    """简单输出文件管理器"""

    def __init__(self, file_path: Path):
        self.file_path = file_path

        # 确保输出目录存在
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # 创建文件并写入头部
        self.file = file_path.open("w", encoding="utf-8")
        self.file.write("pid\tauthor\tauthorid\tuseip\thit_word\tcontext\n")

    def write_match(self, row: dict[str, str], word: str, context: str) -> None:
        """写入匹配结果"""
        self.file.write(
            f"{row.get('pid', '')}\t{row.get('author', '')}\t"
            f"{row.get('authorid', '')}\t{row.get('useip', '')}\t"
            f"{word}\t{context}\n"
        )

    def close(self) -> None:
        """关闭文件"""
        if self.file:
            self.file.close()


def process_text_field(text: str, matcher: SensitiveMatcher) -> Iterator[tuple[str, str]]:
    """处理文本字段，返回匹配的敏感词和上下文"""
    text = str(text or "")
    matches = list(matcher.find(text))

    for word, start, end in matches:
        # 提取上下文（前后20个字符）
        context_start = max(0, start - 20)
        context_end = min(len(text), end + 20)
        context = text[context_start:context_end]

        # 处理换行符 - 统一转换为空格，保持可读性
        context = context.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        # 清理多余空格
        context = ' '.join(context.split())

        yield word, context


def process_single_post(row: dict[str, str], matcher: SensitiveMatcher) -> list[tuple[str, str]]:
    """处理单个帖子的敏感词匹配"""
    matches = []
    text_fields = ["subject", "message"]

    for field in text_fields:
        try:
            text = row.get(field, "")
            for word, context in process_text_field(text, matcher):
                matches.append((word, context))
        except Exception as e:
            # 记录错误但继续处理
            print(f"处理字段 {field} 时出错 (pid={row.get('pid', 'unknown')}): {e}")
            continue

    return matches


def scan_posts(
    cursor,
    matcher: SensitiveMatcher,
    output_file: SimpleOutputFile,
    chunk_size: int,
    limit: int | None = None
) -> tuple[int, int]:
    """扫描帖子主流程"""
    total_records = get_total_records(cursor, limit)
    scanned_count = 0
    hit_count = 0

    for row in tqdm(
        iter_rows(cursor, chunk_size, limit),
        desc="扫描进度",
        total=total_records,
        unit="条",
        unit_scale=True,
        unit_divisor=1000
    ):
        scanned_count += 1

        try:
            # 处理单个帖子
            matches = process_single_post(row, matcher)

            # 写入匹配结果
            for word, context in matches:
                output_file.write_match(row, word, context)
                hit_count += 1

        except Exception as e:
            # 记录错误但继续处理
            print(f"处理帖子时出错 (pid={row.get('pid', 'unknown')}): {e}")
            continue

        # 定期更新进度信息
        if scanned_count % 1000 == 0:
            tqdm.write(f"已扫描: {scanned_count}/{total_records} | 命中: {hit_count}")

    return scanned_count, hit_count


def main() -> None:
    """主函数：批量扫描论坛帖子"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="批量扫描论坛帖子中的敏感词")
    parser.add_argument("--out", default="data/tsv/batch_scan.tsv", help="输出文件路径")
    parser.add_argument("--chunk", type=int, default=1000, help="每批处理的行数")
    parser.add_argument("--test", type=int, help="测试模式：指定扫描记录数量")
    args = parser.parse_args()

    # 加载环境变量
    load_dotenv()

    # 初始化敏感词匹配器
    matcher = SensitiveMatcher("data/sensitive.txt")

    # 设置扫描限制
    limit = args.test if args.test else None

    try:
        with get_database_connection() as conn:
            with conn.cursor(dictionary=True, buffered=False) as cursor:
                # 创建输出文件
                output_file = SimpleOutputFile(Path(args.out))

                try:
                    # 执行扫描
                    scanned_count, hit_count = scan_posts(
                        cursor, matcher, output_file, args.chunk, limit
                    )

                    # 输出最终统计
                    print("\n扫描完成！")
                    print(f"总扫描记录: {scanned_count}")
                    print(f"总命中次数: {hit_count}")
                    print(f"输出文件: {args.out}")

                finally:
                    output_file.close()

    except Exception as e:
        print(f"❌ 数据库连接或初始化错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

```

#### 1‑3‑2 文件职责和主要处理逻辑

**文件职责和作用：**

- 批量扫描数据库中的论坛帖子内容，支持400万+记录处理
- 提取敏感词上下文信息并保存到文件，为后续标注提供数据
- 支持大规模数据处理和进度显示，确保处理过程可监控

**主要处理逻辑：**

- 从环境变量读取数据库连接配置，支持灵活的部署环境
- 分批迭代数据库记录，避免内存溢出，支持大规模数据处理
- 对每个帖子的 **subject** 和 **message** 字段进行敏感词检测
- 提取敏感词前后 20 个字符作为上下文，平衡信息完整性和文件大小
- 提供测试模式，支持小规模验证和调试
- 实时显示扫描进度和统计信息，提供处理过程的可视化监控

---

### 1‑4 字段校验测试

1. 新建 `tests/conftest.py` 并输入以下内容，提供通用的测试 fixtures 和配置，供所有测试文件使用：

   ```python
   import os
   import pytest
   import mysql.connector
   from dotenv import load_dotenv
   
   
   @pytest.fixture(scope="session", autouse=True)
   def load_env():
       load_dotenv()
   
   
   @pytest.fixture(scope="session")
   def db_config():
       return {
           'host': os.getenv('DB_HOST', 'localhost'),
           'port': int(os.getenv('DB_PORT', '3306')),
           'user': os.getenv('DB_USER', 'root'),
           'password': os.getenv('DB_PASSWORD', ''),
           'database': os.getenv('DB_NAME', 'bbsdb'),
           'charset': os.getenv('DB_CHARSET', 'utf8mb4')
       }
   
   
   @pytest.fixture(scope="session")
   def db_connection(db_config):
       conn = mysql.connector.connect(**db_config)
       yield conn
       conn.close()
   
   
   @pytest.fixture
   def db_cursor(db_connection):
       cursor = db_connection.cursor()
       yield cursor
       cursor.close()
   
   ```

   主要处理逻辑如下：

   - 自动加载 `.env` 环境变量文件
   - 提供数据库连接配置 fixture
   - 提供数据库连接对象 fixture（session 级别）
   - 提供数据库游标 fixture（每个测试用例级别）
   - 自动管理数据库连接的创建和关闭

2. 新建 `tests/test_batch_scan_fields.py` 并输入以下内容，测试数据表中的字段是否正确：

   ```python
   # 内容过滤功能所需的必要数据库字段
   import pytest
   
   REQUIRED_FIELDS = {"pid", "author", "authorid", "subject", "message", "useip"}
   
   
   @pytest.mark.integration
   def test_fields_exist(db_cursor):
       """
       测试数据库表是否包含所有必需字段
   
       Args:
           db_cursor: 数据库游标 fixture
       """
       # 查询表结构
       db_cursor.execute("SHOW COLUMNS FROM dzx_forum_post")
   
       # 提取列名并转换为字符串集合
       cols = {str(row[0]) for row in db_cursor.fetchall()}  # type: ignore
   
       # 检查缺失的字段
       missing = REQUIRED_FIELDS - cols
   
       # 断言所有必需字段都存在
       assert not missing, f"缺少字段: {missing}"
   
   ```

---

### 1‑5 运行与提交

#### 1‑5‑1 执行扫描脚本

```bash
python scripts/batch_scan.py
```

- **单机 5 分钟**：在实际测试中，4044030 条记录约 5 分钟扫描完成，处理速度约 13,480 条/秒
- **命中统计**：642224 次敏感词命中，命中率约 15.9%，脚本自动生成的 TSV 文件保存在 `data/tsv` 目录下

#### 1‑5‑2 运行测试与静态检查

```bash
pytest -q tests/test_batch_scan_fields.py
flake8 scripts/batch_scan.py
```

#### 1‑5‑3 提交 PR

```bash
git switch -c db-batch-scan
git add scripts tests requirements.txt data/.gitkeep
git commit -m "feat: add db batch scan script"
git push -u origin db-batch-scan
# GitHub: 创建 PR ➜ base: dev ← compare: db-batch-scan
# CI 全绿 ➜ Merge ➜ Delete branch
# 本地: git switch dev && git branch -d db-batch-scan
```

---

## 2 附加说明

- **单进程 vs 多进程**：先单进程，如速度不足再用 `multiprocessing.Pool` 或分段跑多实例。当前单进程性能已满足需求。
- **长文本截断**：命中词前后各 20 字已足够人工判别上下文，同时避免 TSV 过大。可根据实际需求调整上下文长度。
- **后续标注**：TSV 可导入 Excel、Label Studio 等工具；若需 JSONL，可在脚本末尾更换 `writer`。支持多种标注工具格式。

---

## 总结

1. 单进程流式扫描 `dzx_forum_post`，输出 TSV 样本，支持400万+记录的高效处理。
2. 进度条实时可视，方便估算剩余时间，提供处理过程的可视化监控。
3. 字段校验测试确保数据库结构满足预期，保证数据处理的可靠性。
4. PR 合并后，即可进行人工标注或 LoRA 训练数据准备，为后续模型训练奠定基础。

> **一句话总结**：快速、可追溯地批量提取命中文本，为模型训练奠定数据基础。本步骤实现了高效的大规模数据处理能力，为整个内容过滤系统提供了坚实的数据支撑。

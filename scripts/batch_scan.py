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

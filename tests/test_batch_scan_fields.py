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

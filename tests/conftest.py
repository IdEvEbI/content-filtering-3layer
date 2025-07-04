import os
import pytest
from dotenv import load_dotenv

# 条件导入，只在需要时导入 mysql.connector
try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False


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
    if not MYSQL_AVAILABLE:
        pytest.skip("mysql.connector not available")
    conn = mysql.connector.connect(**db_config)
    yield conn
    conn.close()


@pytest.fixture
def db_cursor(db_connection):
    cursor = db_connection.cursor()
    yield cursor
    cursor.close()

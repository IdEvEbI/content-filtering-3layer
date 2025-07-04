# 企业级项目测试最佳实践

## 测试分层策略

### 1. 单元测试 (Unit Tests)

**目标：** 测试独立的代码单元，快速反馈
**特点：**

- 运行速度快（毫秒级）
- 不依赖外部服务
- 每次代码提交都运行
- 覆盖核心业务逻辑

**示例：**

```python
# tests/unit/test_ac_core.py
def test_sensitive_matcher_find():
    matcher = SensitiveMatcher("test_dict.txt")
    result = matcher.find("测试文本")
    assert len(result) == 0

def test_sensitive_matcher_skip_base64():
    matcher = SensitiveMatcher("test_dict.txt")
    base64_text = "SGVsbG8gV29ybGQ="  # base64编码
    result = matcher.find(base64_text)
    assert len(result) == 0  # 应该跳过base64内容
```

### 2. 集成测试 (Integration Tests)

**目标：** 测试组件间的交互
**特点：**

- 运行速度中等（秒级）
- 依赖外部服务（数据库、API等）
- 在集成测试阶段运行
- 验证系统集成正确性

**当前文件定位：**

```python
# tests/integration/test_database_schema.py (建议重命名)
def test_required_fields_exist():
    """验证数据库表结构是否包含必需字段"""
    # 这是一个集成测试，依赖数据库连接
```

### 3. 端到端测试 (E2E Tests)

**目标：** 测试完整的用户流程
**特点：**

- 运行速度慢（分钟级）
- 测试完整业务流程
- 在部署前运行
- 验证系统整体功能

## 推荐的测试目录结构

```ini
tests/
├── unit/                   # 单元测试
│   ├── test_ac_core.py     # 核心算法测试
│   └── test_utils.py       # 工具函数测试
├── integration/            # 集成测试
│   ├── test_database.py    # 数据库集成测试
│   └── test_api.py         # API集成测试
├── e2e/                    # 端到端测试
│   └── test_batch_scan.py  # 完整流程测试
├── fixtures/               # 测试数据
│   ├── test_data.json
│   └── sample_sensitive.txt
└── conftest.py             # 共享配置
```

## CI/CD 流水线配置

### 1. 开发阶段 (Development)

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Unit Tests
        run: |
          pip install -r requirements-dev.txt
          pytest tests/unit/ -v --cov=src
```

### 2. 集成测试阶段 (Integration)

```yaml
  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    services:
      mysql:
        image: mysql:8.0
        env:
          MYSQL_ROOT_PASSWORD: test
          MYSQL_DATABASE: testdb
    steps:
      - name: Run Integration Tests
        run: |
          pytest tests/integration/ -v
```

### 3. 部署前测试 (Pre-deployment)

```yaml
  e2e-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    steps:
      - name: Run E2E Tests
        run: |
          pytest tests/e2e/ -v
```

## 测试数据管理

### 1. 测试数据策略

```python
# tests/fixtures/test_data.py
import pytest
from pathlib import Path

@pytest.fixture
def sample_sensitive_dict():
    """提供测试用的敏感词词典"""
    return Path("tests/fixtures/sample_sensitive.txt")

@pytest.fixture
def test_database_schema():
    """提供测试数据库结构"""
    return {
        'table_name': 'test_posts',
        'required_fields': ['pid', 'author', 'subject', 'message']
    }
```

### 2. 环境隔离

```python
# tests/conftest.py
import os
import pytest

@pytest.fixture(scope="session", autouse=True)
def test_environment():
    """设置测试环境"""
    os.environ['TESTING'] = 'true'
    os.environ['DB_NAME'] = 'testdb'
    yield
    # 清理测试环境
```

## 最佳实践建议

### 1. 测试命名规范

- 单元测试：`test_<function_name>_<scenario>`
- 集成测试：`test_<component>_<integration_point>`
- E2E测试：`test_<user_story>_<workflow>`

### 2. 测试覆盖率目标

- 单元测试：80%+
- 集成测试：关键路径100%
- E2E测试：核心业务流程100%

### 3. 测试执行策略

- **开发时**：只运行单元测试（快速反馈）
- **提交时**：运行单元测试 + 代码质量检查
- **合并时**：运行所有测试 + 安全扫描
- **部署前**：运行完整测试套件

### 4. 当前项目改进建议

1. **重命名文件**：

   ```bash
   mv tests/test_batch_scan_fields.py tests/integration/test_database_schema.py
   ```

2. **添加单元测试**：

   ```python
   # tests/unit/test_ac_core.py
   def test_sensitive_matcher_initialization():
       matcher = SensitiveMatcher("test_dict.txt")
       assert matcher is not None
   ```

3. **配置测试环境**：

   ```python
   # pytest.ini
   [pytest]
   testpaths = tests
   python_files = test_*.py
   python_classes = Test*
   python_functions = test_*
   addopts = -v --tb=short
   ```

4. **分离测试配置**：

   ```python
   # tests/conftest.py
   @pytest.fixture(scope="session")
   def test_db_config():
       """测试数据库配置"""
       return {
           'host': 'localhost',
           'database': 'testdb',
           # ... 其他测试配置
       }
   ```

## 总结

当前的 `test_batch_scan_fields.py` 作为集成测试是合理的，但建议：

1. **明确测试类型**：将其归类为集成测试
2. **优化执行时机**：不在每次提交时运行
3. **补充单元测试**：为核心逻辑添加快速单元测试
4. **完善测试结构**：按测试类型组织目录结构

这样既能保证代码质量，又能提高开发效率。

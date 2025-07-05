# 项目安装指南

## 环境要求

- **操作系统**: macOS (darwin 24.5.0)
- **Python版本**: 3.12
- **Shell**: /bin/zsh
- **项目路径**: `/.../content-filtering-3layer`

## 安装步骤

### 1. 检查Python版本

```bash
# 检查当前Python版本
python3 --version

# 如果没有Python 3.12，使用homebrew安装
brew install python@3.12
```

### 2. 进入项目目录

```bash
cd /.../content-filtering-3layer
```

### 3. 创建虚拟环境

```bash
# 创建Python 3.12虚拟环境
python -m venv .venv
```

### 4. 激活虚拟环境

```bash
# 激活虚拟环境
source .venv/bin/activate

# 验证激活成功（命令提示符前应该显示(.venv)）
which python
```

### 5. 升级pip

```bash
# 升级pip到最新版本
pip install --upgrade pip
```

### 6. 安装生产依赖

```bash
# 安装项目运行所需的核心依赖
pip install -r requirements.txt
```

**核心依赖包括**:

- `pyahocorasick==2.0.0` - AC自动机算法
- `fastapi==0.115.14` - Web框架
- `httpx==0.28.1` - HTTP客户端
- `uvicorn==0.35.0` - ASGI服务器
- `mysql-connector-python==9.3.0` - MySQL连接器
- `tqdm==4.67.1` - 进度条

### 7. 安装开发依赖

```bash
# 安装开发工具和测试依赖
pip install -r requirements-dev.txt
```

**开发依赖包括**:

- `flake8==7.3.0` - 代码风格检查
- `pytest==8.4.1` - 测试框架
- `python-dotenv==1.1.1` - 环境变量管理
- 其他代码质量工具

## 验证安装

### 1. 验证Python版本

```bash
python --version
# 应该显示: Python 3.12.x
```

### 2. 检查已安装的包

```bash
pip list
```

### 3. 验证核心依赖

```bash
# 测试核心模块导入
python -c "import pyahocorasick, fastapi, tqdm; print('✅ 核心依赖安装成功')"
```

### 4. 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_ac_core.py -v
```

## 环境配置

### 1. 环境变量配置

```bash
# 复制环境变量模板
cp env.example .env

# 编辑环境变量（根据需要修改）
nano .env
```

### 2. 激活虚拟环境（每次使用前）

```bash
# 在项目根目录执行
source .venv/bin/activate
```

### 3. 退出虚拟环境

```bash
deactivate
```

## 常见问题

### 问题1: Python版本不匹配

**错误**: `python3.12: command not found`

**解决方案**:

```bash
# 安装Python 3.12
brew install python@3.12

# 或者使用pyenv
pyenv install 3.12.0
pyenv local 3.12.0
```

### 问题2: 依赖安装失败

**错误**: `ERROR: Could not find a version that satisfies the requirement`

**解决方案**:

```bash
# 升级pip
pip install --upgrade pip

# 清理缓存
pip cache purge

# 重新安装
pip install -r requirements.txt
```

### 问题3: 权限问题

**错误**: `Permission denied`

**解决方案**:

```bash
# 使用用户安装
pip install --user -r requirements.txt
```

## 开发工作流

### 1. 日常开发流程

```bash
# 1. 激活虚拟环境
source .venv/bin/activate

# 2. 运行代码风格检查
flake8 src/ scripts/ tests/

# 3. 运行测试
pytest tests/ -v

# 4. 运行项目
python ac_service/main.py
```

### 2. 代码质量检查

```bash
# 代码风格检查
flake8 src/ scripts/ tests/

# 类型检查（如果使用mypy）
mypy src/

# 运行所有检查
pytest tests/ --flake8 --mypy
```

## 项目结构

```text
content-filtering-3layer/
├── .venv/                      # 虚拟环境目录
├── ac_service/                 # AC服务
├── data/                       # 数据目录
├── docs/                       # 文档目录
│   ├── note/                   # 开发笔记
│   └── *.md                    # ChatGPT编写的文档
├── scripts/                    # 脚本目录
├── src/                        # 源代码
├── tests/                      # 测试文件
├── requirements.txt            # 生产依赖
├── requirements-dev.txt        # 开发依赖
└── .env                        # 环境变量
```

---

**注意**: 每次重新克隆项目后，都需要重新执行上述安装步骤。

**最后更新**: 2025-07-02

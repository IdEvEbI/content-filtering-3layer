# Phase A 步骤 3 — AC FastAPI 服务

> 本笔记讲解 **Phase A · Step 3**：基于前一步的 `SensitiveMatcher`，封装可供外部调用的 **FastAPI 服务**，并以 Docker 镜像形式交付。

---

## 1 实操步骤

> 四块内容：**依赖安装 → 服务代码 → 单元 / 集成测试 → Docker 化与提交**。

### 1‑1 依赖安装

**目的**：引入 REST 框架、ASGI 服务器以及 HTTP 客户端（用于集成测试）。

```bash
# 在虚拟环境中安装运行依赖
pip install fastapi
pip install "uvicorn[standard]"
pip install httpx

# 将运行依赖锁定到文件，确保 CI 环境一致
pip freeze | grep -E "(fastapi|uvicorn|httpx)" >> requirements.txt
```

> **FastAPI**：现代高性能 Web 框架
> **Uvicorn**：ASGI 服务器，运行 FastAPI 应用
> **httpx**：异步/同步皆可的 HTTP 客户端，用于编写集成测试

---

### 1‑2 服务代码（`ac_service/main.py`）

#### 1‑2‑1 创建文件

**目的**：提供 `POST /ac-match` 接口，返回命中情况。

```python
"""FastAPI service wrapping SensitiveMatcher."""
from fastapi import FastAPI
from pydantic import BaseModel
import os

from src.ac_core import SensitiveMatcher

DICT_PATH = os.getenv("DICT_PATH", "data/sensitive.txt")
matcher = SensitiveMatcher(DICT_PATH)

app = FastAPI(title="AC Match Service", version="0.1.0")


class Req(BaseModel):
    text: str


@app.post("/ac-match")
def ac_match(req: Req):
    hits = [
        {"word": w, "start": s, "end": e}
        for w, s, e in matcher.find(req.text)
    ]
    return {
        "matched": bool(hits),
        "hit_count": len(hits),
        "hits": hits,
    }

```

> **依赖注入技巧**：`matcher` 是全局单例，利用 FastAPI 的生命周期函数也可延迟加载；此处简化直接在模块级初始化。

---

### 1‑3 单元 / 集成测试（`tests/test_ac_api.py`）

```python
from fastapi.testclient import TestClient
from ac_service.main import app

client = TestClient(app)


def test_ac_match_hit():
    resp = client.post("/ac-match", json={"text": "包含 大boss 敏感词"})
    data = resp.json()
    assert resp.status_code == 200
    assert data["matched"] is True
    assert any(hit["word"] == "大boss" for hit in data["hits"])


def test_ac_match_no_hit():
    resp = client.post("/ac-match", json={"text": "safe text"})
    data = resp.json()
    assert resp.status_code == 200
    assert data == {"matched": False, "hit_count": 0, "hits": []}

```

---

### 1‑4 Docker 化与提交

#### 1‑4‑1 本地验证

```bash
# 启动服务 (热重载)
uvicorn ac_service.main:app --reload --port 8001

# 另终端测试
curl -X POST http://127.0.0.1:8001/ac-match \
  -H "Content-Type: application/json" \
  -d '{"text":"586 大boss test"}'
```

#### 1‑4‑2 创建 Dockerfile

```Dockerfile
FROM python:3.12-alpine
WORKDIR /app

# 安装编译依赖（pyahocorasick 需要）
RUN apk add --no-cache build-base gcc musl-dev

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8001
CMD ["uvicorn", "ac_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

> **镜像说明**：采用 `alpine` 基础镜像 + 编译工具链，镜像体积更小；`--no-cache-dir` 减少层大小。

#### 1‑4‑3 运行测试 & 构建镜像

```bash
pytest -q
flake8

# 构建镜像
docker build -t ac-match-service:0.1 .
# 查看镜像大小
docker images ac-match-service:0.1

# 启动容器
docker run -d -p 8001:8001 --name ac-match-service ac-match-service:0.1
sleep 5  # 等待服务就绪

# 检查容器状态
docker ps | grep ac-match-service

# 测试 API
curl -s -X POST http://localhost:8001/ac-match \
  -H "Content-Type: application/json" \
  -d '{"text":"包含 586 大boss 敏感词"}' | jq .

curl -s -X POST http://localhost:8001/ac-match \
  -H "Content-Type: application/json" \
  -d '{"text":"这是安全的文本内容"}' | jq .

# 查看容器资源占用（可选）
docker stats ac-match-service --no-stream

# 停止并清理
docker stop ac-match-service && docker rm ac-match-service
```

> 可选：新建 `.dockerignore`，排除 `.venv/`, `.git/`, `__pycache__/` 等目录，加速构建并减小镜像体积。

#### 1‑4‑4 提交 PR

```bash
git switch -c ac-service-api
git add .
git commit -m "feat: add ac match fastapi service"
git push -u origin ac-service-api
# GitHub: 创建 PR ➜ base: dev ← compare: ac-service-api
# CI 全绿 ➜ Merge ➜ Delete branch
# 本地: git switch dev && git branch -d ac-service-api
```

---

## 2 附加说明

- **端口与路由**：默认 `8001` 与 `/ac-match`，如需调整可在环境变量或启动参数中覆盖。
- **响应字段**：包含 `hit_count` 方便前端快速展示命中数量；如需 `elapsed_ms` 可在代码中计算并返回。
- **白名单过滤**：如果后续需要，可在 `matcher.find` 结果返回前进行二次过滤。

---

## 总结

1. FastAPI 服务封装 `SensitiveMatcher`，提供 `/ac-match` JSON API。
2. 集成测试验证命中与否；Dockerfile 实现可移植部署。
3. PR 合并至 `dev` 后，即可被其他业务系统通过 HTTP 调用。

> **一句话总结**：把高性能 AC 匹配包装成 REST 服务，为系统集成打通最后一公里。

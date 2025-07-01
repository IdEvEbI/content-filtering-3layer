# Phase A 步骤 0 — 初始化仓库与 CI 流程

> 本笔记讲解 **Phase A · Step 0**：创建 GitHub 仓库、配置本地开发环境并跑通最小 CI 流程。完成后，你将拥有一套符合 DevOps 流程的起始代码库，可支撑后续迭代。

---

## 1 实操步骤

> 整体分为三大块：**Git 仓库设置 → 本地环境设置 → PR 与分支流程**。每段先说明 *目的*，再给出 *操作* 命令。

### 1‑1 Git 仓库设置

#### 1‑1‑1 初始化远程仓库

**目的**：统一代码源，预设忽略规则，明确开源许可。

**操作**：

```bash
# GitHub Web 操作
# 1. 点击 New ➜ Repository name: content-filtering-3layer
# 2. 选择 MIT License
# 3. 勾选 Add README 与 Python .gitignore
```

---

#### 1‑1‑2 保护 main 并创建 dev 分支

**目的**：`main` 仅保存稳定发布代码；`dev` 作为日常集成分支。

**操作**：

```bash
# 保护 main（GitHub Web）
# Settings ➜ Branches ➜ Add rule ➜ Branch name pattern: main
# 勾选 Require status checks to pass before merging

# 本地克隆并推送 dev 分支
git clone git@github.com:<org>/content-filtering-3layer.git
cd content-filtering-3layer
git switch -c dev
git push -u origin dev
```

---

#### 1‑1‑3 创建功能分支

**目的**：保证每个功能在独立分支上开发，便于 Review 与回滚。

**操作**：

```bash
git switch -c init-repo        # 从 dev 切出
```

---

### 1‑2 本地环境设置

#### 1‑2‑1 规划目录结构

**目的**：一次性确定顶层目录，避免后期频繁移动文件，并对每个目录的职责做到心中有数。

**操作**：

```bash
mkdir -p src ac_service semantic_service scripts tests data bench
touch src/__init__.py
```

**目录职责简述**：

* **src/** — 项目公共代码根包，后续可在此暴露统一 API。
* **ac\_service/** — 敏感词 AC 自动机服务代码。
* **semantic\_service/** — LoRA‑BERT 语义过滤服务代码。
* **scripts/** — 一次性或运维脚本（如词库清洗）。
* **tests/** — pytest 单元测试代码。
* **data/** — 小规模样例数据 / 词库原始文件。
* **bench/** — 压测与性能基准脚本。

---

#### 1‑2‑2 配置 Python 3.12 环境

**目的**：确保本地 Python 版本与 CI 保持一致；安装基础开发依赖。

**操作**：

```bash
pyenv install 3.12.11          # 如未安装
pyenv local 3.12.11
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pytest flake8
pip freeze > requirements-dev.txt
```

> flake8 = *静态代码分析*，提前发现 PEP 8 与潜在错误；pytest = *单元测试框架*。

---

#### 1‑2‑3 添加占位测试

**目的**：让 pytest 能通过，验证 CI 流程。

**操作**：

1. 在 `tests` 目录 **新建** `test_placeholder.py` 文件。
2. 输入以下内容并保存：

   ```python
   def test_placeholder():
       assert True
   ```

---

#### 1‑2‑4 创建 GitHub Actions 工作流

**目的**：在 PR / push 时自动运行 flake8 + pytest，形成质量门禁。

**操作**：

1. 新建目录 `.github/workflows`（如不存在）。
2. 在其中 **新建** `ci.yml`，输入以下内容并保存：

   ```yaml
   name: CI

   on:
     pull_request:
       branches: [dev, main]
     push:
       branches: [dev, main]

   jobs:
     build:
       runs-on: ubuntu-latest
       strategy:
         matrix:
           python-version: ["3.12"]
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with:
             python-version: ${{ matrix.python-version }}
         - name: Install deps
           run: |
             python -m pip install --upgrade pip
             pip install -r requirements-dev.txt
         - name: Lint & Test
           run: |
             flake8
             pytest -q
   ```

---

### 1‑3 PR 与分支流程

#### 1‑3‑1 提交并推送功能分支

**目的**：触发远端 CI，开始代码审查。

**操作**：

```bash
git add .
git commit -m "feat: init repo skeleton & CI"
git push -u origin init-repo
```

---

#### 1‑3‑2 创建 Pull Request

**目的**：将功能分支合并到 **dev**，记录讨论历史。

**操作**：

```bash
# GitHub Web
# Compare & pull request ➜ base: dev ← compare: init-repo
# 标题：feat: init repo skeleton & CI
# 确认 CI 全绿 ➜ Merge pull request ➜ Delete branch
```

---

#### 1‑3‑3 清理本地功能分支

**目的**：防止本地积累过多过期分支。

**操作**：

```bash
git switch dev
git branch -d init-repo        # 本地删除已合并分支
```

> **分支模型回顾**：功能分支 ➜ PR ➜ merge **dev**（集成）➜ 周期性 Release ➜ PR ➜ merge **main**（发布）➜ 自动部署。

---

## 2 附加说明

* **flake8 插件**：可扩展 `flake8-bugbear` / `flake8-comprehensions`，在 `setup.cfg` 自定义规则。
* **工作流模板**：GitHub *Starter Workflows* 可生成初始 YAML；本地可用 `act` 调试。
* **模块重导出技巧**：未来在 `src/__init__.py` 内重导出核心函数，让外部调用路径简洁。

---

## 总结

1. 完成仓库初始化并确定分支策略（main 受保护、dev 集成）。
2. 建立本地 Python 3.12 环境，添加占位测试。
3. 配置 GitHub Actions，实现 flake8 + pytest 双门禁。
4. 跑通第一次 PR 流程，合并到 dev 并清理分支。

> **一句话总结**：分支模型 + 自动质量门禁 = 稳健的 DevOps 起跑线。

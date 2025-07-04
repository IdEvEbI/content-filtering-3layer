# Phase B 步骤 0 — 数据集整理与标注启动

> 本笔记讲解 **Phase B · Step 0**：从 AC 命中 TSV 中抽样，制作三分类标注数据（0 合规 · 1 疑似 · 2 违规），并在 Label Studio 中完成首批 3 k train / 3 k eval 标注，为后续 LoRA‑BERT 微调做准备。

---

## 1 实操步骤

> 四块内容：**依赖安装 → 数据抽样脚本 → Label Studio 标注 → 提交 PR**。

### 1‑1 依赖安装

建议在虚拟环境中安装依赖，确保环境隔离：

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas label-studio

# 锁定依赖
pip freeze | grep -E "(pandas|label-studio)" >> requirements.txt
```

- **pandas**：用于数据抽样、格式转换等数据处理任务
- **label-studio**：Web 标注平台（推荐用 Docker 部署，见下文）

---

### 1‑2 数据抽样脚本（`scripts/tsv2jsonl.py`）

#### 1‑2‑1 导入基础数据

1. 将大模型检测得到的 **疑似违规** 数据（如 LLM 诊断结果）导入到 `data/llm_diagnosis_processed/` 目录。
2. 将人工审核确认的 **违规** 数据导入到 `data/human_review_processed/` 目录。

#### 1‑2‑2 创建脚本文件

编写数据采样脚本 `scripts/tsv2jsonl.py`，实现如下采样策略：

1. 包含所有 LLM 诊断数据（violation 标签）
2. 包含所有人工审核数据（violation 标签）
3. 从 `batch_scan` TSV 中随机采样非重叠的 pids 补充到 6000 样本
4. 标签分配：**violation**（人工审核+LLM诊断）、**suspicious**（仅LLM诊断）、**normal**（TSV采样）
5. 按 **80% train** / **10% validation** / **10% test** 比例分割

#### 1‑2‑3 代码逻辑描述与采样数据说明

##### 代码逻辑描述

- 数据加载：分别加载 LLM 诊断、人工审核、批量扫描三类数据
- 样本分类：优先合并人工审核和 LLM 诊断，剩余用批量扫描补齐
- 数据集分割：按标签分层，80/10/10 划分训练/验证/测试集
- 数据保存：输出为 JSONL 格式，并生成统计信息

##### 采样数据说明

- 数据分层：人工审核 > LLM 诊断 > 批量扫描
- 去重机制：人工审核优先，LLM 诊断与批量扫描去重
- 数量控制：总样本数严格等于 6000，若 violation+suspicious 超过则按比例采样

#### 1‑2‑4 运行实例与输出结果

**运行示例：**

```bash
python scripts/tsv2jsonl.py
```

**主要输出文件：**

- `data/annotations/train.jsonl`：训练集
- `data/annotations/validation.jsonl`：验证集
- `data/annotations/test.jsonl`：测试集
- `data/annotations/dataset_stats.json`：统计信息

**标签分布统计（示例）：**

```json
{
  "train": {"normal": 4397, "suspicious": 163, "violation": 239},
  "validation": {"normal": 549, "suspicious": 20, "violation": 29},
  "test": {"normal": 551, "suspicious": 21, "violation": 31}
}
```

---

### 1‑4 提交 PR

**提交 PR 流程：**

```bash
git switch -c semantic-dataset
git add scripts data/annotations/*.jsonl requirements.txt
git commit -m "feat: add initial 6k labeled dataset & tsv→jsonl converter"
git push -u origin semantic-dataset
```

- 在 GitHub 上创建 PR，base: dev ← compare: semantic-dataset
- CI 仅跑 flake8 / pytest，数据文件不影响
- 如有大文件（>50MB）建议用 Git LFS 管理

---

## 2 附加说明

- 已有 300 条种子数据可放入 `data/annotations/seed.jsonl`，与抽样数据一同导入，提升模型早期表现
- 标签指南建议在项目 wiki 维护，明确三分类标准
- 推荐采用活跃学习：先标注 1k，微调 LoRA，模型再打伪标签，人工快审扩充

## 总结

1. 抽样 6k 行，生成 train/validation/test JSONL
2. Label Studio 标注三分类，导出后可直接用于 LoRA 微调
3. PR 合并至 dev，数据与脚本版本化管理，便于后续迭代

> **一句话总结**：先拿 6k 三分类标注，奠定 LoRA 微调数据基线。

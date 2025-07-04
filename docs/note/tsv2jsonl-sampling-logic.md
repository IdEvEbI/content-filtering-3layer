# tsv2jsonl.py 数据采样逻辑说明

> 本文档详细说明 `scripts/tsv2jsonl.py` 的数据采样处理逻辑，以及关键问题的解答。

---

## 1. 脚本功能概述

`tsv2jsonl.py` 是一个数据采样脚本，用于从多个数据源生成训练数据集，主要功能包括：

- 从 LLM 诊断数据、人工审核数据、批量扫描数据中采样
- 按标签分类：violation（违规）、suspicious（疑似）、normal（正常）
- 生成训练集、验证集、测试集（80%/10%/10% 比例）
- 确保总样本数严格等于 6000

---

## 2. 数据源说明

### 2.1 输入数据文件

- **LLM 诊断数据**：`data/llm_diagnosis_processed/llm_diagnosis_all.jsonl`
  - 来源：大模型检测的疑似违规内容
  - 格式：JSONL，包含 `text`、`meta` 字段
  - 记录数：870 条，唯一 pids：848 个

- **人工审核数据**：`data/human_review_processed/human_review_all.jsonl`
  - 来源：经过人工审核确认的违规内容
  - 格式：JSONL，包含 `text`、`meta` 字段
  - 记录数：297 条，唯一 pids：275 个

- **批量扫描数据**：`data/tsv/batch_scan.tsv`
  - 来源：AC 自动机扫描论坛帖子得到的敏感词命中
  - 格式：TSV，包含 pid、author、context 等字段
  - 记录数：630,170 条，唯一 pids：174,842 个

### 2.2 数据关系

- **LLM 诊断数据** 包含 **人工审核数据**（即人工审核的样本也在 LLM 诊断中）
- **人工审核数据** 是 **LLM 诊断数据** 的子集
- **批量扫描数据** 与上述两个数据集有部分重叠

---

## 3. 采样策略详解

### 3.1 标签定义

- **violation（违规）**：人工审核确认的违规内容 + LLM 诊断但未人工审核的内容
- **suspicious（疑似）**：仅 LLM 诊断，未被人工审核的内容
- **normal（正常）**：从批量扫描中随机采样的内容（排除已使用的 pids）

### 3.2 采样流程

1. **加载数据**：读取三个数据源文件
2. **获取 PID 集合**：提取各数据源的唯一 pids
3. **分类处理**：
   - 人工审核数据 → violation 样本
   - LLM 诊断数据（排除人工审核）→ violation 样本
   - LLM 诊断数据（排除人工审核）→ suspicious 样本
   - 批量扫描数据（排除已使用 pids）→ normal 样本
4. **数量控制**：确保总样本数严格等于 6000
5. **数据集分割**：按 80%/10%/10% 比例分割为训练/验证/测试集

### 3.3 去重逻辑

```python
# 人工审核数据全部加入 violation
for _, row in human_df.iterrows():
    violation_samples.append({...})

# LLM 诊断数据，排除已在人工审核中的
for _, row in llm_df.iterrows():
    if str(row['pid']) not in human_pids:
        violation_samples.append({...})
```

---

## 4. 关键问题解答

### 4.1 数据重复问题

**问题**：LLM 诊断数据包含人工审核数据，采样时是否会有重复？

**答案**：**不会有重复**。脚本实现了严格的去重逻辑：
- 人工审核数据优先加入 violation 样本
- LLM 诊断数据中，只有不在人工审核中的样本才会被加入
- 同一个 pid 不会在 violation 样本中出现两次

### 4.2 标签处理问题

**问题**：人工审核的数据和 LLM 诊断的数据标签是否正确？

**答案**：**标签处理正确**：
- 人工审核数据：`label='violation'`, `source='human_review'`
- LLM 诊断数据（未人工审核）：`label='violation'`, `source='llm_diagnosis'`
- LLM 诊断数据（未人工审核）：`label='suspicious'`, `source='llm_diagnosis'`
- 批量扫描数据：`label='normal'`, `source='batch_scan'`

### 4.3 是否需要 Label Studio 标注

**问题**：生成的样本数据是否还需要用 Label Studio 进行人工标注？

**答案**：**不是必须的**，但可选：
- **直接使用**：如果信任历史标签质量，可以直接用于模型训练
- **人工复核**：如果想提升数据质量，特别是 suspicious/normal 标签，可以用 Label Studio 进行人工复核

---

## 5. 输出文件说明

### 5.1 生成的文件

- `data/annotations/train.jsonl`：训练集
- `data/annotations/validation.jsonl`：验证集
- `data/annotations/test.jsonl`：测试集
- `data/annotations/dataset_stats.json`：统计信息

### 5.2 样本格式

```json
{
    "text": "样本文本内容",
    "label": "violation|suspicious|normal",
    "source": "human_review|llm_diagnosis|batch_scan",
    "pid": "帖子ID",
    "meta": {
        "author": "作者",
        "authorid": "作者ID",
        "useip": "IP地址",
        "hit_word": "命中敏感词",
        "context": "上下文",
        "confidence": "置信度",
        "reasoning": "推理过程"
    }
}
```

---

## 6. 使用示例

### 6.1 基本使用

```bash
# 使用默认参数生成 6000 样本
python scripts/tsv2jsonl.py
```

### 6.2 自定义参数

```bash
# 自定义输出目录和随机种子
python scripts/tsv2jsonl.py --output-dir data/my_annotations --seed 123
```

---

## 7. 注意事项

1. **样本数量**：脚本确保总样本数严格等于 6000
2. **随机性**：使用固定随机种子确保结果可重现
3. **数据质量**：人工审核数据的标签质量最高，LLM 诊断次之
4. **扩展性**：如需调整样本数量，可修改 `classify_samples` 函数的 `target_size` 参数

---

## 总结

`tsv2jsonl.py` 实现了智能的数据采样策略，确保了：
- 数据不重复
- 标签正确
- 数量可控
- 质量分层

生成的样本数据可以直接用于模型训练，也可以根据需要进行人工复核。 
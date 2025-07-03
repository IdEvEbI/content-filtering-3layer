# Phase B 步骤 0 — 数据集整理与标注启动

> 本笔记讲解 **Phase B · Step 0**：从 AC 命中 TSV 中抽样，制作三分类标注数据（0 合规 · 1 疑似 · 2 违规），并在 Label Studio 中完成首批 3 k train / 3 k eval 标注，为后续 LoRA‑BERT 微调做准备。

---

## 1 实操步骤

> 四块内容：**依赖安装 → 数据抽样脚本 → Label Studio 标注 → 提交 PR**。

### 1‑1 依赖安装

```bash
pip install pandas label-studio

# 锁定依赖
pip freeze | grep -E "(pandas|label-studio)" >> requirements.txt
```

* **pandas**：抽样、格式转换
* **label‑studio**：Web 标注平台（Docker 形式文档稍后给出）

---

### 1‑2 数据抽样脚本（`scripts/tsv2jsonl.py`）

#### 1‑2‑1 创建脚本文件

```python
"""
数据采样脚本：从多个数据源生成训练数据集

采样策略：
1. 包含所有 LLM 诊断数据（violation 标签）
2. 包含所有人工审核数据（violation 标签）
3. 从 batch_scan TSV 中随机采样非重叠的 pids 来补充到 6000 样本
4. 标签分配：violation（人工审核+LLM诊断）、suspicious（仅LLM诊断）、normal（TSV采样）
5. 按 80% train / 10% validation / 10% test 比例分割
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def load_llm_diagnosis_data(file_path: str) -> pd.DataFrame:
    """加载 LLM 诊断数据"""
    print(f"加载 LLM 诊断数据: {file_path}")
    df = pd.read_json(file_path, lines=True)
    print(f"  - 总记录数: {len(df)}")
    print(f"  - 唯一 pids: {df['meta'].apply(lambda x: x.get('pid', '')).nunique()}")
    return df


def load_human_review_data(file_path: str) -> pd.DataFrame:
    """加载人工审核数据"""
    print(f"加载人工审核数据: {file_path}")
    df = pd.read_json(file_path, lines=True)
    print(f"  - 总记录数: {len(df)}")
    print(f"  - 唯一 pids: {df['meta'].apply(lambda x: x.get('pid', '')).nunique()}")
    return df


def load_batch_scan_data(file_path: str) -> pd.DataFrame:
    """加载批量扫描数据"""
    print(f"加载批量扫描数据: {file_path}")
    df = pd.read_csv(file_path, sep='\t')
    print(f"  - 总记录数: {len(df)}")
    print(f"  - 唯一 pids: {df['pid'].nunique()}")
    return df


def classify_samples(
    llm_df: pd.DataFrame,
    human_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    target_size: int = 6000
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    对样本进行分类

    Returns:
        Tuple[List[Dict], List[Dict], List[Dict]]: (violation_samples, suspicious_samples, normal_samples)
    """
    print("\n开始样本分类...")

    # 获取所有已使用的 pids
    llm_pids = set(llm_df['meta'].apply(lambda x: str(x.get('pid', ''))))
    human_pids = set(human_df['meta'].apply(lambda x: str(x.get('pid', ''))))
    batch_pids = set(batch_df['pid'].astype(str))

    print(f"  - LLM 诊断 pids: {len(llm_pids)}")
    print(f"  - 人工审核 pids: {len(human_pids)}")
    print(f"  - 批量扫描 pids: {len(batch_pids)}")

    # 1. violation 样本：人工审核 + LLM 诊断（去重）
    violation_pids = human_pids.union(llm_pids)
    violation_samples = []

    # 添加人工审核数据
    for _, row in human_df.iterrows():
        meta = row['meta'] if isinstance(row['meta'], dict) else {}
        meta = dict(meta)  # 确保是字典类型
        violation_samples.append({
            'text': row['text'],
            'label': 'violation',
            'source': 'human_review',
            'pid': str(meta.get('pid', '')),
            'meta': {
                'author': meta.get('author', ''),
                'authorid': meta.get('authorid', ''),
                'useip': meta.get('useip', ''),
                'hit_word': meta.get('hit_word', ''),
                'context': meta.get('context', ''),
                'confidence': meta.get('confidence', ''),
                'reasoning': meta.get('reasoning', '')
            }
        })

    # 添加 LLM 诊断数据（排除已在人工审核中的）
    for _, row in llm_df.iterrows():
        meta = row['meta'] if isinstance(row['meta'], dict) else {}
        meta = dict(meta)  # 确保是字典类型
        pid = str(meta.get('pid', ''))
        if pid not in human_pids:
            violation_samples.append({
                'text': row['text'],
                'label': 'violation',
                'source': 'llm_diagnosis',
                'pid': pid,
                'meta': {
                    'author': meta.get('author', ''),
                    'authorid': meta.get('authorid', ''),
                    'useip': meta.get('useip', ''),
                    'hit_word': meta.get('hit_word', ''),
                    'context': meta.get('context', ''),
                    'confidence': meta.get('confidence', ''),
                    'reasoning': meta.get('reasoning', '')
                }
            })

    # 2. suspicious 样本：仅 LLM 诊断（排除人工审核中的）
    suspicious_samples = []
    for _, row in llm_df.iterrows():
        meta = row['meta'] if isinstance(row['meta'], dict) else {}
        meta = dict(meta)  # 确保是字典类型
        pid = str(meta.get('pid', ''))
        if pid not in human_pids:
            suspicious_samples.append({
                'text': row['text'],
                'label': 'suspicious',
                'source': 'llm_diagnosis',
                'pid': pid,
                'meta': {
                    'author': meta.get('author', ''),
                    'authorid': meta.get('authorid', ''),
                    'useip': meta.get('useip', ''),
                    'hit_word': meta.get('hit_word', ''),
                    'context': meta.get('context', ''),
                    'confidence': meta.get('confidence', ''),
                    'reasoning': meta.get('reasoning', '')
                }
            })

    # 计算各标签的目标数量（按比例分配）
    total_violation = len(violation_samples)
    total_suspicious = len(suspicious_samples)

    # 如果 violation + suspicious 超过目标数量，需要采样
    if total_violation + total_suspicious > target_size:
        # 按比例采样
        violation_ratio = total_violation / (total_violation + total_suspicious)
        target_violation = int(target_size * violation_ratio)
        target_suspicious = target_size - target_violation

        # 随机采样
        random.shuffle(violation_samples)
        random.shuffle(suspicious_samples)
        violation_samples = violation_samples[:target_violation]
        suspicious_samples = suspicious_samples[:target_suspicious]

        print(f"  - 采样后 violation: {len(violation_samples)}")
        print(f"  - 采样后 suspicious: {len(suspicious_samples)}")
        print("  - 不再需要 normal 样本")

        return violation_samples, suspicious_samples, []

        # 3. normal 样本：从批量扫描中采样（排除已使用的 pids）
    available_pids = batch_pids - violation_pids
    print(f"  - 可用于 normal 采样的 pids: {len(available_pids)}")

    # 计算需要采样的数量
    total_violation = len(violation_samples)
    total_suspicious = len(suspicious_samples)
    needed_normal = target_size - total_violation - total_suspicious

    if needed_normal > len(available_pids):
        print(f"  ⚠️  警告：需要 {needed_normal} 个 normal 样本，但只有 {len(available_pids)} 个可用")
        needed_normal = len(available_pids)

    # 随机采样
    sampled_pids = random.sample(list(available_pids), needed_normal)
    sampled_batch_df = batch_df[batch_df['pid'].astype(str).isin(sampled_pids)]

    normal_samples = []
    for _, row in sampled_batch_df.iterrows():
        normal_samples.append({
            'text': row['context'],
            'label': 'normal',
            'source': 'batch_scan',
            'pid': str(row['pid']),
            'meta': {
                'author': row.get('author', ''),
                'authorid': row.get('authorid', ''),
                'useip': row.get('useip', ''),
                'hit_word': row.get('hit_word', ''),
                'context': row.get('context', '')
            }
        })

        print("\n样本分类完成:")
    print(f"  - violation 样本: {len(violation_samples)}")
    print(f"  - suspicious 样本: {len(suspicious_samples)}")
    print(f"  - normal 样本: {len(normal_samples)}")
    print(f"  - 总计: {len(violation_samples) + len(suspicious_samples) + len(normal_samples)}")

    # 确保总样本数严格等于 target_size
    all_samples = violation_samples + suspicious_samples + normal_samples
    if len(all_samples) > target_size:
        print(f"  - 采样到目标数量: {target_size}")
        random.shuffle(all_samples)
        all_samples = all_samples[:target_size]

        # 重新分类
        violation_samples = [s for s in all_samples if s['label'] == 'violation']
        suspicious_samples = [s for s in all_samples if s['label'] == 'suspicious']
        normal_samples = [s for s in all_samples if s['label'] == 'normal']

        print(f"  - 最终 violation: {len(violation_samples)}")
        print(f"  - 最终 suspicious: {len(suspicious_samples)}")
        print(f"  - 最终 normal: {len(normal_samples)}")
        print(f"  - 最终总计: {len(all_samples)}")

    return violation_samples, suspicious_samples, normal_samples


def split_dataset(
    violation_samples: List[Dict],
    suspicious_samples: List[Dict],
    normal_samples: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    按比例分割数据集，确保每个标签在训练/验证/测试集中都有代表性
    """
    print(f"\n开始数据集分割 (train:{train_ratio:.0%}, val:{val_ratio:.0%}, test:{test_ratio:.0%})...")

    # 按标签分组
    samples_by_label = {
        'violation': violation_samples,
        'suspicious': suspicious_samples,
        'normal': normal_samples
    }

    train_samples = []
    val_samples = []
    test_samples = []

    for label, samples in samples_by_label.items():
        if not samples:
            continue

        # 随机打乱
        random.shuffle(samples)

        # 计算分割点
        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # 分割
        train_samples.extend(samples[:train_end])
        val_samples.extend(samples[train_end:val_end])
        test_samples.extend(samples[val_end:])

        print(f"  - {label}: {len(samples)} 样本 -> train:{train_end}, val:{val_end-train_end}, test:{n-val_end}")

    # 最终打乱
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    print("\n数据集分割完成:")
    print(f"  - 训练集: {len(train_samples)} 样本")
    print(f"  - 验证集: {len(val_samples)} 样本")
    print(f"  - 测试集: {len(test_samples)} 样本")

    return train_samples, val_samples, test_samples


def save_jsonl(samples: List[Dict], file_path: str) -> None:
    """保存样本到 JSONL 文件"""
    print(f"保存到: {file_path}")

    # 确保目录存在
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')


def main():
    parser = argparse.ArgumentParser(description="数据采样脚本")
    parser.add_argument("--llm", default="data/llm_diagnosis_processed/llm_diagnosis_all.jsonl",
                        help="LLM 诊断数据文件路径")
    parser.add_argument("--human", default="data/human_review_processed/human_review_all.jsonl",
                        help="人工审核数据文件路径")
    parser.add_argument("--batch", default="data/tsv/batch_scan.tsv",
                        help="批量扫描数据文件路径")
    parser.add_argument("--output-dir", default="data/annotations",
                        help="输出目录")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)

    print("=" * 60)
    print("数据采样脚本启动")
    print("=" * 60)

    # 加载数据
    llm_df = load_llm_diagnosis_data(args.llm)
    human_df = load_human_review_data(args.human)
    batch_df = load_batch_scan_data(args.batch)

    # 样本分类
    violation_samples, suspicious_samples, normal_samples = classify_samples(
        llm_df, human_df, batch_df
    )

    # 数据集分割
    train_samples, val_samples, test_samples = split_dataset(
        violation_samples, suspicious_samples, normal_samples
    )

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(train_samples, str(output_dir / "train.jsonl"))
    save_jsonl(val_samples, str(output_dir / "validation.jsonl"))
    save_jsonl(test_samples, str(output_dir / "test.jsonl"))

    # 保存统计信息
    stats = {
        "total_samples": len(train_samples) + len(val_samples) + len(test_samples),
        "train_samples": len(train_samples),
        "validation_samples": len(val_samples),
        "test_samples": len(test_samples),
        "label_distribution": {
            "train": {},
            "validation": {},
            "test": {}
        }
    }

    # 统计标签分布
    for split_name, samples in [("train", train_samples), ("validation", val_samples), ("test", test_samples)]:
        label_counts = {}
        for sample in samples:
            label = sample['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        stats["label_distribution"][split_name] = label_counts

    with open(output_dir / "dataset_stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("数据采样完成！")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"训练集: {len(train_samples)} 样本")
    print(f"验证集: {len(val_samples)} 样本")
    print(f"测试集: {len(test_samples)} 样本")
    print(f"统计信息: {output_dir / 'dataset_stats.json'}")


if __name__ == "__main__":
    main()

```

**运行示例**：

```bash
python scripts/tsv2jsonl.py
```

**输出结果：**

```bash
============================================================
数据采样脚本启动
============================================================
加载 LLM 诊断数据: data/llm_diagnosis_processed/llm_diagnosis_all.jsonl
  - 总记录数: 870
  - 唯一 pids: 848
加载人工审核数据: data/human_review_processed/human_review_all.jsonl
  - 总记录数: 297
  - 唯一 pids: 275
加载批量扫描数据: data/tsv/batch_scan.tsv
  - 总记录数: 630170
  - 唯一 pids: 174842

开始样本分类...
  - LLM 诊断 pids: 848
  - 人工审核 pids: 275
  - 批量扫描 pids: 174842
  - 可用于 normal 采样的 pids: 174309

样本分类完成:
  - violation 样本: 870
  - suspicious 样本: 573
  - normal 样本: 15957
  - 总计: 17400
  - 采样到目标数量: 6000
  - 最终 violation: 299
  - 最终 suspicious: 204
  - 最终 normal: 5497
  - 最终总计: 6000

开始数据集分割 (train:80%, val:10%, test:10%)...
  - violation: 299 样本 -> train:239, val:29, test:31
  - suspicious: 204 样本 -> train:163, val:20, test:21
  - normal: 5497 样本 -> train:4397, val:549, test:551

数据集分割完成:
  - 训练集: 4799 样本
  - 验证集: 598 样本
  - 测试集: 603 样本
保存到: data/annotations/train.jsonl
保存到: data/annotations/validation.jsonl
保存到: data/annotations/test.jsonl

============================================================
数据采样完成！
============================================================
输出目录: data/annotations
训练集: 4799 样本
验证集: 598 样本
测试集: 603 样本
统计信息: data/annotations/dataset_stats.json
```

**生成文件：**

* `data/annotations/train.jsonl`：训练集（4799 样本）
* `data/annotations/validation.jsonl`：验证集（598 样本）
* `data/annotations/test.jsonl`：测试集（603 样本）
* `data/annotations/dataset_stats.json`：统计信息

**标签分布统计：**

```json
{
  "total_samples": 6000,
  "train_samples": 4799,
  "validation_samples": 598,
  "test_samples": 603,
  "label_distribution": {
    "train": {
      "normal": 4397,
      "suspicious": 163,
      "violation": 239
    },
    "validation": {
      "normal": 549,
      "suspicious": 20,
      "violation": 29
    },
    "test": {
      "normal": 551,
      "violation": 31,
      "suspicious": 21
    }
  }
}
```

---

### 1‑3 Label Studio 标注

#### 1‑3‑1 启动服务（Docker）

```bash
docker run -d --name label-studio -p 8080:8080 \
  -e LABEL_STUDIO_DISABLE_SIGNUP=true \
  heartexlabs/label-studio:latest
```

浏览器访问 `http://localhost:8080`，首次需注册管理员账号。

#### 1‑3‑2 创建项目与标签设置

* 新建项目，进入 Settings → Labeling Interface。
* 推荐标签模板（与采样数据一致，英文 value）：

```xml
<View>
  <Text name="text" value="$text"/>
  <Choices name="label" toName="text" choice="single">
    <Choice value="normal">normal</Choice>
    <Choice value="suspicious">suspicious</Choice>
    <Choice value="violation">violation</Choice>
  </Choices>
</View>
```

#### 1‑3‑3 数据导入

* 推荐将采样数据（如 `train.jsonl`）转换为标准 JSON 数组（`train.json`），再导入。
* 项目页面点击 Import，上传 `train.json`。

#### 1‑3‑4 标注与导出

* 完成标注后，点击 Export，选择 JSON 格式导出（如 `train_export.json`）。
* 如需用于训练，可用脚本转换为 JSONL 格式：

  ```bash
  python scripts/jsonl_json_convert.py data/annotations/train_export.json --reverse
  ```

* 导出数据只包含已标注样本，且会包含额外元数据字段，训练前可用脚本过滤。

---

### 1‑4 提交 PR

```bash
git switch -c semantic-dataset

git add scripts data/annotations/*.jsonl requirements.txt

git commit -m "feat: add initial 6k labeled dataset & tsv→jsonl converter"

git push -u origin semantic-dataset
# GitHub: 创建 PR ➜ base: dev ← compare: semantic-dataset
# CI 仅跑 flake8 / pytest；数据文件不影响
```

---

## 2 附加说明

* **已有 300 条种子**：放入 `data/annotations/seed.jsonl`，与抽样一起导入，可提升模型早期效果。
* **标签指南**：建议在项目 wiki 写一页标准：技术术语→0，模糊暗示→1，明确违法→2。
* **活跃学习**：标注 1 k 即可跑首轮 LoRA，模型再打伪标签 → 人工快审 → 扩充。

---

## 总结

1. 抽样 6 k 行，生成 `train / eval` JSONL。
2. Label Studio 一键标注三分类，导出后即可用于 LoRA 微调。
3. PR 合并至 `dev`，数据与脚本版本化管理，便于后续迭代。

> **一句话总结**：先拿 6 k 三分类标注，奠定 LoRA 微调数据基线。

---

## 建议文档名

`phase-b-step0-dataset`

## 建议保存目录路径

`docs/phase-b`

## 推荐 Commit Message

`feat: add initial 6k labeled dataset & converter`

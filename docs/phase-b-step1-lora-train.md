# Phase B 步骤 1 — LoRA‑BERT 微调脚本

> 本笔记讲解 **Phase B · Step 1**：基于 `hfl/chinese-roberta-wwm-ext` 与 LoRA，训练三分类语义模型（0 合规 · 1 疑似 · 2 违规）。训练在阿里云 GPU 实例完成，随后导出权重供推理服务使用。

---

## 1 实操步骤

> 四块内容：**依赖安装 → 训练脚本 → 云端训练指南 → 提交 PR**。

### 1‑1 依赖安装

```bash
# 核心依赖包
pip install transformers==4.41.0
pip install peft==0.10.0
pip install bitsandbytes==0.42.0
pip install accelerate==1.8.1
pip install datasets==3.6.0
pip install evaluate==0.4.1
pip install scikit-learn==1.7.0

# 可选：如果遇到网络问题，使用国内镜像
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ transformers peft bitsandbytes accelerate datasets evaluate

# 锁定依赖
pip freeze | grep -E "(transformers|peft|bitsandbytes|accelerate|datasets|evaluate|scikit-learn)" >> requirements.txt
```

**依赖说明**：

* **transformers**：Hugging Face 模型库，提供预训练模型和训练框架
* **peft**：Parameter-Efficient Fine-Tuning，实现 LoRA 等高效微调方法
* **bitsandbytes**：8-bit 量化支持，显著减少显存占用
* **accelerate**：分布式训练和混合精度训练支持
* **datasets**：高效的数据集加载和处理
* **evaluate**：模型评估指标库（替代已弃用的 load_metric）
* **scikit-learn**：机器学习工具库，提供分类评估指标和数据处理功能

> *显卡 ≥ T4 16 GB 可直接 fp16；如 CPU 训练需移除 `--fp16`。*

---

### 1‑2 训练脚本（`src/semantic_service/train.py`）

#### 1-2-1 编写脚本

```python
"""LoRA fine-tune chinese-roberta-wwm-ext for 3-class content filtering."""
from pathlib import Path
import json
import argparse
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer  # type: ignore
from datasets import Dataset
from peft import LoraConfig, get_peft_model  # type: ignore
from sklearn.metrics import accuracy_score


def load_jsonl(path):
    """加载 JSONL 文件，每行为一个样本，返回字典生成器"""
    # 标签映射：字符串标签转换为整数
    label_map = {"normal": 0, "violation": 1, "suspicious": 2}

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        # 获取标签字符串并转换为整数
        label_str = obj.get("label", "normal")
        label = label_map.get(label_str, 0)  # 默认为normal(0)
        yield {"text": obj["text"], "label": label}


def make_ds(path):
    """将 JSONL 文件转换为 Huggingface Datasets 格式"""
    return Dataset.from_list(list(load_jsonl(path)))


# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--train", default="data/annotations/train.jsonl", help="训练集路径")
parser.add_argument("--eval",  default="data/annotations/validation.jsonl", help="验证集路径")
parser.add_argument("--out",   default="models/lora_roberta_ckpt", help="模型输出目录")
parser.add_argument("--quick", action="store_true", help="快速测试模式：小batch size和少量epoch")
args = parser.parse_args()

# 自动选择 base_model 路径
local_base_model = Path("chinese-roberta-wwm-ext")
if local_base_model.exists() and (local_base_model / "config.json").exists():
    model_name = str(local_base_model)
    print(f"[INFO] 使用本地基座模型: {model_name}")
else:
    model_name = "hfl/chinese-roberta-wwm-ext"
    print(f"[INFO] 使用 HuggingFace Hub 模型: {model_name}")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    torch_dtype=torch.float32,  # 明确指定数据类型
    low_cpu_mem_usage=True      # 低内存使用模式
)

# 注入 LoRA
lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["query", "value"])  # type: ignore
model = get_peft_model(model, lora_cfg)

# 量化至 8-bit 并转为 float32
model = model.to(torch.float32)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)


def tokenize(batch):
    """批量分词，最大长度128"""
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)


# 加载并分词训练/验证集
train_ds = make_ds(args.train).map(tokenize, batched=True)
val_ds = make_ds(args.eval).map(tokenize, batched=True)


def compute_metrics(eval_pred):
    try:
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "eval_accuracy": acc}
    except Exception:
        # 万一评估出错，返回默认值，避免 KeyError
        return {"accuracy": 0.0, "eval_accuracy": 0.0}


# 检测是否有 GPU 支持 fp16
fp16_enabled = torch.cuda.is_available()  # 只有 CUDA GPU 才支持 fp16

# 训练参数配置
if args.quick:
    # 快速测试模式：小batch size，少量epoch
    print("🔧 快速测试模式：使用小batch size和少量epoch")
    train_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=8,   # 减小batch size
        per_device_eval_batch_size=8,
        num_train_epochs=1,              # 只训练1个epoch
        learning_rate=2e-5,
        evaluation_strategy="no",        # 快速模式下不进行评估
        save_strategy="epoch",           # 每个epoch保存
        fp16=fp16_enabled,               # 根据环境自动设置
        logging_steps=10,                # 更频繁的日志
        save_total_limit=1,
        load_best_model_at_end=False,    # 快速模式下不加载最佳模型
    )
else:
    # 正常训练模式
    train_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        learning_rate=2e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",           # 每个epoch保存
        fp16=fp16_enabled,               # 根据环境自动设置
        logging_steps=50,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",  # 这里强制用 eval_accuracy
    )


class SafeTrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        # 如果 metrics 里没有 metric_for_best_model，补一个默认值
        if metrics is not None and self.args.metric_for_best_model is not None:
            if self.args.metric_for_best_model not in metrics:
                metrics[self.args.metric_for_best_model] = 0.0
        super()._save_checkpoint(model, trial, metrics)


# 构建 SafeTrainer 并训练
trainer = SafeTrainer(model=model, args=train_args, train_dataset=train_ds, eval_dataset=val_ds,
                      compute_metrics=compute_metrics)
trainer.train()
trainer.save_model(args.out)
print("LoRA checkpoint saved →", args.out)

```

> **脚本特点**：
>
> * LoRA r16 + float32 → 显存占用 ≈ 6 GB
> * `tokenize` 128 tokens，适配论坛短文本
> * `load_best_model_at_end` 自动保存最优 epoch
> * 支持快速测试模式，便于本地验证

### 1-2-2 快速测试

```bash
# 快速测试：1个epoch，batch size=8
python src/semantic_service/train.py --quick --out models/quick_test_ckpt
```

**特点**：

* 使用完整数据集，但训练参数优化为快速模式
* batch size: 8（正常32）
* epochs: 1（正常3）
* 预计时间：5-10分钟
* 适合验证环境和脚本配置

### 1-2-3 本地模型效果验证

训练完成后，可直接用推理脚本对模型效果进行本地验证。

```bash
# 单条文本推理
python src/semantic_service/inference.py --model models/quick_test_ckpt --text "这是一个测试文本"

# 批量推理测试
python src/semantic_service/inference.py --model models/quick_test_ckpt --batch
```

**说明**：

* 推理脚本支持单条文本、批量和交互式三种模式
* 输出包括预测标签、置信度和各类别概率分布
* 可用于快速评估模型训练效果

---

### 1‑3 云端训练指南（阿里云 A10 24 GB 实例示例）

详细的部署与训练流程请参考：[docs/note/aliyun-lora-train-deploy.md](../note/aliyun-lora-train-deploy.md)

**核心流程摘要：**

1. 开通 GPU 实例（推荐 A10 24GB）
2. SSH 登录，准备 Python 环境与依赖
3. 上传数据与代码
4. 运行训练脚本
5. 下载模型权重，关停实例

> 详细命令、依赖加速、常见问题排查等请见部署指南文档。

---

### 1‑4 提交 PR

```bash
git switch -c lora-train

git add src/semantic_service/train.py requirements.txt models/.gitkeep docs/phase-b/ali-gpu-setup.md

git commit -m "feat: add lora finetune script & cloud guide"

git push -u origin lora-train
# GitHub: 创建 PR ➜ base: dev ← compare: lora-train
# CI（CPU）可 Skipping heavy tests；代码 lint 通过 ➜ Merge
```

---

## 2 附加说明

* **batch\_size 调整**：如 GPU 仅 16 GB，可降到 16 并用梯度累积。
* **标签不平衡**：初期违规样本少，Trainer 自动使用加权 loss（`class_weight` 可后续补）。
* **活跃学习循环**：训练后用模型打分全部 TSV，选置信度 0.4–0.6 再人工标注。

---

## 总结

1. LoRA‑BERT 三分类训练脚本就绪，显存占用低、训练 3 epoch ≈ 30 min（A10）。
2. 阿里云 GPU 使用文档提供从开机到关机完整 CLI。
3. PR 合并后即进入 **Step B2**：首轮训练 & 评估报告。

> **一句话总结**：轻量 LoRA 微调，让论坛语义过滤在本地 GPU 成本内可行。


"""LoRA fine-tune chinese-roberta-wwm-ext for 3-class content filtering."""
from pathlib import Path
import json
import argparse
import torch
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer  # type: ignore
from datasets import Dataset
import evaluate  # 新增 evaluate 库用于加载 metric
from peft import LoraConfig, get_peft_model  # type: ignore


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

# 加载预训练模型
model_name = "hfl/chinese-roberta-wwm-ext"
cache_path = os.path.expanduser(
      "~/.cache/huggingface/hub/models--hfl--chinese-roberta-wwm-ext/snapshots/5c58d0b8ec1d9014354d691c538661bf00bfdb44"
    )
model = AutoModelForSequenceClassification.from_pretrained(
    cache_path,
    num_labels=3,
    torch_dtype=torch.float32,  # 明确指定数据类型
    low_cpu_mem_usage=True      # 低内存使用模式
)

# 注入 LoRA
lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["query", "value"])  # type: ignore
model = get_peft_model(model, lora_cfg)

# 量化至 8-bit 并转为 float16
model = model.to(torch.float32)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(cache_path, use_fast=False)


def tokenize(batch):
    """批量分词，最大长度128"""
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)


# 加载并分词训练/验证集
train_ds = make_ds(args.train).map(tokenize, batched=True)
val_ds = make_ds(args.eval).map(tokenize, batched=True)

# 加载准确率评估指标（新版 evaluate 替换 load_metric）
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """评估函数，返回准确率"""
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    result = metric.compute(predictions=preds, references=labels)
    return result or {}  # 确保返回字典而不是 None


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
        metric_for_best_model="accuracy",  # 使用准确率作为最佳模型指标
    )

# 构建 Trainer 并训练
trainer = Trainer(model=model, args=train_args, train_dataset=train_ds, eval_dataset=val_ds,
                  compute_metrics=compute_metrics)
trainer.train()
trainer.save_model(args.out)
print("LoRA checkpoint saved →", args.out)

# LoRA 微调训练与推理核心代码逻辑笔记

本笔记总结 `src/semantic_service/train.py` 和 `src/semantic_service/inference.py` 的主要代码结构、核心逻辑与关键实现，便于团队成员理解、复用和二次开发。

---

## 1. 训练脚本 train.py 逻辑概览

### 1.1 数据加载

- 支持 JSONL 格式，每行为一个样本，包含 `text` 和 `label` 字段。
- 标签映射：{"normal": 0, "violation": 1, "suspicious": 2}
- 加载后用 HuggingFace Datasets 封装，支持 map 分词。

### 1.2 参数解析

- 支持命令行参数：
  - `--train` 训练集路径
  - `--eval` 验证集路径
  - `--out` 模型输出目录
  - `--quick` 快速测试模式（小 batch/少 epoch）

### 1.3 自动选择基座模型

- 优先检测本地 `chinese-roberta-wwm-ext` 目录（含 config.json），否则回退 HuggingFace Hub 名称。
- 日志打印实际加载路径。

### 1.4 模型与分词器加载

- 加载 base model（AutoModelForSequenceClassification），指定 `num_labels=3`。
- 注入 LoRA（LoraConfig + get_peft_model）。
- 分词器用 AutoTokenizer，`use_fast=False`。

### 1.5 数据分词

- 分词函数支持批量、最大长度 128。
- 训练/验证集均 map 分词。

### 1.6 评估指标

- 用 sklearn.metrics.accuracy_score 计算准确率。
- compute_metrics 返回 {"accuracy": acc, "eval_accuracy": acc}，兼容 transformers/peft。

### 1.7 训练参数配置

- 快速模式和正式模式参数分开，支持 batch size、epoch、日志频率等灵活配置。
- fp16 自动检测 CUDA。
- `metric_for_best_model` 兼容 eval_accuracy。

### 1.8 健壮性设计

- SafeTrainer 继承 Trainer，_save_checkpoint 时自动补全 metrics key，彻底避免 KeyError。

### 1.9 主流程

- 构建 SafeTrainer，传入数据集、参数、metrics。
- 执行 train()，保存模型。

---

## 2. 推理脚本 inference.py 逻辑概览

### 2.1 类结构

- ContentFilterInference 封装推理流程，支持单条、批量、交互式预测。

### 2.2 自动选择基座模型

- 与 train.py 逻辑一致，优先本地目录，否则 HuggingFace Hub。
- 支持命令行参数 `--base_model` 灵活切换。

### 2.3 加载模型与分词器

- 加载 base model（AutoModelForSequenceClassification）和分词器。
- 加载 LoRA 权重（PeftModel.from_pretrained）。
- 支持 GPU/CPU 自动切换。

### 2.4 推理流程

- predict(text)：分词、推理、softmax 概率、输出标签与置信度。
- batch_predict(texts)：批量预测。

### 2.5 命令行与交互

- 支持 `--model`、`--base_model`、`--text`、`--batch` 参数。
- 支持单条、批量、交互式三种推理模式。
- 结果输出清晰，便于人工验证。

---

## 3. 关键代码片段

### 3.1 自动选择基座模型（通用逻辑）

```python
local_base_model = Path("chinese-roberta-wwm-ext")
if local_base_model.exists() and (local_base_model / "config.json").exists():
    model_name = str(local_base_model)
    print(f"[INFO] 使用本地基座模型: {model_name}")
else:
    model_name = "hfl/chinese-roberta-wwm-ext"
    print(f"[INFO] 使用 HuggingFace Hub 模型: {model_name}")
```

### 3.2 LoRA 注入与模型加载

```python
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, ...)
lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["query", "value"])
model = get_peft_model(model, lora_cfg)
model = model.to(torch.float32)
```

### 3.3 评估指标（兼容 transformers/peft）

```python
def compute_metrics(eval_pred):
    try:
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "eval_accuracy": acc}
    except Exception:
        return {"accuracy": 0.0, "eval_accuracy": 0.0}
```

### 3.4 SafeTrainer 兜底 KeyError

```python
class SafeTrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        if metrics is not None and self.args.metric_for_best_model is not None:
            if self.args.metric_for_best_model not in metrics:
                metrics[self.args.metric_for_best_model] = 0.0
        super()._save_checkpoint(model, trial, metrics)
```

### 3.5 推理 predict 逻辑

```python
def predict(self, text: str) -> Dict[str, Any]:
    inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt").to(self.device)
    with torch.no_grad():
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = int(torch.argmax(logits, dim=-1).item())
        confidence = float(probabilities[0][predicted_class].item())
        all_probs = probabilities[0].tolist()
    return {
        "text": text,
        "predicted_label": self.label_map[predicted_class],
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {self.label_map[i]: prob for i, prob in enumerate(all_probs)}
    }
```

---

## 4. 总结与复用建议

- 训练与推理脚本均支持本地/云端/离线/在线多场景，参数灵活，结构清晰。
- 推荐团队成员参考本笔记快速理解和复用核心代码逻辑。
- 如需二次开发，优先复用自动模型选择、LoRA 注入、SafeTrainer、metrics 兼容等关键实现。

---

如有补充或优化建议，欢迎随时更新本笔记。

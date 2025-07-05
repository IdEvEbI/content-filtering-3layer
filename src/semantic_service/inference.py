"""LoRA model inference script for content filtering."""
import argparse
import torch
from pathlib import Path
from typing import List, Dict, Any

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel  # type: ignore


class ContentFilterInference:
    """内容过滤推理类"""

    def __init__(self, model_path: str, base_model_name: str = ''):
        """
        初始化推理模型
        Args:
            model_path: LoRA模型路径
            base_model_name: 基础模型名称（可为''，自动选择）
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_map = {0: "normal", 1: "violation", 2: "suspicious"}

        # 自动选择 base_model 路径
        if not base_model_name:
            local_base_model = Path("chinese-roberta-wwm-ext")
            if local_base_model.exists() and (local_base_model / "config.json").exists():
                base_model_name = str(local_base_model)
                print(f"[INFO] 使用本地基座模型: {base_model_name}")
            else:
                base_model_name = "hfl/chinese-roberta-wwm-ext"
                print(f"[INFO] 使用 HuggingFace Hub 模型: {base_model_name}")
        else:
            print(f"加载底座模型路径: {base_model_name}")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

        # 加载基础模型
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=3,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        # 加载LoRA权重
        self.model = PeftModel.from_pretrained(self.base_model, model_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ 模型加载完成，设备: {self.device}")

    def predict(self, text: str) -> Dict[str, Any]:
        """
        对单个文本进行预测
        Args:
            text: 输入文本
        Returns:
            预测结果字典
        """
        # 分词
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

            # 获取预测结果
            predicted_class = int(torch.argmax(logits, dim=-1).item())
            confidence = float(probabilities[0][predicted_class].item())

            # 获取所有类别的概率
            all_probs = probabilities[0].tolist()

        return {
            "text": text,
            "predicted_label": self.label_map[predicted_class],
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": {
                self.label_map[i]: prob for i, prob in enumerate(all_probs)
            }
        }

    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        批量预测
        Args:
            texts: 文本列表
        Returns:
            预测结果列表
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


def load_test_samples() -> List[str]:
    """加载测试样本"""
    return [
        "这是一个正常的帖子内容，没有任何违规信息。",
        "今天天气真好，大家一起来讨论一下。",
        "这个产品真的很不错，推荐给大家使用。",
        "我想了解一下这个技术问题，有谁可以帮忙解答吗？",
        "这个论坛的管理员很负责任，维护得很好。",
        "我觉得这个观点很有道理，值得深入思考。",
        "这个活动很有意思，我也要参加。",
        "谢谢大家的帮助，问题已经解决了。",
        "这个软件的功能很强大，使用起来很方便。",
        "我觉得这个建议很好，可以考虑采纳。"
    ]


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LoRA模型推理测试")
    parser.add_argument("--model", default="models/quick_test_ckpt", help="模型路径")
    parser.add_argument("--base_model", default='', help="本地基础模型路径或名称（留空则自动选择）")
    parser.add_argument("--text", help="单条文本测试")
    parser.add_argument("--batch", action="store_true", help="批量测试模式")
    args = parser.parse_args()

    # 检查模型路径
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ 模型路径不存在: {model_path}")
        return

    # 初始化推理模型
    try:
        base_model_arg = args.base_model if args.base_model else ''
        inference = ContentFilterInference(str(model_path), base_model_name=base_model_arg)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 单条文本测试
    if args.text:
        result = inference.predict(args.text)
        print("\n🔍 单条文本预测结果:")
        print(f"文本: {result['text']}")
        print(f"预测标签: {result['predicted_label']} (置信度: {result['confidence']:.3f})")
        print("各类别概率:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.3f}")
        return

    # 批量测试
    if args.batch:
        test_samples = load_test_samples()
        results = inference.batch_predict(test_samples)

        print("\n🔍 批量测试结果:")
        print("-" * 80)
        for i, result in enumerate(results, 1):
            print(f"{i:2d}. 文本: {result['text'][:50]}...")
            print(f"    预测: {result['predicted_label']} (置信度: {result['confidence']:.3f})")
            print(f"    概率分布: normal={result['probabilities']['normal']:.3f}, "
                  f"violation={result['probabilities']['violation']:.3f}, "
                  f"suspicious={result['probabilities']['suspicious']:.3f}")
            print()
        return

    # 交互式测试
    print("\n🎯 交互式推理测试 (输入 'quit' 退出)")
    print("-" * 50)

    while True:
        text = input("\n请输入要测试的文本: ").strip()
        if text.lower() in ['quit', 'exit', 'q']:
            break

        if not text:
            continue

        try:
            result = inference.predict(text)
            print("\n📊 预测结果:")
            print(f"  标签: {result['predicted_label']}")
            print(f"  置信度: {result['confidence']:.3f}")
            print("  概率分布:")
            for label, prob in result['probabilities'].items():
                bar = "█" * int(prob * 20)
                print(f"    {label:10s}: {prob:.3f} {bar}")
        except Exception as e:
            print(f"❌ 预测失败: {e}")


if __name__ == "__main__":
    main()

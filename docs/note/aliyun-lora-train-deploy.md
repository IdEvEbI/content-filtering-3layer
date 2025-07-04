# 阿里云 LoRA 微调训练部署指南

本指南适用于在阿里云 GPU 实例上部署、训练和验证 LoRA 微调模型，适配 `content-filtering-3layer` 项目。

---

## 1. 实例选择与开通

- 推荐实例：A10 24GB（ecs.gn7i-c8g1.2xlarge）
- 地区建议：华东/华北，带宽≥10Mbps
- 镜像建议：官方 Deep Learning AMI 或 Ubuntu 22.04

**开通命令示例：**

```bash
aliyun ecs RunInstances --InstanceType ecs.gn7i-c8g1.2xlarge --ImageId aliyun_3_22_x64_20G_alibase_20240523.raw --RegionId cn-hangzhou
```

---

## 2. SSH 登录与环境准备

```bash
ssh root@<公网IP>
# 建议新建普通用户，避免直接用 root
```

- 检查 GPU 驱动和 CUDA：

  ```bash
  nvidia-smi
  nvcc --version
  ```

- 推荐使用 Python 3.12（可用 pyenv/conda 安装）

---

## 3. Python 环境与依赖安装

```bash
# 安装 Miniconda（推荐）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# 或用 pyenv 安装 Python 3.12

# 创建虚拟环境
conda create -n lora python=3.12 -y
conda activate lora

# 加速源（可选）
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装依赖
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

---

## 4. 代码与数据同步

- **代码上传**：

  ```bash
  git clone <repo>
  # 或用 scp/rsync 上传本地代码
  # 注意：models/ 目录不包含在 Git 中，需要单独处理
  ```

- **数据上传**：

  ```bash
  scp data/annotations/*.jsonl root@<公网IP>:/root/content-filtering-3layer/data/annotations/
  # 或用 ossutil/ossbrowser 上传大文件
  ```

---

## 5. 训练命令与监控

```bash
# 快速测试
python src/semantic_service/train.py --quick --out models/quick_test_ckpt --fp16

# 完整训练
python src/semantic_service/train.py --out models/lora_roberta_ckpt --fp16
```

- **监控显存/性能**：

  ```bash
  watch -n 2 nvidia-smi
  top
  ```

- **训练日志**：建议用 `tee` 保存日志

  ```bash
  python ... > train.log 2>&1 &
  tail -f train.log
  ```

---

## 6. 模型下载与实例关停

- **下载模型**：

  ```bash
  scp -r models/lora_roberta_ckpt <本地路径>
  # 或上传到 OSS
  ```

- **关停实例**：

  ```bash
  aliyun ecs StopInstance --InstanceId <ID>
  aliyun ecs DeleteInstance --InstanceId <ID>
  ```

---

## 7. 常见问题与排查

- **依赖安装慢/失败**：使用国内镜像源
- **CUDA/驱动不兼容**：建议用官方深度学习镜像
- **显存不足**：调小 batch size 或用 --quick
- **数据同步慢**：用 ossutil/ossbrowser
- **训练中断**：用 nohup/screen/tmux 保持会话
- **模型推理验证**：

  ```bash
  python src/semantic_service/inference.py --model models/lora_roberta_ckpt --text "测试文本"
  ```

---

## 8. 参考链接

- [阿里云 ECS 官方文档](https://help.aliyun.com/zh/ecs/)
- [PyTorch 官方镜像](https://pytorch.org/get-started/locally/)
- [HuggingFace 国内加速](https://hf-mirror.com/)

---

如有问题可随时补充本指南。

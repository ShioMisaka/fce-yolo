# COCO 数据集训练指南

本指南介绍如何使用 `coco_train.py` 在 COCO 2017 数据集上训练 YOLOv11 变体模型。

## 📦 前置准备

### 1. 安装依赖

```bash
pip install -e ".[export,solutions,logging]"
```

### 2. 下载 COCO 数据集

#### 方式一：自动下载（推荐）

首次运行时，Ultralytics 会自动下载 COCO 数据集：

```bash
python script/coco_train.py baseline --scale n --test
```

数据集将下载到 `datasets/coco/` 目录（约 20 GB）。

#### 方式二：手动下载

如果自动下载失败，可以手动下载：

```bash
# 创建数据集目录
mkdir -p datasets/coco/images

# 下载标签
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip
unzip coco2017labels.zip -d datasets/coco/

# 下载图像
cd datasets/coco/images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip train2017.zip
unzip val2017.zip
```

### 3. 验证数据集

确保目录结构如下：

```
datasets/coco/
├── train2017.txt      # 训练集图像列表
├── val2017.txt        # 验证集图像列表
├── labels/
│   ├── train2017/
│   └── val2017/
└── images/
    ├── train2017/
    └── val2017/
```

## 🚀 快速开始

### 快速测试（推荐首次使用）

快速测试验证环境配置是否正确（1 epoch，约 5 分钟）：

```bash
# Baseline-N 快速测试
python script/coco_train.py baseline --scale n --test

# FCE-S 快速测试
python script/coco_train.py fce --scale s --test
```

### 完整训练

#### 训练单个模型

```bash
# Baseline-S（单阶段，300 epochs）
python script/coco_train.py baseline --scale s --epochs 300

# FCE-S（自动两阶段：50 + 300 epochs）
python script/coco_train.py fce --scale s --epochs 300

# BiFPN-M（自动两阶段）
python script/coco_train.py bifpn --scale m --batch 16
```

#### 训练所有模型进行对比

```bash
# 训练 baseline、bifpn、fce 三个模型
python script/coco_train.py --all --scale s --epochs 300
```

## ⚙️ 参数说明

### 模型选择

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `baseline` | 原生 YOLOv11 | 单阶段训练 |
| `bifpn` | BiFPN-YOLOv11 | 两阶段训练 |
| `fce` | FCE-YOLOv11 | 两阶段训练 |

### 模型尺度

| 尺度 | 参数量 | 推荐场景 | 默认 batch |
|------|--------|----------|-----------|
| `n` (nano) | 2.6M | 快速验证、边缘设备 | 32 |
| `s` (small) | 9.4M | 速度与精度平衡 | 16 |
| `m` (medium) | 20.1M | 高精度需求 | 8-12 |
| `l` (large) | 25.3M | 最高精度 | 4-8 |
| `x` (xlarge) | 43.7M | 研究用途 | 2-4 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 300 | 训练轮次 |
| `--batch` | 16 | 批次大小（根据 GPU 调整） |
| `--imgsz` | 640 | 输入图像尺寸 |
| `--device` | 0 | GPU 设备（0, 1, 0,1 等） |
| `--workers` | 8 | 数据加载线程数 |
| `--lr0` | 0.01 | 初始学习率 |
| `--name` | 自动 | 实验名称 |
| `--stage` | None | 仅训练指定阶段（1 或 2） |
| `--test` | False | 快速测试模式 |

## 📊 训练配置

### COCO 数据集标准配置

脚本内置了针对 COCO 数据集优化的训练配置：

```python
# 图像尺寸
imgsz = 640  # COCO 标准尺寸

# 批次大小（根据 GPU 调整）
batch = 16   # RTX 3090: 16-24
            # RTX 3080: 12-16
            # RTX 3070: 8-12
            # GTX 1080Ti: 6-8

# 训练轮次
epochs = 300  # 完整训练

# 数据增强（COCO 标准）
degrees = 0.0       # 不旋转
fliplr = 0.5        # 50% 左右翻转
mosaic = 1.0        # Mosaic 增强
mixup = 0.0         # 不使用 Mixup
```

### 两阶段训练（BiFPN/FCE）

**阶段一（冻结预热）**：
- 训练轮次：50 epochs
- 冻结 backbone 前 10 层
- 学习率：0.01（较高）
- 学习率调度：线性衰减
- Mosaic 关闭：最后 10 epochs

**阶段二（全局微调）**：
- 训练轮次：300 epochs
- 解冻所有层
- 学习率：0.001（较低）
- 学习率调度：余弦退火
- Mosaic 关闭：最后 20 epochs

## 💡 使用示例

### 示例 1：快速验证修复效果

在修复了 FCE 模块缩放 bug 后，快速验证 N 尺度的改进：

```bash
# 修复后的 FCE-N 快速测试
python script/coco_train.py fce --scale n --test

# 修复前的对比（如果保存了旧模型）
# yolo val model=path/to/old_fce_n.pt data=coco.yaml
```

### 示例 2：单 GPU 完整训练

使用单个 GPU 训练 FCE-S 模型：

```bash
python script/coco_train.py fce --scale s --batch 16 --device 0
```

### 示例 3：多 GPU 训练

使用多个 GPU 加速训练（需注意 batch size）：

```bash
# 使用 2 个 GPU，batch size 翻倍
python script/coco_train.py fce --scale s --batch 32 --device 0,1
```

### 示例 4：恢复中断的训练

如果训练中断，可以从检查点恢复：

```bash
# 阶段一中断后恢复
python script/coco_train.py fce --scale s --stage 1

# 阶段二中断后恢复
python script/coco_train.py fce --scale s --stage 2
```

### 示例 5：自定义学习率

使用不同的学习率训练：

```bash
# 较高学习率快速收敛
python script/coco_train.py fce --scale s --lr0 0.02

# 较低学习率精细调整
python script/coco_train.py fce --scale s --lr0 0.005
```

### 示例 6：训练对比实验

训练所有模型进行完整的消融实验：

```bash
# S 尺度完整对比
python script/coco_train.py --all --scale s --epochs 300

# N 尺度验证修复效果
python script/coco_train.py --all --scale n --epochs 300
```

## 📈 监控训练进度

### Tensorboard

```bash
# 启动 Tensorboard
tensorboard --logdir runs/coco

# 浏览器打开
http://localhost:6006
```

### 训练日志

训练日志保存在：
```
runs/coco/{experiment_name}/
├── weights/
│   ├── best.pt          # 最佳模型
│   ├── last.pt          # 最后一个 epoch
│   └── epoch_*.pt       # 定期保存
├── results.csv          # 训练指标
├── args.yaml            # 训练配置
└── events.out.tfevents.* # Tensorboard 日志
```

## 🎯 预期结果

### COCO 数据集性能

参考 YOLOv11 在 COCO 上的官方性能：

| 模型 | mAP@50-95 | mAP@50 | 参数量 |
|------|-----------|--------|--------|
| YOLOv11n | 39.5% | 53.2% | 2.6M |
| YOLOv11s | 47.2% | 62.0% | 9.4M |
| YOLOv11m | 51.5% | 66.8% | 20.1M |
| YOLOv11l | 53.4% | 69.0% | 25.3M |
| YOLOv11x | 54.7% | 70.4% | 43.7M |

### FCE/BiFPN 预期改进

基于之前的实验，预期在 COCO 上：
- **FCE-S**: +2-4% mAP@50-95 vs Baseline-S
- **BiFPN-S**: +3-5% mAP@50-95 vs Baseline-S

## 🔧 故障排除

### 问题 1：CUDA Out of Memory

**解决方案**：
```bash
# 减小批次大小
python script/coco_train.py fce --scale s --batch 8

# 或减小图像尺寸
python script/coco_train.py fce --scale s --imgsz 512
```

### 问题 2：数据集下载失败

**解决方案**：手动下载（见"前置准备"部分）

### 问题 3：训练速度慢

**解决方案**：
```bash
# 启用缓存（需要足够 RAM）
python script/coco_train.py fce --scale s --cache ram

# 减少数据加载线程
python script/coco_train.py fce --scale s --workers 4
```

### 问题 4：mAP 不提升

**检查**：
1. 学习率是否过大/过小
2. 数据增强是否过强
3. 是否需要更长训练

## 📝 引用

如果在论文中使用这些结果，请引用：

```bibtex
@article{yolo11,
  title={YOLOv11: Real-Time Object Detection},
  author={Ultralytics},
  year={2024},
  publisher={Ultralytics}
}
```

## 🤝 贡献

如果发现问题或有改进建议，欢迎提交 Issue 或 Pull Request。

# COCO 高性能训练指南

## 📁 数据集结构

数据集已解压到 `/mnt/ssd1/Dataset/coco2017/`，目录结构如下：

```
/mnt/ssd1/Dataset/coco2017/
├── images/
│   ├── train2017/    # 118,287 张训练图片
│   ├── val2017/      # 5,000 张验证图片
│   └── test2017/     # 40,670 张测试图片
└── labels/
    ├── train2017/    # 117,266 个训练标签
    └── val2017/      # 4,952 个验证标签
```

## 🚀 快速开始

### 1. 快速测试（推荐先运行）

验证配置是否正确，使用最小模型快速测试：

```bash
# 使用新的测试脚本（在 script 目录）
python script/test_train_pro.py
```

**预期输出**：
- 训练 1 个 epoch
- 验证模型正常工作
- 显示 mAP 指标

### 2. 单模型训练

使用 script/train_pro.py 训练单个模型：

```bash
# 使用默认参数（推荐配置）
python script/train_pro.py

# 使用自定义模型配置
python script/train_pro.py --model ultralytics/cfg/models/11/yolo11-fce.yaml

# 使用 baseline 模型对比
python script/train_pro.py --model ultralytics/cfg/models/11/yolo11.yaml --name baseline_coco
```

### 3. 多模型对比训练（推荐）

一次性训练 Baseline、BiFPN、FCE 三个模型，并自动生成对比结果：

```bash
# 训练所有模型（n 尺度，适合快速验证）
python script/train_coco_compare.py --scale n

# 训练所有模型（s 尺度，推荐用于正式实验）
python script/train_coco_compare.py --scale s

# 训练所有模型（m 尺度）
python script/train_coco_compare.py --scale m

# 自定义批次大小（根据显存调整）
python script/train_coco_compare.py --scale s --batch 96

# 快速测试模式（1 epoch，验证流程）
python script/train_coco_compare.py --scale n --test

# 跳过训练，仅生成对比（用于已有结果）
python script/train_coco_compare.py --scale s --skip-train
```

**输出结果**：
- 三个模型的训练结果（BiFPN 和 FCE 使用两阶段训练）
- 对比曲线图（mAP@50-95、mAP@50、Precision、Recall）
- 对比摘要文本文件
- 完整的对比表格

## ⚙️ 高级配置

### 性能优化参数

针对 RTX 5090 + 128GB RAM 的优化参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch` | 128 | 批次大小，设为 -1 自动搜索最大值 |
| `--cache` | ram | 数据缓存：ram/disk/none |
| `--workers` | 24 | 数据加载进程数 |
| `--amp` | True | 自动混合精度训练 |
| `--imgsz` | 640 | 输入图像尺寸 |

### 示例命令

```bash
# 超大批次训练（32GB 显存）
python script/train_pro.py --batch 128 --cache ram --workers 24

# 自动搜索最大批次
python script/train_pro.py --batch -1

# 高分辨率训练
python script/train_pro.py --imgsz 1280 --batch 64

# 从断点恢复训练
python script/train_pro.py --resume

# 调整学习率
python script/train_pro.py --lr0 0.01 --lrf 0.01 --cos-lr

# 自定义实验名称
python script/train_pro.py --name my_experiment --epochs 500

# 单类别训练（调试用）
python script/train_pro.py --single-cls

# 禁用验证加速训练
python script/train_pro.py --no-val
```

### 完整参数列表

```bash
python script/train_pro.py --help
```

## 📊 训练监控

### TensorBoard

```bash
tensorboard --logdir runs/detect
```

### 结果文件位置

训练结果保存在 `runs/detect/<name>/`：

```
runs/detect/coco_train/
├── weights/
│   ├── best.pt       # 最佳模型
│   ├── last.pt       # 最后一个 epoch 的模型
│   └── epoch*.pt     # 定期保存的检查点
├── results.csv       # 训练指标 CSV
├── results.png       # 训练曲线图
├── confusion_matrix.png
├── F1_curve.png
└── PR_curve.png
```

## 🔧 硬件优化说明

### RAM 缓存（`--cache ram`）
- **优点**：彻底消除 SSD IO 瓶颈，训练速度提升 2-3x
- **要求**：128GB RAM 可完全缓存 COCO 数据集（~120GB）
- **显存占用**：约 100-110GB

### Worker 进程（`--workers 24`）
- **推荐值**：CPU 核心数的 1.5x
- **9950X3D**：16 核心 → 推荐使用 24 workers
- **作用**：数据预取和增强，避免 GPU 等待数据

### 批次大小（`--batch`）
- **RTX 5090 32GB**：
  - YOLOv11n (640): 推荐批次 128+
  - YOLOv11s (640): 推荐批次 96
  - YOLOv11m (640): 推荐批次 64
  - YOLOv11l (640): 推荐批次 32
  - YOLOv11x (640): 推荐批次 16
- **自动搜索**：使用 `--batch -1` 自动找到最大批次

### AMP 混合精度（`--amp`）
- **优点**：训练速度提升 1.5-2x，显存占用减少 ~40%
- **缺点**：可能轻微影响数值稳定性（通常可忽略）

## 📈 预期训练时间

基于 RTX 5090 的估算（imgsz=640, cache=ram）：

| 模型 | 1 epoch 时间 | 300 epoch 总时间 |
|------|-------------|-----------------|
| YOLOv11n | ~30 分钟 | ~150 小时 |
| YOLOv11s | ~45 分钟 | ~225 小时 |
| YOLOv11m | ~60 分钟 | ~300 小时 |
| YOLOv11l | ~90 分钟 | ~450 小时 |
| YOLOv11x | ~120 分钟 | ~600 小时 |

## 🐛 常见问题

### 1. CUDA Out of Memory
```bash
# 减小批次大小
python script/train_pro.py --batch 64

# 或使用自动搜索
python script/train_pro.py --batch -1
```

### 2. 数据加载慢
```bash
# 检查是否启用了 RAM 缓存
python script/train_pro.py --cache ram

# 减少 workers（如果 CPU 已经满载）
python script/train_pro.py --workers 16
```

### 3. 验证集指标不更新
```bash
# 确保验证集路径正确
cat ultralytics/cfg/datasets/coco_custom.yaml

# 检查标签文件是否存在
ls /mnt/ssd1/Dataset/coco2017/labels/val2017/ | wc -l
```

### 4. 训练中断恢复
```bash
# 使用 --resume 继续
python script/train_pro.py --resume
```

## 📝 性能基准

参考预期结果（COCO val2017）：

| 模型 | mAP50-95 | mAP50 | 参数量 | FLOPs |
|------|----------|-------|--------|-------|
| YOLOv11n | ~39% | ~53% | 2.6M | 6.1B |
| YOLOv11s | ~47% | ~62% | 9.4M | 21.4B |
| YOLOv11m | ~51% | ~66% | 20.9M | 48.6B |
| YOLOv11l | ~53% | ~68% | 26.5M | 60.9B |
| YOLOv11x | ~55% | ~70% | 58.8M | 135.4B |

**FCE 变体**：预期在相同参数量下提升 1-3% mAP。

## 🎯 训练策略建议

### 多模型对比训练（推荐）

使用 `script/train_coco_compare.py` 一次性训练三个模型并生成对比结果：

**训练流程**：
1. 自动训练 Baseline、BiFPN、FCE 三个模型
2. BiFPN 和 FCE 自动执行两阶段训练（50 epochs + 300 epochs）
3. 训练完成后自动生成对比图表和摘要

**优势**：
- 一次性完成所有训练，无需手动切换
- 自动生成专业对比图表
- 统一的结果目录结构
- 自动输出最佳模型对比

```bash
# S 尺度模型对比（推荐）
python script/train_coco_compare.py --scale s

# 自定义批次大小
python script/train_coco_compare.py --scale s --batch 96

# 查看完整参数
python script/train_coco_compare.py --help
```

**输出目录结构**：
```
runs/detect/baselinevsbifpnvsfce_s_300/
├── baseline_s/               # Baseline 结果
│   ├── weights/best.pt
│   └── results.csv
├── bifpn_s_stage1/           # BiFPN 阶段一结果
├── bifpn_s_stage2/           # BiFPN 阶段二结果（最终）
│   ├── weights/best.pt
│   └── results.csv
├── fce_s_stage1/             # FCE 阶段一结果
├── fce_s_stage2/             # FCE 阶段二结果（最终）
│   ├── weights/best.pt
│   └── results.csv
├── comparison_curves.png     # 对比曲线图
└── comparison_summary.txt    # 对比摘要
```

### 单模型两阶段训练

单独训练某个模型，使用项目的模块化训练脚本：

```bash
# 训练 FCE 模型（自动两阶段）
python script/train.py fce --scale s --batch 128

# 训练 BiFPN 模型（自动两阶段）
python script/train.py bifpn --scale s --batch 128

# 训练 Baseline 模型（单阶段）
python script/train.py baseline --scale s --batch 128
```

**两阶段训练说明**：
- **阶段一（50 epochs）**：冻结预热，学习率 0.01，线性衰减
- **阶段二（300 epochs）**：全局微调，学习率 0.001，余弦退火

### 快速实验

使用 train_pro.py 进行快速实验或单阶段训练：

```bash
# 单阶段训练（100 epochs 快速实验）
python script/train_pro.py --epochs 100 --batch 128

# 测试新模块
python script/train_pro.py --model ultralytics/cfg/models/11/yolo11-bifpn.yaml --epochs 50
```

## 📞 获取帮助

- 查看单模型训练参数：`python script/train_pro.py --help`
- 查看多模型对比参数：`python script/train_coco_compare.py --help`
- 查看项目文档：`cat CLAUDE.md`
- 运行快速测试：`python script/test_train_pro.py`

## 🔄 训练脚本对比

| 脚本 | 用途 | 适用场景 | 两阶段训练 | 对比分析 |
|------|------|---------|-----------|---------|
| `train_pro.py` | 单模型高性能训练 | 单个模型深度训练 | ❌ | ❌ |
| `script/train.py` | 模块化单模型训练 | 特定模型变体训练 | ✅ 自动 | ❌ |
| `script/train_coco_compare.py` | 多模型对比训练 | 完整对比实验 | ✅ 自动 | ✅ 自动 |
| `script/compare.py` | 通用多模型对比 | 任意模型组合对比 | ❌ | ✅ 自动 |
| `script/test_train_pro.py` | 快速配置测试 | 验证环境配置 | ❌ | ❌ |

---

**提示**：首次训练前建议运行 `python script/test_train_pro.py` 验证配置正确性！

# COCO 高性能训练指南

## 数据集结构

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

## 快速开始

### 1. 快速测试（推荐先运行）

验证配置是否正确：

```bash
python script/test.py
```

**预期输出**：`✓ 所有测试通过!`

### 2. 单模型训练

使用统一训练 CLI：

```bash
# 默认数据集训练
python script/train.py baseline --scale n
python script/train.py fce --scale s

# COCO 数据集训练（使用预设）
python script/train.py baseline --scale s --dataset coco

# COCO 高性能训练（使用 coco_hq 预设）
python script/train.py baseline --scale s --dataset coco_hq

# 自定义数据集路径
python script/train.py baseline --scale s --data /path/to/data.yaml
```

### 3. 多模型对比训练（推荐）

```bash
# 默认数据集对比
python script/compare.py --models baseline fce --scale s

# 三模型对比
python script/compare.py --models baseline bifpn fce --scale s

# COCO 数据集对比
python script/compare.py --models baseline bifpn fce --scale s --dataset coco

# COCO 高性能对比
python script/compare.py --models baseline bifpn fce --scale s --dataset coco_hq

# 覆盖训练轮次
python script/compare.py --models baseline fce --scale s --epochs 200

# 跳过训练，仅对比已有结果
python script/compare.py --models baseline bifpn --scale s --skip-train
```

## 高级配置

### 性能优化参数

针对 RTX 5090 + 128GB RAM 的优化参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch` | 128 (coco_hq) | 批次大小 |
| `--cache` | ram (coco_hq) | 数据缓存：ram/disk/false |
| `--workers` | 24 (coco_hq) | 数据加载进程数 |
| `--amp` | True | 自动混合精度训练 |
| `--imgsz` | 640 | 输入图像尺寸 |

### 数据集预设

| 预设 | imgsz | batch | workers | cache |
|------|-------|-------|---------|-------|
| `default` | 1280 | 32 | 16 | ram |
| `coco` | 640 | 16 | 8 | false |
| `coco_hq` | 640 | 128 | 24 | ram |

### 示例命令

```bash
# 超大批次训练
python script/train.py baseline --scale s --dataset coco_hq

# 高分辨率训练
python script/train.py fce --scale s --dataset coco --imgsz 1280

# 调整学习率
python script/train.py fce --scale s --dataset coco --lr0 0.01 --cos-lr

# 自定义训练轮次
python script.train.py fce --scale s --dataset coco --epochs 500

# 切换 IoU 损失
python script/compare.py --models baseline fce --scale s --dataset coco --iou-type WIoU

# 快速测试
python script/train.py fce --scale s --test
```

### 参数分类

| 参数类别 | CLI 示例 | 影响范围 |
|---------|---------|---------|
| 共享参数 | `--batch`, `--imgsz`, `--device`, `--workers`, `--iou-type` | 所有阶段 |
| 阶段参数 | `--epochs`, `--lr0`, `--patience` | 仅 stage2 |
| stage1 覆盖 | `--stage1-epochs`, `--stage1-lr0` | 仅 stage1 |

## 训练监控

### TensorBoard

```bash
tensorboard --logdir runs/detect
```

### 结果文件位置

训练结果保存在 `runs/detect/<name>/`：

```
runs/detect/<name>/
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

## 硬件优化说明

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

## 预期训练时间

基于 RTX 5090 的估算（imgsz=640, cache=ram）：

| 模型 | 1 epoch 时间 | 300 epoch 总时间 |
|------|-------------|-----------------|
| YOLOv11n | ~30 分钟 | ~150 小时 |
| YOLOv11s | ~45 分钟 | ~225 小时 |
| YOLOv11m | ~60 分钟 | ~300 小时 |
| YOLOv11l | ~90 分钟 | ~450 小时 |
| YOLOv11x | ~120 分钟 | ~600 小时 |

## 性能基准

参考预期结果（COCO val2017）：

| 模型 | mAP50-95 | mAP50 | 参数量 | FLOPs |
|------|----------|-------|--------|-------|
| YOLOv11n | ~39% | ~53% | 2.6M | 6.1B |
| YOLOv11s | ~47% | ~62% | 9.4M | 21.4B |
| YOLOv11m | ~51% | ~66% | 20.9M | 48.6B |
| YOLOv11l | ~53% | ~68% | 26.5M | 60.9B |
| YOLOv11x | ~55% | ~70% | 58.8M | 135.4B |

**FCE 变体**：预期在相同参数量下提升 1-3% mAP。

## 常见问题

### CUDA Out of Memory
```bash
python script/train.py baseline --scale s --dataset coco_hq --batch 64
```

### 数据加载慢
```bash
python script/train.py baseline --scale s --dataset coco --cache ram
```

### 查看帮助
```bash
python script/train.py --help
python script/compare.py --help
```

## 训练策略

### 两阶段训练

对于 BiFPN 和 FCE 等包含随机初始化模块的模型，自动执行两阶段训练：

- **阶段一（50 epochs）**：冻结预热，学习率 0.01，线性衰减
- **阶段二（300 epochs）**：全局微调，学习率 0.001，余弦退火

可通过参数覆盖：
```bash
# 覆盖 stage2 轮次（50+200）
python script/train.py fce --scale s --dataset coco --epochs 200

# 显式覆盖 stage1
python script.train.py fce --scale s --dataset coco --stage1-epochs 30 --stage1-lr0 0.005
```

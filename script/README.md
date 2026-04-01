# YOLOv11 训练对比工具

灵活的 YOLOv11 变体模型训练和对比工具包，支持 COCO 数据集高性能训练。

## 🚀 快速开始

### 环境验证（首次使用）

```bash
# 验证训练环境和配置
python script/test_train_pro.py
```

### 训练单个模型

**使用模块化训练脚本（推荐）**：

```bash
# 训练 Baseline 模型
python script/train.py baseline --scale s

# 训练 BiFPN 模型（自动两阶段）
python script/train.py bifpn --scale s

# 训练 FCE 模型（自动两阶段）
python script/train.py fce --scale s

# 自定义训练参数
python script/train.py fce --scale s --batch 16 --imgsz 640 --epochs 100
```

**使用高性能训练脚本（RTX 5090 优化）**：

```bash
# 使用自定义模型配置
python script/train_pro.py --model ultralytics/cfg/models/11/yolo11-fce.yaml

# 使用 Baseline 模型
python script/train_pro.py --model ultralytics/cfg/models/11/yolo11.yaml --name baseline_coco

# 自定义训练参数
python script/train_pro.py --batch 96 --epochs 200 --imgsz 640
```

### COCO 多模型训练对比（推荐）

**针对 RTX 5090 优化的完整对比训练流程**：

```bash
# 训练所有模型（n 尺度，快速验证）
python script/train_coco_compare.py --scale n

# 训练所有模型（s 尺度，正式实验）
python script/train_coco_compare.py --scale s

# 自定义批次大小
python script/train_coco_compare.py --scale s --batch 96

# 快速测试模式
python script/train_coco_compare.py --scale n --test

# 仅生成对比（不训练）
python script/train_coco_compare.py --scale s --skip-train
```

### 通用多模型对比

```bash
# Baseline vs BiFPN 对比
python script/compare.py --models baseline bifpn --scale s

# Baseline vs FCE 对比
python script/compare.py --models baseline fce --scale s

# 三模型对比
python script/compare.py --models baseline bifpn fce --scale s

# 仅对比已有结果（不训练）
python script/compare.py --models baseline fce --scale s --skip-train
```

## 📦 模块结构

```
script/
├── __init__.py              # 包初始化
├── config.py                # 配置管理
├── trainer.py               # 训练器
├── train.py                 # 模块化单模型训练 CLI
├── train_pro.py             # 高性能单模型训练（RTX 5090 优化）
├── compare.py               # 通用多模型对比 CLI
├── train_coco_compare.py    # COCO 多模型训练对比（RTX 5090 优化）
├── test_train_pro.py        # 快速环境测试
├── test_two_stage_config.py # 两阶段配置测试
└── README.md                # 本文档
```

## 📊 支持的模型

| 模型 | 说明 | 训练策略 | YAML 配置 |
|------|------|----------|-----------|
| **baseline** | 标准 YOLOv11 | 单阶段 (300 epochs) | `yolo11.yaml` |
| **bifpn** | YOLOv11 + BiFPN | 两阶段 (30 冻结 + 300 完整) | `yolo11-bifpn.yaml` |
| **fce** | YOLOv11 + BiFPN + 注意力 | 两阶段 (30 冻结 + 300 完整) | `yolo11-fce.yaml` |

## 🔧 参数说明

### 模型尺度

| 参数 | 尺寸 | 参数量 | GFLOPs |
|------|------|--------|--------|
| `n` | nano | ~2.6M | ~6.6 |
| `s` | small | ~9.5M | ~21.7 |
| `m` | medium | ~20.1M | ~68.5 |
| `l` | large | ~25.4M | ~87.6 |
| `x` | xlarge | ~56.9M | ~196.0 |

### 训练参数

```bash
--epochs N        # 训练轮次
--batch N         # 批次大小
--imgsz N         # 输入图像尺寸
--device N        # GPU 设备
--patience N      # 早停耐心值
--exp-name NAME   # 自定义实验名称
--test            # 快速测试模式
```

## 📈 输出目录

### 单模型训练

```
runs/detect/
├── baseline_yolo11s/          # Baseline 结果
│   ├── weights/
│   │   ├── best.pt
│   │   └── last.pt
│   ├── results.csv
│   └── ...
├── bifpn_s_stage1/            # BiFPN 阶段一
├── bifpn_s_stage2/            # BiFPN 阶段二
└── fce_s_stage1/              # FCE 阶段一
└── fce_s_stage2/              # FCE 阶段二
```

### 多模型对比

```
runs/detect/
└── baselinevsfce_s_330/       # 对比输出目录（两阶段总 epochs）
    ├── baseline_yolo11s/      # Baseline 结果（300 epochs）
    ├── fce_s_stage1/          # FCE 阶段一结果（30 epochs）
    ├── fce_s_stage2/          # FCE 阶段二结果（300 epochs）
    ├── comparison_curves.png  # 对比曲线图
    └── comparison_summary.txt # 对比摘要文本
```

## 🔬 两阶段训练策略

### 阶段一：冻结预热

- 冻结层数：前 10 层（backbone）
- 训练轮次：30 epochs
- 学习率：0.01（较大）
- 目的：保护预训练权重，让新模块快速适应

### 阶段二：全局微调

- 冻结层数：无（全层训练）
- 训练轮次：300 epochs（与 Baseline 一致，确保公平对比）
- 学习率：0.001（较小）
- 学习率策略：余弦退火
- 目的：端到端完整训练

**总训练轮次**: 30 + 300 = 330 epochs（其中阶段二的 300 epochs 与 Baseline 的 300 epochs 相当）

## 🛠️ 添加新模型

### 1. 创建模型 YAML

在 `ultralytics/cfg/models/11/` 目录下创建新的模型配置文件。

### 2. 更新配置

在 `script/config.py` 的 `MODEL_CONFIGS` 中添加新模型：

```python
MODEL_CONFIGS["new"] = ModelConfig(
    name="new",
    yaml_path="ultralytics/cfg/models/11/yolo11-new.yaml",
    color="#FF0000",
    display_name=lambda s: f"YOLOv11{s.upper()}-New",
    use_two_stage=False,
    result_pattern="new_yolo11{scale}",
)
```

### 3. 使用新模型

```bash
python script/train.py new --scale s
python script/compare.py --models baseline new --scale s
```

## ⚡ 快速实验

### 单阶段快速训练

使用 `train_pro.py` 进行快速实验和单阶段训练：

```bash
# 单阶段训练（100 epochs 快速实验）
python script/train_pro.py --epochs 100 --batch 128

# 测试新模块
python script/train_pro.py --model ultralytics/cfg/models/11/yolo11-bifpn.yaml --epochs 50

# 超大批次训练（32GB 显存）
python script/train_pro.py --batch 128 --cache ram --workers 24

# 自动搜索最大批次
python script/train_pro.py --batch -1

# 高分辨率训练
python script/train_pro.py --imgsz 1280 --batch 64

# 从断点恢复训练
python script/train_pro.py --resume
```

### 调整学习率和其他参数

```bash
# 调整学习率
python script/train_pro.py --lr0 0.01 --lrf 0.01 --cos-lr

# 自定义实验名称
python script/train_pro.py --name my_experiment --epochs 500

# 单类别训练（调试用）
python script/train_pro.py --single-cls

# 禁用验证加速训练
python script/train_pro.py --no-val
```

## 📝 代码示例

### Python API 使用

```python
from script.config import MODEL_CONFIGS, TrainConfig
from script.trainer import train_model

# 训练模型
results = train_model(
    model_type="fce",
    scale="s",
    config=TrainConfig(batch=16, imgsz=640),
)
```

### 自定义训练配置

```python
from script.config import TrainConfig

config = TrainConfig(
    epochs=100,
    batch=16,
    imgsz=640,
    lr0=0.001,
)

results = train_model("fce", "s", config=config)
```

## 🐛 常见问题

### Q: CUDA OOM

```bash
# 减小批次大小
python script/train.py fce --scale s --batch 8
```

### Q: 训练速度慢

```bash
# 减小图像尺寸或批次大小
python script/train.py fce --scale s --imgsz 640 --batch 16
```

### Q: 路径不一致

```bash
# 运行测试工具检查
python script/test.py --config
```

## 📚 相关文档

- [项目主文档](../CLAUDE.md) - 项目架构和开发指南
- [COCO 训练指南](../TRAINING_GUIDE.md) - COCO 数据集训练完整指南
- [模型配置](../ultralytics/cfg/models/11/) - YOLOv11 模型配置文件
- [自定义模块](../ultralytics/nn/modules/fce_block.py) - FCE 自定义模块实现

## 🔄 脚本选择指南

| 脚本 | 适用场景 | 数据集 | 两阶段训练 | 对比分析 | 硬件优化 |
|------|---------|--------|-----------|---------|---------|
| `train_coco_compare.py` | COCO 完整对比实验 | COCO | ✅ 自动 | ✅ 自动 | ✅ RTX 5090 |
| `train.py` | 模块化单模型训练 | 任意 | ✅ 自动 | ❌ | ❌ |
| `train_pro.py` | 高性能单模型训练 | 任意 | ❌ | ❌ | ✅ RTX 5090 |
| `compare.py` | 通用多模型对比 | 任意 | ❌ | ✅ 自动 | ❌ |
| `test_train_pro.py` | 环境验证 | COCO | ❌ | ❌ | ✅ RTX 5090 |

### 选择建议

- **首次使用**：先运行 `test_train_pro.py` 验证环境
- **完整对比实验**：使用 `train_coco_compare.py`（推荐）
- **训练变体模型（BiFPN/FCE）**：使用 `train.py`（自动两阶段）
- **高性能单模型训练**：使用 `train_pro.py`（RTX 5090 优化）
- **已有结果对比**：使用 `compare.py --skip-train`

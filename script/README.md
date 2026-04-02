# YOLOv11 训练对比工具

统一的 YOLOv11 变体模型训练和对比工具包。

## 架构

```
script/
├── config.py    # 配置系统（StageConfig, TrainConfig, ModelConfig, 数据集预设）
├── trainer.py   # 训练器（YOLOv11Trainer）
├── analysis.py  # 对比分析（指标提取、曲线绘制、结果对比）
├── train.py     # 统一训练 CLI
├── compare.py   # 统一对比 CLI
└── test.py      # 配置测试
```

## 快速开始

```bash
# 验证配置
python script/test.py

# 单模型训练
python script/train.py fce --scale s

# 多模型对比
python script/compare.py --models baseline fce --scale s
```

## 训练 CLI

```bash
python script/train.py <MODEL> [OPTIONS]

# 位置参数
MODEL          模型类型 (baseline, bifpn, fce)

# 常用参数
--scale SCALE  模型尺度 n/s/m/l/x（默认: n）
--dataset PRESET  数据集预设 default/coco/coco_hq（默认: default）
--data PATH    自定义数据集路径（覆盖 --dataset）
--test         快速测试模式（1 epoch, 小图像尺寸）

# 共享参数（所有阶段生效）
--batch INT    批次大小
--imgsz INT    输入图像尺寸
--device DEV   训练设备
--workers INT  工作进程数
--iou-type TYPE  IoU 损失 CIoU/DIoU/GIoU/WIoU
--no-amp       禁用混合精度
--cache CACHE  数据缓存 false/disk/ram

# 阶段参数（覆盖 stage2）
--epochs INT   stage2 训练轮次
--lr0 FLOAT    stage2 初始学习率
--patience INT 早停耐心值
--cos-lr       stage2 余弦退火
--close-mosaic INT  最后 N epochs 关闭 Mosaic

# stage1 覆盖（高级用法）
--stage1-epochs INT
--stage1-lr0 FLOAT
--stage1-patience INT
```

## 对比 CLI

```bash
python script/compare.py --models <MODELS> [OPTIONS]

# 必选
--models MODELS [MODELS ...]  要对比的模型列表

# 常用参数（同训练 CLI）
--scale, --dataset, --data, --batch, --imgsz, --epochs, --iou-type ...

# 对比控制
--skip-train   跳过训练，仅对比已有结果
--output DIR   自定义输出目录
--test         快速测试模式
```

## 配置系统

### 参数分类

| 参数类别 | CLI 示例 | 影响范围 |
|---------|---------|---------|
| 共享参数 | `--batch`, `--imgsz`, `--device`, `--workers` | 所有阶段 |
| 阶段参数 | `--epochs`, `--lr0`, `--patience` | 仅 stage2 |
| stage1 覆盖 | `--stage1-epochs`, `--stage1-lr0` | 仅 stage1 |

- `--epochs 200` 对两阶段模型意味着 50+200，对单阶段模型是 200
- `--batch 16` 对所有阶段生效
- `--stage1-epochs 30` 仅影响 stage1

### 数据集预设

| 预设 | 数据集 | imgsz | batch | workers | cache |
|------|--------|-------|-------|---------|-------|
| `default` | 海西机械手 | 1280 | 32 | 16 | ram |
| `coco` | COCO | 640 | 16 | 8 | false |
| `coco_hq` | COCO 高性能 | 640 | 128 | 24 | ram |

### 两阶段训练

包含随机初始化模块的模型（bifpn, fce）自动执行两阶段训练：

- **阶段一（50 epochs）**：冻结 backbone（前 10 层），lr=0.01，线性衰减
- **阶段二（300 epochs）**：全层训练，lr=0.001，余弦退火

## 使用示例

```bash
# 基本训练
python script/train.py fce --scale s

# 覆盖共享参数
python script/train.py fce --scale s --batch 16 --imgsz 640

# 覆盖 stage2 轮次（50+200）
python script/train.py fce --scale s --epochs 200

# COCO 训练
python script/train.py fce --scale s --dataset coco

# 快速测试
python script/train.py fce --scale s --test

# 多模型对比
python script/compare.py --models baseline bifpn fce --scale s

# 跳过训练，仅对比
python script/compare.py --models baseline fce --scale s --skip-train

# 切换 IoU 损失
python script/compare.py --models baseline fce --scale s --iou-type WIoU
```

## 添加新模型

1. 在 `ultralytics/cfg/models/11/` 创建 YAML
2. 在 `script/config.py` 的 `MODEL_CONFIGS` 中添加配置
3. 设置 `stage1` 启用两阶段，或不设置保持单阶段
4. `python script/train.py your_model --scale n --test` 验证

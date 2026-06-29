# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Running Tests

```bash
# Run all tests (excludes slow tests by default)
pytest tests/

# Run tests including slow tests
pytest --slow tests/

# Run specific test file
pytest tests/test_engine.py -v

# Run tests with coverage
pytest --cov=ultralytics/ --cov-report=xml tests/
```

### Code Quality

```bash
# Format code with Ruff
ruff format .

# Lint code
ruff check .

# Check environment and dependencies
yolo checks
```

### Model Training/Testing

```bash
# Quick test train
yolo train model=yolo11n.pt data=coco8.yaml epochs=1 imgsz=32

# Train with WIoU v3 loss (instead of default CIoU)
yolo train model=yolo11n.pt data=coco8.yaml epochs=100 iou_type=WIoU

# Validate model
yolo val model=yolo11n.pt data=coco8.yaml

# Predict on image
yolo predict model=yolo11n.pt source=path/to/image.jpg

# Export model
yolo export model=yolo11n.pt format=onnx
```

### Installation

```bash
# Development installation
pip install -e .

# Install with extra dependencies
pip install -e ".[export,solutions,logging]"
```

> Git 提交规范请参考：`.claude/rules/git-commit.md`

## High-Level Architecture

### Core Components

**Engine Layer** (`ultralytics/engine/`)

- `Model`: Base class for all models, provides unified API for train/val/predict/export/tune
- `BaseTrainer`: Base class for training loops
- `BaseValidator`: Base class for validation
- `BasePredictor`: Base class for inference
- `Results`: Unified output class for all tasks

**Models Layer** (`ultralytics/models/`)

- `YOLO`: Main YOLO model class that auto-detects task type from filename
- `yolo/`: Task-specific implementations (detect, segment, classify, pose, obb)
- Each task has its own: trainer, validator, predictor, and model class

**Neural Network Layer** (`ultralytics/nn/`)

- `tasks.py`: Contains all model task implementations (DetectionModel, SegmentationModel, etc.)
- `modules/`: Reusable building blocks (Conv, C2f, Transformer blocks, etc.)
- `autobackend.py`: Handles inference on various model formats (PyTorch, ONNX, TensorRT, etc.)

**Configuration** (`ultralytics/cfg/`)

- `default.yaml`: All default training/validation/predict/export parameters
- Model YAMLs in `cfg/models/`: Define model architectures
- Dataset YAMLs in `cfg/datasets/`: Define dataset configurations

### Task System

The codebase supports 5 main tasks through a unified interface:

- `detect`: Object detection
- `segment`: Instance segmentation
- `classify`: Image classification
- `pose`: Pose estimation
- `obb`: Oriented bounding boxes

Each task is mapped to specific implementations via the `task_map` property in model classes.

### Model Variants

The architecture supports multiple YOLO versions and variants:

- YOLOv3, v5, v8, v9, v10, v11 (latest)
- YOLOWorld (vision-language)
- YOLOE (efficient variant)
- RTDETR (transformer-based)
- SAM, FastSAM (segmentation models)

Models are auto-detected from filename patterns (e.g., `yolo11n-seg.pt` loads a YOLO11 segmentation model).

### Configuration System

**Hierarchy**: Default config → Model config → User overrides

Configuration is handled through:

1. `ultralytics/cfg/default.yaml`: Base defaults
2. Model YAML files: Architecture-specific settings
3. CLI arguments/Python kwargs: Runtime overrides

Configuration uses `IterableSimpleNamespace` for convenient attribute access.

### Execution Flow

1. **CLI Entry**: `yolo TASK MODE ARGS` (e.g., `yolo detect train model=yolo11n.pt`)
2. **Argument Parsing**: `ultralytics/cfg/entrypoint()` parses and validates
3. **Model Loading**: `YOLO` class auto-detects model type and loads
4. **Task Dispatch**: Model method calls task-specific trainer/validator/predictor
5. **Results**: Unified `Results` class returns formatted outputs

### Key Abstractions

- **BaseModel** (`ultralytics/nn/tasks.py`): Base class for all model implementations
- **Model** (`ultralytics/engine/model.py`): Base class providing train/val/predict/export API
- **Task Maps**: Dictionary mapping tasks to their model/trainer/validator/predictor classes
- **Module Registry**: All neural network modules imported in `ultralytics/nn/modules/__init__.py`

### Design Patterns

- **Factory Pattern**: Model creation based on filename patterns
- **Strategy Pattern**: Task-specific implementations
- **Template Method**: Base classes with customizable hooks
- **Composition**: Modular architecture with reusable components

## Loss Function System

检测任务支持可配置的 IoU 损失类型，通过 `default.yaml` 中的 `iou_type` 参数控制：

| iou_type | 实现                  | 说明                        |
| -------- | --------------------- | --------------------------- |
| `CIoU`   | `bbox_iou(CIoU=True)` | 默认，Complete IoU          |
| `DIoU`   | `bbox_iou(DIoU=True)` | Distance IoU                |
| `GIoU`   | `bbox_iou(GIoU=True)` | Generalized IoU             |
| `WIoU`   | `bbox_wiou` + v3 聚焦 | Wise-IoU v3，动态非单调聚焦 |

使用方式：`yolo train model=yolo11n.pt data=coco8.yaml iou_type=WIoU`

**关键文件：**

- `ultralytics/utils/metrics.py` — `bbox_iou`、`bbox_wiou`（IoU 计算函数）
- `ultralytics/utils/loss.py` — `BboxLoss`（接收 `iou_type`，路由 IoU 计算）
- `ultralytics/cfg/default.yaml` — `iou_type: CIoU` 配置项

**配置传递链：** `default.yaml` → `model.args.iou_type` → `v8DetectionLoss.__init__` → `BboxLoss(iou_type=...)`

**注意：** `BboxLoss` 被 `v8DetectionLoss`（及其子类 Segmentation/Pose）使用；`RotatedBboxLoss` 使用独立的 `probiou`，不受 `iou_type` 影响。

## Custom FCE Modules

本项目在 `ultralytics/nn/modules/fce_block.py` 中实现了自定义特征增强模块：

| 模块                | 描述                                   | 参考                                             |
| ------------------- | -------------------------------------- | ------------------------------------------------ |
| **BiFPN_Concat**    | 可学习的加权特征融合，支持多尺度输入   | [EfficientDet](https://arxiv.org/abs/1911.09070) |
| **CoordAtt**        | 坐标注意力，分别捕获水平和垂直空间依赖 | [CoordAtt](https://arxiv.org/abs/2103.02907)     |
| **CoordCrossAtt**   | 坐标交叉注意力，跨方向特征交互         | 基于 CoordAtt 改进                               |
| **BiCoordCrossAtt** | 双向坐标交叉注意力，对称 H<->W 交互    | 基于 CoordAtt 改进                               |

### YAML 使用示例

```yaml
# BiFPN_Concat: 多层融合
- [[-1, 6], 1, BiFPN_Concat, []]

# CoordAtt: 坐标注意力
- [-1, 1, CoordAtt, [256, 16]] # oup=256, reduction=16

# CoordCrossAtt: 坐标交叉注意力
- [-1, 1, CoordCrossAtt, [256, 16, 2]] # oup=256, reduction=16, num_heads=2

# BiCoordCrossAtt: 双向坐标交叉注意力
- [-1, 1, BiCoordCrossAtt, [512, 16, 8]] # oup=512, reduction=16, num_heads=8
```

## Training Scripts

项目采用模块化训练架构，支持多种模型变体的训练和对比实验。

### 架构概览

- **`script/config.py`** - 配置系统（StageConfig, TrainConfig, ModelConfig, 数据集预设）
- **`script/trainer.py`** - 训练器（YOLOv11Trainer 类）
- **`script/analysis.py`** - 对比分析（指标提取、曲线绘制、结果对比）
- **`script/train.py`** - 统一训练 CLI
- **`script/compare.py`** - 统一对比 CLI
- **`script/test.py`** - 配置测试脚本

### 配置系统

参数分为三类：

| 参数类别    | CLI 示例                                                    | 影响范围  |
| ----------- | ----------------------------------------------------------- | --------- |
| 共享参数    | `--batch`, `--imgsz`, `--device`, `--workers`, `--iou-type` | 所有阶段  |
| 阶段参数    | `--epochs`, `--lr0`, `--patience`                           | 仅 stage2 |
| stage1 覆盖 | `--stage1-epochs`, `--stage1-lr0`                           | 仅 stage1 |

数据集预设：`--dataset default/coco/coco_hq`，自定义路径：`--data /path/to/data.yaml`

### 支持的模型类型

| 模型类型   | YAML 配置           | 两阶段训练  | 说明               |
| ---------- | ------------------- | ----------- | ------------------ |
| `baseline` | `yolo11.yaml`       | 否          | 原生 YOLOv11 模型  |
| `bifpn`    | `yolo11-bifpn.yaml` | 是 (50+300) | BiFPN 特征融合模型 |
| `fce`      | `yolo11-fce.yaml`   | 是 (50+300) | FCE 特征增强模型   |

### 单模型训练

```bash
# Baseline 训练（单阶段 300 epochs）
python script/train.py baseline --scale n

# FCE 两阶段训练（自动 50+300 epochs）
python script/train.py fce --scale s

# 覆盖共享参数
python script/train.py fce --scale s --batch 16 --imgsz 640

# 覆盖 stage2 轮次（fce: 50+200, baseline: 200）
python script/train.py fce --scale s --epochs 200

# 切换数据集预设
python script/train.py fce --scale s --dataset coco
python script/train.py fce --scale s --dataset coco_hq

# 自定义数据集路径
python script/train.py fce --scale s --data /path/to/data.yaml

# 快速测试（1 epoch，小图像尺寸）
python script/train.py fce --scale s --test
```

### 多模型对比训练

```bash
# Baseline vs FCE 对比
python script/compare.py --models baseline fce --scale s

# 三模型对比
python script/compare.py --models baseline bifpn fce --scale s

# 覆盖 epochs
python script/compare.py --models baseline fce --scale s --epochs 200

# 跳过训练，仅对比已有结果
python script/compare.py --models baseline bifpn --scale s --skip-train

# 切换 IoU 损失
python script/compare.py --models baseline bifpn fce --scale s --iou-type WIoU

# 切换数据集
python script/compare.py --models baseline fce --scale s --dataset coco_hq
```

### 对比结果输出

```
runs/detect/
├── baselinevsfce_s_300/
│   ├── baseline_yolo11s/         # Baseline 训练结果
│   ├── fce_s_stage1/             # FCE 阶段一结果
│   ├── fce_s_stage2/             # FCE 阶段二结果
│   ├── comparison_curves.png     # 对比曲线图（mAP@50-95, mAP@50, Precision, Recall）
│   └── comparison_summary.txt    # 对比摘要文本
```

### 配置测试

```bash
python script/test.py
```

预期输出：`✓ 所有测试通过!`

## Adding a New Task

1. Create `ultralytics/models/yolo/mytask/` directory
2. Implement: `train.py`, `val.py`, `predict.py`, `__init__.py`
3. Add task to `TASKS` in `ultralytics/cfg/__init__.py`
4. Update `task_map` in `ultralytics/models/yolo/model.py`

## Adding a New Model Variant

### 添加自定义模块

> 详细流程请参考：`.claude/skills/add-module/SKILL.md`

简要步骤：

1. 在 `ultralytics/nn/modules/fce_block.py` 中实现模块
2. 更新 `__all__` 导出列表
3. 在 `ultralytics/nn/tasks.py` 中导入模块
4. 在 `parse_model()` 方法中添加参数解析逻辑
5. 在模型 YAML 中使用
6. 更新文档

### 添加模型变体到训练脚本

如果要将自定义模型集成到训练脚本中：

1. **创建模型 YAML 配置**
   - 在 `ultralytics/cfg/models/11/` 目录下创建新的 YAML 文件
   - 定义 backbone 和 head 结构
   - 添加自定义模块

2. **更新模型配置**（`script/config.py`）

   ```python
   MODEL_CONFIGS: Dict[str, ModelConfig] = {
       # ... 现有配置
       "your_model": ModelConfig(
           name="your_model",
           yaml_path="ultralytics/cfg/models/11/yolo11-your-model.yaml",
           color="#FF0000",  # 图表颜色
           display_name=lambda s: f"YOLOv11{s.upper()} YourModel",
           freeze=10,  # stage1 冻结层数
           stage1=StageConfig(epochs=50, patience=20, lr0=0.01, cos_lr=False, close_mosaic=10),
           stage2=StageConfig(epochs=300, patience=50, lr0=0.001, cos_lr=True, close_mosaic=20),
           result_pattern="your_model_{scale}_stage2",
       ),
   }
   ```

3. **确定是否需要两阶段训练**
   - 如果模型包含随机初始化的模块（如 BiFPN、FCE），设置 `stage1=StageConfig(...)`
   - 如果只是结构调整但所有层都使用预训练权重，不设置 `stage1`（默认为 None，即单阶段）

4. **测试模型配置**

   ```bash
   # 快速测试
   python script/train.py your_model --scale n --test

   # 运行配置测试
   python script/test.py
   ```

5. **运行对比实验**
   ```bash
   python script/compare.py --models baseline your_model --scale s
   ```

## Important File Locations

### Core Files

- CLI entry point: `ultralytics/cfg/__init__.py` -> `entrypoint()`
- Model base class: `ultralytics/engine/model.py`
- YOLO model class: `ultralytics/models/yolo/model.py`
- Neural network implementations: `ultralytics/nn/tasks.py`
- Custom FCE modules: `ultralytics/nn/modules/fce_block.py`
- IoU functions: `ultralytics/utils/metrics.py` (`bbox_iou`, `bbox_wiou`)
- Loss functions: `ultralytics/utils/loss.py` (`BboxLoss`, `v8DetectionLoss`)
- Configuration: `ultralytics/cfg/default.yaml`
- Model architectures: `ultralytics/cfg/models/`
- Test configuration: `pytest.ini`, `tests/conftest.py`

### Training Scripts (模块化架构)

- **`script/config.py`** - 配置系统
  - `StageConfig`, `TrainConfig`, `ModelConfig` - 配置数据类
  - `MODEL_CONFIGS` - 预定义模型配置（baseline, bifpn, fce）
  - `DATASET_PRESETS` - 数据集预设（default, coco, coco_hq）
  - `apply_overrides()` - 配置覆盖逻辑
  - `build_overrides_from_namespace()` - argparse 参数转换
- **`script/trainer.py`** - 训练器
  - `YOLOv11Trainer` - YOLOv11 变体模型训练器（自动单阶段/两阶段）
- **`script/analysis.py`** - 对比分析（无状态纯函数）
  - `load_results()`, `extract_metrics()` - 指标提取
  - `print_comparison_table()`, `plot_comparison_curves()` - 对比展示
  - `reorganize_results()` - 结果整理
- **`script/train.py`** - 统一训练 CLI
- **`script/compare.py`** - 统一对比 CLI
- **`script/test.py`** - 配置测试脚本

### Development Logs

- **`analysis_log/`** - 开发分析日志目录
  - `README.md` - 日志目录说明
  - `20??-??-??-*/` - 日期格式的分析日志文件夹（已 gitignore）

### 模型配置文件

- **Baseline:** `ultralytics/cfg/models/11/yolo11.yaml`
- **BiFPN:** `ultralytics/cfg/models/11/yolo11-bifpn.yaml`
- **FCE:** `ultralytics/cfg/models/11/yolo11-fce.yaml`

## Testing Notes

- Tests marked with `@pytest.mark.slow` are skipped by default
- Use `pytest --slow` to run all tests
- Tests automatically clean up temporary files/directories
- GPU tests are in `tests/test_cuda.py`
- CI runs on multiple OS, Python, and PyTorch versions

## Documentation Style

- Use Google-style docstrings
- Type hints encouraged (see CONTRIBUTING.md for examples)
- Docstrings should include: Args, Returns, and Examples sections

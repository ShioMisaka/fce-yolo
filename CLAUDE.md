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

## Git Commit Conventions

所有提交使用中文，遵循 **Conventional Commits** 格式：

### 类型

- `feat:` - 新功能
- `fix:` - 修复 bug
- `docs:` - 文档变更
- `refactor:` - 代码重构（无功能变化）
- `test:` - 添加或更新测试
- `chore:` - 维护任务、依赖、配置

### 规则

1. **标题行**：不超过 50 字符
2. **正文**：多文件变更时必须用列表列出
3. **无 AI 标识**：不包含 "生成于 AI"、"AI 协作" 等信息

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

### 添加新模块

> 详细流程请参考：`.claude/rules/add-modules.md`

简要步骤：

1. 在 `ultralytics/nn/modules/fce_block.py` 中实现模块
2. 更新 `__all__` 导出列表
3. 在 `ultralytics/nn/tasks.py` 中导入模块
4. 在 `parse_model()` 方法中添加参数解析逻辑
5. 在模型 YAML 中使用
6. 更新文档

## Adding a New Task

1. Create `ultralytics/models/yolo/mytask/` directory
2. Implement: `train.py`, `val.py`, `predict.py`, `__init__.py`
3. Add task to `TASKS` in `ultralytics/cfg/__init__.py`
4. Update `task_map` in `ultralytics/models/yolo/model.py`

## Adding a New Model Variant

1. Create model YAML in `ultralytics/cfg/models/`
2. Implement custom trainer/validator/predictor if needed
3. Create model class in `ultralytics/models/`

## Important File Locations

- CLI entry point: `ultralytics/cfg/__init__.py` -> `entrypoint()`
- Model base class: `ultralytics/engine/model.py`
- YOLO model class: `ultralytics/models/yolo/model.py`
- Neural network implementations: `ultralytics/nn/tasks.py`
- Custom FCE modules: `ultralytics/nn/modules/fce_block.py`
- Configuration: `ultralytics/cfg/default.yaml`
- Model architectures: `ultralytics/cfg/models/`
- Test configuration: `pytest.ini`, `tests/conftest.py`

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

# Script 目录重构设计

## 目标

将 `script/` 目录中 5 个训练/对比入口 + 3 套配置系统 + 3 份重复分析代码，重构为统一的模块化架构。

## 当前问题

1. **代码重复**：`load_results`、`extract_metrics`、`plot_comparison_curves` 等在 3 个 compare 脚本中各复制一遍
2. **三套 ModelConfig**：`config.py`（dataclass）、`train_coco_compare.py`（普通 class）、`train_pro.py`（无配置对象）
3. **三套配置系统**：`TrainConfig` dataclass、`COCO_CONFIG` 纯 dict、argparse 裸参数
4. **训练逻辑重复**：`trainer.py`、`coco_train.py`、`train_coco_compare.py` 各自实现两阶段训练
5. **入口过多**：train.py / coco_train.py / train_pro.py / compare.py / train_coco_compare.py / compare_iou.py

## 重构后文件结构

```
script/
├── __init__.py          # 包入口
├── config.py            # 配置系统
├── trainer.py           # 训练逻辑
├── analysis.py          # 对比分析（新增，提取公共逻辑）
├── train.py             # 统一训练 CLI
├── compare.py           # 统一对比 CLI
├── test.py              # 配置测试
└── README.md            # 使用文档
```

### 删除的文件

- `coco_train.py` — 合并到 `train.py`
- `train_pro.py` — 合并到 `train.py`
- `train_coco_compare.py` — 合并到 `compare.py`
- `compare_iou.py` — 合并到 `compare.py`
- `test_train_pro.py` — 一次性脚本
- `verify_fix.py` — 一次性脚本
- `test_two_stage_config.py` — 合并到 `test.py`
- `REFACTOR.md`、`TRAINING_STRATEGY.md`、`COCO_TRAIN_README.md` — 合并到 README.md

## 配置系统 (`config.py`)

### 参数分类

**共享参数**：batch, imgsz, device, workers, amp, cache, optimizer, lrf, momentum, weight_decay, iou_type 等。所有阶段共用，override 直接生效。

**阶段参数**：epochs, lr0, cos_lr, patience, close_mosaic。各阶段有独立默认值，`--epochs N` 等默认只覆盖 stage2。

### 数据类定义

```python
@dataclass
class StageConfig:
    """阶段训练参数."""

    epochs: int = 300
    patience: int = 50
    lr0: float = 0.001
    cos_lr: bool = True
    close_mosaic: int = 20


@dataclass
class TrainConfig:
    """完整训练配置."""

    # 数据集
    data: str = ""
    # 共享参数
    batch: int = 32
    imgsz: int = 640
    device: str = "0"
    workers: int = 8
    amp: bool = True
    cache: str = "false"
    optimizer: str = "AdamW"
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    iou_type: str = "CIoU"
    # 保存和日志
    project: str = "runs/detect"
    save_period: int = -1
    exist_ok: bool = True
    verbose: bool = True
    plots: bool = True
    # 两阶段（stage1=None 表示单阶段）
    stage1: Optional[StageConfig] = None
    stage2: StageConfig = field(default_factory=StageConfig)

    def to_dict(self) -> Dict:
        """转换为 YOLO train() 参数字典."""


@dataclass
class ModelConfig:
    """模型变体配置."""

    name: str
    yaml_path: str
    color: str
    display_name: Callable[[str], str]
    freeze: int = 0
    stage1: Optional[StageConfig] = None
    stage2: StageConfig = field(default_factory=StageConfig)
    result_pattern: str = ""

    def is_two_stage(self) -> bool:
        return self.stage1 is not None
```

### 数据集预设

```python
DATASET_PRESETS = {
    "default": TrainConfig(
        data="/mnt/ssd1/Dataset/haixi_jixieshou/yolo_dataset/data.yaml",
        imgsz=1280,
        batch=32,
        workers=16,
        cache=True,
    ),
    "coco": TrainConfig(
        data="coco.yaml",
        imgsz=640,
        batch=16,
        workers=8,
        cache=False,
    ),
    "coco_hq": TrainConfig(
        data="ultralytics/cfg/datasets/coco_custom.yaml",
        imgsz=640,
        batch=128,
        workers=24,
        cache="ram",
    ),
}
```

### 模型配置

```python
MODEL_CONFIGS = {
    "baseline": ModelConfig(
        name="baseline",
        yaml_path="ultralytics/cfg/models/11/yolo11.yaml",
        color="#0BDBEB",
        display_name=lambda s: f"YOLOv11{s.upper()} Baseline",
        stage1=None,  # 单阶段
        stage2=StageConfig(epochs=300, patience=50, lr0=0.001, cos_lr=True, close_mosaic=20),
        result_pattern="baseline_yolo11{scale}",
    ),
    "bifpn": ModelConfig(
        name="bifpn",
        yaml_path="ultralytics/cfg/models/11/yolo11-bifpn.yaml",
        color="#042AFF",
        display_name=lambda s: f"YOLOv11{s.upper()}-BiFPN",
        freeze=10,
        stage1=StageConfig(epochs=50, patience=20, lr0=0.01, cos_lr=False, close_mosaic=10),
        stage2=StageConfig(epochs=300, patience=50, lr0=0.001, cos_lr=True, close_mosaic=20),
        result_pattern="bifpn_{scale}_stage2",
    ),
    "fce": ModelConfig(
        name="fce",
        yaml_path="ultralytics/cfg/models/11/yolo11-fce.yaml",
        color="#FF6B00",
        display_name=lambda s: f"YOLOv11{s.upper()}-FCE",
        freeze=10,
        stage1=StageConfig(epochs=50, patience=20, lr0=0.01, cos_lr=False, close_mosaic=10),
        stage2=StageConfig(epochs=300, patience=50, lr0=0.001, cos_lr=True, close_mosaic=20),
        result_pattern="fce_{scale}_stage2",
    ),
}
```

### Override 逻辑

```python
def apply_overrides(config: TrainConfig, model_cfg: ModelConfig, shared: dict, stage2: dict, stage1: dict):
    # 1. 从 model_cfg 填充两阶段配置
    if model_cfg.is_two_stage():
        config.stage1 = dataclasses.replace(model_cfg.stage1)
    config.stage2 = dataclasses.replace(model_cfg.stage2)
    # 2. 共享参数直接覆盖
    for k, v in shared.items():
        setattr(config, k, v)
    # 3. 阶段参数只覆盖 stage2
    for k, v in stage2.items():
        setattr(config.stage2, k, v)
    # 4. 显式 stage1 覆盖
    if stage1 and config.stage1:
        for k, v in stage1.items():
            setattr(config.stage1, k, v)
```

CLI 参数到三类覆盖的映射：

| CLI 参数                                                                                       | 分类   | 影响范围 |
| ---------------------------------------------------------------------------------------------- | ------ | -------- |
| `--batch`, `--imgsz`, `--device`, `--workers`, `--amp`, `--cache`, `--optimizer`, `--iou-type` | 共享   | 所有阶段 |
| `--epochs`, `--lr0`, `--patience`, `--cos-lr`                                                  | 阶段   | stage2   |
| `--stage1-epochs`, `--stage1-lr0`, `--stage1-patience`                                         | stage1 | stage1   |

## 训练器 (`trainer.py`)

```python
class YOLOv11Trainer:
    def __init__(self, model_cfg: ModelConfig, scale: str, config: TrainConfig):
        self.model_cfg = model_cfg
        self.scale = scale
        self.config = config
        self.pretrained = f"yolo11{scale}.pt"

    def _build_train_args(self, stage_config: StageConfig, freeze: int = 0, name: str = "") -> dict:
        """合并共享参数 + 阶段参数为 YOLO train() 字典."""

    def train(self) -> Union[Path, Dict[str, Path]]:
        """根据 model_cfg 自动选择单阶段/两阶段."""

    def _train_single_stage(self) -> Path:
        """单阶段训练."""

    def _train_two_stage(self) -> Dict[str, Path]:
        """两阶段训练：stage1(冻结预热) → stage2(全局微调)."""
```

关键设计：

- 配置在构造时已确定，训练方法无配置逻辑
- `_build_train_args` 负责 TrainConfig + StageConfig → dict 的合并
- stage2 通过 `YOLO(s1_weights)` 加载阶段一权重
- 不再提供 `train_model()` 便捷函数，调用方直接构造 `YOLOv11Trainer`

## 对比分析 (`analysis.py`)

从三个 compare 脚本中提取的公共逻辑，全部为无状态纯函数：

```python
# 指标提取
def load_results(csv_path: Path) -> pd.DataFrame
def extract_metrics(df: pd.DataFrame) -> Dict[str, float]

# 对比展示
def print_comparison_table(metrics, names, title="")
def plot_comparison_curves(dataframes, names, colors, save_path, title="")
def save_comparison_summary(output_path, metrics, names, config_info)

# 结果整理
def reorganize_results(result_paths, output_dir) -> Dict[str, Path]
```

不依赖任何特定的 Config 类，用 dict/基础类型传参。

## 统一训练 CLI (`train.py`)

合并 train.py + coco_train.py + train_pro.py。

```bash
# 基本训练
python script/train.py baseline --scale s
python script/train.py fce --scale s

# 覆盖共享参数
python script/train.py fce --scale s --batch 16 --imgsz 640

# 覆盖 stage2 轮次（fce: 50+200, baseline: 200）
python script/train.py fce --scale s --epochs 200

# 显式改 stage1
python script/train.py fce --scale s --stage1-epochs 30

# 切换数据集预设
python script/train.py fce --scale s --dataset coco
python script/train.py fce --scale s --dataset coco_hq

# 自定义数据集路径
python script/train.py fce --scale s --data /path/to/data.yaml

# 快速测试
python script/train.py fce --scale s --test
```

内部流程：parse_args → get_dataset_preset → apply_overrides → YOLOv11Trainer.train()

## 统一对比 CLI (`compare.py`)

合并 compare.py + train_coco_compare.py + compare_iou.py。

```bash
# 基本对比
python script/compare.py --models baseline fce --scale s

# 覆盖 epochs
python script/compare.py --models baseline fce --scale s --epochs 200

# 三模型对比
python script/compare.py --models baseline bifpn fce --scale s

# 跳过训练
python script/compare.py --models baseline fce --scale s --skip-train

# 切换 IoU 损失
python script/compare.py --models baseline bifpn fce --scale s --iou-type WIoU

# 切换数据集
python script/compare.py --models baseline fce --scale s --dataset coco_hq
```

内部流程：逐模型训练 → reorganize_results → analysis.py 展示

## 需要同步更新的文档

- `CLAUDE.md` — Training Scripts 章节
- `notes/TRAINING_GUIDE.md` — 所有命令示例
- `script/README.md` — 完整重写

# Script 目录重构实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `script/` 目录中 5 个训练/对比入口 + 3 套配置系统重构为统一的模块化架构。

**Architecture:** 自底向上重构 — 先重建核心模块（config → analysis → trainer），再重建 CLI 入口（train → compare），最后清理旧文件和更新文档。每个任务完成后可独立验证。

**Tech Stack:** Python 3, dataclasses, argparse, ultralytics YOLO, matplotlib, pandas

**Spec:** `docs/superpowers/specs/2026-04-01-script-refactor-design.md`

---

### Task 1: 重写 `config.py` — 配置系统

**Files:**
- Modify: `script/config.py`（完整重写）
- Reference: `docs/superpowers/specs/2026-04-01-script-refactor-design.md`（配置系统章节）

- [ ] **Step 1: 重写 `script/config.py`**

完整重写为以下内容：

```python
"""
配置管理模块

定义训练配置、模型配置、数据集预设，以及配置覆盖逻辑。
"""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional


# ==================== 阶段配置 ====================

@dataclass
class StageConfig:
    """阶段训练参数（lr, epochs, patience 等各阶段独立的参数）"""
    epochs: int = 300
    patience: int = 50
    lr0: float = 0.001
    cos_lr: bool = True
    close_mosaic: int = 20


# ==================== 训练配置 ====================

@dataclass
class TrainConfig:
    """完整训练配置

    参数分为三类：
    - 共享参数：所有阶段共用（batch, imgsz, device 等）
    - 阶段参数：各阶段独立（通过 stage1/stage2 的 StageConfig 管理）
    - 保存和日志参数
    """
    # 数据集
    data: str = ""

    # 共享参数（所有阶段共用，override 直接生效）
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

    # 两阶段配置（stage1=None 表示单阶段训练）
    stage1: Optional[StageConfig] = None
    stage2: StageConfig = field(default_factory=StageConfig)

    def to_dict(self) -> Dict:
        """转换为 YOLO train() 接受的字典"""
        d = {
            "data": self.data,
            "batch": self.batch,
            "imgsz": self.imgsz,
            "device": self.device,
            "workers": self.workers,
            "amp": self.amp,
            "cache": self.cache if self.cache != "false" else False,
            "optimizer": self.optimizer,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "iou_type": self.iou_type,
            "project": self.project,
            "save_period": self.save_period,
            "exist_ok": self.exist_ok,
            "verbose": self.verbose,
            "plots": self.plots,
        }
        return d


# ==================== 模型配置 ====================

@dataclass
class ModelConfig:
    """模型变体配置"""
    name: str
    yaml_path: str
    color: str
    display_name: Callable[[str], str]
    freeze: int = 0
    stage1: Optional[StageConfig] = None
    stage2: StageConfig = field(default_factory=StageConfig)
    result_pattern: str = ""

    def get_display_name(self, scale: str) -> str:
        """获取显示名称"""
        return self.display_name(scale)

    def is_two_stage(self) -> bool:
        """是否为两阶段训练"""
        return self.stage1 is not None

    def get_result_path(self, scale: str, stage: Optional[int] = None) -> str:
        """获取结果目录路径

        Args:
            scale: 模型尺度
            stage: 阶段编号（1 或 2），None 表示最终结果（stage2 或单阶段）
        """
        pattern = self.result_pattern
        if stage is not None:
            pattern = pattern.replace("_stage2", f"_stage{stage}")
        return pattern.format(scale=scale)


def get_model_config(model_type: str) -> ModelConfig:
    """获取模型配置

    Args:
        model_type: 模型类型

    Returns:
        模型配置对象

    Raises:
        ValueError: 未知模型类型
    """
    if model_type not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"未知模型类型: {model_type}，可选: {available}")
    return MODEL_CONFIGS[model_type]


# ==================== 模型配置表 ====================

MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "baseline": ModelConfig(
        name="baseline",
        yaml_path="ultralytics/cfg/models/11/yolo11.yaml",
        color="#0BDBEB",
        display_name=lambda s: f"YOLOv11{s.upper()} Baseline",
        stage1=None,
        stage2=StageConfig(epochs=300, patience=50, lr0=0.001,
                           cos_lr=True, close_mosaic=20),
        result_pattern="baseline_yolo11{scale}",
    ),
    "bifpn": ModelConfig(
        name="bifpn",
        yaml_path="ultralytics/cfg/models/11/yolo11-bifpn.yaml",
        color="#042AFF",
        display_name=lambda s: f"YOLOv11{s.upper()}-BiFPN",
        freeze=10,
        stage1=StageConfig(epochs=50, patience=20, lr0=0.01,
                           cos_lr=False, close_mosaic=10),
        stage2=StageConfig(epochs=300, patience=50, lr0=0.001,
                           cos_lr=True, close_mosaic=20),
        result_pattern="bifpn_{scale}_stage2",
    ),
    "fce": ModelConfig(
        name="fce",
        yaml_path="ultralytics/cfg/models/11/yolo11-fce.yaml",
        color="#FF6B00",
        display_name=lambda s: f"YOLOv11{s.upper()}-FCE",
        freeze=10,
        stage1=StageConfig(epochs=50, patience=20, lr0=0.01,
                           cos_lr=False, close_mosaic=10),
        stage2=StageConfig(epochs=300, patience=50, lr0=0.001,
                           cos_lr=True, close_mosaic=20),
        result_pattern="fce_{scale}_stage2",
    ),
}


# ==================== 数据集预设 ====================

DATASET_PRESETS: Dict[str, TrainConfig] = {
    "default": TrainConfig(
        data="/mnt/ssd1/Dataset/haixi_jixieshou/yolo_dataset/data.yaml",
        imgsz=1280,
        batch=32,
        workers=16,
        cache="ram",
    ),
    "coco": TrainConfig(
        data="coco.yaml",
        imgsz=640,
        batch=16,
        workers=8,
        cache="false",
    ),
    "coco_hq": TrainConfig(
        data="ultralytics/cfg/datasets/coco_custom.yaml",
        imgsz=640,
        batch=128,
        workers=24,
        cache="ram",
    ),
}


def get_dataset_preset(name: str) -> TrainConfig:
    """获取数据集预设配置

    Args:
        name: 预设名称 (default/coco/coco_hq)

    Returns:
        训练配置副本
    """
    if name not in DATASET_PRESETS:
        available = ", ".join(DATASET_PRESETS.keys())
        raise ValueError(f"未知数据集预设: {name}，可选: {available}")
    return dataclasses.replace(DATASET_PRESETS[name])


def get_quick_test_config() -> TrainConfig:
    """获取快速测试配置"""
    return TrainConfig(
        imgsz=64,
        batch=2,
        save_period=1,
        stage2=StageConfig(
            epochs=1,
            patience=10,
            close_mosaic=0,
        ),
    )


# ==================== 配置覆盖逻辑 ====================

# CLI 参数分类
SHARED_PARAMS = {
    "batch", "imgsz", "device", "workers", "amp", "cache",
    "optimizer", "lrf", "momentum", "weight_decay", "iou_type",
    "project", "save_period", "verbose", "plots",
}

STAGE2_PARAMS = {
    "epochs", "lr0", "patience", "cos_lr", "close_mosaic",
}


def apply_overrides(
    config: TrainConfig,
    model_cfg: ModelConfig,
    shared: Optional[Dict] = None,
    stage2: Optional[Dict] = None,
    stage1: Optional[Dict] = None,
) -> TrainConfig:
    """应用用户覆盖到训练配置

    流程：
    1. 从 model_cfg 填充两阶段配置
    2. 共享参数直接覆盖 config 属性
    3. 阶段参数只覆盖 stage2
    4. 显式 stage1 覆盖

    Args:
        config: 基础训练配置（通常来自数据集预设）
        model_cfg: 模型配置（包含 stage1/stage2 默认值）
        shared: 共享参数覆盖 {param_name: value}
        stage2: stage2 参数覆盖 {param_name: value}
        stage1: stage1 参数覆盖 {param_name: value}

    Returns:
        覆盖后的 TrainConfig（新对象，不修改输入）
    """
    shared = shared or {}
    stage2 = stage2 or {}
    stage1 = stage1 or {}

    # 先创建副本，避免修改输入对象
    config = dataclasses.replace(config)

    # 1. 从 model_cfg 填充两阶段配置
    if model_cfg.is_two_stage():
        config.stage1 = dataclasses.replace(model_cfg.stage1)
    config.stage2 = dataclasses.replace(model_cfg.stage2)

    # 2. 共享参数直接覆盖
    for k, v in shared.items():
        if k in SHARED_PARAMS:
            setattr(config, k, v)
        else:
            # 也接受直接覆盖 data 等非分类参数
            if hasattr(config, k):
                setattr(config, k, v)

    # 3. 阶段参数只覆盖 stage2
    for k, v in stage2.items():
        if hasattr(config.stage2, k):
            setattr(config.stage2, k, v)

    # 4. 显式 stage1 覆盖
    if stage1 and config.stage1 is not None:
        for k, v in stage1.items():
            if hasattr(config.stage1, k):
                setattr(config.stage1, k, v)

    return config


def build_overrides_from_namespace(
    args: argparse.Namespace,
    shared_keys: dict = None,
    stage2_keys: dict = None,
    stage1_prefix: str = "stage1_",
) -> tuple:
    """从 argparse.Namespace 构建三类 override 字典

    Args:
        args: argparse 解析结果
        shared_keys: {arg_attr: config_attr} 映射，默认为标准共享参数
        stage2_keys: {arg_attr: config_attr} 映射，默认为标准阶段参数
        stage1_prefix: stage1 参数的前缀

    Returns:
        (shared_dict, stage2_dict, stage1_dict)
    """
    if shared_keys is None:
        shared_keys = {
            "batch": "batch", "imgsz": "imgsz", "device": "device",
            "workers": "workers", "iou_type": "iou_type", "cache": "cache",
        }
    if stage2_keys is None:
        stage2_keys = {
            "epochs": "epochs", "lr0": "lr0", "patience": "patience",
            "cos_lr": "cos_lr", "close_mosaic": "close_mosaic",
        }

    shared = {}
    for arg_attr, cfg_attr in shared_keys.items():
        val = getattr(args, arg_attr, None)
        if val is not None and val is not False:
            shared[cfg_attr] = val

    # data 参数特殊处理
    if getattr(args, "data", None) is not None:
        shared["data"] = args.data

    # 处理 boolean 标志（如 --no-amp）
    no_flags = {"no_amp": "amp"}
    for flag, cfg_attr in no_flags.items():
        if getattr(args, flag, False):
            shared[cfg_attr] = False

    stage2 = {}
    for arg_attr, cfg_attr in stage2_keys.items():
        val = getattr(args, arg_attr, None)
        if val is not None:
            stage2[cfg_attr] = val

    # 处理 --no-cos-lr 对 stage2 的影响
    if getattr(args, "no_cos_lr", False):
        stage2["cos_lr"] = False
    elif getattr(args, "cos_lr", None):
        stage2["cos_lr"] = True

    stage1 = {}
    # 收集所有 stage1_ 前缀的参数
    for attr in dir(args):
        if attr.startswith(stage1_prefix):
            val = getattr(args, attr, None)
            if val is not None:
                stage1[attr[len(stage1_prefix):]] = val

    return shared, stage2, stage1
```

- [ ] **Step 2: 验证 config.py 可以正常导入**

Run: `cd /home/cll/workspace/my_project/fce-yolo && python -c "from script.config import MODEL_CONFIGS, TrainConfig, StageConfig, ModelConfig, get_model_config, get_dataset_preset, apply_overrides, get_quick_test_config; print('OK')"`
Expected: `OK`

- [ ] **Step 3: 提交**

```bash
git add script/config.py
git commit -m "refactor: 重写配置系统，支持共享/阶段参数分离和数据集预设"
```

---

### Task 2: 创建 `analysis.py` — 对比分析模块

**Files:**
- Create: `script/analysis.py`
- Reference: 旧 `script/compare.py` 的 `load_results`、`extract_metrics`、`print_comparison_table`、`plot_comparison_curves`、`save_comparison_summary`、`reorganize_results`

- [ ] **Step 1: 创建 `script/analysis.py`**

从现有 `compare.py` 中提取并泛化所有对比分析函数：

```python
"""
对比分析模块

提供训练结果的加载、指标提取、对比展示等功能。
所有函数为无状态纯函数，不依赖特定的配置类。
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ==================== 指标提取 ====================

def load_results(csv_path: Path) -> pd.DataFrame:
    """加载训练结果 CSV 文件

    Args:
        csv_path: results.csv 路径

    Returns:
        训练结果 DataFrame
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"结果文件不存在: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ 加载: {csv_path}")
    print(f"  训练轮次: {int(df['epoch'].iloc[-1])} epochs")
    return df


def extract_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """提取关键指标

    Args:
        df: 训练结果 DataFrame

    Returns:
        指标字典（包含 final/best mAP、precision、recall）
    """
    metrics = {}

    last_row = df.iloc[-1]
    metrics["final_epoch"] = int(last_row["epoch"])
    metrics["final_map50_95"] = last_row.get("metrics/mAP50-95(B)", 0)
    metrics["final_map50"] = last_row.get("metrics/mAP50(B)", 0)
    metrics["final_precision"] = last_row.get("metrics/precision(B)", 0)
    metrics["final_recall"] = last_row.get("metrics/recall(B)", 0)

    if "metrics/mAP50-95(B)" in df.columns:
        best_idx = df["metrics/mAP50-95(B)"].idxmax()
        best_row = df.loc[best_idx]
        metrics["best_map50_95"] = best_row["metrics/mAP50-95(B)"]
        metrics["best_map50_95_epoch"] = int(best_row["epoch"])
    else:
        metrics["best_map50_95"] = 0
        metrics["best_map50_95_epoch"] = 0

    if "metrics/mAP50(B)" in df.columns:
        best_idx = df["metrics/mAP50(B)"].idxmax()
        best_row = df.loc[best_idx]
        metrics["best_map50"] = best_row["metrics/mAP50(B)"]
        metrics["best_map50_epoch"] = int(best_row["epoch"])
    else:
        metrics["best_map50"] = 0
        metrics["best_map50_epoch"] = 0

    return metrics


# ==================== 对比展示 ====================

METRICS_TO_COMPARE = [
    ("Best mAP@50-95", "best_map50_95"),
    ("Best mAP@50", "best_map50"),
    ("Final mAP@50-95", "final_map50_95"),
    ("Final mAP@50", "final_map50"),
    ("Final Precision", "final_precision"),
    ("Final Recall", "final_recall"),
]


def print_comparison_table(
    metrics: Dict[str, Dict[str, float]],
    names: Dict[str, str],
    title: str = "",
):
    """打印终端对比表格

    Args:
        metrics: {key: {metric_key: value}}
        names: {key: 显示名称}
        title: 表格标题
    """
    keys = list(metrics.keys())
    width = 20 * (len(keys) + 2)

    print(f"\n{'=' * width}")
    if title:
        print(title)
    print(f"{'=' * width}")

    # 表头
    header = f"{'指标':<20}"
    for k in keys:
        header += f"{names.get(k, k.upper()):<20}"
    header += f"{'最佳':<15}"
    print(header)
    print("-" * width)

    for metric_name, metric_key in METRICS_TO_COMPARE:
        row = f"{metric_name:<20}"
        values = {}
        for k in keys:
            value = metrics[k].get(metric_key, 0)
            values[k] = value
            row += f"{value:<20.4f}"
        best = max(values.keys(), key=lambda k: values[k])
        row += f"{names.get(best, best.upper()):<15}"
        print(row)

    print("=" * width)


def plot_comparison_curves(
    dataframes: Dict[str, pd.DataFrame],
    names: Dict[str, str],
    colors: Dict[str, str],
    save_path: Path,
    title: str = "",
):
    """绘制 4 合 1 对比曲线图

    Args:
        dataframes: {key: DataFrame}
        names: {key: 显示名称}
        colors: {key: 颜色}
        save_path: 图片保存路径
        title: 图表标题
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "lines.linewidth": 2,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), tight_layout=True)

    metrics_config = [
        ("metrics/mAP50-95(B)", "mAP@50-95"),
        ("metrics/mAP50(B)", "mAP@50"),
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall"),
    ]

    for idx, (metric_key, metric_name) in enumerate(metrics_config):
        ax = axes[idx // 2, idx % 2]
        for k, df in dataframes.items():
            ax.plot(
                df["epoch"],
                df[metric_key],
                color=colors.get(k, "#000000"),
                label=names.get(k, k),
                linewidth=2,
            )
        ax.set_title(f"{metric_name} 对比", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.2)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\n✓ 对比曲线已保存: {save_path}")
    plt.close()


def save_comparison_summary(
    output_path: Path,
    metrics: Dict[str, Dict[str, float]],
    names: Dict[str, str],
    config_info: dict,
):
    """保存对比摘要到文本文件

    Args:
        output_path: 输出文件路径
        metrics: {key: {metric_key: value}}
        names: {key: 显示名称}
        config_info: 配置信息（自由格式，写入文件头部）
    """
    keys = list(metrics.keys())

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("训练结果对比\n")
        f.write("=" * 100 + "\n\n")

        # 配置信息
        for k, v in config_info.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

        # 模型信息
        for k in keys:
            f.write(f"  - {names.get(k, k.upper())}\n")
        f.write("\n")

        # 对比表格
        f.write("-" * 100 + "\n")
        header = f"{'指标':<25}"
        for k in keys:
            header += f"{names.get(k, k.upper()):<20}"
        f.write(header + "\n")
        f.write("-" * 100 + "\n")

        for metric_name, metric_key in METRICS_TO_COMPARE:
            row = f"{metric_name:<25}"
            for k in keys:
                value = metrics[k].get(metric_key, 0)
                row += f"{value:<20.4f}"
            f.write(row + "\n")

        f.write("-" * 100 + "\n")
        f.write("=" * 100 + "\n")

    print(f"✓ 对比摘要已保存: {output_path}")


# ==================== 结果整理 ====================

def reorganize_results(
    result_paths: Dict[str, str],
    output_dir: Path,
) -> Dict[str, Path]:
    """移动训练结果到统一目录

    Args:
        result_paths: {key: 原始结果目录路径}
        output_dir: 统一输出目录

    Returns:
        {key: csv_path} 映射
    """
    import shutil

    print("\n" + "=" * 60)
    print("整理训练结果...")
    print("=" * 60)

    results_paths = {}

    for key, src_pattern in result_paths.items():
        src_path = Path("runs/detect") / src_pattern
        dst_path = output_dir / src_pattern

        if src_path.exists():
            if dst_path.exists():
                shutil.rmtree(dst_path)
            shutil.move(str(src_path), str(dst_path))
            print(f"✓ 移动: {src_path.name} -> {dst_path}")

            csv = dst_path / "results.csv"
            if csv.exists():
                results_paths[key] = csv
        else:
            print(f"⚠ 跳过: {src_path} (不存在)")

    return results_paths
```

- [ ] **Step 2: 验证 analysis.py 可以正常导入**

Run: `cd /home/cll/workspace/my_project/fce-yolo && python -c "from script.analysis import load_results, extract_metrics, print_comparison_table, plot_comparison_curves, save_comparison_summary, reorganize_results; print('OK')"`
Expected: `OK`

- [ ] **Step 3: 提交**

```bash
git add script/analysis.py
git commit -m "feat: 添加对比分析模块，提取公共的指标提取和对比展示逻辑"
```

---

### Task 3: 重写 `trainer.py` — 训练器

**Files:**
- Modify: `script/trainer.py`（完整重写）

- [ ] **Step 1: 重写 `script/trainer.py`**

```python
"""
训练器模块

提供 YOLOv11 变体模型的训练功能，支持单阶段和两阶段训练。
"""

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Union

from ultralytics import YOLO

from .config import ModelConfig, StageConfig, TrainConfig


class YOLOv11Trainer:
    """YOLOv11 变体模型训练器

    根据 ModelConfig 自动选择单阶段或两阶段训练。
    所有配置在构造时确定，训练方法内无配置逻辑。
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        scale: str,
        config: TrainConfig,
    ):
        """初始化训练器

        Args:
            model_cfg: 模型配置（从 MODEL_CONFIGS 获取）
            scale: 模型尺度 (n, s, m, l, x)
            config: 训练配置（已合并 override）
        """
        self.model_cfg = model_cfg
        self.scale = scale
        self.config = config
        self.pretrained = f"yolo11{scale}.pt"

    def _build_train_args(
        self,
        stage_config: StageConfig,
        freeze: int = 0,
        name: str = "",
    ) -> dict:
        """构建单次 YOLO train() 的参数

        合并 TrainConfig 的共享参数和 StageConfig 的阶段参数。

        Args:
            stage_config: 阶段配置
            freeze: 冻结层数
            name: 实验名称

        Returns:
            YOLO train() 参数字典
        """
        args = self.config.to_dict()
        args.update(asdict(stage_config))
        if freeze > 0:
            args["freeze"] = freeze
        if name:
            args["name"] = name
        return args

    def train(self) -> Union[Path, Dict[str, Path]]:
        """执行训练

        根据 model_cfg 自动选择单阶段或两阶段。

        Returns:
            单阶段返回 Path，两阶段返回 {"stage1": Path, "stage2": Path}
        """
        if self.model_cfg.is_two_stage():
            return self._train_two_stage()
        return self._train_single_stage()

    def _train_single_stage(self) -> Path:
        """单阶段训练

        Returns:
            训练结果目录路径
        """
        exp_name = self.model_cfg.result_pattern.format(scale=self.scale)

        print("\n" + "=" * 60)
        print(f"训练模型: {self.model_cfg.get_display_name(self.scale)}")
        print(f"实验名称: {exp_name}")
        print("=" * 60)

        model = YOLO(self.model_cfg.yaml_path).load(self.pretrained)
        args = self._build_train_args(self.config.stage2, name=exp_name)
        model.train(**args)

        result_dir = Path(self.config.project) / exp_name
        print(f"\n✓ 训练完成: {result_dir}")
        return result_dir

    def _train_two_stage(self) -> Dict[str, Path]:
        """两阶段训练

        阶段一：冻结预热（freeze backbone，训练新增模块）
        阶段二：全局微调（加载阶段一权重，端到端训练）

        Returns:
            {"stage1": Path, "stage2": Path}
        """
        base = self.model_cfg.result_pattern.format(scale=self.scale)
        base = base.replace("_stage2", "")

        # 阶段一
        s1_name = f"{base}_stage1"

        print("\n" + "=" * 60)
        print(f"两阶段训练: {self.model_cfg.get_display_name(self.scale)}")
        print(f"  阶段一: {s1_name} (冻结 {self.model_cfg.freeze} 层)")
        print("=" * 60)

        model = YOLO(self.model_cfg.yaml_path).load(self.pretrained)
        args = self._build_train_args(
            self.config.stage1,
            freeze=self.model_cfg.freeze,
            name=s1_name,
        )
        model.train(**args)

        s1_path = Path(self.config.project) / s1_name
        s1_weights = s1_path / "weights" / "best.pt"

        if not s1_weights.exists():
            raise FileNotFoundError(f"阶段一权重不存在: {s1_weights}")

        # 阶段二
        s2_name = f"{base}_stage2"

        print("\n" + "=" * 60)
        print(f"  阶段二: {s2_name}")
        print(f"  加载权重: {s1_weights}")
        print("=" * 60)

        model = YOLO(str(s1_weights))
        args = self._build_train_args(self.config.stage2, name=s2_name)
        model.train(**args)

        s2_path = Path(self.config.project) / s2_name
        print(f"\n✓ 两阶段训练完成: {s2_path}")

        return {"stage1": s1_path, "stage2": s2_path}
```

- [ ] **Step 2: 验证 trainer.py 可以正常导入**

Run: `cd /home/cll/workspace/my_project/fce-yolo && python -c "from script.trainer import YOLOv11Trainer; print('OK')"`
Expected: `OK`

- [ ] **Step 3: 提交**

```bash
git add script/trainer.py
git commit -m "refactor: 重写训练器，统一配置传递，简化接口"
```

---

### Task 4: 重写 `train.py` — 统一训练 CLI

**Files:**
- Modify: `script/train.py`（完整重写）

- [ ] **Step 1: 重写 `script/train.py`**

```python
#!/usr/bin/env python3
"""
统一训练 CLI

合并原 train.py + coco_train.py + train_pro.py，支持所有训练场景。

Usage:
    python script/train.py baseline --scale s
    python script/train.py fce --scale s --batch 16 --epochs 200
    python script/train.py fce --scale s --dataset coco
    python script/train.py fce --scale s --test
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from script.config import (
    DATASET_PRESETS,
    MODEL_CONFIGS,
    build_overrides_from_namespace,
    get_dataset_preset,
    get_model_config,
    get_quick_test_config,
    apply_overrides,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="YOLOv11 变体模型训练（统一入口）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本训练
  python script/train.py baseline --scale s
  python script/train.py fce --scale s

  # 覆盖共享参数（所有阶段生效）
  python script/train.py fce --scale s --batch 16 --imgsz 640

  # 覆盖 stage2 轮次（fce: 50+200, baseline: 200）
  python script/train.py fce --scale s --epochs 200

  # 显式改 stage1（高级用法）
  python script/train.py fce --scale s --stage1-epochs 30

  # 切换数据集预设
  python script/train.py fce --scale s --dataset coco
  python script/train.py fce --scale s --dataset coco_hq

  # 自定义数据集路径
  python script/train.py fce --scale s --data /path/to/data.yaml

  # 快速测试
  python script/train.py fce --scale s --test
        """
    )

    # 位置参数
    parser.add_argument(
        "model",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help="模型类型 (baseline, bifpn, fce)",
    )

    # 模型尺度
    parser.add_argument("--scale", type=str, default="n",
                        choices=["n", "s", "m", "l", "x"],
                        help="模型尺度 (默认: n)")

    # 数据集
    parser.add_argument("--dataset", type=str, default="default",
                        choices=list(DATASET_PRESETS.keys()),
                        help="数据集预设 (default/coco/coco_hq)")
    parser.add_argument("--data", type=str, default=None,
                        help="自定义数据集路径（覆盖 --dataset）")

    # 共享参数
    parser.add_argument("--epochs", type=int, default=None,
                        help="stage2 训练轮次（单阶段即总轮次）")
    parser.add_argument("--batch", type=int, default=None,
                        help="批次大小")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="输入图像尺寸")
    parser.add_argument("--device", type=str, default=None,
                        help="训练设备")
    parser.add_argument("--workers", type=int, default=None,
                        help="数据加载工作进程数")
    parser.add_argument("--patience", type=int, default=None,
                        help="早停耐心值")
    parser.add_argument("--lr0", type=float, default=None,
                        help="stage2 初始学习率")
    parser.add_argument("--cos-lr", action="store_true", default=None,
                        help="stage2 余弦退火（不传则使用模型默认值）")
    parser.add_argument("--no-cos-lr", action="store_true", default=False,
                        help="stage2 禁用余弦退火")
    parser.add_argument("--close-mosaic", type=int, default=None,
                        help="stage2 最后 N epochs 关闭 Mosaic")

    # IoU 损失
    parser.add_argument("--iou-type", type=str, default=None,
                        choices=["CIoU", "DIoU", "GIoU", "WIoU"],
                        help="IoU 损失类型")

    # 其他共享参数
    parser.add_argument("--no-amp", action="store_true",
                        help="禁用混合精度训练")
    parser.add_argument("--cache", type=str, default=None,
                        help="数据缓存策略 (false/disk/ram)")

    # stage1 覆盖（高级用法）
    parser.add_argument("--stage1-epochs", type=int, default=None,
                        help="stage1 训练轮次")
    parser.add_argument("--stage1-lr0", type=float, default=None,
                        help="stage1 初始学习率")
    parser.add_argument("--stage1-patience", type=int, default=None,
                        help="stage1 早停耐心值")

    # 快速测试
    parser.add_argument("--test", action="store_true",
                        help="快速测试模式（1 epoch, 小图像尺寸）")

    # 实验名称
    parser.add_argument("--name", type=str, default=None,
                        help="自定义实验名称")

    return parser.parse_args()


def build_overrides(args: argparse.Namespace):
    """从 CLI 参数构建三类 override 字典"""
    shared, stage2, stage1 = build_overrides_from_namespace(args)
    return shared, stage2, stage1


def main():
    """主函数"""
    args = parse_args()

    # 获取模型配置
    model_cfg = get_model_config(args.model)

    # 构建训练配置
    if args.test:
        config = get_quick_test_config()
    else:
        config = get_dataset_preset(args.dataset)

    # 应用 CLI override（包含 --data 处理）
    shared, stage2, stage1 = build_overrides(args)
    config = apply_overrides(config, model_cfg, shared, stage2, stage1)

    # 打印训练信息
    print("\n" + "=" * 80)
    print(f"YOLOv11{args.scale.upper()} - {model_cfg.get_display_name(args.scale)} 训练")
    print("=" * 80)
    print(f"\n模型配置:")
    print(f"  YAML: {model_cfg.yaml_path}")
    print(f"  两阶段训练: {'是' if model_cfg.is_two_stage() else '否'}")
    if model_cfg.is_two_stage():
        print(f"  冻结层数: {model_cfg.freeze}")
        print(f"  阶段一: {config.stage1.epochs} epochs, lr={config.stage1.lr0}, cos_lr={config.stage1.cos_lr}")
    print(f"  阶段二: {config.stage2.epochs} epochs, lr={config.stage2.lr0}, cos_lr={config.stage2.cos_lr}")
    print(f"\n共享参数:")
    print(f"  数据集: {config.data}")
    print(f"  批次大小: {config.batch}")
    print(f"  图像尺寸: {config.imgsz}")
    print(f"  设备: {config.device}")
    print(f"  IoU 损失: {config.iou_type}")
    print("=" * 80)

    # 训练
    from script.trainer import YOLOv11Trainer

    trainer = YOLOv11Trainer(model_cfg, args.scale, config)
    results = trainer.train()

    # 显示结果
    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)

    if isinstance(results, dict):
        print(f"\n  阶段一结果: {results['stage1']}")
        print(f"  阶段二结果: {results['stage2']}")
    else:
        print(f"\n  结果保存在: {results}")

    print("\n运行对比:")
    print(f"  python script/compare.py --models baseline {args.model} --scale {args.scale}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证 --help 正常工作**

Run: `cd /home/cll/workspace/my_project/fce-yolo && python script/train.py --help`
Expected: 显示帮助信息，包含所有参数和示例

- [ ] **Step 3: 验证配置构建逻辑（不实际训练）**

Run: `cd /home/cll/workspace/my_project/fce-yolo && python -c "
from script.config import get_model_config, get_dataset_preset, apply_overrides

# 测试 baseline 单阶段
cfg = get_dataset_preset('default')
m = get_model_config('baseline')
apply_overrides(cfg, m, shared={'batch': 16}, stage2={'epochs': 200}, stage1={})
assert cfg.stage1 is None, 'baseline 应该是单阶段'
assert cfg.stage2.epochs == 200, f'stage2.epochs 应该是 200，实际 {cfg.stage2.epochs}'
assert cfg.batch == 16
print('baseline 单阶段: OK')

# 测试 fce 两阶段
cfg = get_dataset_preset('default')
m = get_model_config('fce')
apply_overrides(cfg, m, shared={'batch': 16}, stage2={'epochs': 200}, stage1={})
assert cfg.stage1 is not None, 'fce 应该是两阶段'
assert cfg.stage1.epochs == 50, f'stage1.epochs 应该保持 50，实际 {cfg.stage1.epochs}'
assert cfg.stage2.epochs == 200, f'stage2.epochs 应该是 200，实际 {cfg.stage2.epochs}'
assert cfg.batch == 16
print('fce 两阶段: OK')

# 测试 stage1 显式覆盖
cfg = get_dataset_preset('default')
m = get_model_config('fce')
apply_overrides(cfg, m, shared={}, stage2={}, stage1={'epochs': 30})
assert cfg.stage1.epochs == 30, f'stage1.epochs 应该是 30，实际 {cfg.stage1.epochs}'
print('stage1 显式覆盖: OK')

print('所有测试通过!')
"`
Expected: `所有测试通过!`

- [ ] **Step 4: 提交**

```bash
git add script/train.py
git commit -m "refactor: 重写统一训练 CLI，合并 train/coco_train/train_pro"
```

---

### Task 5: 重写 `compare.py` — 统一对比 CLI

**Files:**
- Modify: `script/compare.py`（完整重写）

- [ ] **Step 1: 重写 `script/compare.py`**

```python
#!/usr/bin/env python3
"""
统一对比 CLI

合并原 compare.py + train_coco_compare.py + compare_iou.py。

Usage:
    python script/compare.py --models baseline fce --scale s
    python script/compare.py --models baseline fce --scale s --epochs 200
    python script/compare.py --models baseline bifpn fce --scale s --skip-train
    python script/compare.py --models baseline fce --scale s --iou-type WIoU
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from script.config import (
    DATASET_PRESETS,
    MODEL_CONFIGS,
    build_overrides_from_namespace,
    get_dataset_preset,
    get_model_config,
    get_quick_test_config,
    apply_overrides,
)
from script.analysis import (
    load_results,
    extract_metrics,
    print_comparison_table,
    plot_comparison_curves,
    save_comparison_summary,
    reorganize_results,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="YOLOv11 多模型训练对比（统一入口）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
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
        """
    )

    parser.add_argument(
        "--models", type=str, nargs="+", required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="要对比的模型列表",
    )
    parser.add_argument("--scale", type=str, default="n",
                        choices=["n", "s", "m", "l", "x"],
                        help="模型尺度 (默认: n)")
    parser.add_argument("--dataset", type=str, default="default",
                        help="数据集预设 (default/coco/coco_hq)")
    parser.add_argument("--data", type=str, default=None,
                        help="自定义数据集路径（覆盖 --dataset）")

    # 共享参数
    parser.add_argument("--epochs", type=int, default=None,
                        help="stage2 训练轮次")
    parser.add_argument("--batch", type=int, default=None,
                        help="批次大小")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="输入图像尺寸")
    parser.add_argument("--device", type=str, default=None,
                        help="训练设备")
    parser.add_argument("--workers", type=int, default=None,
                        help="数据加载工作进程数")
    parser.add_argument("--patience", type=int, default=None,
                        help="早停耐心值")
    parser.add_argument("--lr0", type=float, default=None,
                        help="stage2 初始学习率")
    parser.add_argument("--cos-lr", action="store_true", default=None,
                        help="stage2 余弦退火")
    parser.add_argument("--no-cos-lr", action="store_true", default=False,
                        help="stage2 禁用余弦退火")
    parser.add_argument("--close-mosaic", type=int, default=None,
                        help="stage2 最后 N epochs 关闭 Mosaic")
    parser.add_argument("--iou-type", type=str, default=None,
                        choices=["CIoU", "DIoU", "GIoU", "WIoU"],
                        help="IoU 损失类型")

    # 对比控制
    parser.add_argument("--skip-train", action="store_true",
                        help="跳过训练，仅对比已有结果")
    parser.add_argument("--output", type=str, default=None,
                        help="自定义输出目录")
    parser.add_argument("--test", action="store_true",
                        help="快速测试模式")

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    scale = args.scale
    models = args.models

    # 构建训练配置
    if args.test:
        config = get_quick_test_config()
    else:
        config = get_dataset_preset(args.dataset)

    # 构建 override（包含 --data 处理）
    shared, stage2, stage1 = build_overrides_from_namespace(args)

    # 输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        tag = "vs".join(models)
        epochs_str = str(args.epochs) if args.epochs else "300"
        output_dir = Path(f"runs/detect/{tag}_{scale}_{epochs_str}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打印信息
    model_names = " vs ".join(m.upper() for m in models)
    print("\n" + "=" * 100)
    print(f"{model_names} 训练对比 (尺度: {scale.upper()})")
    print("=" * 100)
    print(f"\n输出目录: {output_dir}")
    print(f"数据集: {config.data}")
    if args.epochs:
        print(f"覆盖 stage2 轮次: {args.epochs}")
    print()

    # 阶段一：训练所有模型
    if not args.skip_train:
        print("▶" * 50)
        print("训练所有模型")
        print("▶" * 50)

        from script.trainer import YOLOv11Trainer

        for model_type in models:
            model_cfg = get_model_config(model_type)
            model_config = apply_overrides(
                get_dataset_preset(args.dataset) if not args.test else get_quick_test_config(),
                model_cfg, shared, stage2, {},
            )

            print(f"\n{'=' * 60}")
            print(f"训练: {model_cfg.get_display_name(scale)}")
            print(f"{'=' * 60}")

            try:
                trainer = YOLOv11Trainer(model_cfg, scale, model_config)
                trainer.train()
                print(f"✓ {model_type} 训练完成")
            except Exception as e:
                print(f"✗ {model_type} 训练失败: {e}")
                return
    else:
        print("⚠ 跳过训练阶段，使用已有结果")

    # 阶段二：整理结果
    print("\n" + "▶" * 50)
    print("整理训练结果")
    print("▶" * 50)

    result_paths = {}
    for model_type in models:
        model_cfg = get_model_config(model_type)
        result_paths[model_type] = model_cfg.get_result_path(scale)

    csv_paths = reorganize_results(result_paths, output_dir)

    if len(csv_paths) < len(models):
        missing = set(models) - set(csv_paths.keys())
        print(f"\n✗ 缺少训练结果: {missing}")
        return

    # 阶段三：生成对比分析
    print("\n" + "▶" * 50)
    print("生成对比分析")
    print("▶" * 50)

    all_dataframes = {}
    all_metrics = {}
    names = {}
    colors = {}

    for model_type in models:
        model_cfg = get_model_config(model_type)
        names[model_type] = model_cfg.get_display_name(scale)
        colors[model_type] = model_cfg.color

        df = load_results(csv_paths[model_type])
        all_dataframes[model_type] = df
        all_metrics[model_type] = extract_metrics(df)

    print_comparison_table(
        all_metrics, names,
        title=f"{model_names} 训练结果对比表 (尺度: {scale.upper()})",
    )

    plot_comparison_curves(
        all_dataframes, names, colors,
        save_path=output_dir / "comparison_curves.png",
    )

    save_comparison_summary(
        output_path=output_dir / "comparison_summary.txt",
        metrics=all_metrics,
        names=names,
        config_info={
            "模型尺度": scale.upper(),
            "数据集": config.data,
            "IoU 损失": config.iou_type,
        },
    )

    # 完成
    print("\n" + "=" * 100)
    print("训练对比完成!")
    print("=" * 100)
    print(f"\n所有结果已保存到: {output_dir}")
    print()
    for model_type in models:
        path = get_model_config(model_type).get_result_path(scale)
        print(f"  ├── {path}/")
    print(f"  ├── comparison_curves.png")
    print(f"  └── comparison_summary.txt")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证 --help 正常工作**

Run: `cd /home/cll/workspace/my_project/fce-yolo && python script/compare.py --help`
Expected: 显示帮助信息

- [ ] **Step 3: 提交**

```bash
git add script/compare.py
git commit -m "refactor: 重写统一对比 CLI，合并 compare/train_coco_compare/compare_iou"
```

---

### Task 6: 更新 `__init__.py` 和 `test.py`

**Files:**
- Modify: `script/__init__.py`
- Modify: `script/test.py`（完整重写）

- [ ] **Step 1: 更新 `script/__init__.py`**

```python
"""
YOLOv11 训练对比工具包

提供统一的模型训练和对比功能。
"""

__version__ = "2.0.0"

from .config import MODEL_CONFIGS, DATASET_PRESETS, TrainConfig, StageConfig, ModelConfig
from .trainer import YOLOv11Trainer
from .analysis import (
    load_results,
    extract_metrics,
    print_comparison_table,
    plot_comparison_curves,
    save_comparison_summary,
    reorganize_results,
)

__all__ = [
    "MODEL_CONFIGS",
    "DATASET_PRESETS",
    "TrainConfig",
    "StageConfig",
    "ModelConfig",
    "YOLOv11Trainer",
    "load_results",
    "extract_metrics",
    "print_comparison_table",
    "plot_comparison_curves",
    "save_comparison_summary",
    "reorganize_results",
]
```

- [ ] **Step 2: 重写 `script/test.py`**

合并原 test.py + test_two_stage_config.py 的测试逻辑：

```python
#!/usr/bin/env python3
"""
测试工具脚本

验证配置一致性、覆盖逻辑等。
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from script.config import (
    MODEL_CONFIGS,
    TrainConfig,
    StageConfig,
    get_model_config,
    get_dataset_preset,
    get_quick_test_config,
    apply_overrides,
)


def test_imports():
    """测试模块导入"""
    print("=" * 80)
    print("测试 1: 模块导入")
    print("=" * 80)

    try:
        from script.config import MODEL_CONFIGS, TrainConfig, StageConfig, ModelConfig
        print("✓ script.config 导入成功")
    except ImportError as e:
        print(f"✗ script.config 导入失败: {e}")
        return False

    try:
        from script.trainer import YOLOv11Trainer
        print("✓ script.trainer 导入成功")
    except ImportError as e:
        print(f"✗ script.trainer 导入失败: {e}")
        return False

    try:
        from script.analysis import (
            load_results, extract_metrics,
            print_comparison_table, plot_comparison_curves,
            save_comparison_summary, reorganize_results,
        )
        print("✓ script.analysis 导入成功")
    except ImportError as e:
        print(f"✗ script.analysis 导入失败: {e}")
        return False

    return True


def test_model_configs():
    """测试模型配置"""
    print("\n" + "=" * 80)
    print("测试 2: 模型配置")
    print("=" * 80)

    all_ok = True
    scales = ["n", "s", "m", "l", "x"]

    for model_type in MODEL_CONFIGS:
        model_cfg = get_model_config(model_type)
        print(f"\n  {model_type.upper()}:")
        print(f"    YAML: {model_cfg.yaml_path}")
        print(f"    两阶段: {model_cfg.is_two_stage()}")

        for s in scales:
            path = model_cfg.get_result_path(s)
            print(f"    scale={s}: {path}")

    return all_ok


def test_dataset_presets():
    """测试数据集预设"""
    print("\n" + "=" * 80)
    print("测试 3: 数据集预设")
    print("=" * 80)

    for name in ["default", "coco", "coco_hq"]:
        config = get_dataset_preset(name)
        print(f"  {name}: data={config.data}, imgsz={config.imgsz}, batch={config.batch}, cache={config.cache}")

    return True


def test_override_logic():
    """测试配置覆盖逻辑"""
    print("\n" + "=" * 80)
    print("测试 4: 配置覆盖逻辑")
    print("=" * 80)

    # 4.1 单阶段模型：--epochs 不影响 stage1（因为 stage1=None）
    print("\n  4.1 baseline 单阶段 --epochs 200")
    cfg = get_dataset_preset("default")
    m = get_model_config("baseline")
    apply_overrides(cfg, m, shared={}, stage2={"epochs": 200}, stage1={})
    assert cfg.stage1 is None, "baseline 应该是单阶段"
    assert cfg.stage2.epochs == 200
    print("    ✓ stage1=None, stage2.epochs=200")

    # 4.2 两阶段模型：--epochs 只改 stage2
    print("\n  4.2 fce 两阶段 --epochs 200")
    cfg = get_dataset_preset("default")
    m = get_model_config("fce")
    apply_overrides(cfg, m, shared={}, stage2={"epochs": 200}, stage1={})
    assert cfg.stage1.epochs == 50, f"stage1.epochs 应该保持 50，实际 {cfg.stage1.epochs}"
    assert cfg.stage2.epochs == 200
    print("    ✓ stage1.epochs=50, stage2.epochs=200")

    # 4.3 共享参数覆盖两个阶段
    print("\n  4.3 fce --batch 16")
    cfg = get_dataset_preset("default")
    m = get_model_config("fce")
    apply_overrides(cfg, m, shared={"batch": 16}, stage2={}, stage1={})
    assert cfg.batch == 16
    print("    ✓ batch=16")

    # 4.4 显式 stage1 覆盖
    print("\n  4.4 fce --stage1-epochs 30 --stage1-lr0 0.005")
    cfg = get_dataset_preset("default")
    m = get_model_config("fce")
    apply_overrides(cfg, m, shared={}, stage2={}, stage1={"epochs": 30, "lr0": 0.005})
    assert cfg.stage1.epochs == 30
    assert cfg.stage1.lr0 == 0.005
    assert cfg.stage2.epochs == 300  # stage2 不受影响
    print("    ✓ stage1.epochs=30, stage1.lr0=0.005, stage2.epochs=300")

    # 4.5 自定义数据集路径
    print("\n  4.5 --data 覆盖")
    cfg = get_dataset_preset("default")
    m = get_model_config("baseline")
    apply_overrides(cfg, m, shared={"data": "/custom/path.yaml"}, stage2={}, stage1={})
    assert cfg.data == "/custom/path.yaml"
    print("    ✓ data=/custom/path.yaml")

    # 4.6 默认配置不被覆盖
    print("\n  4.6 无 override 时保持模型默认值")
    cfg = get_dataset_preset("default")
    m = get_model_config("fce")
    apply_overrides(cfg, m, shared={}, stage2={}, stage1={})
    assert cfg.stage1.epochs == 50
    assert cfg.stage2.epochs == 300
    assert cfg.stage1.lr0 == 0.01
    assert cfg.stage2.lr0 == 0.001
    print("    ✓ stage1 和 stage2 保持模型默认值")

    print("\n  ✓ 所有覆盖逻辑测试通过!")
    return True


def test_quick_test_config():
    """测试快速测试配置"""
    print("\n" + "=" * 80)
    print("测试 5: 快速测试配置")
    print("=" * 80)

    cfg = get_quick_test_config()
    assert cfg.stage2.epochs == 1
    assert cfg.batch == 2
    assert cfg.imgsz == 64
    assert cfg.stage2.close_mosaic == 0
    print("  ✓ 快速测试配置正确")

    return True


def test_train_config_to_dict():
    """测试 TrainConfig.to_dict()"""
    print("\n" + "=" * 80)
    print("测试 6: TrainConfig.to_dict()")
    print("=" * 80)

    cfg = TrainConfig(data="test.yaml", batch=16)
    d = cfg.to_dict()
    assert d["data"] == "test.yaml"
    assert d["batch"] == 16
    assert "iou_type" in d
    assert "epochs" not in d  # epochs 在 StageConfig 中
    print("  ✓ to_dict() 包含共享参数，不包含阶段参数")

    return True


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("Script 模块测试")
    print("=" * 80)

    success = True
    success = test_imports() and success
    success = test_model_configs() and success
    success = test_dataset_presets() and success
    success = test_override_logic() and success
    success = test_quick_test_config() and success
    success = test_train_config_to_dict() and success

    print("\n" + "=" * 80)
    if success:
        print("✓ 所有测试通过!")
    else:
        print("✗ 部分测试失败")
    print("=" * 80 + "\n")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行测试**

Run: `cd /home/cll/workspace/my_project/fce-yolo && python script/test.py`
Expected: `✓ 所有测试通过!`

- [ ] **Step 4: 提交**

```bash
git add script/__init__.py script/test.py
git commit -m "refactor: 更新包入口和测试脚本"
```

---

### Task 7: 删除旧文件

**Files:**
- Delete: `script/coco_train.py`
- Delete: `script/train_pro.py`
- Delete: `script/train_coco_compare.py`
- Delete: `script/compare_iou.py`
- Delete: `script/test_train_pro.py`
- Delete: `script/verify_fix.py`
- Delete: `script/test_two_stage_config.py`
- Delete: `script/REFACTOR.md`
- Delete: `script/TRAINING_STRATEGY.md`
- Delete: `script/COCO_TRAIN_README.md`

- [ ] **Step 1: 删除所有旧文件**

```bash
cd /home/cll/workspace/my_project/fce-yolo
git rm script/coco_train.py script/train_pro.py script/train_coco_compare.py \
    script/compare_iou.py script/test_train_pro.py script/verify_fix.py \
    script/test_two_stage_config.py script/REFACTOR.md \
    script/TRAINING_STRATEGY.md script/COCO_TRAIN_README.md
```

- [ ] **Step 2: 运行测试确认无破坏**

Run: `cd /home/cll/workspace/my_project/fce-yolo && python script/test.py`
Expected: `✓ 所有测试通过!`

- [ ] **Step 3: 提交**

```bash
git commit -m "refactor: 删除已合并的旧脚本和过时文档"
```

---

### Task 8: 更新项目文档

**Files:**
- Modify: `CLAUDE.md`（Training Scripts 章节）
- Rewrite: `script/README.md`
- Modify: `notes/TRAINING_GUIDE.md`（所有命令示例）

- [ ] **Step 1: 更新 `CLAUDE.md` 的 Training Scripts 章节**

将 CLAUDE.md 中的 Training Scripts 章节更新为新的统一命令。关键改动：
- 单模型训练只保留 `script/train.py`
- 对比只保留 `script/compare.py`
- 删除 `script/config.py` 中对 `ExperimentConfig` 的引用
- 更新所有命令示例
- 更新 File Locations 中的文件列表

- [ ] **Step 2: 重写 `script/README.md`**

完整重写为新的使用文档，覆盖：
- 快速开始
- 统一训练 CLI 完整参数说明
- 统一对比 CLI 完整参数说明
- 配置系统说明（共享参数/阶段参数）
- 数据集预设
- 添加新模型变体的方法

- [ ] **Step 2.5: 更新 `notes/TRAINING_GUIDE.md`**

将所有旧的命令示例更新为新的统一 CLI：
- `script/train_pro.py` → `script/train.py --dataset coco_hq`
- `script/train_coco_compare.py` → `script/compare.py`
- `script/coco_train.py` → `script/train.py --dataset coco`
- 更新参数名称（如 `--no-amp`）

- [ ] **Step 3: 最终验证**

Run: `cd /home/cll/workspace/my_project/fce-yolo && python script/test.py`
Expected: `✓ 所有测试通过!`

Run: `cd /home/cll/workspace/my_project/fce-yolo && python script/train.py --help`
Expected: 显示帮助信息

Run: `cd /home/cll/workspace/my_project/fce-yolo && python script/compare.py --help`
Expected: 显示帮助信息

- [ ] **Step 4: 提交**

```bash
git add CLAUDE.md script/README.md notes/TRAINING_GUIDE.md
git commit -m "docs: 更新项目文档，反映统一训练架构"
```

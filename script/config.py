"""
配置管理模块.

定义训练配置、模型配置、数据集预设，以及配置覆盖逻辑。
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Callable

# ==================== 阶段配置 ====================


@dataclass
class StageConfig:
    """阶段训练参数（lr, epochs, patience 等各阶段独立的参数）."""

    epochs: int = 300
    patience: int = 50
    lr0: float = 0.001
    cos_lr: bool = True
    close_mosaic: int = 20


# ==================== 训练配置 ====================


@dataclass
class TrainConfig:
    """完整训练配置.

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
    stage1: StageConfig | None = None
    stage2: StageConfig = field(default_factory=StageConfig)

    def to_dict(self) -> dict:
        """转换为 YOLO train() 接受的字典."""
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
    """模型变体配置."""

    name: str
    yaml_path: str
    color: str
    display_name: Callable[[str], str]
    freeze: int = 0
    stage1: StageConfig | None = None
    stage2: StageConfig = field(default_factory=StageConfig)
    result_pattern: str = ""

    def get_display_name(self, scale: str) -> str:
        """获取显示名称."""
        return self.display_name(scale)

    def is_two_stage(self) -> bool:
        """是否为两阶段训练."""
        return self.stage1 is not None

    def get_result_path(self, scale: str, stage: int | None = None) -> str:
        """获取结果目录路径.

        Args:
            scale: 模型尺度
            stage: 阶段编号（1 或 2），None 表示最终结果（stage2 或单阶段）
        """
        pattern = self.result_pattern
        if stage is not None:
            pattern = pattern.replace("_stage2", f"_stage{stage}")
        return pattern.format(scale=scale)


def get_model_config(model_type: str) -> ModelConfig:
    """获取模型配置.

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

MODEL_CONFIGS: dict[str, ModelConfig] = {
    "baseline": ModelConfig(
        name="baseline",
        yaml_path="ultralytics/cfg/models/11/yolo11.yaml",
        color="#0BDBEB",
        display_name=lambda s: f"YOLOv11{s.upper()} Baseline",
        stage1=None,
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


# ==================== 数据集预设 ====================

DATASET_PRESETS: dict[str, TrainConfig] = {
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
    """获取数据集预设配置.

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
    """获取快速测试配置."""
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
    "batch",
    "imgsz",
    "device",
    "workers",
    "amp",
    "cache",
    "optimizer",
    "lrf",
    "momentum",
    "weight_decay",
    "iou_type",
    "project",
    "save_period",
    "verbose",
    "plots",
}

STAGE2_PARAMS = {
    "epochs",
    "lr0",
    "patience",
    "cos_lr",
    "close_mosaic",
}


def apply_overrides(
    config: TrainConfig,
    model_cfg: ModelConfig,
    shared: dict | None = None,
    stage2: dict | None = None,
    stage1: dict | None = None,
) -> TrainConfig:
    """应用用户覆盖到训练配置.

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
    shared_keys: dict | None = None,
    stage2_keys: dict | None = None,
    stage1_prefix: str = "stage1_",
) -> tuple:
    """从 argparse.Namespace 构建三类 override 字典.

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
            "batch": "batch",
            "imgsz": "imgsz",
            "device": "device",
            "workers": "workers",
            "iou_type": "iou_type",
            "cache": "cache",
        }
    if stage2_keys is None:
        stage2_keys = {
            "epochs": "epochs",
            "lr0": "lr0",
            "patience": "patience",
            "cos_lr": "cos_lr",
            "close_mosaic": "close_mosaic",
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
                stage1[attr[len(stage1_prefix) :]] = val

    return shared, stage2, stage1

"""
YOLOv11 训练对比工具包.

提供统一的模型训练和对比功能。
"""

__version__ = "2.0.0"

from .analysis import (
    extract_metrics,
    load_results,
    plot_comparison_curves,
    print_comparison_table,
    reorganize_results,
    save_comparison_summary,
)
from .config import DATASET_PRESETS, MODEL_CONFIGS, ModelConfig, StageConfig, TrainConfig
from .trainer import YOLOv11Trainer

__all__ = [
    "DATASET_PRESETS",
    "MODEL_CONFIGS",
    "ModelConfig",
    "StageConfig",
    "TrainConfig",
    "YOLOv11Trainer",
    "extract_metrics",
    "load_results",
    "plot_comparison_curves",
    "print_comparison_table",
    "reorganize_results",
    "save_comparison_summary",
]

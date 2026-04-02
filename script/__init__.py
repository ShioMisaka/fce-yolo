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

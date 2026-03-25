"""
YOLOv11 训练对比工具包

提供灵活的 YOLOv11 变体模型训练和对比功能。
"""

__version__ = "1.0.0"
__author__ = "FCE-YOLOv11 Team"

from .config import MODEL_CONFIGS, TrainConfig, ExperimentConfig
from .trainer import YOLOv11Trainer

__all__ = [
    "MODEL_CONFIGS",
    "TrainConfig",
    "ExperimentConfig",
    "YOLOv11Trainer",
]

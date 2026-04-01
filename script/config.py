"""
配置管理模块

定义所有训练和实验相关的配置。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable


# ==================== 模型配置 ====================

@dataclass
class ModelConfig:
    """单个模型配置"""
    name: str
    yaml_path: str
    color: str
    display_name: Callable[[str], str]
    use_two_stage: bool
    result_pattern: str

    def get_display_name(self, scale: str) -> str:
        """获取显示名称"""
        return self.display_name(scale)

    def get_result_path(self, scale: str, stage: Optional[int] = None) -> str:
        """获取结果目录路径

        Args:
            scale: 模型尺度
            stage: 阶段编号（1 或 2），None 表示单阶段
        """
        pattern = self.result_pattern
        if stage is not None:
            pattern = pattern.replace("_stage2", f"_stage{stage}")
        return pattern.format(scale=scale)


# 预定义模型配置
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "baseline": ModelConfig(
        name="baseline",
        yaml_path="ultralytics/cfg/models/11/yolo11.yaml",
        color="#0BDBEB",  # 青色
        display_name=lambda s: f"YOLOv11{s.upper()} Baseline",
        use_two_stage=False,
        result_pattern="baseline_yolo11{scale}",
    ),
    "bifpn": ModelConfig(
        name="bifpn",
        yaml_path="ultralytics/cfg/models/11/yolo11-bifpn.yaml",
        color="#042AFF",  # 蓝色
        display_name=lambda s: f"YOLOv11{s.upper()}-BiFPN",
        use_two_stage=True,
        result_pattern="bifpn_{scale}_stage2",
    ),
    "fce": ModelConfig(
        name="fce",
        yaml_path="ultralytics/cfg/models/11/yolo11-fce.yaml",
        color="#FF6B00",  # 橙色
        display_name=lambda s: f"YOLOv11{s.upper()}-FCE",
        use_two_stage=True,
        result_pattern="fce_{scale}_stage2",
    ),
}


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


# ==================== 训练配置 ====================

@dataclass
class TrainConfig:
    """训练配置"""
    # 数据集
    data_path: str = "/mnt/ssd1/Dataset/haixi_jixieshou/yolo_dataset/data.yaml"

    # 训练参数
    epochs: int = 300
    patience: int = 50
    imgsz: int = 1280
    batch: int = 32
    device: str = "0"

    # 优化器
    optimizer: str = "AdamW"
    lr0: float = 0.01
    lrf: float = 0.01
    cos_lr: bool = True
    momentum: float = 0.937
    weight_decay: float = 0.0005

    # 数据增强
    close_mosaic: int = 20
    mixup: float = 0.0
    degrees: float = 10.0
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4

    # 硬件
    amp: bool = True
    workers: int = 16
    cache: bool = True

    # 保存
    project: str = "runs/detect"
    save: bool = True
    save_period: int = 50
    exist_ok: bool = True

    # IoU 损失类型
    iou_type: str = "CIoU"

    # 其他
    deterministic: bool = False
    verbose: bool = True
    plots: bool = True

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "data": self.data_path,
            "epochs": self.epochs,
            "patience": self.patience,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "device": self.device,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "cos_lr": self.cos_lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "close_mosaic": self.close_mosaic,
            "mixup": self.mixup,
            "degrees": self.degrees,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "amp": self.amp,
            "workers": self.workers,
            "cache": self.cache,
            "project": self.project,
            "save": self.save,
            "save_period": self.save_period,
            "exist_ok": self.exist_ok,
            "iou_type": self.iou_type,
            "deterministic": self.deterministic,
            "verbose": self.verbose,
            "plots": self.plots,
        }


# ==================== 实验配置 ====================

@dataclass
class ExperimentConfig:
    """对比实验配置"""
    models: List[str]
    scale: str
    total_epochs: int = 300
    skip_train: bool = False
    output_dir: Optional[Path] = None

    def __post_init__(self):
        if self.output_dir is None:
            # 根据参与对比的模型生成输出目录名
            model_tag = "vs".join(self.models)
            self.output_dir = Path(f"runs/detect/{model_tag}_{self.scale}_{self.total_epochs}")

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_model_display_names(self) -> Dict[str, str]:
        """获取所有模型的显示名称"""
        return {
            model_type: get_model_config(model_type).get_display_name(self.scale)
            for model_type in self.models
        }

    def get_model_colors(self) -> Dict[str, str]:
        """获取所有模型的颜色"""
        return {
            model_type: get_model_config(model_type).color
            for model_type in self.models
        }


# ==================== 常用配置预设 ====================

def get_quick_test_config() -> TrainConfig:
    """获取快速测试配置（用于测试）"""
    return TrainConfig(
        epochs=1,
        patience=10,
        imgsz=64,
        batch=2,
        save_period=1,
        close_mosaic=0,
    )


def get_default_config() -> TrainConfig:
    """获取默认训练配置"""
    return TrainConfig()


def get_stage1_config() -> TrainConfig:
    """获取阶段一配置（冻结预热）

    阶段一：快速预热随机初始化的模块
    - 冻结 backbone，只训练新增模块
    - 较高学习率快速适应
    - 较短的训练轮次

    注意：这些参数是基于旧版本（commit f0929c）验证过的有效配置
    """
    return TrainConfig(
        epochs=50,      # 50 epochs（与旧版本一致，充分预热）
        patience=20,    # 20 epochs patience（与旧版本一致）
        lr0=0.01,      # 较高学习率
        cos_lr=False,  # 线性衰减
        close_mosaic=10,  # 最后 10 epochs 关闭 Mosaic（与旧版本一致）
    )


def get_stage2_config() -> TrainConfig:
    """获取阶段二配置（全局微调）

    阶段二：端到端完整训练
    - 解冻所有层
    - 较低学习率精细调整
    - 与 Baseline 相同的训练轮次（确保公平对比）
    """
    return TrainConfig(
        epochs=300,     # 300 epochs 完整训练（与 Baseline 一致）
        patience=50,
        lr0=0.001,     # 较低学习率
        cos_lr=True,   # 余弦退火
        close_mosaic=20,
    )

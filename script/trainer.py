"""
训练器模块

提供 YOLOv11 变体模型的训练功能。
"""

from pathlib import Path
from typing import Dict, Optional, Union
from ultralytics import YOLO

from .config import (
    TrainConfig,
    ModelConfig,
    get_model_config,
    get_stage1_config,
    get_stage2_config,
)


class YOLOv11Trainer:
    """YOLOv11 变体模型训练器"""

    def __init__(
        self,
        model_config: ModelConfig,
        scale: str,
        pretrained_weights: Optional[str] = None,
    ):
        """初始化训练器

        Args:
            model_config: 模型配置
            scale: 模型尺度 (n, s, m, l, x)
            pretrained_weights: 预训练权重路径
        """
        self.model_config = model_config
        self.scale = scale
        self.pretrained_weights = pretrained_weights or f"yolo11{scale}.pt"

    def train_single_stage(
        self,
        exp_name: str,
        config: TrainConfig,
        freeze: Optional[int] = None,
    ) -> Path:
        """单阶段训练

        Args:
            exp_name: 实验名称
            config: 训练配置
            freeze: 冻结前 N 层

        Returns:
            训练结果目录路径
        """
        print("\n" + "=" * 60)
        print(f"训练模型: {self.model_config.name} (尺度: {self.scale})")
        print(f"实验名称: {exp_name}")
        print("=" * 60)

        # 加载模型
        model = YOLO(self.model_config.yaml_path).load(self.pretrained_weights)

        # 构建训练参数
        train_args = config.to_dict()
        train_args["name"] = exp_name

        # 添加冻结参数
        if freeze is not None:
            train_args["freeze"] = freeze

        # 开始训练
        model.train(**train_args)

        # 返回结果目录
        result_dir = Path(config.project) / exp_name
        print(f"\n✓ 训练完成: {result_dir}")

        return result_dir

    def train_two_stage(
        self,
        exp_name_base: str,
        stage1_config: TrainConfig,
        stage2_config: TrainConfig,
        stage1_freeze: int = 10,
    ) -> Dict[str, Path]:
        """两阶段训练

        Args:
            exp_name_base: 实验名称基础
            stage1_config: 阶段一配置
            stage2_config: 阶段二配置
            stage1_freeze: 阶段一冻结层数

        Returns:
            包含 stage1 和 stage2 结果目录路径的字典
        """
        print("\n" + "=" * 60)
        print(f"两阶段训练: {self.model_config.name} (尺度: {self.scale})")
        print("=" * 60)

        # 阶段一：冻结预热
        exp_name_s1 = f"{exp_name_base}_stage1"
        result_s1 = self.train_single_stage(
            exp_name=exp_name_s1,
            config=stage1_config,
            freeze=stage1_freeze,
        )

        # 获取阶段一最佳权重
        weights_s1 = result_s1 / "weights" / "best.pt"

        if not weights_s1.exists():
            raise FileNotFoundError(f"阶段一权重不存在: {weights_s1}")

        # 阶段二：全局微调
        exp_name_s2 = f"{exp_name_base}_stage2"
        print("\n" + "=" * 60)
        print(f"加载阶段一权重: {weights_s1}")
        print("=" * 60)

        # 临时更新预训练权重
        original_weights = self.pretrained_weights
        self.pretrained_weights = str(weights_s1)

        result_s2 = self.train_single_stage(
            exp_name=exp_name_s2,
            config=stage2_config,
            freeze=None,
        )

        # 恢复原始权重
        self.pretrained_weights = original_weights

        return {
            "stage1": result_s1,
            "stage2": result_s2,
        }


def train_model(
    model_type: str,
    scale: str = "n",
    exp_name_prefix: Optional[str] = None,
    config: Optional[TrainConfig] = None,
    **kwargs
) -> Union[Path, Dict[str, Path]]:
    """训练指定类型的模型

    Args:
        model_type: 模型类型
        scale: 模型尺度
        exp_name_prefix: 实验名称前缀
        config: 训练配置（None 则使用默认配置）
        **kwargs: 额外的训练参数

    Returns:
        单阶段返回 Path，两阶段返回 Dict[str, Path]
    """
    # 获取模型配置
    model_cfg = get_model_config(model_type)

    # 创建训练器
    trainer = YOLOv11Trainer(
        model_config=model_cfg,
        scale=scale,
    )

    # 实验名称
    if exp_name_prefix is None:
        if model_cfg.use_two_stage:
            exp_name_prefix = f"{model_type}_{scale}"
        else:
            exp_name_prefix = f"{model_type}_yolo11{scale}"

    # 根据配置选择训练模式
    if model_cfg.use_two_stage:
        # 两阶段训练：总是从阶段特定配置开始
        # 然后应用用户提供的自定义参数
        s1_config = get_stage1_config()
        s2_config = get_stage2_config()

        # 如果用户提供了自定义配置，应用其中的非默认值
        if config is not None:
            default_config = TrainConfig()
            for k, v in config.__dict__.items():
                # 只应用与默认值不同的参数
                default_value = getattr(default_config, k, None)
                if default_value != v:
                    setattr(s1_config, k, v)
                    setattr(s2_config, k, v)

        # 合并额外参数
        for k, v in kwargs.items():
            setattr(s1_config, k, v)
            setattr(s2_config, k, v)

        results = trainer.train_two_stage(
            exp_name_base=exp_name_prefix,
            stage1_config=s1_config,
            stage2_config=s2_config,
        )
        return results
    else:
        # 使用默认的单阶段配置
        train_config = config or TrainConfig()

        # 合并额外参数
        for k, v in kwargs.items():
            setattr(train_config, k, v)

        result = trainer.train_single_stage(
            exp_name=exp_name_prefix,
            config=train_config,
        )
        return result

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

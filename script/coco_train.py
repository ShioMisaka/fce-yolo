#!/usr/bin/env python3
"""
COCO 数据集训练脚本

在 COCO 2017 数据集上训练 YOLOv11 变体模型。

Usage:
    # 训练 Baseline-N (快速测试)
    python script/coco_train.py baseline --scale n --test

    # 训练 FCE-S (完整训练)
    python script/coco_train.py fce --scale s --epochs 300

    # 训练 BiFPN-M (自定义参数)
    python script/coco_train.py bifpn --scale m --batch 16 --device 0

    # 训练所有模型进行对比
    python script/coco_train.py --all --scale s --epochs 300
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from script.config import MODEL_CONFIGS, get_model_config


# ==================== COCO 数据集配置 ====================

COCO_CONFIG = {
    # 数据集路径
    "data": "coco.yaml",

    # COCO 数据集默认训练参数
    "imgsz": 640,          # COCO 标准图像尺寸
    "batch": 16,           # 适中的批次大小（可根据 GPU 调整）
    "epochs": 300,         # 完整训练轮次
    "patience": 50,        # 早停耐心值

    # 优化器配置
    "optimizer": "AdamW",
    "lr0": 0.01,          # 初始学习率
    "lrf": 0.01,          # 最终学习率（初始值的倍数）
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "cos_lr": True,       # 余弦退火学习率

    # 数据增强（COCO 标准配置）
    "hsv_h": 0.015,       # HSV 色调增强
    "hsv_s": 0.7,         # HSV 饱和度增强
    "hsv_v": 0.4,         # HSV 明度增强
    "degrees": 0.0,       # 旋转角度（COCO 通常不旋转）
    "translate": 0.1,     # 平移
    "scale": 0.5,         # 缩放
    "shear": 0.0,         # 剪切（COCO 通常不剪切）
    "perspective": 0.0,   # 透视变换（COCO 通常不使用）
    "flipud": 0.0,        # 上下翻转（COCO 通常不翻转）
    "fliplr": 0.5,        # 左右翻转
    "mosaic": 1.0,        # Mosaic 增强
    "mixup": 0.0,         # Mixup 增强（COCO 通常不使用）
    "copy_paste": 0.0,    # Copy-paste 增强

    # 训练策略
    "close_mosaic": 20,   # 最后 20 epochs 关闭 Mosaic
    "warmup_epochs": 3,   # 预热轮次
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,

    # 硬件配置
    "device": "0",        # GPU 设备
    "workers": 8,         # 数据加载线程数
    "amp": True,          # 混合精度训练
    "cache": False,       # COCO 数据集较大，默认不缓存

    # 保存和日志
    "project": "runs/coco",
    "exist_ok": True,
    "save": True,
    "save_period": 50,
    "verbose": True,
    "plots": True,

    # 其他
    "deterministic": False,
    "seed": 0,
    "single_cls": False,  # COCO 是多类别数据集
    "rect": False,        # 训练时不使用矩形训练
}


# ==================== 快速测试配置 ====================

QUICK_TEST_CONFIG = COCO_CONFIG.copy()
QUICK_TEST_CONFIG.update({
    "epochs": 1,
    "imgsz": 320,
    "batch": 4,
    "save_period": 1,
    "close_mosaic": 0,
    "plots": False,
})


# ==================== 阶段一配置（冻结预热）====================

STAGE1_CONFIG = COCO_CONFIG.copy()
STAGE1_CONFIG.update({
    "epochs": 50,
    "patience": 20,
    "lr0": 0.01,
    "cos_lr": False,      # 线性衰减
    "close_mosaic": 10,
})


# ==================== 阶段二配置（全局微调）====================

STAGE2_CONFIG = COCO_CONFIG.copy()
STAGE2_CONFIG.update({
    "epochs": 300,
    "patience": 50,
    "lr0": 0.001,
    "cos_lr": True,       # 余弦退火
    "close_mosaic": 20,
})


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="YOLOv11 变体模型 COCO 数据集训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速测试（1 epoch）
  python script/coco_train.py baseline --scale n --test

  # 训练 FCE-S 模型
  python script/coco_train.py fce --scale s --epochs 300

  # 训练 BiFPN-M（自定义批次大小）
  python script/coco_train.py bifpn --scale m --batch 8

  # 训练所有模型进行对比
  python script/coco_train.py --all --scale s --epochs 300

  # 使用多 GPU
  python script/coco_train.py baseline --scale s --device 0,1
        """
    )

    # 位置参数（模型类型，可选）
    parser.add_argument(
        "model",
        type=str,
        nargs="?",
        choices=list(MODEL_CONFIGS.keys()) + [None],
        help="模型类型（如果使用 --all 则省略）"
    )

    # 可选参数
    parser.add_argument(
        "--scale",
        type=str,
        default="s",
        choices=["n", "s", "m", "l", "x"],
        help="模型尺度 (默认: s)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="训练所有模型（baseline, bifpn, fce）"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="快速测试模式（1 epoch，小图像尺寸）"
    )

    # 训练参数
    parser.add_argument("--epochs", type=int, default=None, help="训练轮次")
    parser.add_argument("--batch", type=int, default=None, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=None, help="输入图像尺寸")
    parser.add_argument("--device", type=str, default=None, help="GPU 设备")
    parser.add_argument("--workers", type=int, default=None, help="数据加载线程数")
    parser.add_argument("--lr0", type=float, default=None, help="初始学习率")

    # 实验名称
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="实验名称（默认: {model}_{scale}）"
    )

    # 两阶段训练控制
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=None,
        help="仅执行指定阶段（1 或 2），默认执行两阶段"
    )

    return parser.parse_args()


def get_config(args: argparse.Namespace, stage: int = None) -> dict:
    """获取训练配置

    Args:
        args: 命令行参数
        stage: 训练阶段（1, 2, 或 None）

    Returns:
        训练配置字典
    """
    # 选择基础配置
    if args.test:
        config = QUICK_TEST_CONFIG.copy()
    elif stage == 1:
        config = STAGE1_CONFIG.copy()
    elif stage == 2:
        config = STAGE2_CONFIG.copy()
    else:
        config = COCO_CONFIG.copy()

    # 应用命令行参数覆盖
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch is not None:
        config["batch"] = args.batch
    if args.imgsz is not None:
        config["imgsz"] = args.imgsz
    if args.device is not None:
        config["device"] = args.device
    if args.workers is not None:
        config["workers"] = args.workers
    if args.lr0 is not None:
        config["lr0"] = args.lr0

    return config


def train_single_model(
    model_type: str,
    scale: str,
    config: dict,
    name: str = None,
    stage: int = None
) -> str:
    """训练单个模型

    Args:
        model_type: 模型类型（baseline, bifpn, fce）
        scale: 模型尺度（n, s, m, l, x）
        config: 训练配置
        name: 实验名称
        stage: 训练阶段（1, 2, 或 None）

    Returns:
        训练结果目录路径
    """
    # 获取模型配置
    model_config = get_model_config(model_type)

    # 构建模型名称
    model_name = f"yolo11{scale}"
    if model_type == "baseline":
        model_path = f"{model_name}.pt"
    else:
        model_path = model_config.yaml_path

    # 生成实验名称
    if name is None:
        if stage is not None:
            name = f"{model_type}_{scale}_stage{stage}"
        else:
            name = f"{model_type}_{scale}"

    config["name"] = name

    # 打印训练信息
    print("\n" + "=" * 80)
    print(f"  训练模型: {model_config.get_display_name(scale)}")
    if stage is not None:
        print(f"  训练阶段: {'阶段一（冻结预热）' if stage == 1 else '阶段二（全局微调）'}")
    print("=" * 80)
    print(f"  模型路径: {model_path}")
    print(f"  实验名称: {name}")
    print(f"  训练轮次: {config['epochs']}")
    print(f"  批次大小: {config['batch']}")
    print(f"  图像尺寸: {config['imgsz']}")
    print(f"  GPU 设备: {config['device']}")
    print("=" * 80 + "\n")

    # 加载模型
    model = YOLO(model_path)

    # 开始训练
    results = model.train(**config)

    # 返回结果目录
    return results.save_dir


def train_two_stage(
    model_type: str,
    scale: str,
    args: argparse.Namespace
) -> None:
    """两阶段训练流程

    Args:
        model_type: 模型类型
        scale: 模型尺度
        args: 命令行参数
    """
    # 检查是否只训练特定阶段
    if args.stage == 1:
        # 仅训练阶段一
        print(f"\n🎯 仅执行阶段一训练：{model_type}-{scale.upper()}\n")
        config = get_config(args, stage=1)
        train_single_model(model_type, scale, config, stage=1)

    elif args.stage == 2:
        # 仅训练阶段二
        print(f"\n🎯 仅执行阶段二训练：{model_type}-{scale.upper()}\n")
        config = get_config(args, stage=2)

        # 检查阶段一权重是否存在
        stage1_name = f"{model_type}_{scale}_stage1"
        stage1_weights = Path(f"runs/coco/{stage1_name}/weights/best.pt")

        if stage1_weights.exists():
            print(f"✓ 找到阶段一权重：{stage1_weights}")
            config["resume"] = str(stage1_weights)
        else:
            print(f"⚠ 未找到阶段一权重，将从头开始训练")
            print(f"  期望路径：{stage1_weights}")

        train_single_model(model_type, scale, config, stage=2)

    else:
        # 完整两阶段训练
        print(f"\n🚀 开始两阶段训练：{model_type}-{scale.upper()}\n")

        # 阶段一：冻结预热
        print("=" * 80)
        print("  阶段一：冻结预热（50 epochs）")
        print("=" * 80)
        config1 = get_config(args, stage=1)
        stage1_dir = train_single_model(model_type, scale, config1, stage=1)
        stage1_weights = Path(stage1_dir) / "weights" / "best.pt"

        if not stage1_weights.exists():
            raise FileNotFoundError(f"阶段一权重未找到：{stage1_weights}")

        print(f"\n✓ 阶段一完成，权重：{stage1_weights}")

        # 阶段二：全局微调
        print("\n" + "=" * 80)
        print("  阶段二：全局微调（300 epochs）")
        print("=" * 80)
        config2 = get_config(args, stage=2)
        config2["resume"] = str(stage1_weights)
        stage2_dir = train_single_model(model_type, scale, config2, stage=2)

        print(f"\n✓ 两阶段训练完成！")
        print(f"  阶段一结果：{stage1_dir}")
        print(f"  阶段二结果：{stage2_dir}")


def main():
    """主函数"""
    args = parse_args()

    # 检查参数
    if args.all and args.model:
        print("❌ 错误：不能同时指定 --all 和模型类型")
        sys.exit(1)

    if not args.all and not args.model:
        print("❌ 错误：必须指定模型类型或使用 --all")
        print("   可选模型：baseline, bifpn, fce")
        sys.exit(1)

    # 确定要训练的模型列表
    if args.all:
        models = ["baseline", "bifpn", "fce"]
    else:
        models = [args.model]

    # 训练每个模型
    for model_type in models:
        model_config = get_model_config(model_type)

        if model_config.use_two_stage:
            # 两阶段训练
            train_two_stage(model_type, args.scale, args)
        else:
            # 单阶段训练
            config = get_config(args)
            name = args.name or f"{model_type}_{args.scale}"
            train_single_model(model_type, args.scale, config, name)

    print("\n" + "=" * 80)
    print("  ✓ 所有训练任务完成！")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

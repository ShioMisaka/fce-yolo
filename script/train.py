#!/usr/bin/env python3
"""
训练 CLI 脚本

提供命令行接口用于训练单个 YOLOv11 变体模型。

Usage:
    python script/train.py baseline --scale s
    python script/train.py fce --scale m --batch 16
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from script.config import (
    MODEL_CONFIGS,
    TrainConfig,
    get_model_config,
    get_quick_test_config,
    get_stage1_config,
    get_stage2_config,
)
from script.trainer import train_model


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="YOLOv11 变体模型训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 训练 Baseline 模型
  python script/train.py baseline --scale s

  # 训练 BiFPN 模型（自动两阶段）
  python script/train.py bifpn --scale s

  # 训练 FCE 模型（自动两阶段）
  python script/train.py fce --scale s

  # 自定义训练参数
  python script/train.py fce --scale s --batch 16 --imgsz 640 --epochs 100

  # 快速测试
  python script/train.py baseline --scale n --test
        """
    )

    # 位置参数
    parser.add_argument(
        "model",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help="模型类型"
    )

    # 可选参数
    parser.add_argument(
        "--scale",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="模型尺度 (n: nano, s: small, m: medium, l: large, x: xlarge)"
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="自定义实验名称前缀"
    )

    # 训练参数
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮次（覆盖默认值）"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="批次大小（覆盖默认值）"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="输入图像尺寸（覆盖默认值）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="训练设备"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="早停耐心值"
    )

    # 测试模式
    parser.add_argument(
        "--test",
        action="store_true",
        help="快速测试模式（1 epoch, 小图像尺寸）"
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    """根据命令行参数构建训练配置"""
    if args.test:
        config = get_quick_test_config()
        print("⚠ 使用快速测试配置")
    else:
        config = TrainConfig()

    # 应用命令行参数
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch is not None:
        config.batch = args.batch
    if args.imgsz is not None:
        config.imgsz = args.imgsz
    if args.device is not None:
        config.device = args.device
    if args.patience is not None:
        config.patience = args.patience

    return config


def main():
    """主函数"""
    args = parse_args()

    # 获取模型配置
    model_cfg = get_model_config(args.model)

    print("\n" + "=" * 80)
    print(f"YOLOv11{args.scale.upper()} - {model_cfg.name.upper()} 训练")
    print("=" * 80)
    print(f"\n模型配置:")
    print(f"  YAML: {model_cfg.yaml_path}")
    print(f"  两阶段训练: {'是' if model_cfg.use_two_stage else '否'}")
    print(f"  显示名称: {model_cfg.get_display_name(args.scale)}")

    # 构建训练配置
    config = build_config(args)

    print(f"\n训练配置:")
    print(f"  训练轮次: {config.epochs}")
    print(f"  批次大小: {config.batch}")
    print(f"  图像尺寸: {config.imgsz}")
    print(f"  设备: {config.device}")
    print(f"  学习率: {config.lr0}")
    print("=" * 80)

    # 训练模型
    results = train_model(
        model_type=args.model,
        scale=args.scale,
        exp_name_prefix=args.exp_name,
        config=config,
    )

    # 显示结果
    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)

    if isinstance(results, dict):
        print(f"\n阶段一结果: {results['stage1']}")
        print(f"阶段二结果: {results['stage2']}")
    else:
        print(f"\n结果保存在: {results}")

    print("\n运行对比:")
    print(f"  python script/compare.py --models baseline {args.model} --scale {args.scale}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

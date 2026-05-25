#!/usr/bin/env python3
"""
统一训练 CLI.

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
    apply_overrides,
    build_overrides_from_namespace,
    get_dataset_preset,
    get_model_config,
    get_quick_test_config,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
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
        """,
    )

    # 位置参数
    parser.add_argument(
        "model",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help="模型类型 (baseline, bifpn, fce)",
    )

    # 模型尺度
    parser.add_argument("--scale", type=str, default="n", choices=["n", "s", "m", "l", "x"], help="模型尺度 (默认: n)")

    # 数据集
    parser.add_argument(
        "--dataset",
        type=str,
        default="default",
        choices=list(DATASET_PRESETS.keys()),
        help="数据集预设 (default/coco/coco_hq)",
    )
    parser.add_argument("--data", type=str, default=None, help="自定义数据集路径（覆盖 --dataset）")

    # 共享参数
    parser.add_argument("--epochs", type=int, default=None, help="stage2 训练轮次（单阶段即总轮次）")
    parser.add_argument("--batch", type=int, default=None, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=None, help="输入图像尺寸")
    parser.add_argument("--device", type=str, default=None, help="训练设备")
    parser.add_argument("--workers", type=int, default=None, help="数据加载工作进程数")
    parser.add_argument("--patience", type=int, default=None, help="早停耐心值")
    parser.add_argument("--lr0", type=float, default=None, help="stage2 初始学习率")
    parser.add_argument("--cos-lr", action="store_true", default=None, help="stage2 余弦退火（不传则使用模型默认值）")
    parser.add_argument("--no-cos-lr", action="store_true", default=False, help="stage2 禁用余弦退火")
    parser.add_argument("--close-mosaic", type=int, default=None, help="stage2 最后 N epochs 关闭 Mosaic")

    # IoU 损失
    parser.add_argument(
        "--iou-type", type=str, default=None, choices=["CIoU", "DIoU", "GIoU", "WIoU"], help="IoU 损失类型"
    )

    # 其他共享参数
    parser.add_argument("--no-amp", action="store_true", help="禁用混合精度训练")
    parser.add_argument("--cache", type=str, default=None, help="数据缓存策略 (false/disk/ram)")

    # stage1 覆盖（高级用法）
    parser.add_argument("--stage1-epochs", type=int, default=None, help="stage1 训练轮次")
    parser.add_argument("--stage1-lr0", type=float, default=None, help="stage1 初始学习率")
    parser.add_argument("--stage1-patience", type=int, default=None, help="stage1 早停耐心值")

    # 快速测试
    parser.add_argument("--test", action="store_true", help="快速测试模式（1 epoch, 小图像尺寸）")

    # 实验名称
    parser.add_argument("--name", type=str, default=None, help="自定义实验名称")

    return parser.parse_args()


def build_overrides(args: argparse.Namespace):
    """从 CLI 参数构建三类 override 字典."""
    shared, stage2, stage1 = build_overrides_from_namespace(args)
    return shared, stage2, stage1


def main():
    """主函数."""
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
    print("\n模型配置:")
    print(f"  YAML: {model_cfg.yaml_path}")
    print(f"  两阶段训练: {'是' if model_cfg.is_two_stage() else '否'}")
    if model_cfg.is_two_stage():
        print(f"  冻结层数: {model_cfg.freeze}")
        print(f"  阶段一: {config.stage1.epochs} epochs, lr={config.stage1.lr0}, cos_lr={config.stage1.cos_lr}")
    print(f"  阶段二: {config.stage2.epochs} epochs, lr={config.stage2.lr0}, cos_lr={config.stage2.cos_lr}")
    print("\n共享参数:")
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

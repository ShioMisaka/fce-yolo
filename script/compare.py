#!/usr/bin/env python3
"""
统一对比 CLI

合并原 compare.py + train_coco_compare.py + compare_iou.py。

Usage:
    python script/compare.py --models baseline fce --scale s
    python script/compare.py --models baseline fce --scale s --epochs 200
    python script/compare.py --models baseline bifpn fce --scale s --skip-train
    python script/compare.py --models baseline fce --scale s --iou-type WIoU
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from script.config import (
    DATASET_PRESETS,
    MODEL_CONFIGS,
    build_overrides_from_namespace,
    get_dataset_preset,
    get_model_config,
    get_quick_test_config,
    apply_overrides,
)
from script.analysis import (
    load_results,
    extract_metrics,
    print_comparison_table,
    plot_comparison_curves,
    save_comparison_summary,
    reorganize_results,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="YOLOv11 多模型训练对比（统一入口）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本对比
  python script/compare.py --models baseline fce --scale s

  # 覆盖 epochs
  python script/compare.py --models baseline fce --scale s --epochs 200

  # 三模型对比
  python script/compare.py --models baseline bifpn fce --scale s

  # 跳过训练
  python script/compare.py --models baseline fce --scale s --skip-train

  # 切换 IoU 损失
  python script/compare.py --models baseline bifpn fce --scale s --iou-type WIoU

  # 切换数据集
  python script/compare.py --models baseline fce --scale s --dataset coco_hq
        """
    )

    parser.add_argument(
        "--models", type=str, nargs="+", required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="要对比的模型列表",
    )
    parser.add_argument("--scale", type=str, default="n",
                        choices=["n", "s", "m", "l", "x"],
                        help="模型尺度 (默认: n)")
    parser.add_argument("--dataset", type=str, default="default",
                        help="数据集预设 (default/coco/coco_hq)")
    parser.add_argument("--data", type=str, default=None,
                        help="自定义数据集路径（覆盖 --dataset）")

    # 共享参数
    parser.add_argument("--epochs", type=int, default=None,
                        help="stage2 训练轮次")
    parser.add_argument("--batch", type=int, default=None,
                        help="批次大小")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="输入图像尺寸")
    parser.add_argument("--device", type=str, default=None,
                        help="训练设备")
    parser.add_argument("--workers", type=int, default=None,
                        help="数据加载工作进程数")
    parser.add_argument("--patience", type=int, default=None,
                        help="早停耐心值")
    parser.add_argument("--lr0", type=float, default=None,
                        help="stage2 初始学习率")
    parser.add_argument("--cos-lr", action="store_true", default=None,
                        help="stage2 余弦退火")
    parser.add_argument("--no-cos-lr", action="store_true", default=False,
                        help="stage2 禁用余弦退火")
    parser.add_argument("--close-mosaic", type=int, default=None,
                        help="stage2 最后 N epochs 关闭 Mosaic")
    parser.add_argument("--iou-type", type=str, default=None,
                        choices=["CIoU", "DIoU", "GIoU", "WIoU"],
                        help="IoU 损失类型")

    # 对比控制
    parser.add_argument("--skip-train", action="store_true",
                        help="跳过训练，仅对比已有结果")
    parser.add_argument("--output", type=str, default=None,
                        help="自定义输出目录")
    parser.add_argument("--test", action="store_true",
                        help="快速测试模式")

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    scale = args.scale
    models = args.models

    # 构建训练配置
    if args.test:
        config = get_quick_test_config()
    else:
        config = get_dataset_preset(args.dataset)

    # 构建 override（包含 --data 处理）
    shared, stage2, stage1 = build_overrides_from_namespace(args)

    # 输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        tag = "vs".join(models)
        epochs_str = str(args.epochs) if args.epochs else "300"
        output_dir = Path(f"runs/detect/{tag}_{scale}_{epochs_str}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打印信息
    model_names = " vs ".join(m.upper() for m in models)
    print("\n" + "=" * 100)
    print(f"{model_names} 训练对比 (尺度: {scale.upper()})")
    print("=" * 100)
    print(f"\n输出目录: {output_dir}")
    print(f"数据集: {config.data}")
    if args.epochs:
        print(f"覆盖 stage2 轮次: {args.epochs}")
    print()

    # 阶段一：训练所有模型
    if not args.skip_train:
        print("▶" * 50)
        print("训练所有模型")
        print("▶" * 50)

        from script.trainer import YOLOv11Trainer

        for model_type in models:
            model_cfg = get_model_config(model_type)
            model_config = apply_overrides(
                get_dataset_preset(args.dataset) if not args.test else get_quick_test_config(),
                model_cfg, shared, stage2, {},
            )

            print(f"\n{'=' * 60}")
            print(f"训练: {model_cfg.get_display_name(scale)}")
            print(f"{'=' * 60}")

            try:
                trainer = YOLOv11Trainer(model_cfg, scale, model_config)
                trainer.train()
                print(f"✓ {model_type} 训练完成")
            except Exception as e:
                print(f"✗ {model_type} 训练失败: {e}")
                return
    else:
        print("⚠ 跳过训练阶段，使用已有结果")

    # 阶段二：整理结果
    print("\n" + "▶" * 50)
    print("整理训练结果")
    print("▶" * 50)

    result_paths = {}
    for model_type in models:
        model_cfg = get_model_config(model_type)
        result_paths[model_type] = model_cfg.get_result_path(scale)

    csv_paths = reorganize_results(result_paths, output_dir)

    if len(csv_paths) < len(models):
        missing = set(models) - set(csv_paths.keys())
        print(f"\n✗ 缺少训练结果: {missing}")
        return

    # 阶段三：生成对比分析
    print("\n" + "▶" * 50)
    print("生成对比分析")
    print("▶" * 50)

    all_dataframes = {}
    all_metrics = {}
    names = {}
    colors = {}

    for model_type in models:
        model_cfg = get_model_config(model_type)
        names[model_type] = model_cfg.get_display_name(scale)
        colors[model_type] = model_cfg.color

        df = load_results(csv_paths[model_type])
        all_dataframes[model_type] = df
        all_metrics[model_type] = extract_metrics(df)

    print_comparison_table(
        all_metrics, names,
        title=f"{model_names} 训练结果对比表 (尺度: {scale.upper()})",
    )

    plot_comparison_curves(
        all_dataframes, names, colors,
        save_path=output_dir / "comparison_curves.png",
    )

    save_comparison_summary(
        output_path=output_dir / "comparison_summary.txt",
        metrics=all_metrics,
        names=names,
        config_info={
            "模型尺度": scale.upper(),
            "数据集": config.data,
            "IoU 损失": config.iou_type,
        },
    )

    # 完成
    print("\n" + "=" * 100)
    print("训练对比完成!")
    print("=" * 100)
    print(f"\n所有结果已保存到: {output_dir}")
    print()
    for model_type in models:
        path = get_model_config(model_type).get_result_path(scale)
        print(f"  ├── {path}/")
    print(f"  ├── comparison_curves.png")
    print(f"  └── comparison_summary.txt")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()

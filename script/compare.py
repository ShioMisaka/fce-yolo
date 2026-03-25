#!/usr/bin/env python3
"""
对比 CLI 脚本

提供命令行接口用于多模型训练对比。

Usage:
    python script/compare.py --models baseline bifpn --scale s
    python script/compare.py --models baseline fce --scale s --skip-train
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from script.config import (
    MODEL_CONFIGS,
    ExperimentConfig,
    TrainConfig,
    get_model_config,
)
from script.trainer import train_model


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="YOLOv11 多模型训练对比脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Baseline vs BiFPN 对比
  python script/compare.py --models baseline bifpn --scale s

  # Baseline vs FCE 对比
  python script/compare.py --models baseline fce --scale s

  # 三模型对比
  python script/compare.py --models baseline bifpn fce --scale s

  # 仅对比已有结果（不训练）
  python script/compare.py --models baseline bifpn --scale s --skip-train

  # 自定义输出目录
  python script/compare.py --models baseline fce --scale s --output my_experiment
        """
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="要对比的模型列表"
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="模型尺度"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="跳过训练，仅对比已有结果"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="自定义输出目录"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="总训练轮次（仅用于输出目录命名）"
    )

    return parser.parse_args()


def run_training(model_type: str, scale: str) -> bool:
    """运行训练脚本

    Args:
        model_type: 模型类型
        scale: 模型尺度

    Returns:
        训练是否成功
    """
    print("\n" + "=" * 60)
    print(f"开始训练: {model_type} (scale={scale})")
    print("=" * 60)

    try:
        result = subprocess.run(
            [sys.executable, "script/train.py", model_type, "--scale", scale],
            check=True,
            capture_output=False,
        )
        print(f"✓ {model_type} 训练完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {model_type} 训练失败: {e}")
        return False


def reorganize_results(config: ExperimentConfig) -> Dict[str, Path]:
    """重新组织训练结果到统一目录"""
    print("\n" + "=" * 60)
    print("整理训练结果...")
    print("=" * 60)

    results_paths = {}

    for model_type in config.models:
        model_cfg = get_model_config(model_type)

        # 原始结果路径
        src_pattern = model_cfg.get_result_path(config.scale)
        src_path = Path("runs/detect") / src_pattern

        # 目标路径
        dst_path = config.output_dir / src_pattern

        if src_path.exists():
            if dst_path.exists():
                shutil.rmtree(dst_path)

            shutil.move(str(src_path), str(dst_path))
            print(f"✓ 移动: {src_path.name} -> {dst_path}")

            results_csv = dst_path / "results.csv"
            if results_csv.exists():
                results_paths[model_type] = results_csv
        else:
            print(f"⚠ 跳过: {src_path} (不存在)")

    return results_paths


def load_results(csv_path: Path) -> pd.DataFrame:
    """加载训练结果 CSV 文件"""
    if not csv_path.exists():
        raise FileNotFoundError(f"结果文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"✓ 加载: {csv_path}")
    print(f"  训练轮次: {int(df['epoch'].iloc[-1])} epochs")

    return df


def extract_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """提取关键指标"""
    metrics = {}

    last_row = df.iloc[-1]
    metrics["final_epoch"] = int(last_row["epoch"])
    metrics["final_map50_95"] = last_row.get("metrics/mAP50-95(B)", 0)
    metrics["final_map50"] = last_row.get("metrics/mAP50(B)", 0)
    metrics["final_precision"] = last_row.get("metrics/precision(B)", 0)
    metrics["final_recall"] = last_row.get("metrics/recall(B)", 0)

    if "metrics/mAP50-95(B)" in df.columns:
        best_idx = df["metrics/mAP50-95(B)"].idxmax()
        best_row = df.loc[best_idx]
        metrics["best_map50_95"] = best_row["metrics/mAP50-95(B)"]
        metrics["best_map50_95_epoch"] = int(best_row["epoch"])
    else:
        metrics["best_map50_95"] = 0
        metrics["best_map50_95_epoch"] = 0

    if "metrics/mAP50(B)" in df.columns:
        best_idx = df["metrics/mAP50(B)"].idxmax()
        best_row = df.loc[best_idx]
        metrics["best_map50"] = best_row["metrics/mAP50(B)"]
        metrics["best_map50_epoch"] = int(best_row["epoch"])
    else:
        metrics["best_map50"] = 0
        metrics["best_map50_epoch"] = 0

    return metrics


def print_comparison_table(all_metrics: Dict[str, Dict], config: ExperimentConfig):
    """打印对比表格"""
    print("\n" + "=" * 100)
    model_names = " vs ".join([m.upper() for m in config.models])
    print(f"{model_names} 训练结果对比表 (尺度: {config.scale.upper()})")
    print("=" * 100)

    header = f"{'指标':<20}"
    for model_type in config.models:
        header += f"{model_type.upper():<20}"
    header += f"{'最佳模型':<20}"
    print(header)
    print("-" * 100)

    metrics_to_compare = [
        ("Best mAP@50-95", "best_map50_95"),
        ("Best mAP@50", "best_map50"),
        ("Final mAP@50-95", "final_map50_95"),
        ("Final mAP@50", "final_map50"),
        ("Final Precision", "final_precision"),
        ("Final Recall", "final_recall"),
    ]

    for metric_name, metric_key in metrics_to_compare:
        row = f"{metric_name:<20}"

        values = {}
        for model_type in config.models:
            value = all_metrics[model_type][metric_key]
            values[model_type] = value
            row += f"{value:<20.4f}"

        best_model = max(values.keys(), key=lambda k: values[k])
        row += f"{best_model.upper():<20}"

        print(row)

    print("=" * 100)


def plot_comparison_curves(
    all_dataframes: Dict[str, pd.DataFrame],
    save_path: Path,
    config: ExperimentConfig,
):
    """绘制训练对比曲线"""
    plt.rcParams.update({
        "font.size": 11,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "lines.linewidth": 2,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), tight_layout=True)

    metrics_config = [
        ("metrics/mAP50-95(B)", "mAP@50-95"),
        ("metrics/mAP50(B)", "mAP@50"),
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall"),
    ]

    display_names = config.get_model_display_names()
    colors = config.get_model_colors()

    for idx, (metric_key, metric_name) in enumerate(metrics_config):
        ax = axes[idx // 2, idx % 2]

        for model_type in config.models:
            df = all_dataframes[model_type]
            ax.plot(
                df["epoch"],
                df[metric_key],
                color=colors[model_type],
                label=display_names[model_type],
                linewidth=2,
            )

        ax.set_title(f"{metric_name} 对比", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.2)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\n✓ 对比曲线已保存: {save_path}")

    plt.close()


def save_comparison_summary(config: ExperimentConfig, all_metrics: Dict[str, Dict]):
    """保存对比摘要到文本文件"""
    summary_path = config.output_dir / "comparison_summary.txt"

    model_names = " vs ".join([m.upper() for m in config.models])
    display_names = config.get_model_display_names()

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"{model_names} 训练结果对比\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"模型尺度: {config.scale.upper()}\n")
        f.write(f"总训练轮次: {config.total_epochs} epochs\n\n")

        for model_type in config.models:
            model_cfg = get_model_config(model_type)
            f.write(f"{model_type.upper()}: {model_cfg.yaml_path}\n")
        f.write("\n")

        f.write("-" * 100 + "\n")
        header = f"{'指标':<25}"
        for model_type in config.models:
            header += f"{model_type.upper():<20}"
        f.write(header + "\n")
        f.write("-" * 100 + "\n")

        metrics_to_write = [
            ("Best mAP@50-95", "best_map50_95"),
            ("Best mAP@50", "best_map50"),
            ("Final mAP@50-95", "final_map50_95"),
            ("Final mAP@50", "final_map50"),
            ("Final Precision", "final_precision"),
            ("Final Recall", "final_recall"),
        ]

        for metric_name, metric_key in metrics_to_write:
            row = f"{metric_name:<25}"
            for model_type in config.models:
                value = all_metrics[model_type][metric_key]
                row += f"{value:<20.4f}"
            f.write(row + "\n")

        f.write("-" * 100 + "\n")
        f.write("\n目录结构:\n")
        for model_type in config.models:
            result_pattern = get_model_config(model_type).get_result_path(config.scale)
            f.write(f"  - {model_type.upper()} 结果: {result_pattern}/\n")
        f.write(f"  - 对比曲线图: comparison_curves.png\n")
        f.write(f"  - 本摘要文件: comparison_summary.txt\n")
        f.write("=" * 100 + "\n")

    print(f"✓ 对比摘要已保存: {summary_path}")


def main():
    """主函数"""
    args = parse_args()

    # 创建实验配置
    config = ExperimentConfig(
        models=args.models,
        scale=args.scale,
        total_epochs=args.epochs,
        skip_train=args.skip_train,
        output_dir=Path(args.output) if args.output else None,
    )

    model_names = " vs ".join([m.upper() for m in config.models])

    print("\n" + "=" * 100)
    print(f"{model_names} 训练对比 (尺度: {config.scale.upper()})")
    print("=" * 100)
    print(f"\n输出目录: {config.output_dir}")

    # 阶段一：训练所有模型
    if not config.skip_train:
        print("\n" + "▶" * 50)
        print("阶段一：训练所有模型")
        print("▶" * 50)

        all_success = True
        for model_type in config.models:
            success = run_training(model_type, config.scale)
            if not success:
                print(f"\n✗ {model_type} 训练失败，终止流程")
                all_success = False
                break

        if not all_success:
            return
    else:
        print("\n⚠ 跳过训练阶段，使用已有结果")

    # 阶段二：整理结果
    print("\n" + "▶" * 50)
    print("阶段二：整理训练结果")
    print("▶" * 50)

    results_paths = reorganize_results(config)

    if len(results_paths) < len(config.models):
        missing = set(config.models) - set(results_paths.keys())
        print(f"\n✗ 缺少训练结果文件: {missing}")
        return

    # 阶段三：生成对比分析
    print("\n" + "▶" * 50)
    print("阶段三：生成对比分析")
    print("▶" * 50)

    all_dataframes = {}
    all_metrics = {}

    for model_type, csv_path in results_paths.items():
        df = load_results(csv_path)
        all_dataframes[model_type] = df
        all_metrics[model_type] = extract_metrics(df)

    print_comparison_table(all_metrics, config)

    comparison_image = config.output_dir / "comparison_curves.png"
    plot_comparison_curves(all_dataframes, comparison_image, config)

    save_comparison_summary(config, all_metrics)

    # 完成
    print("\n" + "=" * 100)
    print("完整训练对比流程完成!")
    print("=" * 100)
    print(f"\n所有结果已保存到: {config.output_dir}")
    print("\n目录结构:")
    for model_type in config.models:
        result_pattern = get_model_config(model_type).get_result_path(config.scale)
        print(f"  ├── {result_pattern}/")
    print(f"  ├── comparison_curves.png")
    print(f"  └── comparison_summary.txt")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()

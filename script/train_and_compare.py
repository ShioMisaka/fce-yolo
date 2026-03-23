#!/usr/bin/env python3
"""
YOLOv11 Baseline vs BiFPN 完整训练对比脚本

自动执行以下流程：
1. 训练 Baseline 模型
2. 训练 BiFPN 模型（两阶段）
3. 生成对比分析图表

所有结果保存到统一的目录结构中，方便查阅和对比。

Usage:
    # 使用默认 nano 模型
    python script/train_and_compare.py

    # 指定模型尺度
    python script/train_and_compare.py --scale s
    python script/train_and_compare.py --scale m
    python script/train_and_compare.py --scale l
    python script/train_and_compare.py --scale x
"""

import argparse
import os
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


# ==================== 配置 ====================

# 颜色配置（Ultralytics 风格）
COLORS = {
    "baseline": "#0BDBEB",  # 青色
    "bifpn": "#042AFF",     # 蓝色
}


def setup_output_directory(scale: str, total_epochs: int) -> Path:
    """
    设置输出目录结构

    Args:
        scale: 模型尺度
        total_epochs: 总训练轮次

    Returns:
        统一的输出目录路径
    """
    output_dir = Path(f"runs/detect/11vsbifpn_{scale}_{total_epochs}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    return output_dir


def run_training(script_path: str, scale: str) -> bool:
    """
    运行训练脚本

    Args:
        script_path: 训练脚本路径
        scale: 模型尺度

    Returns:
        训练是否成功
    """
    script_name = Path(script_path).stem
    print("\n" + "=" * 60)
    print(f"开始训练: {script_name} (scale={scale})")
    print("=" * 60)

    try:
        result = subprocess.run(
            [sys.executable, script_path, "--scale", scale],
            check=True,
            capture_output=False,
        )
        print(f"✓ {script_name} 训练完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {script_name} 训练失败: {e}")
        return False


def reorganize_results(scale: str, output_dir: Path, total_epochs: int) -> Dict[str, Path]:
    """
    重新组织训练结果到统一目录

    Args:
        scale: 模型尺度
        output_dir: 统一输出目录
        total_epochs: 总训练轮次

    Returns:
        各模型结果 CSV 文件路径字典
    """
    print("\n" + "=" * 60)
    print("整理训练结果...")
    print("=" * 60)

    # 原始结果路径
    baseline_src = Path(f"runs/detect/baseline_yolo11{scale}")
    bifpn_s1_src = Path(f"runs/detect/bifpn_{scale}_stage1_warmup")
    bifpn_s2_src = Path(f"runs/detect/bifpn_{scale}_stage2_finetune")

    # 目标路径
    baseline_dst = output_dir / "baseline_yolo11"
    bifpn_s1_dst = output_dir / "bifpn_stage1_warmup"
    bifpn_s2_dst = output_dir / "bifpn_stage2_finetune"

    # 移动目录
    paths_to_move = [
        (baseline_src, baseline_dst),
        (bifpn_s1_src, bifpn_s1_dst),
        (bifpn_s2_src, bifpn_s2_dst),
    ]

    results_paths = {}

    for src, dst in paths_to_move:
        if src.exists():
            # 如果目标已存在，先删除
            if dst.exists():
                shutil.rmtree(dst)

            shutil.move(str(src), str(dst))
            print(f"✓ 移动: {src.name} -> {dst}")

            # 保存 results.csv 路径
            results_csv = dst / "results.csv"
            if results_csv.exists():
                if "baseline" in dst.name:
                    results_paths["baseline"] = results_csv
                elif "stage2" in dst.name:
                    results_paths["bifpn"] = results_csv
        else:
            print(f"⚠ 跳过: {src} (不存在)")

    return results_paths


def load_results(csv_path: Path) -> pd.DataFrame:
    """
    加载训练结果 CSV 文件

    Args:
        csv_path: CSV 文件路径

    Returns:
        训练结果 DataFrame
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"结果文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"✓ 加载: {csv_path}")
    print(f"  训练轮次: {int(df['epoch'].iloc[-1])} epochs")
    print(f"  数据行数: {len(df)}")

    return df


def extract_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    提取关键指标

    Args:
        df: 训练结果 DataFrame

    Returns:
        包含关键指标的字典
    """
    metrics = {}

    # 最终 epoch 的指标
    last_row = df.iloc[-1]
    metrics["final_epoch"] = int(last_row["epoch"])
    metrics["final_map50_95"] = last_row.get("metrics/mAP50-95(B)", 0)
    metrics["final_map50"] = last_row.get("metrics/mAP50(B)", 0)
    metrics["final_precision"] = last_row.get("metrics/precision(B)", 0)
    metrics["final_recall"] = last_row.get("metrics/recall(B)", 0)

    # 最佳 mAP@50-95
    if "metrics/mAP50-95(B)" in df.columns:
        best_idx = df["metrics/mAP50-95(B)"].idxmax()
        best_row = df.loc[best_idx]
        metrics["best_map50_95"] = best_row["metrics/mAP50-95(B)"]
        metrics["best_map50_95_epoch"] = int(best_row["epoch"])
    else:
        metrics["best_map50_95"] = 0
        metrics["best_map50_95_epoch"] = 0

    # 最佳 mAP@50
    if "metrics/mAP50(B)" in df.columns:
        best_idx = df["metrics/mAP50(B)"].idxmax()
        best_row = df.loc[best_idx]
        metrics["best_map50"] = best_row["metrics/mAP50(B)"]
        metrics["best_map50_epoch"] = int(best_row["epoch"])
    else:
        metrics["best_map50"] = 0
        metrics["best_map50_epoch"] = 0

    return metrics


def print_comparison_table(baseline_metrics: Dict, bifpn_metrics: Dict, scale: str):
    """
    打印对比表格

    Args:
        baseline_metrics: Baseline 模型指标
        bifpn_metrics: BiFPN 模型指标
        scale: 模型尺度
    """
    print("\n" + "=" * 80)
    print(f"YOLOv11{scale.upper()} 训练结果对比表")
    print("=" * 80)
    print(f"{'指标':<20} {'Baseline':<20} {'BiFPN':<20} {'差异':<15}")
    print("-" * 80)

    # mAP@50-95 对比
    baseline_best = baseline_metrics["best_map50_95"]
    bifpn_best = bifpn_metrics["best_map50_95"]
    diff = bifpn_best - baseline_best
    diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
    print(f"{'Best mAP@50-95':<20} {baseline_best:<20.4f} {bifpn_best:<20.4f} {diff_str:<15}")

    # mAP@50 对比
    baseline_best = baseline_metrics["best_map50"]
    bifpn_best = bifpn_metrics["best_map50"]
    diff = bifpn_best - baseline_best
    diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
    print(f"{'Best mAP@50':<20} {baseline_best:<20.4f} {bifpn_best:<20.4f} {diff_str:<15}")

    # 最终 mAP@50-95
    baseline_final = baseline_metrics["final_map50_95"]
    bifpn_final = bifpn_metrics["final_map50_95"]
    diff = bifpn_final - baseline_final
    diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
    print(f"{'Final mAP@50-95':<20} {baseline_final:<20.4f} {bifpn_final:<20.4f} {diff_str:<15}")

    # 最终 Precision
    baseline_final = baseline_metrics["final_precision"]
    bifpn_final = bifpn_metrics["final_precision"]
    diff = bifpn_final - baseline_final
    diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
    print(f"{'Final Precision':<20} {baseline_final:<20.4f} {bifpn_final:<20.4f} {diff_str:<15}")

    # 最终 Recall
    baseline_final = baseline_metrics["final_recall"]
    bifpn_final = bifpn_metrics["final_recall"]
    diff = bifpn_final - baseline_final
    diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
    print(f"{'Final Recall':<20} {baseline_final:<20.4f} {bifpn_final:<20.4f} {diff_str:<15}")

    print("=" * 80)


def plot_comparison_curves(
    baseline_df: pd.DataFrame,
    bifpn_df: pd.DataFrame,
    save_path: Path,
    scale: str,
):
    """
    绘制训练对比曲线

    Args:
        baseline_df: Baseline 训练结果
        bifpn_df: BiFPN 训练结果
        save_path: 图片保存路径
        scale: 模型尺度
    """
    # 设置 matplotlib 风格
    plt.rcParams.update({
        "font.size": 11,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "lines.linewidth": 2,
    })

    # 模型名称
    model_names = {
        "baseline": f"YOLOv11{scale.upper()} Baseline",
        "bifpn": f"YOLOv11{scale.upper()}-BiFPN",
    }

    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), tight_layout=True)

    # 1. mAP 对比
    ax = axes[0, 0]
    ax.plot(baseline_df["epoch"], baseline_df["metrics/mAP50-95(B)"],
            color=COLORS["baseline"], label=model_names["baseline"], linewidth=2)
    ax.plot(bifpn_df["epoch"], bifpn_df["metrics/mAP50-95(B)"],
            color=COLORS["bifpn"], label=model_names["bifpn"], linewidth=2)
    ax.set_title("mAP@50-95 对比", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@50-95")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)

    # 2. mAP@50 对比
    ax = axes[0, 1]
    ax.plot(baseline_df["epoch"], baseline_df["metrics/mAP50(B)"],
            color=COLORS["baseline"], label=model_names["baseline"], linewidth=2)
    ax.plot(bifpn_df["epoch"], bifpn_df["metrics/mAP50(B)"],
            color=COLORS["bifpn"], label=model_names["bifpn"], linewidth=2)
    ax.set_title("mAP@50 对比", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@50")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)

    # 3. Precision 对比
    ax = axes[1, 0]
    ax.plot(baseline_df["epoch"], baseline_df["metrics/precision(B)"],
            color=COLORS["baseline"], label=model_names["baseline"], linewidth=2)
    ax.plot(bifpn_df["epoch"], bifpn_df["metrics/precision(B)"],
            color=COLORS["bifpn"], label=model_names["bifpn"], linewidth=2)
    ax.set_title("Precision 对比", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Precision")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)

    # 4. Recall 对比
    ax = axes[1, 1]
    ax.plot(baseline_df["epoch"], baseline_df["metrics/recall(B)"],
            color=COLORS["baseline"], label=model_names["baseline"], linewidth=2)
    ax.plot(bifpn_df["epoch"], bifpn_df["metrics/recall(B)"],
            color=COLORS["bifpn"], label=model_names["bifpn"], linewidth=2)
    ax.set_title("Recall 对比", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)

    # 保存图片
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\n✓ 对比曲线已保存: {save_path}")

    plt.close()


def save_comparison_summary(
    output_dir: Path,
    scale: str,
    baseline_metrics: Dict,
    bifpn_metrics: Dict,
):
    """
    保存对比摘要到文本文件

    Args:
        output_dir: 输出目录
        scale: 模型尺度
        baseline_metrics: Baseline 模型指标
        bifpn_metrics: BiFPN 模型指标
    """
    summary_path = output_dir / "comparison_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"YOLOv11{scale.upper()} Baseline vs BiFPN 训练结果对比\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"模型尺度: {scale.upper()}\n")
        f.write(f"总训练轮次: 300 epochs (Baseline: 300, BiFPN: 50 + 250)\n\n")

        f.write("-" * 80 + "\n")
        f.write(f"{'指标':<25} {'Baseline':<20} {'BiFPN':<20} {'差异':<15}\n")
        f.write("-" * 80 + "\n")

        # 写入各项指标
        metrics_to_write = [
            ("Best mAP@50-95", "best_map50_95"),
            ("Best mAP@50", "best_map50"),
            ("Final mAP@50-95", "final_map50_95"),
            ("Final mAP@50", "final_map50"),
            ("Final Precision", "final_precision"),
            ("Final Recall", "final_recall"),
        ]

        for name, key in metrics_to_write:
            baseline_val = baseline_metrics[key]
            bifpn_val = bifpn_metrics[key]
            diff = bifpn_val - baseline_val
            diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
            f.write(f"{name:<25} {baseline_val:<20.4f} {bifpn_val:<20.4f} {diff_str:<15}\n")

        f.write("-" * 80 + "\n")
        f.write("\n目录结构:\n")
        f.write(f"  - Baseline 结果: baseline_yolo11/\n")
        f.write(f"  - BiFPN-S1 结果: bifpn_stage1_warmup/\n")
        f.write(f"  - BiFPN-S2 结果: bifpn_stage2_finetune/\n")
        f.write(f"  - 对比曲线图: comparison_curves.png\n")
        f.write(f"  - 本摘要文件: comparison_summary.txt\n")
        f.write("=" * 80 + "\n")

    print(f"✓ 对比摘要已保存: {summary_path}")


def main(scale: str = "n"):
    """
    主函数：执行完整的训练和对比流程

    Args:
        scale: 模型尺度 (n, s, m, l, x)
    """
    print("\n" + "=" * 80)
    print(f"YOLOv11{scale.upper()} Baseline vs BiFPN 完整训练对比")
    print("=" * 80)

    # 总训练轮次
    TOTAL_EPOCHS = 300

    # 设置输出目录
    output_dir = setup_output_directory(scale, TOTAL_EPOCHS)

    # 阶段一：训练 Baseline
    print("\n" + "▶" * 40)
    print("阶段一：训练 Baseline 模型")
    print("▶" * 40)

    baseline_success = run_training("script/train_baseline.py", scale)

    if not baseline_success:
        print("\n✗ Baseline 训练失败，终止流程")
        return

    # 阶段二：训练 BiFPN
    print("\n" + "▶" * 40)
    print("阶段二：训练 BiFPN 模型")
    print("▶" * 40)

    bifpn_success = run_training("script/train_bifpn.py", scale)

    if not bifpn_success:
        print("\n✗ BiFPN 训练失败，终止流程")
        return

    # 阶段三：整理结果
    print("\n" + "▶" * 40)
    print("阶段三：整理训练结果")
    print("▶" * 40)

    results_paths = reorganize_results(scale, output_dir, TOTAL_EPOCHS)

    if "baseline" not in results_paths or "bifpn" not in results_paths:
        print("\n✗ 缺少训练结果文件，无法生成对比")
        return

    # 阶段四：生成对比分析
    print("\n" + "▶" * 40)
    print("阶段四：生成对比分析")
    print("▶" * 40)

    # 加载结果
    print("\n加载训练结果...")
    baseline_df = load_results(results_paths["baseline"])
    bifpn_df = load_results(results_paths["bifpn"])

    # 提取指标
    baseline_metrics = extract_metrics(baseline_df)
    bifpn_metrics = extract_metrics(bifpn_df)

    # 打印对比表格
    print_comparison_table(baseline_metrics, bifpn_metrics, scale)

    # 绘制对比曲线
    comparison_image = output_dir / "comparison_curves.png"
    plot_comparison_curves(baseline_df, bifpn_df, comparison_image, scale)

    # 保存对比摘要
    save_comparison_summary(output_dir, scale, baseline_metrics, bifpn_metrics)

    # 完成
    print("\n" + "=" * 80)
    print("完整训练对比流程完成!")
    print("=" * 80)
    print(f"\n所有结果已保存到: {output_dir}")
    print("\n目录结构:")
    print(f"  ├── baseline_yolo11/           # Baseline 训练结果")
    print(f"  ├── bifpn_stage1_warmup/        # BiFPN 阶段一结果")
    print(f"  ├── bifpn_stage2_finetune/      # BiFPN 阶段二结果")
    print(f"  ├── comparison_curves.png       # 对比曲线图")
    print(f"  └── comparison_summary.txt      # 对比摘要文本")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOv11 Baseline vs BiFPN 完整训练对比脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python script/train_and_compare.py           # 使用默认 nano 模型
  python script/train_and_compare.py --scale s # 使用 small 模型
  python script/train_and_compare.py --scale m # 使用 medium 模型
  python script/train_and_compare.py --scale l # 使用 large 模型
  python script/train_and_compare.py --scale x # 使用 xlarge 模型
        """
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="模型尺度 (n: nano, s: small, m: medium, l: large, x: xlarge)"
    )
    args = parser.parse_args()

    main(scale=args.scale)

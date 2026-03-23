#!/usr/bin/env python3
"""
YOLOv11 Baseline vs BiFPN 训练结果对比脚本

对比分析原生 YOLOv11 和 YOLOv11-BiFPN 的训练结果，
生成对比图表和统计表格。

Usage:
    python script/compare_results.py
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ==================== 配置 ====================

# 结果文件路径
BASELINE_RESULTS = "runs/detect/11vsbifpn_s_300/baseline_yolo11s/results.csv"
BIFPN_RESULTS = "runs/detect/11vsbifpn_s_300/bifpn_s_stage2_finetune/results.csv"

# 输出路径
OUTPUT_DIR = "script"
OUTPUT_IMAGE = "comparison_curve.png"

# 模型名称
MODEL_NAMES = {
    "baseline": "YOLOv11n Baseline",
    "bifpn": "YOLOv11n-BiFPN",
}

# 颜色配置（Ultralytics 风格）
COLORS = {
    "baseline": "#0BDBEB",  # 青色
    "bifpn": "#042AFF",     # 蓝色
}


def load_results(csv_path: str) -> pd.DataFrame:
    """
    加载训练结果 CSV 文件

    Args:
        csv_path: CSV 文件路径

    Returns:
        训练结果 DataFrame
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"结果文件不存在: {csv_path}")

    df = pd.read_csv(path)
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


def print_comparison_table(baseline_metrics: Dict, bifpn_metrics: Dict):
    """
    打印对比表格

    Args:
        baseline_metrics: Baseline 模型指标
        bifpn_metrics: BiFPN 模型指标
    """
    print("\n" + "=" * 80)
    print("训练结果对比表")
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
    save_path: str
):
    """
    绘制训练对比曲线

    Args:
        baseline_df: Baseline 训练结果
        bifpn_df: BiFPN 训练结果
        save_path: 图片保存路径
    """
    # 设置 matplotlib 风格
    plt.rcParams.update({
        "font.size": 11,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "lines.linewidth": 2,
    })

    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), tight_layout=True)

    # 1. mAP 对比
    ax = axes[0, 0]
    ax.plot(baseline_df["epoch"], baseline_df["metrics/mAP50-95(B)"],
            color=COLORS["baseline"], label=MODEL_NAMES["baseline"], linewidth=2)
    ax.plot(bifpn_df["epoch"], bifpn_df["metrics/mAP50-95(B)"],
            color=COLORS["bifpn"], label=MODEL_NAMES["bifpn"], linewidth=2)
    ax.set_title("mAP@50-95 对比", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@50-95")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)

    # 2. mAP@50 对比
    ax = axes[0, 1]
    ax.plot(baseline_df["epoch"], baseline_df["metrics/mAP50(B)"],
            color=COLORS["baseline"], label=MODEL_NAMES["baseline"], linewidth=2)
    ax.plot(bifpn_df["epoch"], bifpn_df["metrics/mAP50(B)"],
            color=COLORS["bifpn"], label=MODEL_NAMES["bifpn"], linewidth=2)
    ax.set_title("mAP@50 对比", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@50")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)

    # 3. Precision 对比
    ax = axes[1, 0]
    ax.plot(baseline_df["epoch"], baseline_df["metrics/precision(B)"],
            color=COLORS["baseline"], label=MODEL_NAMES["baseline"], linewidth=2)
    ax.plot(bifpn_df["epoch"], bifpn_df["metrics/precision(B)"],
            color=COLORS["bifpn"], label=MODEL_NAMES["bifpn"], linewidth=2)
    ax.set_title("Precision 对比", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Precision")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)

    # 4. Recall 对比
    ax = axes[1, 1]
    ax.plot(baseline_df["epoch"], baseline_df["metrics/recall(B)"],
            color=COLORS["baseline"], label=MODEL_NAMES["baseline"], linewidth=2)
    ax.plot(bifpn_df["epoch"], bifpn_df["metrics/recall(B)"],
            color=COLORS["bifpn"], label=MODEL_NAMES["bifpn"], linewidth=2)
    ax.set_title("Recall 对比", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)

    # 保存图片
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\n✓ 对比曲线已保存: {output_path}")

    plt.close()


def main():
    """主函数"""
    print("=" * 80)
    print("YOLOv11 Baseline vs BiFPN 训练结果对比")
    print("=" * 80)

    # 检查文件是否存在
    baseline_path = Path(BASELINE_RESULTS)
    bifpn_path = Path(BIFPN_RESULTS)

    if not baseline_path.exists():
        print(f"\n✗ 错误: Baseline 结果文件不存在")
        print(f"  请先运行: python script/train_baseline.py")
        print(f"  期望路径: {BASELINE_RESULTS}")
        return

    if not bifpn_path.exists():
        print(f"\n✗ 错误: BiFPN 结果文件不存在")
        print(f"  请先运行: python script/train_bifpn.py")
        print(f"  期望路径: {BIFPN_RESULTS}")
        return

    # 加载结果
    print("\n加载训练结果...")
    baseline_df = load_results(BASELINE_RESULTS)
    bifpn_df = load_results(BIFPN_RESULTS)

    # 提取指标
    baseline_metrics = extract_metrics(baseline_df)
    bifpn_metrics = extract_metrics(bifpn_df)

    # 打印对比表格
    print_comparison_table(baseline_metrics, bifpn_metrics)

    # 绘制对比曲线
    output_path = str(Path(OUTPUT_DIR) / OUTPUT_IMAGE)
    plot_comparison_curves(baseline_df, bifpn_df, output_path)

    print("\n" + "=" * 80)
    print("对比分析完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()

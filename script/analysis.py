"""
对比分析模块

提供训练结果的加载、指标提取、对比展示等功能。
所有函数为无状态纯函数，不依赖特定的配置类。
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ==================== 指标提取 ====================

def load_results(csv_path: Path) -> pd.DataFrame:
    """加载训练结果 CSV 文件

    Args:
        csv_path: results.csv 路径

    Returns:
        训练结果 DataFrame
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"结果文件不存在: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ 加载: {csv_path}")
    print(f"  训练轮次: {int(df['epoch'].iloc[-1])} epochs")
    return df


def extract_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """提取关键指标

    Args:
        df: 训练结果 DataFrame

    Returns:
        指标字典（包含 final/best mAP、precision、recall）
    """
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


# ==================== 对比展示 ====================

METRICS_TO_COMPARE = [
    ("Best mAP@50-95", "best_map50_95"),
    ("Best mAP@50", "best_map50"),
    ("Final mAP@50-95", "final_map50_95"),
    ("Final mAP@50", "final_map50"),
    ("Final Precision", "final_precision"),
    ("Final Recall", "final_recall"),
]


def print_comparison_table(
    metrics: Dict[str, Dict[str, float]],
    names: Dict[str, str],
    title: str = "",
):
    """打印终端对比表格

    Args:
        metrics: {key: {metric_key: value}}
        names: {key: 显示名称}
        title: 表格标题
    """
    keys = list(metrics.keys())
    width = 20 * (len(keys) + 2)

    print(f"\n{'=' * width}")
    if title:
        print(title)
    print(f"{'=' * width}")

    # 表头
    header = f"{'指标':<20}"
    for k in keys:
        header += f"{names.get(k, k.upper()):<20}"
    header += f"{'最佳':<15}"
    print(header)
    print("-" * width)

    for metric_name, metric_key in METRICS_TO_COMPARE:
        row = f"{metric_name:<20}"
        values = {}
        for k in keys:
            value = metrics[k].get(metric_key, 0)
            values[k] = value
            row += f"{value:<20.4f}"
        best = max(values.keys(), key=lambda k: values[k])
        row += f"{names.get(best, best.upper()):<15}"
        print(row)

    print("=" * width)


def plot_comparison_curves(
    dataframes: Dict[str, pd.DataFrame],
    names: Dict[str, str],
    colors: Dict[str, str],
    save_path: Path,
    title: str = "",
):
    """绘制 4 合 1 对比曲线图

    Args:
        dataframes: {key: DataFrame}
        names: {key: 显示名称}
        colors: {key: 颜色}
        save_path: 图片保存路径
        title: 图表标题
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    for idx, (metric_key, metric_name) in enumerate(metrics_config):
        ax = axes[idx // 2, idx % 2]
        for k, df in dataframes.items():
            ax.plot(
                df["epoch"],
                df[metric_key],
                color=colors.get(k, "#000000"),
                label=names.get(k, k),
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


def save_comparison_summary(
    output_path: Path,
    metrics: Dict[str, Dict[str, float]],
    names: Dict[str, str],
    config_info: dict,
):
    """保存对比摘要到文本文件

    Args:
        output_path: 输出文件路径
        metrics: {key: {metric_key: value}}
        names: {key: 显示名称}
        config_info: 配置信息（自由格式，写入文件头部）
    """
    keys = list(metrics.keys())

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("训练结果对比\n")
        f.write("=" * 100 + "\n\n")

        # 配置信息
        for k, v in config_info.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

        # 模型信息
        for k in keys:
            f.write(f"  - {names.get(k, k.upper())}\n")
        f.write("\n")

        # 对比表格
        f.write("-" * 100 + "\n")
        header = f"{'指标':<25}"
        for k in keys:
            header += f"{names.get(k, k.upper()):<20}"
        f.write(header + "\n")
        f.write("-" * 100 + "\n")

        for metric_name, metric_key in METRICS_TO_COMPARE:
            row = f"{metric_name:<25}"
            for k in keys:
                value = metrics[k].get(metric_key, 0)
                row += f"{value:<20.4f}"
            f.write(row + "\n")

        f.write("-" * 100 + "\n")
        f.write("=" * 100 + "\n")

    print(f"✓ 对比摘要已保存: {output_path}")


# ==================== 结果整理 ====================

def reorganize_results(
    result_paths: Dict[str, str],
    output_dir: Path,
) -> Dict[str, Path]:
    """移动训练结果到统一目录

    Args:
        result_paths: {key: 原始结果目录路径}
        output_dir: 统一输出目录

    Returns:
        {key: csv_path} 映射
    """
    import shutil

    print("\n" + "=" * 60)
    print("整理训练结果...")
    print("=" * 60)

    results_paths = {}

    for key, src_pattern in result_paths.items():
        src_path = Path("runs/detect") / src_pattern
        dst_path = output_dir / src_pattern

        if src_path.exists():
            if dst_path.exists():
                shutil.rmtree(dst_path)
            shutil.move(str(src_path), str(dst_path))
            print(f"✓ 移动: {src_path.name} -> {dst_path}")

            csv = dst_path / "results.csv"
            if csv.exists():
                results_paths[key] = csv
        else:
            print(f"⚠ 跳过: {src_path} (不存在)")

    return results_paths

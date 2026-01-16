"""
训练结果对比脚本

使用方式: 直接修改下方 __main__ 中的模型配置和参数，然后运行脚本
"""

import os
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(result_path: str) -> pd.DataFrame:
    """加载训练结果数据"""
    path = Path(result_path)
    if not path.exists():
        raise FileNotFoundError(f"结果文件不存在: {result_path}")
    return pd.read_csv(path)


# 预定义的指标组合
METRIC_GROUPS = {
    "all": None,  # 默认四宫格
    "loss": ["train/box_loss", "train/cls_loss", "train/dfl_loss",
             "val/box_loss", "val/cls_loss", "val/dfl_loss"],
    "train_loss": ["train/box_loss", "train/cls_loss", "train/dfl_loss"],
    "val_loss": ["val/box_loss", "val/cls_loss", "val/dfl_loss"],
    "cls_loss": ["train/cls_loss", "val/cls_loss"],
    "box_loss": ["train/box_loss", "val/box_loss"],
    "dfl_loss": ["train/dfl_loss", "val/dfl_loss"],
    "map": ["metrics/mAP50-95(B)", "metrics/mAP50(B)"],
    "pr": ["metrics/precision(B)", "metrics/recall(B)"],
}


def plot_comparison(
    models: List[Dict[str, str]],
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
    dpi: int = 200,
) -> None:
    """
    绘制训练结果对比图表

    Args:
        models: 模型配置列表，每个元素包含 name、result 和可选的 color 字段
        metrics: 要绘制的指标列表，None 表示绘制默认指标。
                 也支持使用预定义组名: 'all', 'loss', 'train_loss', 'val_loss',
                 'cls_loss', 'box_loss', 'dfl_loss', 'map', 'pr'
        save_path: 保存路径，None 表示显示图表
        figsize: 图片尺寸，None 表示自动根据指标数量调整
        dpi: 图片分辨率
    """
    # 默认 Ultralytics 风格配色
    default_colors = [
        "#042AFF", "#0BDBEB", "#00DFB7", "#FF6FDD", "#FF444F",
        "#CCED00", "#00F344", "#BD00FF", "#00B4FF", "#DD00BA",
    ]

    metric_names = {
        "train/box_loss": "Train Box Loss",
        "train/cls_loss": "Train Cls Loss",
        "train/dfl_loss": "Train DFL Loss",
        "val/box_loss": "Val Box Loss",
        "val/cls_loss": "Val Cls Loss",
        "val/dfl_loss": "Val DFL Loss",
        "metrics/precision(B)": "Precision",
        "metrics/recall(B)": "Recall",
        "metrics/mAP50(B)": "mAP@50",
        "metrics/mAP50-95(B)": "mAP@50-95",
        "lr/pg0": "LR PG0",
        "lr/pg1": "LR PG1",
        "lr/pg2": "LR PG2",
    }

    # 设置 matplotlib 参数
    plt.rcParams.update({
        "font.size": 11,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 2,
    })

    # 处理预定义指标组
    if isinstance(metrics, str) and metrics in METRIC_GROUPS:
        metrics = METRIC_GROUPS[metrics]

    # 加载数据并提取颜色
    datas = []
    names = []
    colors = []
    for i, model in enumerate(models):
        datas.append(load_data(model["result"]))
        names.append(model["name"])
        # 使用模型指定的颜色，否则使用默认颜色
        colors.append(model.get("color", default_colors[i % len(default_colors)]))

    # 自动调整图片尺寸
    if figsize is None:
        if metrics is None:
            figsize = (16, 10)
        else:
            n = len(metrics)
            n_cols = min(3, n)
            n_rows = (n + n_cols - 1) // n_cols
            figsize = (5 * n_cols, 4 * n_rows)

    # 绘制图表
    if metrics is None:
        fig, axes = plt.subplots(2, 2, figsize=figsize, tight_layout=True)

        _plot_panel(axes[0, 0], datas, names, colors,
                   ["train/box_loss", "train/cls_loss", "train/dfl_loss"], "Train Loss", metric_names)
        _plot_panel(axes[0, 1], datas, names, colors,
                   ["metrics/mAP50-95(B)", "metrics/mAP50(B)"], "Metrics", metric_names)
        _plot_panel(axes[1, 0], datas, names, colors,
                   ["val/box_loss", "val/cls_loss", "val/dfl_loss"], "Val Loss", metric_names)
        _plot_panel(axes[1, 1], datas, names, colors,
                   ["metrics/precision(B)", "metrics/recall(B)"], "Precision & Recall", metric_names)
    else:
        n = len(metrics)
        n_cols = min(3, n)
        n_rows = (n + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), tight_layout=True)

        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            _plot_panel(axes[i], datas, names, colors, [metric],
                       metric_names.get(metric, metric), metric_names)

        for i in range(n, len(axes)):
            axes[i].set_visible(False)

    if save_path:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()

    plt.close()


def generate_filename(models: List[Dict[str, str]], metrics: Optional[List[str]] = None) -> str:
    """
    生成基于模型名称和指标的文件名

    Args:
        models: 模型配置列表
        metrics: 指标列表

    Returns:
        生成的文件名（不含路径和扩展名）
    """
    # 处理预定义指标组
    if isinstance(metrics, str) and metrics in METRIC_GROUPS:
        metrics = METRIC_GROUPS[metrics]

    # 模型名称部分：用 _vs_ 连接多个模型
    model_names = "_vs_".join([m["name"].replace(" ", "_") for m in models])

    # 指标部分
    if metrics is None:
        metric_suffix = "all"
    else:
        # 从指标名中提取关键部分（去重）
        metric_keys = []
        for m in metrics:
            if "/" in m:
                parts = m.split("/")
                key = parts[-1].replace("(B)", "").replace("loss", "Loss")
                if key not in metric_keys:  # 去重
                    metric_keys.append(key)
            elif "precision" in m.lower():
                if "Precision" not in metric_keys:
                    metric_keys.append("Precision")
            elif "recall" in m.lower():
                if "Recall" not in metric_keys:
                    metric_keys.append("Recall")
            elif "mAP" in m:
                if "mAP" not in metric_keys:
                    metric_keys.append("mAP")
            else:
                key = m.replace(" ", "_")
                if key not in metric_keys:
                    metric_keys.append(key)

        if len(metric_keys) == 1:
            metric_suffix = metric_keys[0]
        else:
            metric_suffix = "_".join(metric_keys[:3])  # 最多取前3个

    return f"{model_names}_{metric_suffix}"


def _plot_panel(ax, datas: List[pd.DataFrame], names: List[str], colors: List[str],
                metrics: List[str], title: str, metric_names: Dict[str, str]) -> None:
    """在单个子图上绘制指标"""
    for i, metric in enumerate(metrics):
        for j, (data, name) in enumerate(zip(datas, names)):
            if metric not in data.columns:
                continue

            x = data["epoch"].values
            y = data[metric].values

            # 多指标时仅第一个显示图例，单指标时都显示
            if len(metrics) > 1:
                label = name if i == 0 else "_nolegend_"
            else:
                label = name

            color = colors[j % len(colors)]

            # 绘制实际数据点
            ax.plot(x, y, marker=".", markersize=4, linestyle="-",
                   color=color, linewidth=1.5, label=label, alpha=0.8)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", framealpha=0.8)

    if "loss" in title.lower():
        ax.set_ylim(bottom=0)


def print_summary(models: List[Dict[str, str]]) -> None:
    """打印训练结果摘要"""
    print("\n" + "=" * 60)
    print("训练结果摘要")
    print("=" * 60)

    for model in models:
        print(f"\n[{model['name']}]")
        print(f"  路径: {model['result']}")

        data = load_data(model["result"])
        if data.empty:
            print("  无数据")
            continue

        last_row = data.iloc[-1]
        print(f"  训练轮次: {int(last_row['epoch'])}")
        print(f"  最终 mAP@50-95: {last_row.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        print(f"  最终 mAP@50: {last_row.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"  最终精确率: {last_row.get('metrics/precision(B)', 'N/A'):.4f}")
        print(f"  最终召回率: {last_row.get('metrics/recall(B)', 'N/A'):.4f}")

        if "metrics/mAP50-95(B)" in data.columns:
            best_idx = data["metrics/mAP50-95(B)"].idxmax()
            best_row = data.loc[best_idx]
            print(f"  最佳 mAP@50-95: {best_row['metrics/mAP50-95(B)']:.4f} (Epoch {int(best_row['epoch'])})")


if __name__ == "__main__":
    # ============================================
    # 模型配置 - 修改这里进行对比
    # ============================================
    colors = [
        "#042AFF", "#0BDBEB", "#00DFB7", "#FF6FDD", "#FF444F",
        "#CCED00", "#00F344", "#BD00FF", "#00B4FF", "#DD00BA",
    ]
    # 每个模型可以指定 name、result 和可选的 color
    # color 为空时自动使用默认颜色
    model1 = {"name": "YOLO11n", "result": "runs/detect/train_yolo11/results.csv", "color": colors[1]}
    model2 = {"name": "FCE-YOLO v1", "result": "runs/detect/train_fce-yolo_1/results.csv", "color": colors[2]}
    model3 = {"name": "FCE-YOLO v2", "result": "runs/detect/train_fce-yolo_2/results.csv", "color": colors[0]}

    MODELS = [model1, model2, model3]

    # ============================================
    # 指标配置 - 选择要绘制的指标
    # ============================================

    # 方式1: 使用预定义指标组（推荐）
    # 可选值: 'all' (默认四宫格), 'loss', 'train_loss', 'val_loss',
    #         'cls_loss', 'box_loss', 'dfl_loss', 'map', 'pr'
    METRICS = "dfl_loss"  # 只绘制分类损失对比

    # 方式2: 自定义指标列表
    # METRICS = ["train/cls_loss", "val/cls_loss"]  # 与 "cls_loss" 等价
    # METRICS = ["metrics/mAP50-95(B)"]  # 只绘制 mAP@50-95

    # 方式3: None 表示默认四宫格
    # METRICS = None

    # ============================================
    # 导出配置
    # ============================================

    # 保存路径（None 表示显示图表）
    # 设置为 "auto" 则自动根据模型名和指标生成文件名
    SAVE_PATH = "auto"  # 会生成如: YOLO11n_vs_FCE-YOLO_v2_cls_Loss.png

    # 或者手动指定完整路径
    # SAVE_PATH = "runs/compare/my_comparison.png"

    # 保存目录（仅在 SAVE_PATH 为 "auto" 时生效）
    SAVE_DIR = "runs/compare"

    # 图片尺寸 (None 表示自动根据指标数量调整)
    FIGSIZE = None

    # 图片分辨率
    DPI = 200

    # ============================================

    # 打印摘要
    print_summary(MODELS)

    # 处理自动文件名
    save_path = SAVE_PATH
    if SAVE_PATH == "auto":
        filename = generate_filename(MODELS, METRICS)
        save_path = str(Path(SAVE_DIR) / f"{filename}.png")
        Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
        print(f"自动生成保存路径: {save_path}")

    # 绘制对比图表
    plot_comparison(MODELS, METRICS, save_path, FIGSIZE, DPI)

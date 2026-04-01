#!/usr/bin/env python3
"""
IoU 损失策略对比脚本

对比 Baseline、BiFPN(CIoU)、BiFPN(WIoU) 三种模式的训练效果。

Usage:
    python script/compare_iou.py --scale s
    python script/compare_iou.py --scale s --skip-train
    python script/compare_iou.py --scale s --output my_experiment
"""

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from script.config import (
    TrainConfig,
    get_model_config,
    get_stage1_config,
    get_stage2_config,
)
from script.trainer import YOLOv11Trainer, train_model


# ==================== 实验定义 ====================

@dataclass
class Experiment:
    """单个实验配置"""
    name: str                    # 实验标识（如 "baseline", "bifpn_ciou", "bifpn_wiou"）
    display_name: str            # 图表中显示的名称
    model_type: str              # 模型类型（对应 MODEL_CONFIGS 的 key）
    color: str                   # 图表颜色
    iou_type: str                # IoU 损失类型
    use_two_stage: bool          # 是否两阶段训练
    result_pattern: str          # 结果目录名称模式

    def get_result_path(self, scale: str) -> str:
        """获取结果目录路径"""
        return self.result_pattern.format(scale=scale)


# 三个对比实验
EXPERIMENTS: Dict[str, Experiment] = {
    "baseline": Experiment(
        name="baseline",
        display_name=lambda s: f"YOLOv11{s.upper()} Baseline (CIoU)",
        model_type="baseline",
        color="#0BDBEB",
        iou_type="CIoU",
        use_two_stage=False,
        result_pattern="baseline_yolo11{scale}",
    ),
    "bifpn_ciou": Experiment(
        name="bifpn_ciou",
        display_name=lambda s: f"YOLOv11{s.upper()}-BiFPN (CIoU)",
        model_type="bifpn",
        color="#042AFF",
        iou_type="CIoU",
        use_two_stage=True,
        result_pattern="bifpn_{scale}_stage2",
    ),
    "bifpn_wiou": Experiment(
        name="bifpn_wiou",
        display_name=lambda s: f"YOLOv11{s.upper()}-BiFPN (WIoU)",
        model_type="bifpn",
        color="#FF6B00",
        iou_type="WIoU",
        use_two_stage=True,
        result_pattern="bifpn_wiou_{scale}_stage2",
    ),
}


def get_experiment_display_name(exp: Experiment, scale: str) -> str:
    """获取实验显示名称"""
    name = exp.display_name
    if callable(name):
        return name(scale)
    return name


# ==================== CLI ====================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="IoU 损失策略对比脚本（Baseline vs BiFPN-CIoU vs BiFPN-WIoU）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行完整对比
  python script/compare_iou.py --scale s

  # 仅对比已有结果（不训练）
  python script/compare_iou.py --scale s --skip-train

  # 自定义输出目录
  python script/compare_iou.py --scale s --output my_experiment
        """
    )

    parser.add_argument(
        "--scale",
        type=str,
        default="s",
        choices=["n", "s", "m", "l", "x"],
        help="模型尺度 (默认: s)"
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
        help="阶段二总训练轮次（默认: 300）"
    )

    return parser.parse_args()


# ==================== 训练 ====================

def run_training(exp: Experiment, scale: str, epochs: int) -> bool:
    """运行单个实验的训练

    Args:
        exp: 实验配置
        scale: 模型尺度
        epochs: 阶段二训练轮次

    Returns:
        训练是否成功
    """
    display_name = get_experiment_display_name(exp, scale)
    print("\n" + "=" * 60)
    print(f"训练: {display_name}")
    print(f"  模型: {exp.model_type}")
    print(f"  IoU 损失: {exp.iou_type}")
    print(f"  两阶段: {'是' if exp.use_two_stage else '否'}")
    print("=" * 60)

    try:
        model_cfg = get_model_config(exp.model_type)
        exp_name_base = exp.result_pattern.replace("_stage2", "").format(scale=scale)

        if exp.use_two_stage:
            # 直接使用 YOLOv11Trainer 避免两阶段配置覆盖问题
            s1_config = get_stage1_config()
            s2_config = get_stage2_config()
            s2_config.epochs = epochs

            # 应用 iou_type 到两个阶段
            s1_config.iou_type = exp.iou_type
            s2_config.iou_type = exp.iou_type

            trainer = YOLOv11Trainer(model_config=model_cfg, scale=scale)
            results = trainer.train_two_stage(
                exp_name_base=exp_name_base,
                stage1_config=s1_config,
                stage2_config=s2_config,
            )
        else:
            # 单阶段训练
            config = TrainConfig(epochs=epochs, iou_type=exp.iou_type)
            train_model(
                model_type=exp.model_type,
                scale=scale,
                exp_name_prefix=exp_name_base,
                config=config,
            )

        print(f"✓ {exp.name} 训练完成")
        return True

    except Exception as e:
        print(f"✗ {exp.name} 训练失败: {e}")
        return False


# ==================== 结果整理 ====================

def reorganize_results(
    experiments: List[Experiment],
    scale: str,
    output_dir: Path,
) -> Dict[str, Path]:
    """重新组织训练结果到统一目录"""
    print("\n" + "=" * 60)
    print("整理训练结果...")
    print("=" * 60)

    results_paths = {}

    for exp in experiments:
        # 原始结果路径
        src_pattern = exp.get_result_path(scale)
        src_path = Path("runs/detect") / src_pattern

        # 目标路径
        dst_path = output_dir / src_pattern

        if src_path.exists():
            if dst_path.exists():
                shutil.rmtree(dst_path)

            shutil.move(str(src_path), str(dst_path))
            print(f"✓ 移动: {src_path.name} -> {dst_path}")

            results_csv = dst_path / "results.csv"
            if results_csv.exists():
                results_paths[exp.name] = results_csv
        else:
            print(f"⚠ 跳过: {src_path} (不存在)")

    return results_paths


# ==================== 指标提取 ====================

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


# ==================== 对比分析 ====================

def print_comparison_table(
    all_metrics: Dict[str, Dict],
    experiments: List[Experiment],
    scale: str,
):
    """打印对比表格"""
    exp_names = " vs ".join([e.name.upper() for e in experiments])
    print("\n" + "=" * 120)
    print(f"{exp_names} 训练结果对比表 (尺度: {scale.upper()})")
    print("=" * 120)

    # 表头
    header = f"{'指标':<22}"
    for exp in experiments:
        header += f"{exp.name.upper():<25}"
    header += f"{'最佳':<15}"
    print(header)
    print("-" * 120)

    metrics_to_compare = [
        ("Best mAP@50-95", "best_map50_95"),
        ("Best mAP@50-95 Epoch", "best_map50_95_epoch"),
        ("Best mAP@50", "best_map50"),
        ("Best mAP@50 Epoch", "best_map50_epoch"),
        ("Final mAP@50-95", "final_map50_95"),
        ("Final mAP@50", "final_map50"),
        ("Final Precision", "final_precision"),
        ("Final Recall", "final_recall"),
    ]

    for metric_name, metric_key in metrics_to_compare:
        row = f"{metric_name:<22}"

        values = {}
        for exp in experiments:
            value = all_metrics[exp.name][metric_key]
            values[exp.name] = value

            # Epoch 类型直接显示整数
            if "epoch" in metric_key.lower():
                row += f"{int(value):<25}"
            else:
                row += f"{value:<25.4f}"

        # 找最佳（Epoch 类型的指标不比较）
        if "epoch" not in metric_key.lower():
            best_exp = max(values.keys(), key=lambda k: values[k])
            row += f"{best_exp.upper():<15}"

        print(row)

    print("=" * 120)


def plot_comparison_curves(
    all_dataframes: Dict[str, pd.DataFrame],
    experiments: List[Experiment],
    scale: str,
    save_path: Path,
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

    for idx, (metric_key, metric_name) in enumerate(metrics_config):
        ax = axes[idx // 2, idx % 2]

        for exp in experiments:
            df = all_dataframes[exp.name]
            display_name = get_experiment_display_name(exp, scale)
            ax.plot(
                df["epoch"],
                df[metric_key],
                color=exp.color,
                label=display_name,
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
    experiments: List[Experiment],
    scale: str,
    epochs: int,
    output_dir: Path,
    all_metrics: Dict[str, Dict],
):
    """保存对比摘要到文本文件"""
    summary_path = output_dir / "comparison_summary.txt"

    exp_names = " vs ".join([e.name.upper() for e in experiments])

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 120 + "\n")
        f.write(f"{exp_names} 训练结果对比\n")
        f.write("=" * 120 + "\n\n")

        f.write(f"模型尺度: {scale.upper()}\n")
        f.write(f"阶段二训练轮次: {epochs} epochs\n\n")

        f.write("实验配置:\n")
        for exp in experiments:
            model_cfg = get_model_config(exp.model_type)
            display_name = get_experiment_display_name(exp, scale)
            f.write(f"  {exp.name.upper()}:\n")
            f.write(f"    显示名称: {display_name}\n")
            f.write(f"    模型 YAML: {model_cfg.yaml_path}\n")
            f.write(f"    IoU 损失: {exp.iou_type}\n")
            f.write(f"    两阶段训练: {'是' if exp.use_two_stage else '否'}\n")
        f.write("\n")

        f.write("-" * 120 + "\n")
        header = f"{'指标':<25}"
        for exp in experiments:
            header += f"{exp.name.upper():<25}"
        f.write(header + "\n")
        f.write("-" * 120 + "\n")

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
            for exp in experiments:
                value = all_metrics[exp.name][metric_key]
                row += f"{value:<25.4f}"
            f.write(row + "\n")

        f.write("-" * 120 + "\n")
        f.write("\n目录结构:\n")
        for exp in experiments:
            result_pattern = exp.get_result_path(scale)
            f.write(f"  - {exp.name.upper()} 结果: {result_pattern}/\n")
        f.write(f"  - 对比曲线图: comparison_curves.png\n")
        f.write(f"  - 本摘要文件: comparison_summary.txt\n")
        f.write("=" * 120 + "\n")

    print(f"✓ 对比摘要已保存: {summary_path}")


# ==================== 主函数 ====================

def main():
    """主函数"""
    args = parse_args()

    scale = args.scale
    exp_list = list(EXPERIMENTS.values())

    # 输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(f"runs/detect/iou_compare_{scale}_{args.epochs}")
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_names = " vs ".join([e.name.upper() for e in exp_list])

    print("\n" + "=" * 100)
    print(f"IoU 损失策略对比: {exp_names} (尺度: {scale.upper()})")
    print("=" * 100)
    print(f"\n输出目录: {output_dir}")
    print("\n实验配置:")
    for exp in exp_list:
        display_name = get_experiment_display_name(exp, scale)
        print(f"  {exp.name:<15} | {display_name:<40} | IoU: {exp.iou_type}")

    # 阶段一：训练所有模型
    if not args.skip_train:
        print("\n" + "▶" * 50)
        print("阶段一：训练所有模型")
        print("▶" * 50)

        all_success = True
        for exp in exp_list:
            success = run_training(exp, scale, args.epochs)
            if not success:
                print(f"\n✗ {exp.name} 训练失败，终止流程")
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

    results_paths = reorganize_results(exp_list, scale, output_dir)

    if len(results_paths) < len(exp_list):
        missing = [e.name for e in exp_list if e.name not in results_paths]
        print(f"\n✗ 缺少训练结果文件: {missing}")
        return

    # 阶段三：生成对比分析
    print("\n" + "▶" * 50)
    print("阶段三：生成对比分析")
    print("▶" * 50)

    all_dataframes = {}
    all_metrics = {}

    for exp in exp_list:
        if exp.name in results_paths:
            df = load_results(results_paths[exp.name])
            all_dataframes[exp.name] = df
            all_metrics[exp.name] = extract_metrics(df)

    print_comparison_table(all_metrics, exp_list, scale)

    comparison_image = output_dir / "comparison_curves.png"
    plot_comparison_curves(all_dataframes, exp_list, scale, comparison_image)

    save_comparison_summary(exp_list, scale, args.epochs, output_dir, all_metrics)

    # 完成
    print("\n" + "=" * 100)
    print("IoU 损失策略对比完成!")
    print("=" * 100)
    print(f"\n所有结果已保存到: {output_dir}")
    print("\n目录结构:")
    for exp in exp_list:
        result_pattern = exp.get_result_path(scale)
        print(f"  ├── {result_pattern}/")
    print(f"  ├── comparison_curves.png")
    print(f"  └── comparison_summary.txt")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()

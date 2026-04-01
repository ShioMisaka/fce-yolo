#!/usr/bin/env python3
"""
COCO 数据集多模型训练对比脚本

针对 RTX 5090 优化的 COCO 数据集多模型训练对比工具。
支持一次性训练 Baseline、BiFPN、FCE 三个模型，并自动生成对比结果。

Usage:
    # 训练所有模型（n 尺度）
    python script/train_coco_compare.py --scale n

    # 训练所有模型（s 尺度，自定义批次）
    python script/train_coco_compare.py --scale s --batch 96

    # 快速测试模式（1 epoch）
    python script/train_coco_compare.py --scale n --test

    # 跳过训练，仅生成对比
    python script/train_coco_compare.py --scale s --skip-train
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import pandas as pd

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from ultralytics.utils import LOGGER


# ==================== 配置类 ====================

class ModelConfig:
    """模型配置类"""

    def __init__(
        self,
        name: str,
        yaml_path: str,
        color: str,
        display_name: str,
        use_two_stage: bool = False,
    ):
        self.name = name
        self.yaml_path = yaml_path
        self.color = color
        self.display_name = display_name
        self.use_two_stage = use_two_stage

    def get_result_path(self, scale: str, stage: Optional[int] = None) -> str:
        """获取训练结果路径"""
        if self.use_two_stage and stage is not None:
            return f"{self.name}_{scale}_stage{stage}"
        return f"{self.name}_{scale}"


class ExperimentConfig:
    """实验配置类"""

    def __init__(
        self,
        models: List[str],
        scale: str,
        data: str,
        epochs: int,
        batch: int,
        imgsz: int,
        workers: int,
        cache: str,
        amp: bool,
        skip_train: bool = False,
        output_dir: Optional[Path] = None,
    ):
        self.models = models
        self.scale = scale
        self.data = data
        self.epochs = epochs
        self.batch = batch
        self.imgsz = imgsz
        self.workers = workers
        self.cache = cache
        self.amp = amp
        self.skip_train = skip_train
        self.output_dir = output_dir or self._generate_output_dir()

    def _generate_output_dir(self) -> Path:
        """生成输出目录名称"""
        model_names = "vs".join(self.models)
        return Path(f"runs/detect/{model_names}_{self.scale}_{self.epochs}")

    def get_model_colors(self) -> Dict[str, str]:
        """获取模型颜色映射"""
        return MODEL_CONFIGS[self.models[0]].color if len(self.models) == 1 else {
            "baseline": "#1f77b4",  # 蓝色
            "bifpn": "#ff7f0e",     # 橙色
            "fce": "#2ca02c",       # 绿色
        }

    def get_model_display_names(self) -> Dict[str, str]:
        """获取模型显示名称"""
        return {m: MODEL_CONFIGS[m].display_name for m in self.models}


# ==================== 模型配置 ====================

MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "baseline": ModelConfig(
        name="baseline",
        yaml_path="ultralytics/cfg/models/11/yolo11.yaml",
        color="#1f77b4",
        display_name="YOLOv11",
        use_two_stage=False,
    ),
    "bifpn": ModelConfig(
        name="bifpn",
        yaml_path="ultralytics/cfg/models/11/yolo11-bifpn.yaml",
        color="#ff7f0e",
        display_name="YOLOv11+BiFPN",
        use_two_stage=True,
    ),
    "fce": ModelConfig(
        name="fce",
        yaml_path="ultralytics/cfg/models/11/yolo11-fce.yaml",
        color="#2ca02c",
        display_name="YOLOv11+FCE",
        use_two_stage=True,
    ),
}


# ==================== 训练函数 ====================

def train_single_model(
    model_cfg: ModelConfig,
    scale: str,
    config: ExperimentConfig,
    stage: Optional[int] = None,
) -> Path:
    """训练单个模型

    Args:
        model_cfg: 模型配置
        scale: 模型尺度
        config: 实验配置
        stage: 训练阶段（用于两阶段训练）

    Returns:
        训练结果目录路径
    """
    # 生成实验名称
    if stage is not None:
        exp_name = f"{model_cfg.name}_{scale}_stage{stage}"
    else:
        exp_name = f"{model_cfg.name}_{scale}"

    # 加载预训练权重（如果是阶段一）
    if stage == 1:
        # 使用预训练的 YOLOv11 权重作为初始权重
        pretrained = f"yolo11{scale}.pt"
        model = YOLO(pretrained)
        LOGGER.info(f"📦 加载预训练权重: {pretrained}")
    else:
        model = YOLO(model_cfg.yaml_path)

    # 构建训练参数
    train_args = {
        "data": config.data,
        "epochs": config.epochs,
        "batch": config.batch,
        "imgsz": config.imgsz,
        "device": 0,
        "workers": config.workers,
        "project": "runs/detect",
        "name": exp_name,
        "cache": config.cache if config.cache != "none" else False,
        "amp": config.amp,
        "cos_lr": True if stage != 1 else False,
        "lr0": 0.001 if stage != 1 else 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "close_mosaic": 15 if stage != 1 else 10,
        "patience": 50 if stage != 1 else 20,
        "val": True,
        "plots": True,
        "save_json": True,
        "exist_ok": True,
        "verbose": True,
    }

    # 如果是两阶段训练的冻结阶段
    if stage == 1:
        train_args["freeze"] = 10  # 冻结前 10 层

    # 开始训练
    LOGGER.info(f"🏋️ 开始训练: {exp_name}")
    LOGGER.info(f"   配置: epochs={config.epochs}, batch={config.batch}, imgsz={config.imgsz}")

    try:
        results = model.train(**train_args)
        result_path = Path(f"runs/detect/{exp_name}")
        LOGGER.info(f"✅ 训练完成: {exp_name}")
        return result_path

    except Exception as e:
        LOGGER.error(f"❌ 训练失败: {exp_name} - {e}")
        raise


def train_two_stage_model(
    model_cfg: ModelConfig,
    scale: str,
    config: ExperimentConfig,
) -> Dict[str, Path]:
    """两阶段训练模型

    Args:
        model_cfg: 模型配置
        scale: 模型尺度
        config: 实验配置

    Returns:
        两阶段训练结果路径字典
    """
    results = {}

    # 阶段一：冻结预热
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info(f"📍 阶段一：冻结预热 ({model_cfg.name.upper()})")
    LOGGER.info("=" * 80)

    stage1_config = config
    stage1_config.epochs = 50  # 阶段一固定 50 epochs

    stage1_path = train_single_model(model_cfg, scale, stage1_config, stage=1)
    results["stage1"] = stage1_path

    # 阶段二：全局微调
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info(f"📍 阶段二：全局微调 ({model_cfg.name.upper()})")
    LOGGER.info("=" * 80)

    # 使用阶段一的权重
    stage1_weights = stage1_path / "weights" / "best.pt"
    if stage1_weights.exists():
        LOGGER.info(f"📦 加载阶段一权重: {stage1_weights}")
        model = YOLO(str(stage1_weights))
    else:
        LOGGER.warning(f"⚠ 阶段一权重不存在，使用 YAML 配置")
        model = YOLO(model_cfg.yaml_path)

    stage2_config = config
    stage2_config.epochs = config.epochs  # 恢复原始 epoch 数

    # 更新训练参数
    train_args = {
        "data": config.data,
        "epochs": config.epochs,
        "batch": config.batch,
        "imgsz": config.imgsz,
        "device": 0,
        "workers": config.workers,
        "project": "runs/detect",
        "name": f"{model_cfg.name}_{scale}_stage2",
        "cache": config.cache if config.cache != "none" else False,
        "amp": config.amp,
        "cos_lr": True,
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "close_mosaic": 15,
        "patience": 50,
        "val": True,
        "plots": True,
        "save_json": True,
        "exist_ok": True,
        "verbose": True,
    }

    LOGGER.info(f"🏋️ 开始训练: {model_cfg.name}_{scale}_stage2")
    LOGGER.info(f"   配置: epochs={config.epochs}, batch={config.batch}, imgsz={config.imgsz}")

    try:
        model.train(**train_args)
        stage2_path = Path(f"runs/detect/{model_cfg.name}_{scale}_stage2")
        LOGGER.info(f"✅ 训练完成: {model_cfg.name}_{scale}_stage2")
        results["stage2"] = stage2_path
    except Exception as e:
        LOGGER.error(f"❌ 训练失败: {model_cfg.name}_{scale}_stage2 - {e}")
        raise

    return results


def train_all_models(config: ExperimentConfig) -> Dict[str, Path]:
    """训练所有模型

    Args:
        config: 实验配置

    Returns:
        模型训练结果路径字典
    """
    results = {}

    for model_type in config.models:
        model_cfg = MODEL_CONFIGS[model_type]

        LOGGER.info("\n" + "▶" * 50)
        LOGGER.info(f"🚀 开始训练: {model_cfg.display_name} ({model_type.upper()})")
        LOGGER.info("▶" * 50)

        try:
            if model_cfg.use_two_stage:
                # 两阶段训练
                stage_results = train_two_stage_model(model_cfg, config.scale, config)
                # 使用阶段二的结果作为最终结果
                results[model_type] = stage_results["stage2"]
            else:
                # 单阶段训练
                result_path = train_single_model(model_cfg, config.scale, config)
                results[model_type] = result_path

            LOGGER.info(f"✅ {model_type.upper()} 训练完成\n")

        except Exception as e:
            LOGGER.error(f"❌ {model_type.upper()} 训练失败: {e}")
            raise

    return results


# ==================== 结果分析函数 ====================

def reorganize_results(config: ExperimentConfig) -> Dict[str, Path]:
    """重新组织训练结果到统一目录

    Args:
        config: 实验配置

    Returns:
        模型 results.csv 路径字典
    """
    print("\n" + "=" * 80)
    print("📁 整理训练结果...")
    print("=" * 80)

    results_paths = {}

    for model_type in config.models:
        model_cfg = MODEL_CONFIGS[model_type]

        # 确定源路径
        if model_cfg.use_two_stage:
            src_pattern = f"{model_cfg.name}_{config.scale}_stage2"
        else:
            src_pattern = f"{model_cfg.name}_{config.scale}"

        src_path = Path("runs/detect") / src_pattern
        dst_path = config.output_dir / src_pattern

        if src_path.exists():
            # 移动到统一目录
            if dst_path.exists():
                shutil.rmtree(dst_path)
            shutil.move(str(src_path), str(dst_path))
            print(f"✓ 移动: {src_path.name} -> {dst_path}")

            # 检查 results.csv
            results_csv = dst_path / "results.csv"
            if results_csv.exists():
                results_paths[model_type] = results_csv
            else:
                print(f"⚠ 警告: {results_csv} 不存在")
        else:
            print(f"⚠ 跳过: {src_path} (不存在)")

    return results_paths


def load_results(csv_path: Path) -> pd.DataFrame:
    """加载训练结果 CSV 文件

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

    return df


def extract_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """提取关键指标

    Args:
        df: 训练结果 DataFrame

    Returns:
        指标字典
    """
    metrics = {}

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


def print_comparison_table(all_metrics: Dict[str, Dict], config: ExperimentConfig):
    """打印对比表格

    Args:
        all_metrics: 所有模型的指标字典
        config: 实验配置
    """
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
    """绘制训练对比曲线

    Args:
        all_dataframes: 所有模型的训练结果 DataFrame
        save_path: 保存路径
        config: 实验配置
    """
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
    """保存对比摘要到文本文件

    Args:
        config: 实验配置
        all_metrics: 所有模型的指标字典
    """
    summary_path = config.output_dir / "comparison_summary.txt"

    model_names = " vs ".join([m.upper() for m in config.models])
    display_names = config.get_model_display_names()

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"{model_names} 训练结果对比\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"数据集: COCO 2017\n")
        f.write(f"模型尺度: {config.scale.upper()}\n")
        f.write(f"总训练轮次: {config.epochs} epochs\n")
        f.write(f"批次大小: {config.batch}\n")
        f.write(f"图像尺寸: {config.imgsz}\n")
        f.write(f"数据缓存: {config.cache}\n")
        f.write(f"混合精度: {config.amp}\n\n")

        for model_type in config.models:
            model_cfg = MODEL_CONFIGS[model_type]
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
            model_cfg = MODEL_CONFIGS[model_type]
            if model_cfg.use_two_stage:
                result_pattern = f"{model_cfg.name}_{config.scale}_stage2"
            else:
                result_pattern = f"{model_cfg.name}_{config.scale}"
            f.write(f"  - {model_type.upper()} 结果: {result_pattern}/\n")
        f.write(f"  - 对比曲线图: comparison_curves.png\n")
        f.write(f"  - 本摘要文件: comparison_summary.txt\n")
        f.write("=" * 100 + "\n")

    print(f"✓ 对比摘要已保存: {summary_path}")


# ==================== 命令行接口 ====================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="COCO 数据集多模型训练对比脚本（RTX 5090 优化）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 训练所有模型（n 尺度）
  python script/train_coco_compare.py --scale n

  # 训练所有模型（s 尺度）
  python script/train_coco_compare.py --scale s

  # 自定义批次大小
  python script/train_coco_compare.py --scale s --batch 96

  # 快速测试模式（1 epoch）
  python script/train_coco_compare.py --scale n --test

  # 跳过训练，仅生成对比
  python script/train_coco_compare.py --scale s --skip-train

  # 自定义训练轮次
  python script/train_coco_compare.py --scale m --epochs 200
        """
    )

    parser.add_argument(
        "--scale",
        type=str,
        default="s",
        choices=["n", "s", "m", "l", "x"],
        help="模型尺度"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="ultralytics/cfg/datasets/coco_custom.yaml",
        help="数据集配置文件"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="训练轮次（仅对两阶段训练的阶段二有效）"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=128,
        help="批次大小（RTX 5090 32GB 显存推荐 128）"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="输入图像尺寸"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="数据加载工作进程数（AMD 9950X3D 推荐使用 24）"
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="ram",
        choices=["ram", "disk", "none"],
        help="数据缓存策略"
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="禁用自动混合精度训练"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="跳过训练，仅生成对比分析"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="快速测试模式（1 epoch，小图像尺寸）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="自定义输出目录"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建实验配置
    config = ExperimentConfig(
        models=["baseline", "bifpn", "fce"],
        scale=args.scale,
        data=args.data,
        epochs=1 if args.test else args.epochs,
        batch=16 if args.test else args.batch,
        imgsz=320 if args.test else args.imgsz,
        workers=8 if args.test else args.workers,
        cache=args.cache,
        amp=not args.no_amp,
        skip_train=args.skip_train,
        output_dir=Path(args.output) if args.output else None,
    )

    model_names = " vs ".join([m.upper() for m in config.models])

    print("\n" + "=" * 100)
    print(f"🚀 {model_names} COCO 数据集训练对比 (尺度: {config.scale.upper()})")
    print("=" * 100)
    print(f"\n数据集: {config.data}")
    print(f"输出目录: {config.output_dir}")
    print(f"\n训练配置:")
    print(f"  训练轮次: {config.epochs}")
    print(f"  批次大小: {config.batch}")
    print(f"  图像尺寸: {config.imgsz}")
    print(f"  工作进程: {config.workers}")
    print(f"  数据缓存: {config.cache}")
    print(f"  混合精度: {config.amp}")

    # 阶段一：训练所有模型
    if not config.skip_train:
        print("\n" + "▶" * 50)
        print("阶段一：训练所有模型")
        print("▶" * 50)

        try:
            train_all_models(config)
        except Exception as e:
            LOGGER.error(f"❌ 训练失败: {e}")
            sys.exit(1)
    else:
        print("\n⚠ 跳过训练阶段，使用已有结果")

    # 阶段二：整理结果
    print("\n" + "▶" * 50)
    print("阶段二：整理训练结果")
    print("▶" * 50)

    results_paths = reorganize_results(config)

    if len(results_paths) < len(config.models):
        missing = set(config.models) - set(results_paths.keys())
        print(f"\n❌ 缺少训练结果文件: {missing}")
        sys.exit(1)

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
    print("✅ 完整训练对比流程完成!")
    print("=" * 100)
    print(f"\n所有结果已保存到: {config.output_dir}")
    print("\n目录结构:")
    for model_type in config.models:
        model_cfg = MODEL_CONFIGS[model_type]
        if model_cfg.use_two_stage:
            result_pattern = f"{model_cfg.name}_{config.scale}_stage2"
        else:
            result_pattern = f"{model_cfg.name}_{config.scale}"
        print(f"  ├── {result_pattern}/")
    print(f"  ├── comparison_curves.png")
    print(f"  └── comparison_summary.txt")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()

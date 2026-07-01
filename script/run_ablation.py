#!/usr/bin/env python3
"""
公平消融实验编排器（Fair Ablation Orchestrator）

按 ablation_config.yaml 配方一键训练 4 类模型 × 多尺度，强制全部两阶段
（含 baseline，公平对齐），产出受控对比表与论文图表。

核心设计：
- 配方驱动：所有统一变量集中在 ablation_config.yaml，重做实验只改 YAML
- 公平注入：运行时用 dataclasses.replace 给 baseline 注入 stage1+freeze，
  不改 config.py 既有定义
- 断点续跑：stage2/best.pt 存在 + results.csv 行数足够则跳过
- 复用基础设施：YOLOv11Trainer / analysis / paper_figs

用法（在 fce-yolo 环境下）：
    conda activate fce-yolo
    cd <项目根>/fce-yolo
    python script/run_ablation.py --dry-run                  # 仅预览矩阵
    python script/run_ablation.py                            # 跑全配方
    python script/run_ablation.py --scales m                 # 只跑 m
    python script/run_ablation.py --models baseline fce      # 子集
    python script/run_ablation.py --skip-train               # 仅整理+出图
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import yaml

# 项目根（fce-yolo 仓库根）入 path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from script.config import (  # noqa: E402
    MODEL_CONFIGS,
    ModelConfig,
    StageConfig,
    TrainConfig,
    get_model_config,
)
from script.analysis import load_results, extract_metrics  # noqa: E402

# 论文项目根（Visual Guidance Robotic Arm/，fce-yolo 上一级）
PAPER_ROOT = PROJECT_ROOT.parent
DEFAULT_RECIPE = PROJECT_ROOT / "script" / "ablation_config.yaml"


# ============================================================
# 配方加载
# ============================================================

def load_recipe(yaml_path: Path) -> dict:
    """加载配方 YAML 并做基本校验。

    Returns:
        配方 dict
    Raises:
        FileNotFoundError / ValueError
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"配方文件不存在: {yaml_path}")
    with open(yaml_path, encoding="utf-8") as f:
        recipe = yaml.safe_load(f)

    # 必需字段校验
    required = ["shared", "stage1", "stage2", "freeze", "scales", "models"]
    missing = [k for k in required if k not in recipe]
    if missing:
        raise ValueError(f"配方缺少必需字段: {missing}")

    # model_key 合法性
    bad = [m for m in recipe["models"] if m not in MODEL_CONFIGS]
    if bad:
        raise ValueError(f"未知模型: {bad}，可选: {list(MODEL_CONFIGS.keys())}")

    # scales 合法性
    valid_scales = ["n", "s", "m", "l", "x"]
    bad_s = [s for s in recipe["scales"] if s not in valid_scales]
    if bad_s:
        raise ValueError(f"未知尺度: {bad_s}，可选: {valid_scales}")

    return recipe


# ============================================================
# 公平性注入
# ============================================================

def build_model_cfg_with_fairness(model_key: str, recipe: dict) -> ModelConfig:
    """给任意 model（含 baseline）注入统一的两阶段配置。

    - 所有模型统一 stage1 + stage2 + freeze（含 baseline，实现公平对齐）
    - 不改 config.py 全局 MODEL_CONFIGS，返回新对象
    """
    base_cfg = get_model_config(model_key)
    new_cfg = replace(
        base_cfg,
        stage1=StageConfig(**recipe["stage1"]),
        stage2=StageConfig(**recipe["stage2"]),
        freeze=recipe["freeze"],
    )
    return new_cfg


def build_train_config(recipe: dict, model_key: str) -> TrainConfig:
    """从配方的 shared 字段构建 TrainConfig，并按 iou_override 设置 iou_type。"""
    shared = dict(recipe["shared"])
    iou_override = recipe.get("iou_override", {}) or {}
    iou_type = iou_override.get(model_key, "CIoU")

    # shared 里可能有 TrainConfig 不认识的字段（degrees/seed/deterministic）
    # 这些通过 to_dict 额外注入，先把 TrainConfig 认识的字段挑出来
    train_fields = {f.name for f in TrainConfig.__dataclass_fields__.values()}
    recognized = {k: v for k, v in shared.items() if k in train_fields}
    # cache: 配方里 'ram'/'true' 是字符串，TrainConfig.cache 是 str 字段
    # to_dict 会把 "false" 转 False，其它原样传，ultralytics 会处理 'ram'/'true'
    config = TrainConfig(**recognized)
    config.iou_type = iou_type
    return config


# ============================================================
# 预览
# ============================================================

def print_matrix_preview(recipe: dict, scales: list, models: list):
    """打印实验矩阵 + 统一变量摘要，供用户 Ctrl+C 复核。"""
    print("\n" + "=" * 80)
    print("公平消融实验矩阵预览")
    print("=" * 80)

    # 矩阵
    print(f"\n尺度 × 模型（共 {len(scales) * len(models)} 组训练）：")
    header = "  ".join(f"{m:<14}" for m in models)
    print(f"{'尺度':<8}{header}")
    for s in scales:
        row = f"{s:<8}" + "  ".join(f"{'✓':<14}" for _ in models)
        print(row)

    # 统一变量
    sh = recipe["shared"]
    print(f"\n统一变量：")
    print(f"  imgsz={sh.get('imgsz')}, batch={sh.get('batch')}, optimizer={sh.get('optimizer')}, lr0(stage2)={recipe['stage2']['lr0']}")
    print(f"  seed={sh.get('seed')}, deterministic={sh.get('deterministic')}, degrees={sh.get('degrees')}, cache={sh.get('cache')}")
    print(f"  两阶段: stage1({recipe['stage1']['epochs']}ep, freeze={recipe['freeze']}) + stage2({recipe['stage2']['epochs']}ep) = {recipe.get('total_epochs', recipe['stage1']['epochs']+recipe['stage2']['epochs'])}ep")
    print(f"  IoU 映射: {recipe.get('iou_override', {})}")
    print(f"  输出: {PAPER_ROOT / recipe['output_root']}")
    print("\n（--dry-run 仅预览，不训练；去掉 --dry-run 开始训练）")
    print("=" * 80 + "\n")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="公平消融实验编排器（一键训练 4 模型 × 多尺度）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE, help="配方 YAML 路径")
    p.add_argument("--scales", type=str, nargs="+", default=None, help="覆盖配方的 scales")
    p.add_argument("--models", type=str, nargs="+", default=None, help="覆盖配方的 models")
    p.add_argument("--skip-train", action="store_true", help="跳过训练，仅整理+出图")
    p.add_argument("--dry-run", action="store_true", help="仅打印矩阵，不训练")
    return p.parse_args()


def main():
    args = parse_args()
    recipe = load_recipe(args.recipe)
    scales = args.scales or recipe["scales"]
    models = args.models or recipe["models"]

    print_matrix_preview(recipe, scales, models)
    if args.dry_run:
        return

    # 占位：后续 Task 实现
    print("⚠ 训练/整理/出图逻辑尚未实现（见后续 Task）")


if __name__ == "__main__":
    main()

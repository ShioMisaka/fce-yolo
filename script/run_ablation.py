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
from script.trainer import YOLOv11Trainer  # noqa: E402

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
    required = ["shared", "stage1", "stage2", "freeze", "scales", "models", "output_root"]
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
    return replace(
        base_cfg,
        stage1=StageConfig(**recipe["stage1"]),
        stage2=StageConfig(**recipe["stage2"]),
        freeze=recipe["freeze"],
    )


def build_train_config(recipe: dict, model_key: str) -> TrainConfig:
    """从配方的 shared 字段构建 TrainConfig，并按 iou_override 设置 iou_type。

    shared 中 TrainConfig 认识的字段直接填入；不认识的（seed/deterministic/degrees
    等共享超参）通过 extra_args 注入，由 to_dict() 展开到 train() kwargs。
    """
    shared = dict(recipe["shared"])
    iou_override = recipe.get("iou_override", {}) or {}
    iou_type = iou_override.get(model_key, "CIoU")

    train_fields = {f for f in TrainConfig.__dataclass_fields__}
    recognized = {k: v for k, v in shared.items() if k in train_fields}
    extra = {k: v for k, v in shared.items() if k not in train_fields}

    config = TrainConfig(**recognized)
    config.iou_type = iou_type
    config.extra_args = extra
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
# IoU 校验
# ============================================================

def verify_iou_type(stage2_dir: Path, expected_iou: str) -> bool:
    """校验训练产物 args.yaml 里记录的 iou_type 与预期一致。

    防止 fce_wiou 漏加 --iou-type WIoU 退化成 CIoU（旧 bug 重现）。

    Args:
        stage2_dir: stage2 结果目录
        expected_iou: 预期 IoU 类型字符串（'WIoU'/'CIoU' 等）
    Returns:
        True 一致；False 不一致或读不到
    """
    args_yaml = stage2_dir / "args.yaml"
    if not args_yaml.exists():
        return False
    with open(args_yaml, encoding="utf-8") as f:
        # args.yaml 是 ultralytics 写的 YAML
        content = yaml.safe_load(f)
    actual = str(content.get("iou_type", "")).strip()
    if actual != expected_iou:
        print(f"✗ IoU 校验失败: {stage2_dir.name} 期望 {expected_iou} 实际 {actual}")
        return False
    return True


# ============================================================
# 单组实验执行
# ============================================================

def get_model_dir_name(model_key: str, scale: str) -> str:
    """根据 model_key 返回规范目录名（用于 main_ablation_fair/<scale>/0X_<name>）。"""
    mapping = {
        "baseline": "01_baseline_yolo11" + scale,
        "bifpn": "02_bifpn_" + scale,
        "fce": "03_fce_ciou_" + scale,
        "fce_wiou": "04_fce_wiou_" + scale,
    }
    return mapping[model_key]


def is_experiment_complete(scale: str, model_key: str, recipe: dict) -> bool:
    """判断某 (scale, model) 组合是否已完成（断点续跑判定）。

    判据：stage2/best.pt 存在 且 results.csv 行数 >= stage2.epochs * 0.9（容忍早停）。
    """
    from script.config import get_model_config
    base_cfg = get_model_config(model_key)
    # result_pattern 已含 _stage2 后缀（baseline 例外：原是单阶段无后缀，但我们注入了两阶段，
    # 训练后会生成 baseline_yolo11{scale}_stage2，因此这里统一查 _stage2 目录）
    if model_key == "baseline":
        s2_full = Path("runs/detect") / f"baseline_yolo11{scale}_stage2"
    else:
        s2_full = Path("runs/detect") / base_cfg.result_pattern.format(scale=scale)
    best_pt = s2_full / "weights" / "best.pt"
    csv = s2_full / "results.csv"
    if not (best_pt.exists() and csv.exists()):
        return False
    import pandas as pd
    try:
        df = pd.read_csv(csv)
    except Exception:
        return False
    min_rows = int(recipe["stage2"]["epochs"] * 0.9)
    return len(df) >= min_rows


def run_one_experiment(model_key: str, scale: str, recipe: dict) -> dict:
    """训练单组 (scale, model)，返回 {"stage1": Path, "stage2": Path}。

    断点续跑：若已完成则直接返回路径不重训。
    """
    # 断点续跑
    if is_experiment_complete(scale, model_key, recipe):
        base_cfg0 = get_model_config(model_key)
        s2 = Path("runs/detect") / (base_cfg0.result_pattern.format(scale=scale)
                                    if model_key != "baseline"
                                    else f"baseline_yolo11{scale}_stage2")
        s1 = Path(str(s2).replace("_stage2", "_stage1"))
        print(f"✓ 已完成，跳过 {scale}/{model_key}")
        return {"stage1": s1, "stage2": s2}

    # 公平注入
    model_cfg = build_model_cfg_with_fairness(model_key, recipe)
    config = build_train_config(recipe, model_key)

    print(f"\n▶ 训练 {scale}/{model_key}  (iou={config.iou_type})")
    trainer = YOLOv11Trainer(model_cfg, scale, config)
    result = trainer.train()  # 两阶段 -> {"stage1":..., "stage2":...}

    # IoU 校验（仅 fce_wiou 等非 CIoU 模型严格校验）
    iou_override = recipe.get("iou_override", {}) or {}
    expected_iou = iou_override.get(model_key, "CIoU")
    if expected_iou != "CIoU":
        s2_dir = result["stage2"]
        if not verify_iou_type(s2_dir, expected_iou):
            raise RuntimeError(
                f"{model_key} 训练后 IoU 校验失败：args.yaml 未记录 {expected_iou}，"
                f"可能退化成 CIoU，请检查配方 iou_override 与 build_train_config"
            )

    return result


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="公平消融实验编排器（一键训练 4 模型 × 多尺度）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE, help="配方 YAML 路径")
    parser.add_argument("--scales", type=str, nargs="+", default=None, help="覆盖配方的 scales")
    parser.add_argument("--models", type=str, nargs="+", default=None, help="覆盖配方的 models")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练，仅整理+出图")
    parser.add_argument("--dry-run", action="store_true", help="仅打印矩阵，不训练")
    return parser.parse_args()


def main():
    args = parse_args()
    recipe = load_recipe(args.recipe)
    scales = args.scales or recipe["scales"]
    models = args.models or recipe["models"]

    print_matrix_preview(recipe, scales, models)
    if args.dry_run:
        return

    # 阶段一：训练
    if not args.skip_train:
        print("\n" + "=" * 80)
        print("阶段一：训练")
        print("=" * 80)
        total = len(scales) * len(models)
        i = 0
        for scale in scales:
            for model_key in models:
                i += 1
                print(f"\n[{i}/{total}] ===== {scale} / {model_key} =====")
                try:
                    run_one_experiment(model_key, scale, recipe)
                except Exception as e:
                    print(f"✗ {scale}/{model_key} 失败: {e}")
                    failed_log = PAPER_ROOT / recipe["output_root"] / "failed.log"
                    failed_log.parent.mkdir(parents=True, exist_ok=True)
                    with open(failed_log, "a", encoding="utf-8") as f:
                        f.write(f"{scale}/{model_key}: {e}\n")
    else:
        print("⚠ --skip-train：跳过训练")

    # 阶段二/三（整理 + 出图）：后续 Task 实现
    print("\n⚠ 整理/出图逻辑尚未实现（见后续 Task）")


if __name__ == "__main__":
    main()

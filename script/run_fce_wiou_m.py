#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第④组实验（完整 FCE + WIoU）一键训练脚本

在工作站上通过 ssh 执行，完成：训练前自检 → 调用 train.py 两阶段训练 →
训练后校验（防漏传 --iou-type 退化成 CIoU）→ 自动打包结果。

背景：
  fce_wiou 配置（config.py）与 fce 逐字相同，仅靠 --iou-type WIoU 切换损失。
  若漏传该参数，会静默退化为 CIoU 训练（结果目录名仍是 fce_wiou_*，极具迷惑性）。
  本脚本通过训练后校验 args.yaml 杜绝此隐患。

用法（工作站 ssh 下）：
    python script/run_fce_wiou_m.py             # 完整流程：自检+训练+校验+打包
    python script/run_fce_wiou_m.py --check     # 仅训练前自检，不实际训练（dry-run）
    python script/run_fce_wiou_m.py --skip-train  # 跳过训练，只校验+打包（训练已完成时用）

说明：
  本脚本不改 train.py/config.py/trainer.py 任何现有代码，仅用 subprocess 调用
  `python script/train.py fce_wiou --scale m --iou-type WIoU`，复用全部两阶段逻辑。
"""

import argparse
import subprocess
import sys
from pathlib import Path

# 项目根目录（fce-yolo/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_NAME = "fce_wiou_m_stage2"  # 两阶段训练的 stage2 结果目录名

# 期望的训练配置（与③ fce 严格一致，仅 iou_type 不同）——用于训练后校验
EXPECT_MODEL_YAML = "ultralytics/cfg/models/11/yolo11-fce.yaml"
EXPECT_IOU_TYPE = "WIoU"
MIN_EPOCHS = 250  # 正常应跑满 300 或 patience 早停，过少视为异常


# ==================== 训练前自检 ====================

def pre_check() -> bool:
    """训练前自检：确认关键文件存在、配置已注册、数据集可访问。"""
    print("\n" + "=" * 60)
    print("【训练前自检】")
    print("=" * 60)
    ok = True

    # 1. FCE 模型 yaml 存在
    yaml = PROJECT_ROOT / EXPECT_MODEL_YAML
    if yaml.exists():
        print(f"✓ 模型 yaml 存在: {yaml.relative_to(PROJECT_ROOT)}")
    else:
        print(f"✗ 模型 yaml 缺失: {yaml}")
        ok = False

    # 2. train.py 存在且 fce_wiou 已注册
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from script.config import MODEL_CONFIGS
        if "fce_wiou" in MODEL_CONFIGS:
            cfg = MODEL_CONFIGS["fce_wiou"]
            print(f"✓ fce_wiou 配置已注册")
            print(f"    yaml_path: {cfg.yaml_path}")
            print(f"    freeze: {cfg.freeze}, 两阶段: {cfg.is_two_stage()}")
            print(f"    result_pattern: {cfg.result_pattern}")
        else:
            print("✗ fce_wiou 未在 MODEL_CONFIGS 注册")
            ok = False
    except Exception as e:
        print(f"✗ 无法导入 config: {e}")
        ok = False

    # 3. train.py 入口存在
    train_py = PROJECT_ROOT / "script" / "train.py"
    if train_py.exists():
        print(f"✓ train.py 存在")
    else:
        print(f"✗ train.py 缺失: {train_py}")
        ok = False

    # 4. 数据集可访问性（从默认配置推断，不强依赖 data.yaml 路径）
    #    实际训练时 train.py 会读 config，这里只做软提醒
    print(f"ℹ 数据集路径将由 config.py 的 DATASET_PRESETS 提供（训练时若报错请检查 data.yaml）")

    if ok:
        print("\n✅ 自检通过，可开始训练")
    else:
        print("\n❌ 自检失败，请修复上述问题后再试")
    return ok


# ==================== 训练后校验 ====================

def parse_args_yaml(args_path: Path) -> dict:
    """简单解析 ultralytics 的 args.yaml（key: value 格式，无需 pyyaml）。"""
    d = {}
    for line in args_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            k, _, v = line.partition(":")
            d[k.strip()] = v.strip().strip("'\"")
    return d


def post_check(runs_detect: Path) -> bool:
    """训练后校验：确认确实用了 WIoU、正确的模型、训练轮数充足。"""
    print("\n" + "=" * 60)
    print("【训练后校验】")
    print("=" * 60)

    exp_dir = runs_detect / EXP_NAME
    if not exp_dir.exists():
        print(f"✗ 结果目录不存在: {exp_dir}")
        print("  训练可能未完成，或 runs 目录位置不同（检查 ultralytics settings 的 runs_dir）")
        return False

    ok = True

    # 1. args.yaml 校验（核心：防漏传 --iou-type）
    args_yaml = exp_dir / "args.yaml"
    if not args_yaml.exists():
        print(f"✗ args.yaml 缺失: {args_yaml}")
        return False
    args = parse_args_yaml(args_yaml)

    model = args.get("model", "")
    iou = args.get("iou_type", "")
    print(f"  model: {model}")
    print(f"  iou_type: {iou}")

    if iou != EXPECT_IOU_TYPE:
        print(f"✗ iou_type 不是 {EXPECT_IOU_TYPE}！实际为 {iou!r}")
        print(f"  这意味着训练时漏传了 --iou-type WIoU，已静默退化为 CIoU。结果无效，需重跑。")
        ok = False
    else:
        print(f"✓ iou_type = WIoU，损失函数正确")

    if model != EXPECT_MODEL_YAML:
        print(f"✗ model 不是 {EXPECT_MODEL_YAML}！实际为 {model!r}")
        ok = False
    else:
        print(f"✓ model 配置正确")

    # 2. results.csv 存在 + 轮数充足
    csv = exp_dir / "results.csv"
    if not csv.exists():
        print(f"✗ results.csv 缺失: {csv}")
        return False
    try:
        import pandas as pd
        df = pd.read_csv(csv)
        df.columns = [c.strip() for c in df.columns]
        total = int(df["epoch"].iloc[-1])
        print(f"  训练轮数: {total}")
        if total < MIN_EPOCHS:
            print(f"⚠ 训练轮数 {total} < {MIN_EPOCHS}，可能异常（正常应 300 或早停）")
            # 不直接判失败，早停也可能合理，仅警告
        else:
            print(f"✓ 训练轮数充足")

        # best 指标预览
        col = "metrics/mAP50-95(B)"
        if col in df.columns:
            bi = df[col].idxmax()
            row = df.loc[bi]
            print(f"\n  📊 best 指标预览（ep{int(row['epoch'])}）:")
            print(f"     P={row['metrics/precision(B)']*100:.2f} "
                  f"R={row['metrics/recall(B)']*100:.2f} "
                  f"mAP50={row['metrics/mAP50(B)']*100:.2f} "
                  f"mAP50-95={row[col]*100:.2f}")
            print(f"     对照 ③ fce_m: mAP50-95=80.53（看 ④ 是否 > ③）")
    except Exception as e:
        print(f"⚠ 读取 results.csv 失败: {e}")

    if ok:
        print("\n✅ 校验通过，结果有效")
    else:
        print("\n❌ 校验失败，结果不可用于论文（见上方原因）")
    return ok


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(
        description="第④组（FCE+WIoU）一键训练 + 自检 + 打包")
    parser.add_argument("--check", action="store_true",
                        help="仅训练前自检（dry-run，不训练）")
    parser.add_argument("--skip-train", action="store_true",
                        help="跳过训练（训练已完成），只做校验 + 打包")
    parser.add_argument("--runs-dir", default="runs/detect",
                        help="runs 目录，默认 runs/detect")
    args = parser.parse_args()

    runs_detect = (PROJECT_ROOT / args.runs_dir).resolve()

    # 1. 训练前自检
    if not pre_check():
        sys.exit(1)
    if args.check:
        print("\n--check 模式：自检完成，不执行训练。")
        return

    # 2. 训练（调用现有 train.py，不重写）
    if not args.skip_train:
        print("\n" + "=" * 60)
        print("【开始训练】第④组 fce_wiou_m（WIoU）")
        print("=" * 60)
        cmd = [
            sys.executable, "script/train.py",
            "fce_wiou", "--scale", "m", "--iou-type", "WIoU",
        ]
        print(f"执行: {' '.join(cmd)}")
        print(f"工作目录: {PROJECT_ROOT}")
        print("（两阶段：stage1 50ep freeze=10 + stage2 300ep，预计 6-10 小时）\n")
        r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if r.returncode != 0:
            print(f"\n✗ 训练失败，返回码 {r.returncode}")
            sys.exit(r.returncode)
        print("\n✓ 训练命令执行完毕")

    # 3. 训练后校验
    if not post_check(runs_detect):
        print("\n⚠ 校验未通过，不执行打包（避免把无效结果传回本机）")
        sys.exit(1)

    # 4. 自动打包
    print("\n" + "=" * 60)
    print("【打包结果】")
    print("=" * 60)
    pack_script = PROJECT_ROOT / "script" / "pack_results.py"
    if not pack_script.exists():
        print(f"✗ 打包脚本缺失: {pack_script}")
        sys.exit(1)
    cmd = [sys.executable, str(pack_script), EXP_NAME, "--runs-dir", str(runs_detect)]
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if r.returncode != 0:
        print(f"\n✗ 打包失败，返回码 {r.returncode}")
        sys.exit(r.returncode)

    print("\n" + "=" * 60)
    print("🎉 全部完成！")
    print("=" * 60)
    print(f"\n下载方式（在本机 Windows 执行）:")
    print(f"  scp <工作站>:~/workspace/my_project/fce-yolo/runs/{EXP_NAME}.zip .")
    print(f"\n解压后放到（本机）:")
    print(f"  D:\\Box\\Project\\Visual Guidance Robotic Arm\\实验\\3月测试\\baselinevsbifpnvsfce_m_300\\{EXP_NAME}\\")
    print(f"\n然后本机分析:")
    print(f"  1. python 工具脚本/regen_metrics_summary.py   # ④纳入汇总")
    print(f"  2. python script/paper_plots.py               # 重出 4 线对比图")


if __name__ == "__main__":
    main()

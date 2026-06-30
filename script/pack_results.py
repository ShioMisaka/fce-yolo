#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练结果打包器（通用）

把 runs/detect/<name>/ 整个复制并压缩成 runs/<name>.zip，
方便通过 scp / 网盘一次性下载到本机分析。

特点：
  - 复制 + 压缩，不移动/删除工作站原文件
  - 保留 best.pt / last.pt / epoch*.pt 权重（本机分析需要）
  - 排除 __pycache__ 等临时文件
  - 自动生成 _manifest.txt：打包时间、git commit、文件清单、best 指标摘要

用法：
    python script/pack_results.py                       # 默认打包 fce_wiou_m_stage2
    python script/pack_results.py fce_m_stage2          # 打包指定实验
    python script/pack_results.py fce_wiou_m_stage2 --runs-dir runs/detect
"""

import argparse
import datetime
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# 项目根目录（fce-yolo/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 打包时排除的文件名/扩展名
EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints"}
EXCLUDE_EXTS = {".pyc", ".pyo"}


def get_git_commit() -> str:
    """获取当前 git commit hash（失败返回 unknown）。"""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


def extract_best_metrics(csv_path: Path) -> dict:
    """从 results.csv 提取 best 指标（按 mAP50-95 最高轮）。失败返回空 dict。"""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        col = "metrics/mAP50-95(B)"
        if col not in df.columns:
            return {}
        best_idx = df[col].idxmax()
        row = df.loc[best_idx]
        return {
            "best_epoch": int(row["epoch"]),
            "mAP50-95": round(float(row[col]) * 100, 2),
            "mAP50": round(float(row["metrics/mAP50(B)"]) * 100, 2),
            "P": round(float(row["metrics/precision(B)"]) * 100, 2),
            "R": round(float(row["metrics/recall(B)"]) * 100, 2),
            "total_epochs": int(df["epoch"].iloc[-1]),
        }
    except Exception as e:
        return {"error": str(e)}


def build_manifest(src_dir: Path, name: str, metrics: dict) -> str:
    """生成 manifest 文本。"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"训练结果打包清单：{name}")
    lines.append("=" * 60)
    lines.append(f"打包时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"git commit: {get_git_commit()}")
    lines.append(f"源目录: {src_dir}")
    lines.append("")

    lines.append("--- best 指标摘要（results.csv）---")
    if metrics and "error" not in metrics:
        lines.append(f"  总训练轮数: {metrics['total_epochs']}")
        lines.append(f"  best 轮次: {metrics['best_epoch']}")
        lines.append(f"  Precision: {metrics['P']}")
        lines.append(f"  Recall:    {metrics['R']}")
        lines.append(f"  mAP50:     {metrics['mAP50']}")
        lines.append(f"  mAP50-95:  {metrics['mAP50-95']}")
    elif metrics:
        lines.append(f"  ⚠ 读取失败: {metrics.get('error')}")
    else:
        lines.append("  ⚠ 未找到 results.csv")
    lines.append("")

    lines.append("--- 文件清单 ---")
    file_count = 0
    total_size = 0
    for f in sorted(src_dir.rglob("*")):
        if f.is_file():
            # 过滤排除项
            if any(part in EXCLUDE_DIRS for part in f.parts):
                continue
            if f.suffix in EXCLUDE_EXTS:
                continue
            rel = f.relative_to(src_dir)
            size_mb = f.stat().st_size / (1024 * 1024)
            lines.append(f"  {rel}  ({size_mb:.2f} MB)")
            file_count += 1
            total_size += f.stat().st_size
    lines.append("")
    lines.append(f"共 {file_count} 个文件，合计 {total_size / (1024 * 1024):.2f} MB")
    lines.append("")
    lines.append("--- 本机分析指引（下载后）---")
    lines.append("1. 解压到: 实验/3月测试/baselinevsbifpnvsfce_m_300/<name>/")
    lines.append("2. 重算汇总: python 工具脚本/regen_metrics_summary.py")
    lines.append("3. 重出图表: python script/paper_plots.py")
    return "\n".join(lines)


def pack(name: str, runs_dir: Path) -> Path:
    """打包 runs_dir/<name> 为 runs_dir/../<name>.zip。返回 zip 路径。"""
    src = runs_dir / name
    if not src.exists():
        print(f"✗ 实验目录不存在: {src}")
        sys.exit(1)

    # 输出 zip 放在 runs/ 下（runs_dir 的父级，或 runs_dir 本身取决于布局）
    # 约定：runs/detect/<name> → runs/<name>.zip
    out_dir = runs_dir.parent if runs_dir.name == "detect" else runs_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{name}.zip"

    # 先生成 manifest 到临时内存（不污染源目录）
    csv = src / "results.csv"
    metrics = extract_best_metrics(csv) if csv.exists() else {}
    manifest = build_manifest(src, name, metrics)

    print(f"正在打包: {src} → {zip_path}")
    # 复制方式压缩（不删源），手动遍历以应用过滤
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in src.rglob("*"):
            if not f.is_file():
                continue
            if any(part in EXCLUDE_DIRS for part in f.parts):
                continue
            if f.suffix in EXCLUDE_EXTS:
                continue
            arcname = f.relative_to(src.parent)  # 顶层保留 <name>/ 目录名
            zf.write(f, arcname)

    # 把 manifest 写进 zip 根目录
    with zipfile.ZipFile(zip_path, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("_manifest.txt", manifest)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ 打包完成: {zip_path} ({size_mb:.2f} MB)")
    print("\n" + manifest)
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="训练结果打包器")
    parser.add_argument("name", nargs="?", default="fce_wiou_m_stage2",
                        help="实验目录名（runs/detect/<name>），默认 fce_wiou_m_stage2")
    parser.add_argument("--runs-dir", default="runs/detect",
                        help="runs 目录路径，默认 runs/detect")
    args = parser.parse_args()

    runs_dir = (PROJECT_ROOT / args.runs_dir).resolve()
    if not runs_dir.exists():
        # 兼容 ultralytics settings 里自定义 runs_dir 的情况，给出提示
        print(f"✗ runs 目录不存在: {runs_dir}")
        print("  若你的 runs 在别处，用 --runs-dir 指定，例如 --runs-dir /home/you/workspace/runs/detect")
        sys.exit(1)

    pack(args.name, runs_dir)


if __name__ == "__main__":
    main()

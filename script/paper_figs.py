"""
Paper figure generator for FCE-YOLOv11 ablation experiments.

Reads experiment groups described in paper_figs_config.yaml, loads results.csv
of each model, and produces publication-grade comparison figures highlighting
the progressive improvement M1 -> M2 -> M3 -> M4.

Produces 4 categories (all by default; use --only to select):
  A. Training curves (metrics 4-panel + loss 4-panel)
  B. Ablation analysis (B1 gain bar / B2 performance radar / B3 convergence /
     B ablation table CSV+MD)
  C. Detection visualization (val_batch0 GT + 4-model predictions, stacked)
  D. PR / F1 / confusion matrix montage

Extensibility: "which experiments / colors / paths" are all driven by YAML;
re-running an experiment only requires editing the YAML, never the code.

Usage (under the fce-yolo conda env):
    conda activate fce-yolo
    cd <project_root>/fce-yolo
    python script/paper_figs.py                       # produce all
    python script/paper_figs.py --only A,B            # produce A and B only
    python script/paper_figs.py --config script/xxx.yaml   # custom config

Data integrity: all numbers strictly come from the best epoch (idxmax of
mAP50-95) of results.csv; never fabricated. GFLOPs/Params are computed from
best.pt on the fly, falling back to N/A on failure.

NOTE: All output file/dir names and in-figure text are English-only so the
script runs on both Windows (local) and Linux (workstation, no CJK fonts).
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# 项目根目录入 path（fce-yolo 仓库根），便于复用 script.analysis
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from script.analysis import load_results, extract_metrics  # noqa: E402

# 论文项目根（Visual Guidance Robotic Arm/，即 fce-yolo 的上一级）
PAPER_ROOT = PROJECT_ROOT.parent


# ============================================================
# 配置加载
# ============================================================

def _resolve_path(rel_path: str) -> Path:
    """把 YAML 里的相对路径解析成绝对路径，双根兼容。

    优先在 PROJECT_ROOT（fce-yolo 仓库根）下找；找不到再回退 PAPER_ROOT
    （项目根）。这样既能读 runs/outputs（新工作流，工作站产物在仓库内），
    又能兼容旧 paper_figs_config.yaml 里的中文 `实验/论文正式实验/...` 路径。
    """
    p_project = (PROJECT_ROOT / rel_path).resolve()
    if p_project.exists():
        return p_project
    p_paper = (PAPER_ROOT / rel_path).resolve()
    return p_paper


def _detect_root(rel_path: str, reference_abs: Path, reference_rel: str) -> Path:
    """从已解析的同级路径推断 rel_path 应所在的根。

    reference_abs 已正确解析（如某实验的 abs_dir），reference_rel 是它在 YAML 里的
    相对路径。两者共享前缀，故 reference_abs 去掉 reference_rel 各级后即得"根"，
    再拼 rel_path 得到目标绝对路径。用于 out_dir 这类尚不存在、无法靠 exists 判根的路径。
    """
    root = reference_abs
    for _ in Path(reference_rel).parts:
        root = root.parent
    return (root / rel_path).resolve()


def load_config(config_path):
    """加载 YAML 配置，返回 (experiments, settings)。

    experiments: 按 order 排序的实验列表，每项含 key/dir/display/color 等
    settings: 全局设置 dict
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exps = cfg["experiments"]
    # 转为有序列表（按 order 字段），便于消融递进展示
    exp_list = []
    for key, e in exps.items():
        e = dict(e)
        e["key"] = key
        e["abs_dir"] = _resolve_path(e["dir"])
        exp_list.append(e)
    exp_list.sort(key=lambda x: x["order"])

    settings = cfg["settings"]
    # out_dir 与实验 dir 共享前缀（同一 out_prefix），数据在哪根下，图表就写哪根下。
    # out_dir 尚不存在，无法靠 exists 判根，故从首个已解析的实验路径推断根。
    if exp_list:
        settings["abs_out_dir"] = _detect_root(
            settings["out_dir"], exp_list[0]["abs_dir"], exp_list[0]["dir"]
        )
    else:
        settings["abs_out_dir"] = (PROJECT_ROOT / settings["out_dir"]).resolve()
    return exp_list, settings


def load_all_dfs(exp_list):
    """批量加载所有实验的 results.csv，返回 {key: DataFrame}。"""
    dfs = {}
    for e in exp_list:
        csv = e["abs_dir"] / "results.csv"
        if not csv.exists():
            print(f"warn: results.csv missing: {csv}, skipping {e['key']}")
            continue
        dfs[e["key"]] = load_results(csv)
    return dfs


# ============================================================
# Font configuration (cross-platform, English-only output)
# ============================================================

def setup_cn_font(dpi=300):
    """Configure matplotlib fonts for cross-platform rendering.

    In-figure text is English-only now, so any sans-serif font works. We still
    prefer CJK-capable fonts when present (harmless), but DejaVu Sans (matplotlib
    default on Linux) is the guaranteed fallback. This lets the script run on a
    Linux workstation without any CJK font installed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # Order: prefer CJK-capable (in case of mixed content), fall back to DejaVu
    # (ships with matplotlib on every platform).
    candidates = [
        "Microsoft YaHei", "SimHei", "Microsoft YaHei UI", "SimSun",  # Windows
        "Noto Sans CJK SC", "Noto Sans CJK", "WenQuanYi Zen Hei",     # Linux
        "PingFang SC", "Heiti SC",                                     # macOS
        "DejaVu Sans",                                                 # universal fallback
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), "DejaVu Sans")

    plt.rcParams.update({
        "font.sans-serif": [chosen] + candidates,
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 2,
        "figure.dpi": 150,
        "savefig.dpi": dpi,
    })
    print(f"font: {chosen}")
    return plt


# ============================================================
# 通用：按 order 排序的实验元数据访问
# ============================================================

def _ordered(dfs, exp_list):
    """Return items from dfs in exp_list order as (key, df, meta)."""
    out = []
    for e in exp_list:
        if e["key"] in dfs:
            out.append((e["key"], dfs[e["key"]], e))
    return out


# ============================================================
# Cross-platform PIL font resolution
# ============================================================

def _load_pil_font(size=28, bold=False):
    """Find a usable TrueType font for PIL across Windows/Linux/macOS.

    In-figure text is English-only, so any TTF works. We probe common per-OS
    font paths and fall back to PIL's default bitmap font (always available).
    """
    from PIL import ImageFont

    # Bold-biased file names first when bold=True; regular-biased first otherwise.
    win_bold = ["msyhbd.ttc", "segoeuib.ttf", "arialbd.ttf"]
    win_reg = ["msyh.ttc", "segoeui.ttf", "arial.ttf", "simhei.ttf"]
    linux_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    mac_paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]

    if sys.platform.startswith("win"):
        base = Path("C:/Windows/Fonts")
        prefer = (win_bold if bold else []) + (win_reg if not bold else []) + \
                 (win_reg if bold else []) + (win_bold if not bold else [])
        candidates = [str(base / f) for f in prefer]
    elif sys.platform == "darwin":
        candidates = mac_paths if not bold else mac_paths
    else:
        candidates = linux_paths

    for fp in candidates:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ============================================================
# Output A: training curves (metrics 4-panel + loss 4-panel)
# ============================================================

def _plot_4panels(dfs, exp_list, col_config, save_path, fig_title, settings):
    """Generic 2x2 comparison plotter.

    col_config: [(csv_column, subplot_title, ylabel), ...] 4 items
    """
    plt = setup_cn_font(settings["dpi"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), tight_layout=True)
    items = _ordered(dfs, exp_list)
    for idx, (col, subtitle, ylabel) in enumerate(col_config):
        ax = axes[idx // 2, idx % 2]
        for key, df, meta in items:
            if col not in df.columns:
                continue
            ax.plot(
                df["epoch"], df[col],
                color=meta["color"], linestyle=meta["linestyle"],
                label=meta["display"], linewidth=2,
            )
        ax.set_title(subtitle, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.25)
    fig.suptitle(fig_title, fontsize=15, fontweight="bold", y=1.0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"saved: {save_path}")


def produce_A(dfs, exp_list, settings):
    """A: training-curve comparison (metrics + loss)."""
    print("\n" + "=" * 60)
    print("Output A: training curves")
    print("=" * 60)
    out = settings["abs_out_dir"] / "A_curves"

    metric_cfg = [
        ("metrics/mAP50-95(B)", "mAP@[50-95] Comparison", "mAP@[50-95]"),
        ("metrics/mAP50(B)", "mAP@50 Comparison", "mAP@50"),
        ("metrics/precision(B)", "Precision Comparison", "Precision"),
        ("metrics/recall(B)", "Recall Comparison", "Recall"),
    ]
    _plot_4panels(
        dfs, exp_list, metric_cfg,
        out / "A1_metrics_4panel.png",
        "Training Metrics Comparison (FCE-YOLOv11 Ablation, scale-m)",
        settings,
    )

    loss_cfg = [
        ("train/box_loss", "Train Box Loss", "Box Loss"),
        ("train/cls_loss", "Train Cls Loss", "Cls Loss"),
        ("train/dfl_loss", "Train DFL Loss", "DFL Loss"),
        ("val/box_loss", "Val Box Loss", "Box Loss"),
    ]
    _plot_4panels(
        dfs, exp_list, loss_cfg,
        out / "A2_loss_4panel.png",
        "Training/Validation Loss Comparison (FCE-YOLOv11 Ablation, scale-m)",
        settings,
    )


# ============================================================
# 产出 B：消融分析（柱状图 + 雷达图 + 收敛速度 + 表格）
# ============================================================

def _best_metrics(df):
    """提取 best 轮（mAP50-95 最高）的 4 个指标，返回 dict（0~1 原值）。"""
    col = "metrics/mAP50-95(B)"
    if col not in df.columns:
        return None
    best = df.loc[df[col].idxmax()]
    return {
        "epoch": int(best["epoch"]),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "mAP50": float(best["metrics/mAP50(B)"]),
        "mAP50_95": float(best[col]),
    }


def produce_B1(dfs, exp_list, settings):
    """B1: ablation gain bar chart (mAP50-95 progression + delta annotation)."""
    print("\n--- B1: ablation gain bar chart ---")
    plt = setup_cn_font(settings["dpi"])
    import numpy as np

    items = _ordered(dfs, exp_list)
    names, vals, colors = [], [], []
    for key, df, meta in items:
        m = _best_metrics(df)
        if m is None:
            continue
        names.append(meta["display"])
        vals.append(m["mAP50_95"] * 100)
        colors.append(meta["color"])

    fig, ax = plt.subplots(figsize=(10, 6.5), tight_layout=True)
    x = np.arange(len(vals))
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.8, width=0.55)

    # absolute value on top of each bar
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.2f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    # delta between adjacent bars
    for i in range(1, len(vals)):
        delta = vals[i] - vals[i - 1]
        midx = (x[i - 1] + x[i]) / 2
        y = max(vals[i - 1], vals[i]) + 2.2
        ax.annotate(
            f"+{delta:.2f}", xy=(midx, y), ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#C62828",
            arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.2),
            xytext=(midx, y + 1.0),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("mAP@[50-95] (%)", fontsize=12)
    ax.set_title("Ablation: Progressive Improvement on mAP@[50-95]",
                 fontsize=14, fontweight="bold")
    ymin = min(vals) - 5
    ax.set_ylim(ymin, max(vals) + 6)
    ax.grid(True, axis="y", alpha=0.3)
    out = settings["abs_out_dir"] / "B_ablation"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "B1_ablation_gain_bar.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"saved: {p}")


def produce_B2(dfs, exp_list, settings):
    """B2: performance radar (P/R/mAP50/mAP50-95, 4 dims)."""
    print("\n--- B2: performance radar ---")
    plt = setup_cn_font(settings["dpi"])
    import numpy as np

    dims = ["Precision", "Recall", "mAP@50", "mAP@[50-95]"]
    keys = ["precision", "recall", "mAP50", "mAP50_95"]
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), tight_layout=True)
    items = _ordered(dfs, exp_list)
    for key, df, meta in items:
        m = _best_metrics(df)
        if m is None:
            continue
        vals = [m[k] * 100 for k in keys]
        vals += vals[:1]
        ax.plot(angles, vals, color=meta["color"], linestyle=meta["linestyle"],
                linewidth=2.2, label=meta["display"])
        ax.fill(angles, vals, color=meta["color"], alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=12, fontweight="bold")
    ax.set_ylim(70, 100)
    ax.set_yticks([75, 80, 85, 90, 95, 100])
    ax.set_yticklabels(["75", "80", "85", "90", "95", "100"], fontsize=9, color="gray")
    ax.set_title("Performance Radar (best metrics, 4-model comparison)",
                 fontsize=14, fontweight="bold", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), framealpha=0.9)
    ax.grid(True, alpha=0.4)
    out = settings["abs_out_dir"] / "B_ablation"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "B2_performance_radar.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"saved: {p}")


def produce_B3(dfs, exp_list, settings):
    """B3: convergence-speed comparison (epochs to first reach mAP50-95 threshold)."""
    print("\n--- B3: convergence speed ---")
    plt = setup_cn_font(settings["dpi"])
    import numpy as np

    thr = settings.get("convergence_threshold", 0.75)
    col = "metrics/mAP50-95(B)"
    items = _ordered(dfs, exp_list)

    names, epochs, colors = [], [], []
    for key, df, meta in items:
        if col not in df.columns:
            continue
        reached = df[df[col] >= thr]
        if reached.empty:
            print(f"  warn: {meta['display']} never reached threshold {thr}; using total epochs")
            ep = int(df["epoch"].iloc[-1])
        else:
            ep = int(reached.iloc[0]["epoch"])
        names.append(meta["display"])
        epochs.append(ep)
        colors.append(meta["color"])

    if not epochs:
        print("  warn: no usable data, skipping B3")
        return

    fig, ax = plt.subplots(figsize=(10, 5.5), tight_layout=True)
    y = np.arange(len(epochs))
    bars = ax.barh(y, epochs, color=colors, edgecolor="black", linewidth=0.8, height=0.55)
    for b, ep in zip(bars, epochs):
        ax.text(ep + 2, b.get_y() + b.get_height() / 2, f"{ep} ep",
                va="center", fontsize=11, fontweight="bold")

    # speedup of fastest vs slowest
    if len(epochs) >= 2:
        fastest, slowest = min(epochs), max(epochs)
        speedup = (1 - fastest / slowest) * 100
        ax.text(0.98, 0.04,
                f"Fastest is {speedup:.0f}% quicker than slowest\n"
                f"(threshold mAP@[50-95]={thr})",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, color="#C62828",
                bbox=dict(boxstyle="round,pad=0.4", fc="#FFF3E0", ec="#FF6B00"))

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()  # M1 at top
    ax.set_xlabel(f"Epochs to first reach mAP@[50-95]>={thr}", fontsize=12)
    ax.set_title("Convergence Speed (shorter is faster)", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    out = settings["abs_out_dir"] / "B_ablation"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "B3_convergence_speed.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"saved: {p}")


def _compute_model_complexity(weights_path, imgsz=1280):
    """计算 Params 与 GFLOPs（CPU 可跑）。失败返回 (None, None)。"""
    import contextlib
    import io
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    torch.set_num_threads(4)

    m = YOLO(str(weights_path))
    inner = m.model
    n_params = sum(p.numel() for p in inner.parameters())
    with contextlib.redirect_stdout(io.StringIO()):
        gflops = get_flops(inner, imgsz=imgsz)
    return n_params / 1e6, gflops


def produce_B_table(dfs, exp_list, settings):
    """B ablation table: CSV + Markdown (with Params/GFLOPs and delta gain)."""
    print("\n--- B ablation table: CSV + Markdown ---")
    rows = []
    prev_map = None
    for key, df, meta in _ordered(dfs, exp_list):
        m = _best_metrics(df)
        if m is None:
            continue
        cur = round(m["mAP50_95"] * 100, 2)
        delta = "-" if prev_map is None else f"+{cur - prev_map:.2f}"
        prev_map = cur

        weights = meta["abs_dir"] / "weights" / "best.pt"
        params_m, gflops = (None, None)
        if weights.exists():
            print(f"  computing complexity of {meta['display']} ...")
            try:
                params_m, gflops = _compute_model_complexity(weights, settings["imgsz"])
            except Exception as ex:
                print(f"  warn: complexity computation failed: {ex}")
        else:
            print(f"  warn: best.pt missing: {weights}")

        rows.append({
            "No": f"M{meta['order']}" if meta["order"] <= 4 else str(meta["order"]),
            "Model": meta["display"],
            "FCE_Module": meta["fce_module"],
            "Loss": meta["loss"],
            "Best_Epoch": m["epoch"],
            "Precision": round(m["precision"] * 100, 2),
            "Recall": round(m["recall"] * 100, 2),
            "mAP50": round(m["mAP50"] * 100, 2),
            "mAP50-95": cur,
            "d_mAP50-95": delta,
            "Params(M)": round(params_m, 2) if params_m else "N/A",
            "GFLOPs": round(gflops, 1) if gflops else "N/A",
        })

    df_out = pd.DataFrame(rows)
    out = settings["abs_out_dir"] / "B_ablation"
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "B_ablation_results_table.csv"
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"saved: {csv_path}")

    # Markdown table
    md = ["# Table 4-1 Ablation Results (scale-m, imgsz=1280, best metrics)\n",
          "> best metrics = the epoch with the highest val mAP50-95 (standard YOLO reporting)\n",
          "> Training: AdamW, lr0=0.001, batch=32, cosine annealing, 300 epochs, two-stage\n"]
    cols = list(df_out.columns)
    md.append("| " + " | ".join(cols) + " |")
    md.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in df_out.iterrows():
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    md_path = out / "B_ablation_results_table.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"saved: {md_path}")
    return df_out


def produce_B(dfs, exp_list, settings):
    """B: ablation analysis (full set)."""
    print("\n" + "=" * 60)
    print("Output B: ablation analysis")
    print("=" * 60)
    produce_B1(dfs, exp_list, settings)
    produce_B2(dfs, exp_list, settings)
    produce_B3(dfs, exp_list, settings)
    produce_B_table(dfs, exp_list, settings)


# ============================================================
# Output C: detection visualization (val_batch0 GT + 4-model preds)
# ============================================================

def produce_C(dfs, exp_list, settings):
    """C: stacked val_batch0 prediction comparison."""
    print("\n" + "=" * 60)
    print("Output C: detection visualization")
    print("=" * 60)
    from PIL import Image, ImageDraw

    panels = [("Ground Truth (val_batch0_labels)",
               exp_list[0]["abs_dir"] / "val_batch0_labels.jpg")]
    for key, df, meta in _ordered(dfs, exp_list):
        panels.append((f"{meta['display']} (val_batch0_pred)",
                       meta["abs_dir"] / "val_batch0_pred.jpg"))

    for title, p in panels:
        if not p.exists():
            print(f"warn: missing image: {p}")
            return

    imgs = [Image.open(p) for _, p in panels]
    w = max(im.width for im in imgs)
    title_h, pad = 48, 8
    total_h = sum(im.height for im in imgs) + (title_h + pad) * len(imgs) + pad
    canvas = Image.new("RGB", (w + pad * 2, total_h), "white")

    font = _load_pil_font(size=30)

    draw = ImageDraw.Draw(canvas)
    y = pad
    for (title, _), im in zip(panels, imgs):
        draw.text((pad, y + 8), title, fill="black", font=font)
        y += title_h
        canvas.paste(im, (pad, y))
        y += im.height + pad

    out = settings["abs_out_dir"] / "C_detection"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "C_val_batch0_4model_compare.png"
    canvas.save(p, "PNG", dpi=(settings["dpi"], settings["dpi"]))
    print(f"saved: {p}")


# ============================================================
# 产出 D：PR/F1/混淆矩阵 中文化拼接
# ============================================================

def _hstack_with_titles(panels, out_path, fig_title, settings, subtitle_h=60):
    """Horizontally stack images with English subtitles + a main title."""
    from PIL import Image, ImageDraw
    for _, p in panels:
        if not p.exists():
            print(f"warn: missing image: {p}")
            return False
    imgs = [Image.open(p) for _, p in panels]
    h = min(im.height for im in imgs)
    imgs = [im.resize((int(im.width * h / im.height), h)) for im in imgs]

    title_h, pad = 50, 10
    total_w = sum(im.width for im in imgs) + pad * (len(imgs) + 1)
    total_h = h + title_h + pad * 3
    canvas = Image.new("RGB", (total_w, total_h), "white")

    font = _load_pil_font(size=28)
    title_font = _load_pil_font(size=34, bold=True)

    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 8), fig_title, fill="black", font=title_font)
    y = title_h
    x = pad
    for (subtitle, _), im in zip(panels, imgs):
        draw.text((x, y), subtitle, fill="black", font=font)
        canvas.paste(im, (x, y + 28))
        x += im.width + pad

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "PNG", dpi=(settings["dpi"], settings["dpi"]))
    print(f"saved: {out_path}")
    return True


def produce_D(dfs, exp_list, settings):
    """D: PR / F1 / confusion matrix montage."""
    print("\n" + "=" * 60)
    print("Output D: PR / F1 / confusion montage")
    print("=" * 60)
    out = settings["abs_out_dir"] / "D_pr_confusion"
    items = _ordered(dfs, exp_list)

    # D1 PR
    _hstack_with_titles(
        [(m["display"], m["abs_dir"] / "BoxPR_curve.png") for _, _, m in items],
        out / "D1_PR_4model_compare.png",
        "PR Curve Comparison (4 models, scale-m)", settings,
    )
    # D2 F1
    _hstack_with_titles(
        [(m["display"], m["abs_dir"] / "BoxF1_curve.png") for _, _, m in items],
        out / "D2_F1_4model_compare.png",
        "F1 Curve Comparison (4 models, scale-m)", settings,
    )
    # D3 confusion matrix (full FCE = highest order)
    last = items[-1][2]
    cn = settings.get("class_names", [])
    cls_note = ", ".join(f"{i}={n}" for i, n in enumerate(cn)) if cn else ""
    note_suffix = f" [{cls_note}]" if cls_note else ""
    _hstack_with_titles(
        [(f"{last['display']} (Normalized Confusion Matrix)",
          last["abs_dir"] / "confusion_matrix_normalized.png")],
        out / "D3_FCE_confusion_matrix.png",
        f"{last['display']} Normalized Confusion Matrix{note_suffix}", settings,
    )


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Paper figure generator (FCE-YOLOv11 ablation)")
    parser.add_argument(
        "--config", default=str(Path(__file__).parent / "paper_figs_config.yaml"),
        help="YAML config file path",
    )
    parser.add_argument(
        "--only", default="", type=str,
        help="Produce only the given categories (comma-separated, e.g. A,B,C,D); default all",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"error: config not found: {config_path}")
        sys.exit(1)

    print("=" * 60)
    print(f"config: {config_path}")
    print("=" * 60)
    exp_list, settings = load_config(config_path)
    print(f"  experiments: {[e['key'] for e in exp_list]}")
    print(f"  output_dir: {settings['abs_out_dir']}")

    dfs = load_all_dfs(exp_list)
    if not dfs:
        print("error: no usable data, exiting")
        sys.exit(1)

    targets = {t.strip().upper() for t in args.only.split(",") if t.strip()} or \
              {"A", "B", "C", "D"}

    if "A" in targets:
        produce_A(dfs, exp_list, settings)
    if "B" in targets:
        produce_B(dfs, exp_list, settings)
    if "C" in targets:
        produce_C(dfs, exp_list, settings)
    if "D" in targets:
        produce_D(dfs, exp_list, settings)

    print("\n" + "=" * 60)
    print(f"DONE. output_dir: {settings['abs_out_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

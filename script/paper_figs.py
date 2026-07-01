"""
论文图表生成脚本（FCE-YOLOv11 消融实验）

基于 paper_figs_config.yaml 描述的实验组，从各模型 results.csv 读取真实数据，
产出多维度的论文级对比图表，凸显 ①→②→③→④ 的递进优化效果。

产出 8 类（默认全部生成，可用 --only 选择）：
  A. 训练曲线对比（metrics 4合1 + loss 4合1）
  B. 消融分析（B1 提升柱状图 / B2 性能雷达图 / B3 收敛速度 / B_消融表 CSV+MD）
  C. 检测可视化对比（val_batch0 GT + 四模型 pred 竖排拼图）
  D. PR/F1/混淆矩阵 中文化拼接

扩展性：所有"哪些实验/配色/路径"由 YAML 配置；重做实验只需改 YAML，不改代码。

用法（在 fce-yolo 环境下）：
    conda activate fce-yolo
    cd <项目根>/fce-yolo
    python script/paper_figs.py                # 产出全部
    python script/paper_figs.py --only A,B     # 仅产出 A、B
    python script/paper_figs.py --config script/xxx.yaml   # 指定其他配置

数据真实性红线：所有数值严格来自 results.csv 的 best 轮（idxmax mAP50-95），
绝不编造。GFLOPs/Params 从 best.pt 真实计算，失败降级为 N/A。
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
        e["abs_dir"] = (PAPER_ROOT / e["dir"]).resolve()
        exp_list.append(e)
    exp_list.sort(key=lambda x: x["order"])

    settings = cfg["settings"]
    settings["abs_out_dir"] = (PAPER_ROOT / settings["out_dir"]).resolve()
    return exp_list, settings


def load_all_dfs(exp_list):
    """批量加载所有实验的 results.csv，返回 {key: DataFrame}。"""
    dfs = {}
    for e in exp_list:
        csv = e["abs_dir"] / "results.csv"
        if not csv.exists():
            print(f"⚠ 缺 results.csv: {csv}，跳过 {e['key']}")
            continue
        dfs[e["key"]] = load_results(csv)
    return dfs


# ============================================================
# 中文字体配置
# ============================================================

def setup_cn_font(dpi=300):
    """配置 matplotlib 中文字体，避免中文渲染成方框。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidates = ["Microsoft YaHei", "SimHei", "Microsoft YaHei UI", "SimSun", "KaiTi"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)

    plt.rcParams.update({
        "font.sans-serif": [chosen] if chosen else candidates,
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
    if chosen:
        print(f"✓ 中文字体: {chosen}")
    else:
        print("⚠ 未找到候选中文字体，中文可能显示为方框。")
    return plt


# ============================================================
# 通用：按 order 排序的实验元数据访问
# ============================================================

def _ordered(dfs, exp_list):
    """从 dfs 中按 exp_list 的 order 顺序取出 (key, df, meta)。"""
    out = []
    for e in exp_list:
        if e["key"] in dfs:
            out.append((e["key"], dfs[e["key"]], e))
    return out


# ============================================================
# 产出 A：训练曲线对比（metrics 4合1 + loss 4合1）
# ============================================================

def _plot_4panels(dfs, exp_list, col_config, save_path, fig_title, settings):
    """通用 2x2 对比绘图。

    col_config: [(csv列名, 子图标题, y轴标签), ...] 4 项
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
        ax.set_xlabel("训练轮次 Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.25)
    fig.suptitle(fig_title, fontsize=15, fontweight="bold", y=1.0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"✓ 已保存: {save_path}")


def produce_A(dfs, exp_list, settings):
    """A：训练曲线对比图（metrics + loss 各一张）"""
    print("\n" + "=" * 60)
    print("产出 A：训练曲线对比图")
    print("=" * 60)
    out = settings["abs_out_dir"] / "A_训练曲线"

    metric_cfg = [
        ("metrics/mAP50-95(B)", "mAP@[50-95] 对比", "mAP@[50-95]"),
        ("metrics/mAP50(B)", "mAP@50 对比", "mAP@50"),
        ("metrics/precision(B)", "精确率 Precision 对比", "Precision"),
        ("metrics/recall(B)", "召回率 Recall 对比", "Recall"),
    ]
    _plot_4panels(
        dfs, exp_list, metric_cfg,
        out / "A1_metrics_4合1对比曲线.png",
        "四模型训练过程指标对比（FCE-YOLOv11 消融，m 规模）",
        settings,
    )

    loss_cfg = [
        ("train/box_loss", "训练 Box Loss", "Box Loss"),
        ("train/cls_loss", "训练 Cls Loss", "Cls Loss"),
        ("train/dfl_loss", "训练 DFL Loss", "DFL Loss"),
        ("val/box_loss", "验证 Box Loss", "Box Loss"),
    ]
    _plot_4panels(
        dfs, exp_list, loss_cfg,
        out / "A2_loss_4合1对比曲线.png",
        "四模型训练/验证损失对比（FCE-YOLOv11 消融，m 规模）",
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
    """B1：消融提升柱状图（mAP50-95 递进 + Δ增量标注）"""
    print("\n--- B1：消融提升柱状图 ---")
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

    # 柱顶标绝对值
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.2f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    # 柱间标 Δ 增量（相对前一根）
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
    ax.set_title("消融实验：各模块对 mAP@[50-95] 的递进提升",
                 fontsize=14, fontweight="bold")
    ymin = min(vals) - 5
    ax.set_ylim(ymin, max(vals) + 6)
    ax.grid(True, axis="y", alpha=0.3)
    out = settings["abs_out_dir"] / "B_消融分析"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "B1_消融提升柱状图.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"✓ 已保存: {p}")


def produce_B2(dfs, exp_list, settings):
    """B2：综合性能雷达图（P/R/mAP50/mAP50-95 四维）"""
    print("\n--- B2：综合性能雷达图 ---")
    plt = setup_cn_font(settings["dpi"])
    import numpy as np

    dims = ["Precision", "Recall", "mAP@50", "mAP@[50-95]"]
    keys = ["precision", "recall", "mAP50", "mAP50_95"]
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

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
    ax.set_title("综合性能雷达图（best 指标，四模型对比）",
                 fontsize=14, fontweight="bold", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), framealpha=0.9)
    ax.grid(True, alpha=0.4)
    out = settings["abs_out_dir"] / "B_消融分析"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "B2_综合性能雷达图.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"✓ 已保存: {p}")


def produce_B3(dfs, exp_list, settings):
    """B3：收敛速度对比（mAP50-95 首次达阈值所需轮数）"""
    print("\n--- B3：收敛速度对比 ---")
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
            print(f"  ⚠ {meta['display']} 未达到阈值 {thr}，记为总轮数")
            ep = int(df["epoch"].iloc[-1])
        else:
            ep = int(reached.iloc[0]["epoch"])
        names.append(meta["display"])
        epochs.append(ep)
        colors.append(meta["color"])

    if not epochs:
        print("  ⚠ 无可用数据，跳过 B3")
        return

    fig, ax = plt.subplots(figsize=(10, 5.5), tight_layout=True)
    y = np.arange(len(epochs))
    bars = ax.barh(y, epochs, color=colors, edgecolor="black", linewidth=0.8, height=0.55)
    for b, ep in zip(bars, epochs):
        ax.text(ep + 2, b.get_y() + b.get_height() / 2, f"{ep} 轮",
                va="center", fontsize=11, fontweight="bold")

    # 标注最快相对最慢的提速比
    if len(epochs) >= 2:
        fastest, slowest = min(epochs), max(epochs)
        speedup = (1 - fastest / slowest) * 100
        ax.text(0.98, 0.04,
                f"最快模型相对最慢提速 {speedup:.0f}%\n"
                f"(阈值 mAP@[50-95]={thr})",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, color="#C62828",
                bbox=dict(boxstyle="round,pad=0.4", fc="#FFF3E0", ec="#FF6B00"))

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()  # ① 在最上
    ax.set_xlabel(f"首次达到 mAP@[50-95]≥{thr} 所需训练轮次", fontsize=12)
    ax.set_title("收敛速度对比（越短越快）", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    out = settings["abs_out_dir"] / "B_消融分析"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "B3_收敛速度对比.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"✓ 已保存: {p}")


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
    """B 消融表：CSV + Markdown（含 Params/GFLOPs + Δ提升）"""
    print("\n--- B 消融表：CSV + Markdown ---")
    rows = []
    prev_map = None
    for key, df, meta in _ordered(dfs, exp_list):
        m = _best_metrics(df)
        if m is None:
            continue
        cur = round(m["mAP50_95"] * 100, 2)
        delta = "—" if prev_map is None else f"+{cur - prev_map:.2f}"
        prev_map = cur

        weights = meta["abs_dir"] / "weights" / "best.pt"
        params_m, gflops = (None, None)
        if weights.exists():
            print(f"  计算 {meta['display']} 复杂度 ...")
            try:
                params_m, gflops = _compute_model_complexity(weights, settings["imgsz"])
            except Exception as ex:
                print(f"  ⚠ 复杂度计算失败: {ex}")
        else:
            print(f"  ⚠ 缺 best.pt: {weights}")

        rows.append({
            "序号": f"{'①②③④'[meta['order']-1]}" if meta["order"] <= 4 else str(meta["order"]),
            "模型": meta["display"],
            "FCE模块": meta["fce_module"],
            "损失": meta["loss"],
            "best轮次": m["epoch"],
            "Precision": round(m["precision"] * 100, 2),
            "Recall": round(m["recall"] * 100, 2),
            "mAP50": round(m["mAP50"] * 100, 2),
            "mAP50-95": cur,
            "ΔmAP50-95": delta,
            "Params(M)": round(params_m, 2) if params_m else "N/A",
            "GFLOPs": round(gflops, 1) if gflops else "N/A",
        })

    df_out = pd.DataFrame(rows)
    out = settings["abs_out_dir"] / "B_消融分析"
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "B_消融实验结果表.csv"
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✓ 已保存: {csv_path}")

    # Markdown 表格
    md = ["# 表4-1 消融实验结果（m 规模，imgsz=1280，best 指标）\n",
          "> best 指标定义：验证集 mAP50-95 最高那一轮（YOLO 标准报告方式）\n",
          "> 训练配置：AdamW, lr0=0.001, batch=32, 余弦退火, 300 epochs, 两阶段\n"]
    cols = list(df_out.columns)
    md.append("| " + " | ".join(cols) + " |")
    md.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in df_out.iterrows():
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    md_path = out / "B_消融实验结果表.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"✓ 已保存: {md_path}")
    return df_out


def produce_B(dfs, exp_list, settings):
    """B：消融分析全套"""
    print("\n" + "=" * 60)
    print("产出 B：消融分析")
    print("=" * 60)
    produce_B1(dfs, exp_list, settings)
    produce_B2(dfs, exp_list, settings)
    produce_B3(dfs, exp_list, settings)
    produce_B_table(dfs, exp_list, settings)


# ============================================================
# 产出 C：检测可视化对比（val_batch0 GT + 四模型 pred）
# ============================================================

def produce_C(dfs, exp_list, settings):
    """C：val_batch0 预测图竖排对比"""
    print("\n" + "=" * 60)
    print("产出 C：检测可视化对比图")
    print("=" * 60)
    from PIL import Image, ImageDraw, ImageFont

    panels = [("真值标注 GT (val_batch0_labels)",
               exp_list[0]["abs_dir"] / "val_batch0_labels.jpg")]
    for key, df, meta in _ordered(dfs, exp_list):
        panels.append((f"{meta['display']} (val_batch0_pred)",
                       meta["abs_dir"] / "val_batch0_pred.jpg"))

    for title, p in panels:
        if not p.exists():
            print(f"⚠ 缺图: {p}")
            return

    imgs = [Image.open(p) for _, p in panels]
    w = max(im.width for im in imgs)
    title_h, pad = 48, 8
    total_h = sum(im.height for im in imgs) + (title_h + pad) * len(imgs) + pad
    canvas = Image.new("RGB", (w + pad * 2, total_h), "white")

    font = None
    for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
               "C:/Windows/Fonts/msyhbd.ttc"]:
        if Path(fp).exists():
            font = ImageFont.truetype(fp, 30)
            break
    if font is None:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)
    y = pad
    for (title, _), im in zip(panels, imgs):
        draw.text((pad, y + 8), title, fill="black", font=font)
        y += title_h
        canvas.paste(im, (pad, y))
        y += im.height + pad

    out = settings["abs_out_dir"] / "C_检测对比"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "C_val_batch0_四模型对比.png"
    canvas.save(p, "PNG", dpi=(settings["dpi"], settings["dpi"]))
    print(f"✓ 已保存: {p}")


# ============================================================
# 产出 D：PR/F1/混淆矩阵 中文化拼接
# ============================================================

def _hstack_with_titles(panels, out_path, fig_title, settings, subtitle_h=60):
    """横向拼接图片并加中文小标题。"""
    from PIL import Image, ImageDraw, ImageFont
    for _, p in panels:
        if not p.exists():
            print(f"⚠ 缺图: {p}")
            return False
    imgs = [Image.open(p) for _, p in panels]
    h = min(im.height for im in imgs)
    imgs = [im.resize((int(im.width * h / im.height), h)) for im in imgs]

    title_h, pad = 50, 10
    total_w = sum(im.width for im in imgs) + pad * (len(imgs) + 1)
    total_h = h + title_h + pad * 3
    canvas = Image.new("RGB", (total_w, total_h), "white")

    font = None
    for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
        if Path(fp).exists():
            font = ImageFont.truetype(fp, 28)
            break
    if font is None:
        font = ImageFont.load_default()
    title_font = None
    for fp in ["C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/msyh.ttc"]:
        if Path(fp).exists():
            title_font = ImageFont.truetype(fp, 34)
            break
    if title_font is None:
        title_font = font

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
    print(f"✓ 已保存: {out_path}")
    return True


def produce_D(dfs, exp_list, settings):
    """D：PR/F1/混淆矩阵 中文化拼接"""
    print("\n" + "=" * 60)
    print("产出 D：PR/F1/混淆矩阵 中文化拼接")
    print("=" * 60)
    out = settings["abs_out_dir"] / "D_PR混淆"
    items = _ordered(dfs, exp_list)

    # D1 PR
    _hstack_with_titles(
        [(m["display"], m["abs_dir"] / "BoxPR_curve.png") for _, _, m in items],
        out / "D1_PR曲线_四模型对比.png",
        "四模型 PR 曲线对比（m 规模）", settings,
    )
    # D2 F1
    _hstack_with_titles(
        [(m["display"], m["abs_dir"] / "BoxF1_curve.png") for _, _, m in items],
        out / "D2_F1曲线_四模型对比.png",
        "四模型 F1 曲线对比（m 规模）", settings,
    )
    # D3 混淆矩阵（取 order 最大的，即完整 FCE）
    last = items[-1][2]
    cn = settings.get("class_names", [])
    cls_note = "，".join(f"{i}={n}" for i, n in enumerate(cn)) if cn else ""
    _hstack_with_titles(
        [(f"{last['display']}（归一化混淆矩阵）",
          last["abs_dir"] / "confusion_matrix_normalized.png")],
        out / "D3_FCE混淆矩阵.png",
        f"{last['display']} 归一化混淆矩阵（{cls_note}）", settings,
    )


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="论文图表生成（FCE-YOLOv11 消融）")
    parser.add_argument(
        "--config", default=str(Path(__file__).parent / "paper_figs_config.yaml"),
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--only", default="", type=str,
        help="只产出指定项，逗号分隔，如 A,B,C,D；默认全部",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"✗ 配置文件不存在: {config_path}")
        sys.exit(1)

    print("=" * 60)
    print(f"配置文件: {config_path}")
    print("=" * 60)
    exp_list, settings = load_config(config_path)
    print(f"  实验组: {[e['key'] for e in exp_list]}")
    print(f"  产出目录: {settings['abs_out_dir']}")

    dfs = load_all_dfs(exp_list)
    if not dfs:
        print("✗ 无可用数据，退出")
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
    print(f"✅ 全部完成。产出目录: {settings['abs_out_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

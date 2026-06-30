"""
论文交融实验图表生成模块（FCE-YOLOv11）

基于本机已有训练产物（results.csv / best.pt / 训练期图片），产出论文级图表：
  A. 训练曲线对比图（metrics 4合1 + loss 4合1）
  B. 消融指标表格（Params/GFLOPs + best 指标，Markdown + CSV）
  C. 检测可视化对比图（val_batch 竖排拼图）
  D. PR/F1/混淆矩阵中文化处理

数据真实性红线：所有数值严格来自 results.csv 的 best 轮（idxmax mAP50-95），
绝不编造；fce_wiou_m（完整FCE+WIoU）行明确留占位符，待训练机回传。

用法（在 fce-yolo 环境下，项目根目录运行）：
    python script/paper_plots.py           # 产出全部
    python script/paper_plots.py --only A  # 仅产出 A
    python script/paper_plots.py --only B,C
"""

import argparse
import contextlib
import io
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# 项目根目录入 path，便于复用 script.analysis
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from script.analysis import load_results, extract_metrics  # noqa: E402

# ==================== 全局路径与配色常量 ====================

# 论文项目根（Visual Guidance Robotic Arm）
PAPER_ROOT = Path(__file__).resolve().parents[2]

# m 规模三组实验目录（主推规模）
EXP_ROOT = PAPER_ROOT / "实验" / "3月测试" / "baselinevsbifpnvsfce_m_300"

# 三组结果目录与权重
RUNS = {
    "baseline": {
        "dir": EXP_ROOT / "baseline_yolo11m",
        "csv": EXP_ROOT / "baseline_yolo11m" / "results.csv",
        "weights": EXP_ROOT / "baseline_yolo11m" / "weights" / "best.pt",
    },
    "bifpn": {
        "dir": EXP_ROOT / "bifpn_m_stage2",
        "csv": EXP_ROOT / "bifpn_m_stage2" / "results.csv",
        "weights": EXP_ROOT / "bifpn_m_stage2" / "weights" / "best.pt",
    },
    "fce": {
        "dir": EXP_ROOT / "fce_m_stage2",
        "csv": EXP_ROOT / "fce_m_stage2" / "results.csv",
        "weights": EXP_ROOT / "fce_m_stage2" / "weights" / "best.pt",
    },
}

# 论文用显示名（中文）与配色（沿用 config.py MODEL_CONFIGS）
DISPLAY = {
    "baseline": "YOLOv11（基线）",
    "bifpn": "+BiFPN",
    "fce": "FCE（完整）",
}
COLORS = {
    "baseline": "#0BDBEB",  # 青
    "bifpn": "#042AFF",     # 蓝
    "fce": "#FF6B00",       # 橙
}
LINESTYLES = {"baseline": "--", "bifpn": "-.", "fce": "-"}

# 产出根目录
OUT_ROOT = PAPER_ROOT / "图表" / "交融实验产出"
OUT_DIRS = {
    "A": OUT_ROOT / "训练曲线",
    "B": OUT_ROOT / "消融表",
    "C": OUT_ROOT / "检测对比",
    "D": OUT_ROOT / "PR混淆",
}

# 数据集类别名（从 best.pt 读出，本任务固定 2 类）
# circle_base = 圆形底座 / square_workpiece = 方形工件
CLASS_NAMES_CN = {0: "圆形底座", 1: "方形工件"}


# ==================== 中文字体配置 ====================

def setup_cn_font():
    """配置 matplotlib 中文字体，避免中文渲染成方框。

    Windows 优先用系统自带的中文字体；找不到则回退并提示。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 候选字体（按优先级），均为 Windows 常见中文字体
    candidates = ["Microsoft YaHei", "SimHei", "Microsoft YaHei UI", "SimSun", "KaiTi"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)

    plt.rcParams.update({
        "font.sans-serif": [chosen] if chosen else candidates,
        "font.family": "sans-serif",
        "axes.unicode_minus": False,   # 负号正常显示
        "font.size": 11,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 2,
        "figure.dpi": 150,
        "savefig.dpi": 300,            # 论文级分辨率
    })
    if chosen:
        print(f"✓ 中文字体: {chosen}")
    else:
        print("⚠ 未找到候选中文字体，中文可能显示为方框。请安装 Microsoft YaHei / SimHei")
    return plt


# ==================== 产出 A：训练曲线对比 ====================

def _metric_ylabels():
    """metrics 4合1 的列配置：(csv列名, 中文标题, y轴标签)"""
    return [
        ("metrics/mAP50-95(B)", "mAP@[50-95] 对比", "mAP@[50-95]"),
        ("metrics/mAP50(B)", "mAP@50 对比", "mAP@50"),
        ("metrics/precision(B)", "精确率 Precision 对比", "Precision"),
        ("metrics/recall(B)", "召回率 Recall 对比", "Recall"),
    ]


def _loss_subplots():
    """loss 4合1 的列配置：4个子图（train 三个 loss + val box loss）"""
    return [
        ("train/box_loss", "训练 Box Loss", "Box Loss"),
        ("train/cls_loss", "训练 Cls Loss", "Cls Loss"),
        ("train/dfl_loss", "训练 DFL Loss", "DFL Loss"),
        ("val/box_loss", "验证 Box Loss", "Box Loss"),
    ]


def plot_comparison(dfs, col_config, save_path, fig_title):
    """通用 2x2 对比绘图（共用配色/线型/字体）。

    Args:
        dfs: {key: DataFrame}
        col_config: [(csv列名, 子图标题, y轴标签), ...] 4 项
        save_path: 输出 png
        fig_title: 整图标题
    """
    plt = setup_cn_font()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), tight_layout=True)
    for idx, (col, subtitle, ylabel) in enumerate(col_config):
        ax = axes[idx // 2, idx % 2]
        for k, df in dfs.items():
            if col not in df.columns:
                continue
            ax.plot(
                df["epoch"], df[col],
                color=COLORS[k], linestyle=LINESTYLES[k],
                label=DISPLAY[k], linewidth=2,
            )
        ax.set_title(subtitle, fontsize=13, fontweight="bold")
        ax.set_xlabel("训练轮次 Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.25)
    fig.suptitle(fig_title, fontsize=15, fontweight="bold", y=1.0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ 已保存: {save_path}")


def produce_A():
    """产出 A：训练曲线对比图（metrics + loss 各一张）"""
    print("\n" + "=" * 60)
    print("产出 A：训练曲线对比图")
    print("=" * 60)
    keys = ["baseline", "bifpn", "fce"]
    dfs = {k: load_results(RUNS[k]["csv"]) for k in keys}

    out = OUT_DIRS["A"]
    # A1：metrics
    plot_comparison(
        dfs, _metric_ylabels(),
        out / "A1_metrics_对比曲线.png",
        "m 规模三模型训练过程指标对比（FCE-YOLOv11 消融）",
    )
    # A2：loss
    plot_comparison(
        dfs, _loss_subplots(),
        out / "A2_loss_对比曲线.png",
        "m 规模三模型训练/验证损失对比（FCE-YOLOv11 消融）",
    )


# ==================== 产出 B：消融指标表格 ====================

def _compute_model_complexity(weights_path, imgsz=1280):
    """计算 Params 与 GFLOPs（CPU 可跑）。

    ultralytics 8.3.x 的 model.info() 返回 None（直接打印），因此这里：
      - params 用 sum(p.numel()) 直接统计
      - gflops 用 torch_utils.get_flops(model, imgsz)
    返回 (params_M, gflops)。失败返回 (None, None)。
    """
    import contextlib, io
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    torch.set_num_threads(4)  # CPU 限线程，避免资源告警

    m = YOLO(str(weights_path))
    inner = m.model
    n_params = sum(p.numel() for p in inner.parameters())
    with contextlib.redirect_stdout(io.StringIO()):  # get_flops 会打印，静默之
        gflops = get_flops(inner, imgsz=imgsz)
    return n_params / 1e6, gflops


def produce_B():
    """产出 B：消融指标表格（MD + CSV）"""
    print("\n" + "=" * 60)
    print("产出 B：消融指标表格")
    print("=" * 60)

    rows = []
    # 行①②③：真实数据
    spec = [
        ("①", "baseline", "YOLOv11 基线", "标准结构 + CIoU", "—"),
        ("②", "bifpn",    "+BiFPN",        "BiFPN 加权融合 + CIoU", "F"),
        ("③", "fce",      "+BiFPN+注意力", "BiFPN + BiCoordCrossAtt×2 + CIoU", "F+C"),
    ]
    for idx, key, name, improve, module in spec:
        df = load_results(RUNS[key]["csv"])
        mt = extract_metrics(df)
        # best 指标（按 mAP50-95 最高轮）
        best_ep = mt["best_map50_95_epoch"]
        best_row = df.loc[df["metrics/mAP50-95(B)"].idxmax()]
        p50 = best_row["metrics/mAP50(B)"]
        p5095 = best_row["metrics/mAP50-95(B)"]
        prec = best_row["metrics/precision(B)"]
        rec = best_row["metrics/recall(B)"]
        # 复杂度
        print(f"  计算 {name} 复杂度 (best.pt) ...")
        params_m, gflops = _compute_model_complexity(RUNS[key]["weights"], imgsz=1280)
        rows.append({
            "序号": idx, "模型": name, "改进": improve, "FCE模块": module,
            "损失": "CIoU",
            "best轮次": best_ep,
            "Precision": round(prec * 100, 2),
            "Recall": round(rec * 100, 2),
            "mAP50": round(p50 * 100, 2),
            "mAP50-95": round(p5095 * 100, 2),
            "Params(M)": round(params_m, 2) if params_m else "N/A",
            "GFLOPs": round(gflops, 1) if gflops else "N/A",
        })

    # 行④：占位（待训练机 fce_wiou_m 回传）
    rows.append({
        "序号": "④", "模型": "+BiFPN+注意力+WIoU（完整FCE）",
        "改进": "BiFPN + BiCoordCrossAtt×2 + WIoU",
        "FCE模块": "F+C+E", "损失": "WIoU",
        "best轮次": "—",
        "Precision": "⏳待测", "Recall": "⏳待测",
        "mAP50": "⏳待测", "mAP50-95": "⏳待测",
        "Params(M)": "≈同③", "GFLOPs": "≈同③",
    })

    df_out = pd.DataFrame(rows)
    out_dir = OUT_DIRS["B"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "B_消融实验结果表.csv"
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✓ 已保存: {csv_path}")

    # 手动生成 Markdown 表格（不依赖 tabulate，避免给环境加包）
    md_path = out_dir / "B_消融实验结果表.md"
    cols = list(df_out.columns)
    lines = []
    lines.append("# 表4-1 消融实验结果（m 规模，imgsz=1280，best 指标）\n")
    lines.append("> 训练配置：AdamW, lr0=0.001, batch=32, 余弦退火, 300 epochs, "
                 "两阶段(stage1 warmup 50ep freeze=10 + stage2 finetune 250ep), "
                 "close_mosaic=20, patience=50")
    lines.append(">")
    lines.append("> 数据集：haixi_jixieshou/yolo_dataset（2 类：圆形底座、方形工件）")
    lines.append("> best 指标定义：验证集 mAP50-95 最高那一轮（YOLO 标准报告方式）\n")
    # 表头
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df_out.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    lines.append("\n## 说明")
    lines.append("- ①→② 证明 BiFPN 有效；②→③ 证明 BiCoordCrossAtt 注意力有效；"
                 "③→④ 证明 WIoU 边缘约束有效（待测）")
    lines.append("- 第④行 ⏳ 为占位符，需在训练机执行后回传 `runs/detect/fce_wiou_m_stage2/`\n")
    lines.append("## 训练机补跑命令")
    lines.append("```bash")
    lines.append("cd ~/workspace/my_project/fce-yolo")
    lines.append("python script/train.py fce_wiou --scale m --iou-type WIoU")
    lines.append("```")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✓ 已保存: {md_path}")
    return df_out


# ==================== 产出 C：检测可视化对比 ====================

def produce_C():
    """产出 C：val_batch0 预测图竖排对比（GT + 三模型）"""
    print("\n" + "=" * 60)
    print("产出 C：检测可视化对比图")
    print("=" * 60)
    from PIL import Image, ImageDraw, ImageFont

    # 四张图：labels(GT) + 三模型 pred。同一 seed 同一批次，可对比
    panels = [
        ("真值标注 GT (val_batch0_labels)", RUNS["baseline"]["dir"] / "val_batch0_labels.jpg"),
        ("YOLOv11 基线 (val_batch0_pred)", RUNS["baseline"]["dir"] / "val_batch0_pred.jpg"),
        ("+BiFPN (val_batch0_pred)",        RUNS["bifpn"]["dir"] / "val_batch0_pred.jpg"),
        ("FCE 完整 (val_batch0_pred)",      RUNS["fce"]["dir"] / "val_batch0_pred.jpg"),
    ]
    for title, p in panels:
        if not p.exists():
            print(f"⚠ 缺图: {p}")
            return

    imgs = [Image.open(p) for _, p in panels]
    w = max(im.width for im in imgs)
    # 每张图上方留标题条
    title_h = 48
    pad = 8
    total_h = sum(im.height for im in imgs) + (title_h + pad) * len(imgs) + pad
    canvas = Image.new("RGB", (w + pad * 2, total_h), "white")

    # 字体（中文）
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

    out_dir = OUT_DIRS["C"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "C_检测对比_val_batch0.png"
    canvas.save(out_path, "PNG", dpi=(300, 300))
    print(f"✓ 已保存: {out_path}")


# ==================== 产出 D：PR/F1/混淆矩阵中文化 ====================

def _hstack_with_titles(panels, out_path, fig_title, subtitle_h=60):
    """横向拼接图片并在每张上方加中文小标题。

    panels: [(subtitle, path), ...]
    """
    from PIL import Image, ImageDraw, ImageFont
    for _, p in panels:
        if not p.exists():
            print(f"⚠ 缺图: {p}")
            return False
    imgs = [Image.open(p) for _, p in panels]
    # 统一到同一高度（取最小，避免失真太多）
    h = min(im.height for im in imgs)
    imgs = [im.resize((int(im.width * h / im.height), h)) for im in imgs]

    title_h = 50
    pad = 10
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

    draw = ImageDraw.Draw(canvas)
    # 大标题
    title_font = None
    for fp in ["C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/msyh.ttc"]:
        if Path(fp).exists():
            title_font = ImageFont.truetype(fp, 34)
            break
    if title_font is None:
        title_font = font
    draw.text((pad, 8), fig_title, fill="black", font=title_font)

    y = title_h
    x = pad
    for (subtitle, _), im in zip(panels, imgs):
        draw.text((x, y), subtitle, fill="black", font=font)
        canvas.paste(im, (x, y + 28))
        x += im.width + pad

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "PNG", dpi=(300, 300))
    print(f"✓ 已保存: {out_path}")
    return True


def produce_D():
    """产出 D：PR/F1/混淆矩阵 中文化拼接

    说明：results.csv 只有逐 epoch 聚合指标，不含逐阈值 PR 数据点；
    本机又无数据集无法重跑 val() 生成中文 PR 曲线。
    故采用「三模型图横向拼接 + 中文标题」方案：曲线本身保留 YOLO 原图，
    顶部加中文小标题与整体大标题，论文图注用中文解释。
    """
    print("\n" + "=" * 60)
    print("产出 D：PR/F1/混淆矩阵 中文化拼接")
    print("=" * 60)
    out_dir = OUT_DIRS["D"]

    # D1：三模型 PR 曲线横向拼接
    _hstack_with_titles(
        [
            ("YOLOv11 基线", RUNS["baseline"]["dir"] / "BoxPR_curve.png"),
            ("+BiFPN",        RUNS["bifpn"]["dir"] / "BoxPR_curve.png"),
            ("FCE 完整",      RUNS["fce"]["dir"] / "BoxPR_curve.png"),
        ],
        out_dir / "D1_PR曲线_三模型对比.png",
        "三模型 PR 曲线对比（m 规模）",
    )

    # D2：三模型 F1 曲线横向拼接
    _hstack_with_titles(
        [
            ("YOLOv11 基线", RUNS["baseline"]["dir"] / "BoxF1_curve.png"),
            ("+BiFPN",        RUNS["bifpn"]["dir"] / "BoxF1_curve.png"),
            ("FCE 完整",      RUNS["fce"]["dir"] / "BoxF1_curve.png"),
        ],
        out_dir / "D2_F1曲线_三模型对比.png",
        "三模型 F1 曲线对比（m 规模）",
    )

    # D3：FCE 的归一化混淆矩阵（单图，加中文标题；类别名见论文图注）
    _hstack_with_titles(
        [("FCE 完整（归一化混淆矩阵）",
          RUNS["fce"]["dir"] / "confusion_matrix_normalized.png")],
        out_dir / "D3_FCE_混淆矩阵.png",
        "FCE 归一化混淆矩阵（类别：0=圆形底座, 1=方形工件）",
    )


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(description="论文交融实验图表生成")
    parser.add_argument(
        "--only", default="", type=str,
        help="只产出指定项，逗号分隔，如 A,B,C,D；默认全部",
    )
    args = parser.parse_args()
    targets = {t.strip().upper() for t in args.only.split(",") if t.strip()} or {"A", "B", "C", "D"}

    if "A" in targets:
        produce_A()
    if "B" in targets:
        produce_B()
    if "C" in targets:
        produce_C()
    if "D" in targets:
        produce_D()

    print("\n" + "=" * 60)
    print(f"✅ 全部完成。产出目录: {OUT_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()

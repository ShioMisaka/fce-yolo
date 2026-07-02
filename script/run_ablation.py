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
    """从配方构建 TrainConfig（含 stage1/stage2），并按 iou_override 设置 iou_type。

    - shared 中 TrainConfig 认识的字段直接填入；不认识的（seed/deterministic/degrees
      等共享超参）通过 extra_args 注入，由 to_dict() 展开到 train() kwargs。
    - stage1/stage2 必须写入 TrainConfig：trainer._build_train_args 从 self.config.stage1/stage2
      取阶段配置（而非从 model_cfg），所以这里不注入会导致 asdict(None) 崩溃。
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
    # 关键：trainer 从 config.stage1/stage2 取阶段配置，必须用配方的 StageConfig 覆盖默认值
    config.stage1 = StageConfig(**recipe["stage1"])
    config.stage2 = StageConfig(**recipe["stage2"])
    return config


# ============================================================
# 预览
# ============================================================

def print_matrix_preview(recipe: dict, scales: list, models: list):
    """Print the experiment matrix + unified-variable summary for review."""
    print("\n" + "=" * 80)
    print("Fair ablation matrix preview")
    print("=" * 80)

    # matrix
    print(f"\nscale x model ({len(scales) * len(models)} runs):")
    header = "  ".join(f"{m:<14}" for m in models)
    print(f"{'scale':<8}{header}")
    for s in scales:
        row = f"{s:<8}" + "  ".join(f"{'x':<14}" for _ in models)
        print(row)

    # unified variables
    sh = recipe["shared"]
    print(f"\nunified variables:")
    print(f"  imgsz={sh.get('imgsz')}, batch={sh.get('batch')}, optimizer={sh.get('optimizer')}, lr0(stage2)={recipe['stage2']['lr0']}")
    print(f"  seed={sh.get('seed')}, deterministic={sh.get('deterministic')}, degrees={sh.get('degrees')}, cache={sh.get('cache')}")
    print(f"  two-stage: stage1({recipe['stage1']['epochs']}ep, freeze={recipe['freeze']}) + stage2({recipe['stage2']['epochs']}ep) = {recipe.get('total_epochs', recipe['stage1']['epochs']+recipe['stage2']['epochs'])}ep")
    print(f"  IoU mapping: {recipe.get('iou_override', {})}")
    print(f"  export_root: {PROJECT_ROOT / recipe.get('export_root', 'fair_runs')}")
    print(f"  output: {PAPER_ROOT / recipe['output_root']}")
    print("\n(--dry-run previews only; drop it to start training)")
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

    策略说明（review I1）：0.9 阈值同时覆盖"正常早停"和"stage2 训练到 90% 后被中断"两种
    情形——后者按"接受现状、视作完成"处理。理由：best.pt 已在 best 轮保存，论文用 best
    指标（非末轮），所以即便 stage2 末段缺失也不影响 best 指标采集；重训反而浪费数小时。
    若需严格"必须训完 stage2.epochs 轮"，把 0.9 改为 1.0 即可（但早停触发时永远 < epochs）。
    """
    s2_full = _stage2_run_dir(model_key, scale)
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
        s2 = _stage2_run_dir(model_key, scale)
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
# 结果收集与归档
# ============================================================

def _stage2_run_dir(model_key: str, scale: str) -> Path:
    """返回 runs/detect 下某 (scale, model) 的 stage2 目录路径。

    统一 baseline 特例（其 result_pattern 无 _stage2 后缀，但公平注入后训练会生成
    baseline_yolo11{scale}_stage2）。集中在此，避免多处重复。
    """
    if model_key == "baseline":
        return Path("runs/detect") / f"baseline_yolo11{scale}_stage2"
    base_cfg = get_model_config(model_key)
    return Path("runs/detect") / base_cfg.result_pattern.format(scale=scale)


def _resolve_stage2_source(model_key: str, scale: str, source: str, recipe: dict) -> Path:
    """Resolve where to read stage2 results from.

    source:
      - "runs"      -> fce-yolo/runs/detect/<exp>_stage2 (training-machine default)
      - "fair_runs" -> fce-yolo/<export_root>/<scale>/<0X_name>/stage2 (the English
                       export package, used locally after unpacking a workstation tar)
    """
    if source == "fair_runs":
        export_root = PROJECT_ROOT / recipe.get("export_root", "fair_runs")
        return export_root / scale / get_model_dir_name(model_key, scale) / "stage2"
    # default: runs/detect (relative to PROJECT_ROOT for portability)
    s2 = _stage2_run_dir(model_key, scale)
    if not s2.is_absolute():
        s2 = PROJECT_ROOT / s2
    return s2


def archive_one(scale: str, model_key: str, recipe: dict, source: str = "runs"):
    """把 stage2 结果复制到 output_root/<scale>/0X_<name>/stage2。

    source 见 _resolve_stage2_source。默认仅复制 stage2（copy_stage1: false）；
    目录自包含、可独立归档。
    """
    import shutil
    output_root = PAPER_ROOT / recipe["output_root"]
    scale_dir = output_root / scale
    model_dir_name = get_model_dir_name(model_key, scale)
    dst_model_dir = scale_dir / model_dir_name
    scale_dir.mkdir(parents=True, exist_ok=True)

    s2_src = _resolve_stage2_source(model_key, scale, source, recipe)

    copy_stage1 = recipe.get("copy_stage1", False)

    # 复制 stage2
    dst_s2 = dst_model_dir / "stage2"
    if s2_src.exists():
        if dst_s2.exists():
            shutil.rmtree(dst_s2)
        shutil.copytree(str(s2_src), str(dst_s2))
        print(f"  + stage2 -> {dst_s2.relative_to(PAPER_ROOT)}")
    else:
        print(f"  warn: stage2 source missing: {s2_src}")

    # 可选复制 stage1
    if copy_stage1:
        s1_src = Path(str(s2_src).replace("_stage2", "_stage1"))
        if s1_src.exists():
            dst_s1 = dst_model_dir / "stage1"
            if dst_s1.exists():
                shutil.rmtree(dst_s1)
            shutil.copytree(str(s1_src), str(dst_s1))
            print(f"  + stage1 -> {dst_s1.relative_to(PAPER_ROOT)}")

    return dst_s2


def collect_results(scales: list, models: list, recipe: dict, source: str = "runs") -> dict:
    """遍历 (scale, model)，复制结果并汇总 results.csv 路径。

    Returns:
        {scale: {model_key: {"stage2_dir": Path, "csv": Path, "df": DataFrame, "metrics": dict}}}
    """
    print("\n" + "=" * 80)
    print(f"Stage 2: collect training results (source={source})")
    print("=" * 80)

    all_results = {}
    for scale in scales:
        all_results[scale] = {}
        for model_key in models:
            dst_s2 = archive_one(scale, model_key, recipe, source=source)
            csv = dst_s2 / "results.csv"
            if not csv.exists():
                print(f"  warn: results.csv missing: {csv}")
                continue
            df = load_results(csv)
            all_results[scale][model_key] = {
                "stage2_dir": dst_s2,
                "csv": csv,
                "df": df,
                "metrics": extract_metrics(df),
            }
    return all_results


# ============================================================
# 对比表生成
# ============================================================

# 模型展示名 + 消融序号（与 paper_figs_config 对齐）
# 序号用 M1/M2/M3/M4 而非 ①②③④，避免 Linux 无对应字形时渲染异常
MODEL_DISPLAY = {
    "baseline":   ("M1", "YOLOv11 (baseline)", "CIoU"),
    "bifpn":      ("M2", "+BiFPN",             "CIoU"),
    "fce":        ("M3", "+BiFPN+Attn",        "CIoU"),
    "fce_wiou":   ("M4", "FCE (+WIoU)",        "WIoU"),
}


def compute_params_gflops(best_pt: Path, imgsz: int = 1280) -> tuple:
    """从 best.pt 真实计算 Params/GFLOPs。

    失败返回 (None, None)。
    """
    try:
        from ultralytics import YOLO
        m = YOLO(str(best_pt))
        params = sum(p.numel() for p in m.model.parameters())
        try:
            from ultralytics.utils.torch_utils import get_flops
            flops = get_flops(m.model, imgsz=imgsz)
        except Exception:
            flops = None
        return params / 1e6, flops
    except Exception as e:
        print(f"  ⚠ Params/GFLOPs 计算失败 {best_pt.name}: {e}")
        return None, None


def write_comparison_table(scale: str, scale_results: dict, recipe: dict):
    """为单尺度生成对比表 CSV + MD（按 best 指标，含 Δ 列）。"""
    output_root = PAPER_ROOT / recipe["output_root"]
    cmp_dir = output_root / "comparison"
    cmp_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    prev_map = None
    for model_key in ["baseline", "bifpn", "fce", "fce_wiou"]:
        if model_key not in scale_results:
            continue
        r = scale_results[model_key]
        m = r["metrics"]
        best_pt = r["stage2_dir"] / "weights" / "best.pt"
        params, gflops = (None, None)
        if best_pt.exists():
            params, gflops = compute_params_gflops(best_pt, recipe["shared"].get("imgsz", 1280))
        delta = ""
        if prev_map is not None:
            delta = m["best_map50_95"] - prev_map
        seq, disp, loss = MODEL_DISPLAY[model_key]
        # 口径统一：P/R/mAP50/mAP50-95 全部取自同一个 best mAP50-95 行，
        # 与 paper_figs._best_metrics 对齐，杜绝表内列与列来自不同 epoch 的错位
        # （AGENTS.md §7/§8 数据真实性红线）。
        best_idx = r["df"]["metrics/mAP50-95(B)"].idxmax()
        best_row = r["df"].loc[best_idx]
        rows.append({
            "No": seq,
            "Model": disp,
            "Loss": loss,
            "Best_Epoch": m["best_map50_95_epoch"],
            "P": best_row["metrics/precision(B)"],
            "R": best_row["metrics/recall(B)"],
            "mAP50": best_row["metrics/mAP50(B)"],
            "mAP50_95": best_row["metrics/mAP50-95(B)"],
            "d_mAP50_95": delta,
            "Params(M)": params,
            "GFLOPs": gflops,
            "Total_Ep": recipe["stage1"]["epochs"] + recipe["stage2"]["epochs"],
        })
        prev_map = m["best_map50_95"]

    # CSV
    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = cmp_dir / f"{scale}_comparison_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  + {csv_path.relative_to(PAPER_ROOT)}")

    # MD (with data-integrity disclaimer)
    md_path = cmp_dir / f"{scale}_comparison_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Scale-{scale.upper()} Fair Ablation Comparison\n\n")
        f.write(f"> Source: `main_ablation_fair/{scale}/` stage2 `results.csv` (by best epoch mAP50-95)\n")
        f.write(f"> Reading convention: strict column-name lookup in results.csv (AGENTS.md §8); P/R/mAP50/mAP50-95 are all taken from the same best epoch (max mAP50-95 row), aligned with the paper_figs B table\n")
        f.write(f"> **Data integrity (AGENTS.md §7): the following are real training results, not fabricated; if M1->M4 is not strictly monotonic, it is recorded as-is.**\n\n")
        f.write("| No | Model | Loss | Best_Epoch | P | R | mAP50 | mAP50-95 | d(mAP50-95) | Params(M) | GFLOPs | Total_Ep |\n")
        f.write("|------|------|------|--------|---|---|-------|----------|-------------|-----------|--------|------|\n")

        def fmt(x, d=4):
            if isinstance(x, str) or x is None:
                return "-" if x is None else str(x)
            return f"{x:.{d}f}"

        for row in rows:
            f.write(f"| {row['No']} | {row['Model']} | {row['Loss']} | {row['Best_Epoch']} | "
                    f"{fmt(row['P'])} | {fmt(row['R'])} | {fmt(row['mAP50'])} | "
                    f"{fmt(row['mAP50_95'])} | {fmt(row['d_mAP50_95']) if row['d_mAP50_95']!='' else '-'} | "
                    f"{fmt(row['Params(M)'],2) if row['Params(M)'] else 'N/A'} | "
                    f"{fmt(row['GFLOPs'],1) if row['GFLOPs'] else 'N/A'} | {row['Total_Ep']} |\n")
    print(f"  + {md_path.relative_to(PAPER_ROOT)}")
    return df


def write_cross_scale_summary(all_results: dict, recipe: dict):
    """Cross-scale summary: best-mAP50-95 matrix of each model across n/s/m, to
    inspect scale-stability of the improvement."""
    output_root = PAPER_ROOT / recipe["output_root"]
    cmp_dir = output_root / "comparison"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    rows = []
    for model_key in ["baseline", "bifpn", "fce", "fce_wiou"]:
        row = {"Model": MODEL_DISPLAY[model_key][1]}
        for scale in all_results.keys():
            if model_key in all_results[scale]:
                row[f"{scale}_mAP50_95"] = all_results[scale][model_key]["metrics"]["best_map50_95"]
            else:
                row[f"{scale}_mAP50_95"] = None
        rows.append(row)
    df = pd.DataFrame(rows)
    csv = cmp_dir / "cross_scale_summary.csv"
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    print(f"  ✓ {csv.relative_to(PAPER_ROOT)}")
    return df


# ============================================================
# paper_figs 衔接
# ============================================================

def generate_paper_figs_config(scale: str, recipe: dict) -> Path:
    """为指定尺度生成适配 main_ablation_fair 的 paper_figs_config YAML。

    复用 paper_figs.py 的配置驱动机制，零重复造轮子。
    display/loss 从 MODEL_DISPLAY 派生（单一事实来源，保证表与图例标签一致）；
    color/linestyle/fce_module 为 paper_figs 渲染专用配置。
    """
    cfg_dir = PROJECT_ROOT / "script"
    cfg_path = cfg_dir / f"paper_figs_config_fair_{scale}.yaml"

    # 渲染专用配置（color/linestyle/fce_module 不在 MODEL_DISPLAY 中，此处独立维护）
    _render = {
        "baseline":  {"color": "#0BDBEB", "linestyle": "--",  "fce_module": "-"},
        "bifpn":     {"color": "#042AFF", "linestyle": "-.",  "fce_module": "F"},
        "fce":       {"color": "#FF6B00", "linestyle": ":",   "fce_module": "F+C"},
        "fce_wiou":  {"color": "#E91E63", "linestyle": "-",   "fce_module": "F+C+E"},
    }
    experiments = {}
    for order, key in enumerate(["baseline", "bifpn", "fce", "fce_wiou"], start=1):
        seq, display, loss = MODEL_DISPLAY[key]
        experiments[key] = {
            "dir": f"{recipe['output_root']}/{scale}/{get_model_dir_name(key, scale)}/stage2",
            "display": display,
            "loss": loss,
            "order": order,
            **_render[key],
        }
    settings = {
        "imgsz": recipe["shared"].get("imgsz", 1280),
        # 英文目录名，保证 Linux 工作站能正常创建/打包
        "out_dir": f"{recipe['output_root']}/figures/{scale}",
        "convergence_threshold": 0.75,
        # 类名英文化（避免 Linux 无 CJK 字体时混淆矩阵标注渲染异常）
        "class_names": ["round_base", "square_part"],
        "dpi": 300,
    }
    cfg = {"experiments": experiments, "settings": settings}
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"  + paper_figs config: {cfg_path.relative_to(PROJECT_ROOT)}")
    return cfg_path


def generate_figures(scales: list, recipe: dict):
    """为每个尺度调用 paper_figs.py 出图（A/B/C/D 全套）。"""
    print("\n" + "=" * 80)
    print("Stage 4: generate figures (paper_figs.py)")
    print("=" * 80)
    import subprocess
    for scale in scales:
        cfg_path = generate_paper_figs_config(scale, recipe)
        cmd = [
            sys.executable, "script/paper_figs.py",
            "--config", str(cfg_path),
        ]
        print(f"\n> figures {scale}: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        except subprocess.CalledProcessError as e:
            print(f"  warn: {scale} figure generation failed: {e}")
        except FileNotFoundError:
            print(f"  warn: paper_figs.py not found or subprocess error, skipping {scale}")


# ============================================================
# export 子命令：把 runs/detect 产物打包成英文目录，便于跨平台回传
# ============================================================

def run_export(scales: list, models: list, recipe: dict):
    """把 runs/detect 下各 (scale, model) 的 stage2 复制到 <export_root>/<scale>/。

    用途：在 Linux 训练工作站上训练完成后，运行本子命令把结果整理到一个纯英文
    目录（相对 fce-yolo 仓库根），然后 tar/zip 回传本地，本地再用
    `--skip-train --source fair_runs` 整理+出图。彻底解耦训练机路径（中文 项目目录）
    与结果归档目录。

    每个 (scale, model) 的 stage2 复制到：
        <export_root>/<scale>/<0X_name>/stage2/
    并在该 scale 目录下写一份 MANIFEST.txt 记录源路径、行数、best 指标，便于核对。
    """
    import shutil
    import pandas as pd

    export_root = PROJECT_ROOT / recipe.get("export_root", "fair_runs")
    export_root.mkdir(parents=True, exist_ok=True)

    for scale in scales:
        scale_dir = export_root / scale
        scale_dir.mkdir(parents=True, exist_ok=True)
        manifest_lines = [
            f"# MANIFEST for scale={scale}",
            f"# generated by run_ablation.py export",
            f"# source: runs/detect/<exp>_stage2",
            f"# export root: {export_root}",
            "",
        ]
        for model_key in models:
            model_dir_name = get_model_dir_name(model_key, scale)
            dst_model_dir = scale_dir / model_dir_name
            s2_src = _stage2_run_dir(model_key, scale)
            if not s2_src.is_absolute():
                s2_src = PROJECT_ROOT / s2_src
            dst_s2 = dst_model_dir / "stage2"

            if not s2_src.exists():
                msg = f"  warn: source missing, skipped: {s2_src}"
                print(msg)
                manifest_lines.append(f"[{model_key}] {model_dir_name}: MISSING ({s2_src})")
                continue

            if dst_s2.exists():
                shutil.rmtree(dst_s2)
            shutil.copytree(str(s2_src), str(dst_s2))

            # 记录核对信息：行数 + best 指标
            csv = dst_s2 / "results.csv"
            n_rows = 0
            best_str = "n/a"
            if csv.exists():
                try:
                    df = pd.read_csv(csv)
                    n_rows = len(df)
                    if "metrics/mAP50-95(B)" in df.columns:
                        bi = df["metrics/mAP50-95(B)"].idxmax()
                        br = df.loc[bi]
                        best_str = (
                            f"best@ep{int(br['epoch'])} "
                            f"P={br['metrics/precision(B)']:.4f} "
                            f"R={br['metrics/recall(B)']:.4f} "
                            f"mAP50={br['metrics/mAP50(B)']:.4f} "
                            f"mAP50-95={br['metrics/mAP50-95(B)']:.4f}"
                        )
                except Exception as ex:
                    best_str = f"read error: {ex}"

            print(f"  + {model_dir_name}/stage2 ({n_rows} rows, {best_str})")
            manifest_lines.append(
                f"[{model_key}] {model_dir_name}: {n_rows} rows | {best_str}"
            )

        manifest_path = scale_dir / "MANIFEST.txt"
        manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        print(f"  + {manifest_path.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 80)
    print("Export done.")
    print(f"  export root: {export_root}")
    for scale in scales:
        print(f"  pack scale {scale}:  tar -czf fair_runs_{scale}.tar.gz "
              f"{export_root.relative_to(PROJECT_ROOT)}/{scale}")
    print("=" * 80)


# ============================================================
# README 自动生成
# ============================================================

def generate_readme(recipe: dict, all_results: dict):
    """Generate main_ablation_fair/README.md (recipe summary + real metrics +
    diff vs the legacy main_ablation_m)."""
    output_root = PAPER_ROOT / recipe["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)  # ensure dir exists on full failure
    readme = output_root / "README.md"
    sh = recipe["shared"]

    lines = []
    lines.append("# Fair Ablation (main_ablation_fair)\n")
    lines.append("> Auto-generated by `script/run_ablation.py` from `script/ablation_config.yaml`.\n")
    lines.append("> **Data integrity (AGENTS.md §7)**: the following are real training results, not fabricated; if M1->M4 is not strictly monotonic, it is recorded as-is.\n\n")

    lines.append("## Recipe\n")
    lines.append(f"- scales: {', '.join(recipe['scales'])}")
    lines.append(f"- models: {' / '.join(MODEL_DISPLAY[m][1] for m in recipe['models'])}")
    lines.append(f"- **all two-stage**: stage1({recipe['stage1']['epochs']}ep, freeze={recipe['freeze']}, lr0={recipe['stage1']['lr0']}) + stage2({recipe['stage2']['epochs']}ep, lr0={recipe['stage2']['lr0']}, cos_lr={recipe['stage2']['cos_lr']}) = {recipe.get('total_epochs','?')}ep")
    lines.append(f"- unified variables: seed={sh.get('seed')}, deterministic={sh.get('deterministic')}, degrees={sh.get('degrees')}, optimizer={sh.get('optimizer')}, imgsz={sh.get('imgsz')}, batch={sh.get('batch')}, cache={sh.get('cache')}")
    lines.append(f"- **baseline also two-stage** (fair alignment: fully symmetric with the FCE structure)")
    lines.append(f"- IoU mapping: {recipe.get('iou_override', {})}\n")

    lines.append("## Real metrics\n")
    for scale in recipe["scales"]:
        if scale in all_results and all_results[scale]:
            md_path = output_root / "comparison" / f"{scale}_comparison_summary.md"
            lines.append(f"### Scale {scale.upper()}")
            lines.append(f"See `{md_path.relative_to(PAPER_ROOT)}`.\n")

    lines.append("## Diff vs legacy main_ablation_m\n")
    lines.append("- **legacy `main_ablation_m/`**: variables not unified (baseline single-stage / lr0=0.01, M4 with degrees off, inconsistent cache/deterministic, etc.), kept only for historical reference, now archived.")
    lines.append("- **this `main_ablation_fair/`**: controlled comparison (unified seed/deterministic/augmentation/two-stage); the final data adopted by the paper.\n")

    lines.append("## Reproduction\n")
    lines.append("```bash")
    lines.append("conda activate fce-yolo")
    lines.append("cd <project_root>/fce-yolo")
    lines.append("python script/run_ablation.py --recipe script/ablation_config.yaml")
    lines.append("```\n")

    with open(readme, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  + {readme.relative_to(PAPER_ROOT)}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fair ablation orchestrator (one-click train 4 models x scales)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python script/run_ablation.py --dry-run            # preview matrix only
  python script/run_ablation.py                      # train full recipe
  python script/run_ablation.py --scales m           # train m only
  python script/run_ablation.py export --scales m    # pack runs/ -> fair_runs/m
  python script/run_ablation.py --skip-train --source fair_runs --scales m
        """,
    )
    sub = parser.add_subparsers(dest="command")

    # --- common args (top-level, used when no subcommand) ---
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE, help="recipe YAML path")
    parser.add_argument("--scales", type=str, nargs="+", default=None, help="override recipe scales")
    parser.add_argument("--models", type=str, nargs="+", default=None, help="override recipe models")
    parser.add_argument("--skip-train", action="store_true", help="skip training, only collect + plot")
    parser.add_argument(
        "--source", type=str, default="runs", choices=["runs", "fair_runs"],
        help="where to read stage2 results when collecting: "
             "'runs' (fce-yolo/runs/detect, default) or 'fair_runs' "
             "(the English export package, for local use after unpacking a workstation tar)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print matrix only, no training")

    # --- export subcommand ---
    p_export = sub.add_parser(
        "export",
        help="pack runs/detect/<exp>_stage2 into <export_root>/<scale>/ (English), "
             "for transferring from a Linux workstation back to local",
    )
    p_export.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE, help="recipe YAML path")
    p_export.add_argument("--scales", type=str, nargs="+", default=None, help="override recipe scales")
    p_export.add_argument("--models", type=str, nargs="+", default=None, help="override recipe models")

    return parser.parse_args()


def main():
    args = parse_args()
    recipe = load_recipe(args.recipe)

    # --- export subcommand: only pack, no train/collect/plot ---
    if args.command == "export":
        scales = args.scales or recipe["scales"]
        models = args.models or recipe["models"]
        run_export(scales, models, recipe)
        return

    scales = args.scales or recipe["scales"]
    models = args.models or recipe["models"]

    print_matrix_preview(recipe, scales, models)
    if args.dry_run:
        return

    # Stage 1: train
    if not args.skip_train:
        print("\n" + "=" * 80)
        print("Stage 1: training")
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
                    print(f"x {scale}/{model_key} failed: {e}")
                    failed_log = PAPER_ROOT / recipe["output_root"] / "failed.log"
                    failed_log.parent.mkdir(parents=True, exist_ok=True)
                    with open(failed_log, "a", encoding="utf-8") as f:
                        f.write(f"{scale}/{model_key}: {e}\n")
    else:
        print("warn: --skip-train: skipping training")

    # Stage 2: collect results (respect --source)
    all_results = collect_results(scales, models, recipe, source=args.source)

    # Stage 3: comparison tables
    print("\n" + "=" * 80)
    print("Stage 3: generate comparison tables")
    print("=" * 80)
    for scale in scales:
        if scale in all_results and all_results[scale]:
            write_comparison_table(scale, all_results[scale], recipe)
    write_cross_scale_summary(all_results, recipe)

    # README
    print("\n> generate README")
    generate_readme(recipe, all_results)

    # Stage 4: figures
    generate_figures(scales, recipe)

    print("\n" + "=" * 80)
    print("ALL DONE")
    print("=" * 80)
    print(f"results dir: {PAPER_ROOT / recipe['output_root']}")


if __name__ == "__main__":
    main()

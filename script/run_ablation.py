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

DEFAULT_RECIPE = PROJECT_ROOT / "script" / "ablation_config.yaml"

# 时间戳父目录：每次完整运行在其下创建 runs/outputs/fair_<YYYYMMDD_HHMMSS>/，
# 收纳 stage2 副本 + 对比表 + 图表 + README，自包含、打包回传一条命令。
WORK_BASE_ROOT = PROJECT_ROOT / "runs" / "outputs"


def make_run_dir(base: Path = None) -> Path:
    """创建 runs/outputs/fair_<YYYYMMDD_HHMMSS>/ 并返回。

    每次完整运行（训练/整理）创建一个独立时间戳文件夹，收纳全部产物，
    避免不同次实验互相覆盖。--replot 模式不调用此函数，直接复用已有文件夹。
    """
    from datetime import datetime
    if base is None:
        base = WORK_BASE_ROOT
    ts = datetime.now().strftime("fair_%Y%m%d_%H%M%S")
    run_dir = base / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _rel(p: Path) -> Path:
    """返回相对 PROJECT_ROOT 的路径（用于打印）。无法相对时原样返回。"""
    try:
        return p.relative_to(PROJECT_ROOT)
    except ValueError:
        return p


def _rel_posix(p: Path) -> str:
    """同 _rel 但返回 posix 风格字符串（正斜杠），用于写入 YAML 避免分隔符混乱。"""
    return _rel(p).as_posix()


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
    """给任意 model 注入统一的训练配置（单阶段或两阶段，由配方决定）。

    - recipe.freeze == 0 或 recipe.stage1.epochs == 0 时：单阶段（stage1=None），
      所有模型走 config.py 的单阶段分支，排除 freeze 对新增模块的干扰。
    - 否则：两阶段，统一 stage1 + stage2 + freeze。
    - 不改 config.py 全局 MODEL_CONFIGS，返回新对象。
    """
    base_cfg = get_model_config(model_key)
    freeze = recipe["freeze"]
    stage1_cfg = recipe.get("stage1", {})
    is_single_stage = (freeze == 0 or stage1_cfg.get("epochs", 50) == 0)
    return replace(
        base_cfg,
        stage1=None if is_single_stage else StageConfig(**stage1_cfg),
        stage2=StageConfig(**recipe["stage2"]),
        freeze=freeze,
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

def print_matrix_preview(recipe: dict, scales: list, models: list, output_root: Path = None):
    """Print the experiment matrix + unified-variable summary for review."""
    if output_root is None:
        output_root = WORK_BASE_ROOT
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
    print(f"  imgsz={sh.get('imgsz')}, batch={sh.get('batch')}, optimizer={sh.get('optimizer')}, lr0={recipe['stage2']['lr0']}")
    print(f"  seed={sh.get('seed')}, deterministic={sh.get('deterministic')}, degrees={sh.get('degrees')}, cache={sh.get('cache')}")
    # 训练策略描述：单阶段（freeze=0 或 stage1.epochs=0）vs 两阶段
    freeze = recipe["freeze"]
    s1_ep = recipe["stage1"]["epochs"]
    if freeze == 0 or s1_ep == 0:
        print(f"  training: single-stage ({recipe['stage2']['epochs']}ep, lr0={recipe['stage2']['lr0']}, cos_lr={recipe['stage2']['cos_lr']}, close_mosaic={recipe['stage2']['close_mosaic']}, no freeze)")
    else:
        print(f"  training: two-stage stage1({s1_ep}ep, freeze={freeze}) + stage2({recipe['stage2']['epochs']}ep) = {recipe.get('total_epochs', s1_ep+recipe['stage2']['epochs'])}ep")
    print(f"  IoU mapping: {recipe.get('iou_override', {})}")
    print(f"  output_root: {output_root}")
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
        final = _stage2_run_dir(model_key, scale)
        # 单阶段无 stage1，返回 None 占位以保持返回结构一致
        s1 = Path(str(final).replace("_stage2", "_stage1")) if "_stage2" in final.name else None
        print(f"✓ done, skip {scale}/{model_key}")
        return {"stage1": s1, "stage2": final}

    # 公平注入
    model_cfg = build_model_cfg_with_fairness(model_key, recipe)
    config = build_train_config(recipe, model_key)

    print(f"\n> train {scale}/{model_key}  (iou={config.iou_type})")
    trainer = YOLOv11Trainer(model_cfg, scale, config)
    result = trainer.train()
    # 单阶段返回 Path，两阶段返回 {"stage1":..., "stage2":...}；统一成 dict 结构
    if isinstance(result, dict):
        result_dict = result
    else:
        result_dict = {"stage1": None, "stage2": result}

    # IoU 校验（仅 fce_wiou 等非 CIoU 模型严格校验）
    iou_override = recipe.get("iou_override", {}) or {}
    expected_iou = iou_override.get(model_key, "CIoU")
    if expected_iou != "CIoU":
        s2_dir = result_dict["stage2"]
        if not verify_iou_type(s2_dir, expected_iou):
            raise RuntimeError(
                f"{model_key} IoU verify failed: args.yaml not {expected_iou}, "
                f"may degrade to CIoU, check recipe iou_override & build_train_config"
            )

    return result_dict


# ============================================================
# 结果收集与归档
# ============================================================

def _stage2_run_dir(model_key: str, scale: str) -> Path:
    """返回 runs/detect 下某 (scale, model) 的最终结果目录路径。

    单阶段（stage1=None）：目录名即 result_pattern（如 baseline_yolo11m、bifpn_m）。
    两阶段：result_pattern 含 _stage2 后缀（如 bifpn_m_stage2）。
    统一从 config.py 的 result_pattern 推导，无需 baseline 特例。
    """
    base_cfg = get_model_config(model_key)
    return Path("runs/detect") / base_cfg.result_pattern.format(scale=scale)


def _resolve_stage2_source(model_key: str, scale: str, source: str, recipe: dict) -> Path:
    """Resolve where to read stage2 results from.

    source:
      - "runs"    -> fce-yolo/runs/detect/<exp> (workstation default, just trained)
      - "outputs" -> fce-yolo/runs/outputs/<scale>/<0X_name>/stage2 (after unpacking
                     a workstation zip locally; the zip carries runs/outputs)
    """
    if source == "outputs":
        return WORK_OUTPUT_ROOT / scale / get_model_dir_name(model_key, scale) / "stage2"
    # default: runs/detect (relative to PROJECT_ROOT for portability)
    s2 = _stage2_run_dir(model_key, scale)
    if not s2.is_absolute():
        s2 = PROJECT_ROOT / s2
    return s2


def archive_one(scale: str, model_key: str, recipe: dict, source: str = "runs",
                output_root: Path = None):
    """把 stage2 结果复制到 output_root/<scale>/0X_<name>/stage2。

    source 见 _resolve_stage2_source。output_root 为 None 时用 WORK_OUTPUT_ROOT。
    默认仅复制 stage2（copy_stage1: false）；目录自包含、可独立归档。
    """
    import shutil
    if output_root is None:
        output_root = WORK_BASE_ROOT
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
        print(f"  + stage2 -> {_rel(dst_s2)}")
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
            print(f"  + stage1 -> {_rel(dst_s1)}")

    return dst_s2


def collect_results(scales: list, models: list, recipe: dict, source: str = "runs",
                    output_root: Path = None) -> dict:
    """遍历 (scale, model)，复制结果并汇总 results.csv 路径。

    Returns:
        {scale: {model_key: {"stage2_dir": Path, "csv": Path, "df": DataFrame, "metrics": dict}}}
    """
    if output_root is None:
        output_root = WORK_BASE_ROOT
    print("\n" + "=" * 80)
    print(f"Stage 2: collect training results (source={source}, output={_rel(output_root)})")
    print("=" * 80)

    all_results = {}
    for scale in scales:
        all_results[scale] = {}
        for model_key in models:
            dst_s2 = archive_one(scale, model_key, recipe, source=source, output_root=output_root)
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


def load_existing_results(scales: list, models: list, output_root: Path) -> dict:
    """从 output_root/<scale>/0X_*/stage2 直接加载 csv（--replot 用，不做复制）。

    与 collect_results 的区别：不调 archive_one（不复制 stage2），直接读已整理好的目录。
    用于 --replot 场景：时间戳文件夹已含完整 stage2 副本，只需重新读 csv 出图。
    """
    all_results = {}
    for scale in scales:
        all_results[scale] = {}
        for model_key in models:
            s2 = output_root / scale / get_model_dir_name(model_key, scale) / "stage2"
            csv = s2 / "results.csv"
            if not csv.exists():
                print(f"  warn: {csv} missing, skipping")
                continue
            df = load_results(csv)
            all_results[scale][model_key] = {
                "stage2_dir": s2,
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


def write_comparison_table(scale: str, scale_results: dict, recipe: dict,
                           output_root: Path = None):
    """为单尺度生成对比表 CSV + MD（按 best 指标，含 Δ 列）。"""
    if output_root is None:
        output_root = WORK_BASE_ROOT
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
    print(f"  + {_rel(csv_path)}")

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
    print(f"  + {_rel(md_path)}")
    return df


def write_cross_scale_summary(all_results: dict, recipe: dict, output_root: Path = None):
    """Cross-scale summary: best-mAP50-95 matrix of each model across n/s/m, to
    inspect scale-stability of the improvement."""
    if output_root is None:
        output_root = WORK_BASE_ROOT
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
    print(f"  + {_rel(csv)}")
    return df


# ============================================================
# paper_figs 衔接
# ============================================================

def generate_paper_figs_config(scale: str, recipe: dict, output_root: Path = None) -> Path:
    """为指定尺度生成适配 main_ablation_fair 的 paper_figs_config YAML。

    output_root 决定 dir/out_dir 的相对前缀（工作站 runs/outputs 或本地中文目录）。
    paper_figs.py 的 _resolve_path 会按 PROJECT_ROOT→PAPER_ROOT 顺序解析，所以这里
    写成"相对 output_root 所在根"的路径即可两边都对齐。
    """
    if output_root is None:
        output_root = WORK_BASE_ROOT
    # output_root 相对其所在根（PROJECT_ROOT 或 PAPER_ROOT）的相对前缀（posix 正斜杠）
    out_prefix = _rel_posix(output_root)
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
            "dir": f"{out_prefix}/{scale}/{get_model_dir_name(key, scale)}/stage2",
            "display": display,
            "loss": loss,
            "order": order,
            **_render[key],
        }
    settings = {
        "imgsz": recipe["shared"].get("imgsz", 1280),
        "out_dir": f"{out_prefix}/figures/{scale}",
        "convergence_threshold": 0.75,
        # 类名英文化（避免 Linux 无 CJK 字体时混淆矩阵标注渲染异常）
        "class_names": ["round_base", "square_part"],
        "dpi": 300,
    }
    cfg = {"experiments": experiments, "settings": settings}
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"  + paper_figs config: {_rel(cfg_path)}")
    return cfg_path


def generate_figures(scales: list, recipe: dict, output_root: Path = None):
    """为每个尺度调用 paper_figs.py 出图（A/B/C/D 全套）。"""
    print("\n" + "=" * 80)
    print("Stage 4: generate figures (paper_figs.py)")
    print("=" * 80)
    import subprocess
    for scale in scales:
        cfg_path = generate_paper_figs_config(scale, recipe, output_root=output_root)
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
# README 自动生成
# ============================================================

def generate_readme(recipe: dict, all_results: dict, output_root: Path = None):
    """Generate <output_root>/README.md (recipe summary + real metrics +
    diff vs the legacy main_ablation_m)."""
    if output_root is None:
        output_root = WORK_BASE_ROOT
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
    # 训练策略描述：单阶段 vs 两阶段
    freeze = recipe["freeze"]
    s1 = recipe.get("stage1", {})
    s2 = recipe["stage2"]
    if freeze == 0 or s1.get("epochs", 0) == 0:
        lines.append(f"- **single-stage**: {s2['epochs']}ep, lr0={s2['lr0']}, cos_lr={s2['cos_lr']}, close_mosaic={s2['close_mosaic']}, no freeze (excludes freeze interference on the new attention module)")
    else:
        lines.append(f"- **all two-stage**: stage1({s1['epochs']}ep, freeze={freeze}, lr0={s1.get('lr0','?')}) + stage2({s2['epochs']}ep, lr0={s2['lr0']}, cos_lr={s2['cos_lr']}) = {recipe.get('total_epochs','?')}ep")
        lines.append(f"- **baseline also two-stage** (fair alignment: fully symmetric with the FCE structure)")
    lines.append(f"- unified variables: seed={sh.get('seed')}, deterministic={sh.get('deterministic')}, degrees={sh.get('degrees')}, optimizer={sh.get('optimizer')}, imgsz={sh.get('imgsz')}, batch={sh.get('batch')}, cache={sh.get('cache')}")
    lines.append(f"- IoU mapping: {recipe.get('iou_override', {})}\n")

    lines.append("## Real metrics\n")
    for scale in recipe["scales"]:
        if scale in all_results and all_results[scale]:
            md_path = output_root / "comparison" / f"{scale}_comparison_summary.md"
            lines.append(f"### Scale {scale.upper()}")
            lines.append(f"See `{_rel(md_path)}`.\n")

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
    print(f"  + {_rel(readme)}")


# ============================================================
# CLI
# ============================================================

def parse_scale_arg(raw: str) -> list:
    """解析 --scale 连写：'nsm' -> ['n','s','m']；'m' -> ['m']。

    每个字符必须 ∈ {n,s,m,l,x}，不允许重复，不允许空字符串。
    """
    valid = {"n", "s", "m", "l", "x"}
    scales = list(raw)
    bad = [c for c in scales if c not in valid]
    if bad:
        raise ValueError(f"invalid scale chars {bad} in '--scale {raw}'; valid: n/s/m/l/x")
    if len(scales) != len(set(scales)):
        raise ValueError(f"duplicate scale chars in '--scale {raw}'")
    if not scales:
        raise ValueError("--scale cannot be empty")
    return scales


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fair ablation orchestrator (one-click train 4 models x scales)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python script/run_ablation.py --dry-run            # preview matrix only
  python script/run_ablation.py --scale m            # train m, output -> runs/outputs/fair_<ts>/
  python script/run_ablation.py --scale nsm          # train n,s,m scales
  python script/run_ablation.py                      # train full recipe
  # after finish, pack for transfer:
  #   zip -r fair_<ts>.zip runs/outputs/fair_<ts>
  # local replot (after unpacking):
  #   python script/run_ablation.py --replot fair_<ts>
        """,
    )

    # --- common args ---
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE, help="recipe YAML path")
    parser.add_argument("--scale", type=str, default=None,
                        help="scales concatenated, e.g. 'm' or 'nsm'; default: recipe scales")
    parser.add_argument("--models", type=str, nargs="+", default=None, help="override recipe models")
    parser.add_argument("--skip-train", action="store_true", help="skip training, only collect + plot")
    parser.add_argument("--replot", type=str, default=None, metavar="TIMESTAMP",
                        help="re-generate figures from an existing run dir "
                             "(e.g. fair_20260702_153000); reads <run_dir>/<scale>/0X_*/stage2, "
                             "writes figures/README back to the same dir")
    parser.add_argument("--dry-run", action="store_true", help="print matrix only, no training")

    return parser.parse_args()


def main():
    args = parse_args()
    recipe = load_recipe(args.recipe)

    # --- scale 解析（连写）---
    if args.scale:
        scales = parse_scale_arg(args.scale)
    else:
        scales = recipe["scales"]
    models = args.models or recipe["models"]

    # --- output_root 确定 ---
    if args.replot:
        # --replot 模式：复用已有时间戳文件夹，只重出图 + README
        output_root = WORK_BASE_ROOT / args.replot
        if not output_root.exists():
            sys.exit(f"error: run dir not found: {output_root}")
        print(f"\nreplot mode: {output_root}")

        all_results = load_existing_results(scales, models, output_root)
        if not any(all_results.values()):
            sys.exit("error: no usable results.csv found in run dir")
        generate_readme(recipe, all_results, output_root=output_root)
        generate_figures(scales, recipe, output_root=output_root)
        print(f"\nDONE. figures refreshed in: {output_root}")
        return

    # 正常模式：dry-run 用占位路径预览（不建目录），非 dry-run 才真正创建时间戳文件夹
    output_root = WORK_BASE_ROOT / "fair_<ts>" if args.dry_run else make_run_dir()

    print_matrix_preview(recipe, scales, models, output_root)
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
                    failed_log = output_root / "failed.log"
                    with open(failed_log, "a", encoding="utf-8") as f:
                        f.write(f"{scale}/{model_key}: {e}\n")
    else:
        print("warn: skipping training")

    # Stage 2: collect results
    all_results = collect_results(scales, models, recipe, output_root=output_root)

    # Stage 3: comparison tables
    print("\n" + "=" * 80)
    print("Stage 3: generate comparison tables")
    print("=" * 80)
    for scale in scales:
        if scale in all_results and all_results[scale]:
            write_comparison_table(scale, all_results[scale], recipe, output_root=output_root)
    write_cross_scale_summary(all_results, recipe, output_root=output_root)

    # README
    print("\n> generate README")
    generate_readme(recipe, all_results, output_root=output_root)

    # Stage 4: figures
    generate_figures(scales, recipe, output_root=output_root)

    print("\n" + "=" * 80)
    print("ALL DONE")
    print("=" * 80)
    print(f"results dir: {output_root}")
    print(f"\nPack for transfer to local:")
    print(f"  zip -r {output_root.name}.zip {_rel(output_root)}")


if __name__ == "__main__":
    main()

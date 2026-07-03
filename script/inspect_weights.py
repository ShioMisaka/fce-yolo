#!/usr/bin/env python3
"""
模块权重诊断工具

从训练好的 best.pt 提取 BiFPN_Concat / BiCoordCrossAtt 模块的可学习权重，
判断「改进模块是否真的学动了」——这是消融实验倒挂时定位根因的关键。

背景（2026-07-03）：
  公平消融实验中 +BiFPN / +Attn 多个尺度跑不过 baseline，但单看最终 mAP
  无法区分是「模块设计有问题」还是「训练没学到位」。本工具直接读权重：
    - BiFPN_Concat.w：可学习融合权重，若训练后仍 ≈[0.5,0.5] 说明等价普通
      Concat，没学到有用加权。
    - BiCoordCrossAtt：检查 out_h/out_w 投影权重是否非平凡、门控值是否
      脱离 0.5（均匀门控=没学到空间注意力）。

用法：
    python script/inspect_weights.py runs/detect/bifpn_m_stage2/weights/best.pt
    python script/inspect_weights.py path/to/best.pt --input-shape 2,512,160,160
    python script/inspect_weights.py fair_<ts>/m/04_fce_wiou_m/stage2/weights/best.pt
"""

import argparse
import sys
from pathlib import Path

import torch

# 项目根入 path，复用 ultralytics 的模型加载
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _format_tensor_stats(name: str, t: torch.Tensor) -> str:
    """格式化张量的关键统计量（均值/标准差/最小/最大/L2范数）。"""
    t = t.detach().float().flatten()
    return (f"    {name}: shape={list(t.shape)}, "
            f"mean={t.mean().item():+.4f}, std={t.std().item():.4f}, "
            f"min={t.min().item():+.4f}, max={t.max().item():+.4f}, "
            f"norm={t.norm().item():.4f}")


def _collect_modules(model) -> dict:
    """遍历模型，按类型收集 BiFPN_Concat / BiCoordCrossAtt 模块。"""
    from ultralytics.nn.modules.fce_block import BiFPN_Concat, BiCoordCrossAtt
    modules = {"bifpn": [], "bicoord": []}
    for name, mod in model.named_modules():
        if isinstance(mod, BiFPN_Concat):
            modules["bifpn"].append((name, mod))
        elif isinstance(mod, BiCoordCrossAtt):
            modules["bicoord"].append((name, mod))
    return modules


def inspect_bifpn(name: str, mod) -> list:
    """检查 BiFPN_Concat 的可学习融合权重 w。"""
    lines = [f"  [BiFPN_Concat] {name}"]
    w = mod.w.detach().float()
    w_normed = torch.relu(w) / (torch.relu(w).sum() + mod.epsilon)
    lines.append(f"    原始 w: {w.tolist()}")
    lines.append(f"    归一化 w (relu+softmax): {[round(x, 4) for x in w_normed.tolist()]}")
    # 判断是否学动：归一化后若所有分量都接近 1/n，说明等价平均融合
    n = len(w)
    uniform = 1.0 / n
    max_dev = (w_normed - uniform).abs().max().item()
    if max_dev < 0.02:
        verdict = "≈ 等权融合（接近 1/n，没学到偏好，可能等价普通 Concat）"
    elif max_dev < 0.10:
        verdict = "轻微偏好（学到一点加权）"
    else:
        verdict = "明显偏好（学到有意义的融合权重）"
    lines.append(f"    偏离均匀融合: {max_dev:.4f}  → {verdict}")
    return lines


def inspect_bicoord(name: str, mod) -> list:
    """检查 BiCoordCrossAtt 的投影权重与门控活跃度。

    仅看静态权重不够，还需看前向门控值的分布。但前向需要输入，这里先报静态权重
    统计；若想看门控动态值，用 --input-shape 触发前向探测。
    """
    lines = [f"  [BiCoordCrossAtt] {name}"]
    lines.append(f"    num_heads={mod.num_heads}, dim_head={mod.dim_head}, mid_dim={mod.mid_dim}")
    # out_h / out_w 是 mid_dim→oup 的投影，它们的权重范数反映分支强度
    lines.append(_format_tensor_stats("out_h.weight", mod.out_h.weight))
    lines.append(_format_tensor_stats("out_w.weight", mod.out_w.weight))
    has_id = hasattr(mod.identity, "weight")
    if has_id:
        lines.append(_format_tensor_stats("identity.weight", mod.identity.weight))
    else:
        lines.append("    identity: Identity() (inp==oup, 无可学习参数)")
    return lines


def probe_gate_values(mod, input_shape: tuple) -> list:
    """前向探测 BiCoordCrossAtt 的门控值分布（判断注意力是否活跃）。"""
    lines = ["    --- 门控值前向探测 ---"]
    mod.eval()
    n, c = input_shape[0], input_shape[1]
    h = w = int(input_shape[2]) if len(input_shape) >= 4 else 64
    if hasattr(mod, "pool_h"):  # 确认是 BiCoordCrossAtt
        x = torch.randn(n, mod.proj_q_h.in_channels, h, w)
        with torch.no_grad():
            # 复算门控（与 forward 一致，但单独提取 gate）
            x_h = mod.pool_h(x)
            x_w = mod.pool_w(x)
            hh, ww = x_h.shape[2], x_w.shape[3]
            q_h = mod.proj_q_h(x_h).view(n, mod.num_heads, mod.dim_head, hh).permute(0, 1, 3, 2)
            k_h = mod.proj_k_h(x_w).view(n, mod.num_heads, mod.dim_head, ww)
            v_h = mod.proj_v_h(x_w).view(n, mod.num_heads, mod.dim_head, ww).permute(0, 1, 3, 2)
            attn_h = (q_h @ k_h * mod.scale).softmax(-1)
            y_h = (attn_h @ v_h).permute(0, 1, 3, 2).reshape(n, mod.mid_dim, hh, 1)
            gate_h = mod.out_h(y_h)
            q_w = mod.proj_q_w(x_w).view(n, mod.num_heads, mod.dim_head, ww).permute(0, 1, 3, 2)
            k_w = mod.proj_k_w(x_h).view(n, mod.num_heads, mod.dim_head, hh)
            v_w = mod.proj_v_w(x_h).view(n, mod.num_heads, mod.dim_head, hh).permute(0, 1, 3, 2)
            attn_w = (q_w @ k_w * mod.scale).softmax(-1)
            y_w = (attn_w @ v_w).permute(0, 1, 3, 2).reshape(n, mod.mid_dim, 1, ww)
            gate_w = mod.out_w(y_w)
            gate = mod.gate(gate_h + gate_w)
        lines.append(f"      gate 值域: [{gate.min().item():.4f}, {gate.max().item():.4f}] "
                     f"均值={gate.mean().item():.4f} 标准差={gate.std().item():.4f}")
        if gate.std().item() < 0.01:
            lines.append("      → 门控几乎均匀（std<0.01），注意力未激活，模块空转")
        else:
            lines.append("      → 门控有空间变化，注意力已激活")
    return lines


def inspect_model(weight_path: Path, input_shape: tuple = None) -> None:
    """加载 best.pt 并打印所有 BiFPN/BiCoordCrossAtt 模块的权重诊断。"""
    from ultralytics import YOLO

    print("=" * 70)
    print(f"权重诊断: {weight_path}")
    print("=" * 70)

    model = YOLO(str(weight_path))
    # YOLO 包装的底层 nn.Module
    nn_model = model.model

    modules = _collect_modules(nn_model)
    n_bifpn, n_bicoord = len(modules["bifpn"]), len(modules["bicoord"])
    print(f"\n找到 {n_bifpn} 个 BiFPN_Concat，{n_bicoord} 个 BiCoordCrossAtt\n")

    if n_bifpn == 0 and n_bicoord == 0:
        print("⚠ 该模型不含 FCE 改进模块（可能是 baseline 或权重路径错误）")
        return

    print("--- BiFPN 融合权重 ---")
    if not modules["bifpn"]:
        print("  (无)")
    for name, mod in modules["bifpn"]:
        for line in inspect_bifpn(name, mod):
            print(line)

    print("\n--- BiCoordCrossAtt 注意力权重 ---")
    if not modules["bicoord"]:
        print("  (无)")
    for name, mod in modules["bicoord"]:
        for line in inspect_bicoord(name, mod):
            print(line)
        if input_shape is not None:
            for line in probe_gate_values(mod, input_shape):
                print(line)


def parse_shape(s: str) -> tuple:
    """解析 '2,512,160,160' → (2,512,160,160)。"""
    parts = [int(x.strip()) for x in s.split(",")]
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect BiFPN/BiCoordCrossAtt weights in a trained best.pt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python script/inspect_weights.py runs/detect/bifpn_m_stage2/weights/best.pt
  python script/inspect_weights.py best.pt --input-shape 2,512,160,160
        """,
    )
    parser.add_argument("weight", type=Path, help="path to best.pt")
    parser.add_argument("--input-shape", type=parse_shape, default=None,
                        help="e.g. 2,512,160,160; if given, also probe gate values via forward")
    args = parser.parse_args()

    if not args.weight.exists():
        sys.exit(f"error: weight file not found: {args.weight}")

    inspect_model(args.weight, input_shape=args.input_shape)


if __name__ == "__main__":
    main()

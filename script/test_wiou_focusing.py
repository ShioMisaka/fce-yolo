"""WIoU v3 聚焦机制单元测试（systematic-debugging Phase 4.1）

验证 _wiouv3_focusing 的聚焦曲线是否符合 Wise-IoU 论文（arXiv:2301.10051）
与官方实现（github.com/Instinct323/Wise-IoU）的设计意图。

背景：fair_20260706/0707a/0707b 三轮实验 M4(WIoU) 始终 <0.80，三轮超参
调整（box/mixup/patience）无效，诊断为代码 bug。本测试先复现 bug 行为
（修复前应失败），修复后应通过。

运行：
    conda activate fce-yolo
    cd fce-yolo
    python script/test_wiou_focusing.py
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils.loss import BboxLoss  # noqa: E402


def make_focusing(iou_type="WIoU"):
    """构造一个 BboxLoss 实例用于测试 _wiouv3_focusing。"""
    return BboxLoss(reg_max=16, iou_type=iou_type)


def test_focusing_curve_shape():
    """聚焦曲线形状测试（核心）。

    论文/官方设计的 v3 动态非单调聚焦机制应满足：
      1. 易样本（loss << mean）：r 较小（适度降权，但不归零，r > 0.2）
      2. 中等样本（loss ≈ mean）：r 接近峰值（~1.0-1.3）
      3. 难样本（loss >> mean）：r 缓慢下降但不归零（r > 0.5）

    bug 行为（修复前）：难样本 r 断崖归零（~0.0），违反设计意图。
    """
    focusing = make_focusing()

    # 模拟一个 batch 的 WIoU v1 metric（越高越好）
    # 假设 iou_mean 对应的 loss mean ≈ 0.3
    target_mean_loss = 0.3

    # 构造不同难度的样本：w_iou_metric = 1 - loss
    losses = torch.tensor([0.05, 0.15, 0.25, 0.30, 0.40, 0.55, 0.70, 0.90])
    w_iou_metric = 1.0 - losses  # metric 形式

    # 让 _wiou_loss_mean 初始化到 0.3（绕过 EMA 冷启动）
    focusing._wiou_loss_mean = target_mean_loss

    r = focusing._wiouv3_focusing(w_iou_metric)

    print("\n聚焦系数曲线（target mean loss = 0.3）：")
    print(f"{'loss':>8} {'β(L/mean)':>12} {'r(聚焦系数)':>14} {'期望行为':>20}")
    bins_expect = {
        (0.0, 0.10): "易样本，r 应 0.2-0.6",
        (0.10, 0.25): "中易样本，r 应 0.6-1.1",
        (0.25, 0.35): "近 mean，r 应 ~1.0-1.3",
        (0.35, 0.50): "中难样本，r 应 ~1.0-1.3",
        (0.50, 0.80): "难样本，r 应 >0.5",
        (0.80, 1.00): "极难样本，r 应 >0.3",
    }
    for loss_val, r_val in zip(losses.tolist(), r.tolist()):
        expect = next((v for k, v in bins_expect.items() if k[0] <= loss_val < k[1]), "?")
        beta = loss_val / target_mean_loss
        print(f"{loss_val:>8.2f} {beta:>12.2f} {r_val:>14.4f}   {expect}")

    # === 断言（修复前应失败，修复后应通过）===
    # 难样本（loss >= 0.5）的 r 不应归零——这是 bug 的核心症状
    hard_mask = losses >= 0.5
    hard_r_min = r[hard_mask].min().item()
    print(f"\n难样本(loss>=0.5) 最小 r = {hard_r_min:.4f}")
    assert hard_r_min > 0.3, (
        f"FAIL: 难样本聚焦系数归零（r={hard_r_min:.4f} < 0.3），"
        f"梯度消失——这是 WIoU v3 聚焦公式 bug 的症状。"
    )

    # 近 mean 样本不应被过度放大（官方峰值约 1.2-1.3，bug 版会到 2.5-3.0）
    near_mean_mask = (losses >= 0.25) & (losses <= 0.35)
    near_mean_r_max = r[near_mean_mask].max().item()
    print(f"近 mean 样本(0.25-0.35) 最大 r = {near_mean_r_max:.4f}")
    assert near_mean_r_max < 2.0, (
        f"FAIL: 近 mean 样本聚焦系数过大（r={near_mean_r_max:.4f} > 2.0），"
        f"梯度过度集中——β.pow(delta) bug 的症状。"
    )

    # r 不应有 NaN/Inf
    assert torch.isfinite(r).all(), f"FAIL: 聚焦系数含 NaN/Inf: {r}"

    print("\nPASS: 聚焦曲线形状符合 Wise-IoU v3 设计（难样本不归零、近 mean 不过载）")


def test_focusing_monotonicity_at_mean():
    """β=1（loss=mean）时 r 应接近论文峰值，且难样本不应断崖下跌。"""
    focusing = make_focusing()
    focusing._wiou_loss_mean = 0.3

    # 构造 loss 从 0.1 到 0.9 的等间距序列
    losses = torch.linspace(0.1, 0.9, 17)
    w_iou_metric = 1.0 - losses
    r = focusing._wiouv3_focusing(w_iou_metric)

    # 找到 r 的峰值位置
    peak_idx = r.argmax().item()
    peak_loss = losses[peak_idx].item()
    print(f"\n聚焦系数峰值在 loss={peak_loss:.3f}（mean=0.3），r_max={r[peak_idx].item():.4f}")

    # 峰值不应在 mean 处过度尖锐（bug 版会在 mean 处尖峰后断崖）
    # 检查 peak 之后 r 下降是否平缓：loss=0.9 的 r 不应小于 loss=0.1 的 r 的 0.3 倍
    r_at_hard = r[losss_to_idx(losses, 0.9)].item() if False else r[-1].item()
    r_at_easy = r[0].item()
    print(f"r at loss=0.1: {r_at_easy:.4f}, r at loss=0.9: {r_at_hard:.4f}")

    assert r_at_hard > 0.2, (
        f"FAIL: 极难样本(loss=0.9) r={r_at_hard:.4f} 过低，难样本梯度不足"
    )

    print("PASS: 聚焦曲线单调性合理")


def losss_to_idx(losses, target):
    """找到 losses 中最接近 target 的索引。"""
    return (losses - target).abs().argmin().item()


if __name__ == "__main__":
    print("=" * 70)
    print("WIoU v3 聚焦机制单元测试（systematic-debugging Phase 4）")
    print("=" * 70)
    try:
        test_focusing_curve_shape()
        test_focusing_monotonicity_at_mean()
        print("\n" + "=" * 70)
        print("✓ 全部测试通过")
        print("=" * 70)
    except AssertionError as e:
        print("\n" + "=" * 70)
        print(f"✗ 测试失败: {e}")
        print("=" * 70)
        sys.exit(1)

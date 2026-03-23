#!/usr/bin/env python3
"""
YOLOv11-BiFPN 两阶段训练脚本

阶段一：冻结主干网络，预热训练 BiFPN 模块
阶段二：解冻所有层，全局微调

Usage:
    python script/train_bifpn.py
"""

from ultralytics import YOLO
from pathlib import Path
import argparse


def stage1_warmup(scale="n"):
    """
    阶段一：冻结主干网络预热训练

    Args:
        scale: 模型尺度 (n, s, m, l, x)

    目的：
    - 保护预训练的主干网络权重不被破坏
    - 让随机初始化的 BiFPN 模块适应特征尺度
    - 使用较大学习率快速收敛

    冻结策略：
    - 冻结前 10 层（backbone 层）
    - 仅训练 head 中的 BiFPN 和检测层
    """
    print("\n" + "=" * 60)
    print("阶段一：冻结主干网络预热训练")
    print("=" * 60)

    # ==================== 阶段一配置 ====================
    DATA_PATH = "/mnt/ssd1/Dataset/haixi_jixieshou/yolo_dataset/data.yaml"

    # 使用 BiFPN 配置，加载官方预训练权重
    MODEL_YAML = "ultralytics/cfg/models/11/yolo11-bifpn.yaml"
    PRETRAINED_WEIGHTS = f"yolo11{scale}.pt"

    # 阶段一训练参数
    EPOCHS_S1 = 50
    PATIENCE_S1 = 20
    IMG_SIZE = 1280
    BATCH_SIZE = 32
    DEVICE = "0"

    # 保存路径
    PROJECT_NAME = "runs/detect"
    EXPERIMENT_NAME_S1 = f"bifpn_{scale}_stage1_warmup"

    # 冻结层数：前 10 层（backbone）
    FREEZE_LAYERS = 10

    print(f"\n阶段一配置 (模型尺度: {scale}):")
    print(f"  模型: {MODEL_YAML}")
    print(f"  预训练权重: {PRETRAINED_WEIGHTS}")
    print(f"  冻结层数: {FREEZE_LAYERS} (backbone)")
    print(f"  训练轮次: {EPOCHS_S1}")
    print(f"  学习率: 0.01 (较大)")
    print(f"  保存路径: {Path(PROJECT_NAME) / EXPERIMENT_NAME_S1}")
    print("=" * 60 + "\n")

    # 加载模型并应用预训练权重
    # 注意：BiFPN 层会随机初始化，其他匹配的层使用预训练权重
    model = YOLO(MODEL_YAML).load(PRETRAINED_WEIGHTS)

    # 阶段一训练：冻结 backbone
    model.train(
        data=DATA_PATH,

        # 训练轮次与早停
        epochs=EPOCHS_S1,
        patience=PATIENCE_S1,

        # 硬件性能配置
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        amp=True,
        workers=16,
        cache=True,

        # 冻结配置
        freeze=FREEZE_LAYERS,    # 冻结前 10 层

        # 优化器配置（较大学习率）
        optimizer="AdamW",
        lr0=0.01,                # 较大学习率快速预热
        cos_lr=False,            # 线性学习率衰减

        # 数据增强
        close_mosaic=10,         # 阶段一较短，提前关闭 Mosaic
        mixup=0.0,
        degrees=10.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # 保存配置
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME_S1,
        exist_ok=True,
        save=True,
        save_period=50,

        deterministic=False,
        verbose=True,
    )

    # 返回阶段一最佳权重路径
    best_weights_s1 = Path(PROJECT_NAME) / EXPERIMENT_NAME_S1 / "weights" / "best.pt"

    print("\n" + "=" * 60)
    print("阶段一完成!")
    print(f"最佳权重: {best_weights_s1}")
    print("=" * 60)

    return best_weights_s1


def stage2_finetune(weights_path, scale="n"):
    """
    阶段二：全局微调

    Args:
        weights_path: 阶段一训练得到的权重路径
        scale: 模型尺度 (n, s, m, l, x)

    目的：
    - 解冻所有层进行端到端训练
    - 使用较小学习率精细调整
    - 余弦退火平滑收敛

    策略：
    - 加载阶段一的 best.pt 权重
    - 不使用 freeze 参数（全层训练）
    - 使用余弦退火学习率
    """
    print("\n" + "=" * 60)
    print("阶段二：全局微调")
    print("=" * 60)

    # ==================== 阶段二配置 ====================
    DATA_PATH = "/mnt/ssd1/Dataset/haixi_jixieshou/yolo_dataset/data.yaml"

    MODEL_YAML = "ultralytics/cfg/models/11/yolo11-bifpn.yaml"

    # 阶段二训练参数
    EPOCHS_S2 = 250
    PATIENCE_S2 = 50
    IMG_SIZE = 1280
    BATCH_SIZE = 32
    DEVICE = "0"

    # 保存路径
    PROJECT_NAME = "runs/detect"
    EXPERIMENT_NAME_S2 = f"bifpn_{scale}_stage2_finetune"

    print(f"\n阶段二配置 (模型尺度: {scale}):")
    print(f"  阶段一权重: {weights_path}")
    print(f"  训练轮次: {EPOCHS_S2}")
    print(f"  学习率: 0.001 (较小)")
    print(f"  余弦退火: True")
    print(f"  保存路径: {Path(PROJECT_NAME) / EXPERIMENT_NAME_S2}")
    print("=" * 60 + "\n")

    # 加载阶段一训练得到的权重
    model = YOLO(MODEL_YAML).load(str(weights_path))

    # 阶段二训练：全局微调
    model.train(
        data=DATA_PATH,

        # 训练轮次与早停
        epochs=EPOCHS_S2,
        patience=PATIENCE_S2,

        # 硬件性能配置
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        amp=True,
        workers=16,
        cache=True,

        # 不使用 freeze 参数（全层训练）
        # freeze=None  # 默认不冻结

        # 优化器配置（较小学习率）
        optimizer="AdamW",
        lr0=0.001,               # 较小学习率精细调整
        cos_lr=True,             # 余弦退火

        # 数据增强
        close_mosaic=20,         # 最后 20 个 epoch 关闭 Mosaic
        mixup=0.0,
        degrees=10.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # 保存配置
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME_S2,
        exist_ok=True,
        save=True,
        save_period=50,

        deterministic=False,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("阶段二完成!")
    print(f"最终结果保存在: {Path(PROJECT_NAME) / EXPERIMENT_NAME_S2}")
    print("=" * 60)


def main(scale="n"):
    """执行两阶段训练流程

    Args:
        scale: 模型尺度 (n, s, m, l, x)
    """
    print("\n" + "=" * 60)
    print(f"YOLOv11{scale}-BiFPN 两阶段训练")
    print("=" * 60)
    print("\n训练策略:")
    print("  阶段一: 冻结 backbone (前10层) → 预热 BiFPN")
    print("  阶段二: 解冻所有层 → 全局微调")
    print("\n总训练轮次: 50 + 250 = 300 epochs")
    print(f"模型尺度: {scale}")
    print("=" * 60)

    # 阶段一：冻结预热
    best_weights_s1 = stage1_warmup(scale)

    # 阶段二：全局微调
    stage2_finetune(best_weights_s1, scale)

    print("\n" + "=" * 60)
    print("两阶段训练全部完成!")
    print("=" * 60)
    print(f"\n结果对比 (模型尺度: {scale}):")
    print(f"  Baseline: runs/detect/baseline_yolo11{scale}")
    print(f"  BiFPN-S1: runs/detect/bifpn_{scale}_stage1_warmup")
    print(f"  BiFPN-S2: runs/detect/bifpn_{scale}_stage2_finetune")
    print("\n运行对比脚本:")
    print("  python script/compare_results.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11-BiFPN 两阶段训练脚本")
    parser.add_argument(
        "--scale",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="模型尺度 (n: nano, s: small, m: medium, l: large, x: xlarge)"
    )
    args = parser.parse_args()
    main(scale=args.scale)

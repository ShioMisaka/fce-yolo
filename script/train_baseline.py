#!/usr/bin/env python3
"""
YOLOv11 Baseline 训练脚本

训练原生 YOLOv11n 模型作为基线对比。
使用官方预训练权重和标准配置。

Usage:
    python script/train_baseline.py
"""

from ultralytics import YOLO
from pathlib import Path


def main():
    """训练原生 YOLOv11n 模型"""

    # ==================== 配置参数 ====================
    # 数据集路径
    DATA_PATH = "/mnt/ssd1/Dataset/haixi_jixieshou/yolo_dataset/data.yaml"

    # 模型配置
    MODEL_YAML = "ultralytics/cfg/models/11/yolo11.yaml"
    PRETRAINED_WEIGHTS = "yolo11n.pt"

    # 训练参数
    EPOCHS = 300
    PATIENCE = 50
    IMG_SIZE = 1280
    BATCH_SIZE = 32
    DEVICE = "0"  # GPU 0

    # 保存路径
    PROJECT_NAME = "runs/detect"
    EXPERIMENT_NAME = "baseline_yolo11n"

    # ==================== 加载模型 ====================
    print("=" * 60)
    print("YOLOv11n Baseline 训练")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  模型: {MODEL_YAML}")
    print(f"  预训练权重: {PRETRAINED_WEIGHTS}")
    print(f"  数据集: {DATA_PATH}")
    print(f"  训练轮次: {EPOCHS}")
    print(f"  早停耐心值: {PATIENCE}")
    print(f"  图像尺寸: {IMG_SIZE}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  设备: GPU {DEVICE}")
    print(f"  保存路径: {Path(PROJECT_NAME) / EXPERIMENT_NAME}")
    print("=" * 60 + "\n")

    # 加载模型并应用预训练权重
    model = YOLO(MODEL_YAML).load(PRETRAINED_WEIGHTS)

    # ==================== 开始训练 ====================
    model.train(
        data=DATA_PATH,

        # 训练轮次与早停
        epochs=EPOCHS,
        patience=PATIENCE,

        # 硬件性能配置
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        amp=True,                # 混合精度训练
        workers=16,              # 数据加载线程数
        cache=True,              # 缓存数据集到内存

        # 优化器配置
        optimizer="AdamW",
        lr0=0.01,                # 初始学习率
        cos_lr=True,             # 余弦退火学习率

        # 数据增强
        close_mosaic=20,         # 最后 20 个 epoch 关闭 Mosaic
        mixup=0.0,               # 禁用 Mixup（避免物体遮挡）
        degrees=10.0,            # 旋转角度范围
        hsv_h=0.015,             # 色调扰动
        hsv_s=0.7,               # 饱和度扰动
        hsv_v=0.4,               # 明度扰动

        # 保存配置
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        exist_ok=True,           # 允许覆盖
        save=True,               # 保存检查点
        save_period=10,          # 每 10 个 epoch 保存一次

        # 其他
        deterministic=False,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"结果保存在: {Path(PROJECT_NAME) / EXPERIMENT_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    main()

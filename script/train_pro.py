#!/usr/bin/env python3
"""
高性能 COCO 数据集训练脚本
针对 RTX 5090 (32GB) + AMD 9950X3D (16C/32T) + 128GB RAM 优化
"""

import os
import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import LOGGER


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='高性能 COCO 训练脚本')
    parser.add_argument(
        '--model',
        type=str,
        default='ultralytics/cfg/models/11/yolo11-fce.yaml',
        help='模型 YAML 配置文件路径'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='ultralytics/cfg/datasets/coco_custom.yaml',
        help='数据集 YAML 配置文件路径'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=300,
        help='训练轮次'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=128,
        help='批次大小（-1 表示自动搜索最大值）'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='输入图像尺寸'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='GPU 设备 ID'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=24,
        help='数据加载工作进程数'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='runs/detect',
        help='项目保存目录'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='coco_train',
        help='实验名称'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='从上次中断处继续训练'
    )
    parser.add_argument(
        '--cache',
        type=str,
        default='ram',
        choices=['ram', 'disk', 'none'],
        help='数据缓存策略: ram(内存), disk(磁盘), none(不缓存)'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        default=True,
        help='启用自动混合精度训练 (AMP)'
    )
    parser.add_argument(
        '--close-mosaic',
        type=int,
        default=15,
        help='在最后 N 个 epoch 关闭 mosaic 数据增强'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='早停耐心值'
    )
    parser.add_argument(
        '--save-period',
        type=int,
        default=-1,
        help='每 N 个 epoch 保存一次检查点 (-1 表示仅保存最佳和最后一个)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='随机种子'
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='强制确定性训练（略微影响性能）'
    )
    parser.add_argument(
        '--single-cls',
        action='store_true',
        help='将多类别训练为单类别'
    )
    parser.add_argument(
        '--rect',
        action='store_true',
        help='验证时使用矩形训练'
    )
    parser.add_argument(
        '--cos-lr',
        action='store_true',
        default=True,
        help='使用余弦学习率调度器'
    )
    parser.add_argument(
        '--lr0',
        type=float,
        default=0.001,
        help='初始学习率'
    )
    parser.add_argument(
        '--lrf',
        type=float,
        default=0.01,
        help='最终学习率 (lr0 * lrf)'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.937,
        help='SGD 动量/Adam beta1'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='优化器权重衰减'
    )
    parser.add_argument(
        '--warmup-epochs',
        type=float,
        default=3.0,
        help='预热轮次'
    )
    parser.add_argument(
        '--warmup-momentum',
        type=float,
        default=0.8,
        help='预热初始动量'
    )
    parser.add_argument(
        '--warmup-bias-lr',
        type=float,
        default=0.1,
        help='预热初始偏置学习率'
    )
    parser.add_argument(
        '--box',
        type=float,
        default=7.5,
        help='框损失增益'
    )
    parser.add_argument(
        '--cls',
        type=float,
        default=0.5,
        help='cls 损失增益'
    )
    parser.add_argument(
        '--dfl',
        type=float,
        default=1.5,
        help='dfl 损失增益'
    )
    parser.add_argument(
        '--pose',
        type=float,
        default=12.0,
        help='pose 损失增益'
    )
    parser.add_argument(
        '--kobj',
        type=float,
        default=1.0,
        help='关键点对象损失增益'
    )
    parser.add_argument(
        '--label-smoothing',
        type=float,
        default=0.0,
        help='标签平滑 epsilon'
    )
    parser.add_argument(
        '--nbs',
        type=int,
        default=64,
        help='名义批次大小用于累积梯度'
    )
    parser.add_argument(
        '--overlap-mask',
        action='store_true',
        default=True,
        help='在训练期间掩码应该重叠（仅分割）'
    )
    parser.add_argument(
        '--mask-ratio',
        type=int,
        default=4,
        help='掩码下采样比例（仅分割）'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.0,
        help='使用 dropout 正则化'
    )
    parser.add_argument(
        '--val',
        action='store_true',
        default=True,
        help='在训练期间验证/测试'
    )
    parser.add_argument(
        '--plots',
        action='store_true',
        default=True,
        help='在训练期间保存训练图和结果图'
    )
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='将 results.json 保存到 runs 目录'
    )
    parser.add_argument(
        '--exist-ok',
        action='store_true',
        help='允许覆盖项目/名称（可选）'
    )

    return parser.parse_args()


def main():
    """主训练函数"""
    args = parse_args()

    # 打印配置信息
    LOGGER.info('=' * 80)
    LOGGER.info('🚀 高性能 COCO 训练脚本启动')
    LOGGER.info('=' * 80)
    LOGGER.info(f'模型配置: {args.model}')
    LOGGER.info(f'数据集配置: {args.data}')
    LOGGER.info(f'训练轮次: {args.epochs}')
    LOGGER.info(f'批次大小: {args.batch}')
    LOGGER.info(f'图像尺寸: {args.imgsz}')
    LOGGER.info(f'工作进程: {args.workers}')
    LOGGER.info(f'缓存策略: {args.cache}')
    LOGGER.info(f'混合精度: {"启用" if args.amp else "禁用"}')
    LOGGER.info('=' * 80)

    # 加载模型
    LOGGER.info(f'📦 加载模型: {args.model}')
    model = YOLO(args.model)

    # 构建训练参数字典
    train_args = {
        # 数据配置
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,

        # 系统配置
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'resume': args.resume,
        'seed': args.seed,

        # 性能优化参数
        'cache': args.cache if args.cache != 'none' else False,  # RAM 缓存整个数据集
        'amp': args.amp,  # 自动混合精度

        # 数据增强
        'close_mosaic': args.close_mosaic,  # 最后 15 个 epoch 关闭 mosaic
        'single_cls': args.single_cls,
        'rect': args.rect,

        # 学习率调度
        'cos_lr': args.cos_lr,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'warmup_momentum': args.warmup_momentum,
        'warmup_bias_lr': args.warmup_bias_lr,

        # 损失函数权重
        'box': args.box,
        'cls': args.cls,
        'dfl': args.dfl,
        'pose': args.pose,
        'kobj': args.kobj,

        # 正则化
        'label_smoothing': args.label_smoothing,
        'dropout': args.dropout,

        # 分割特定参数
        'overlap_mask': args.overlap_mask,
        'mask_ratio': args.mask_ratio,

        # 验证和日志
        'val': args.val,
        'plots': args.plots,
        'save_json': args.save_json,
        'save_period': args.save_period,
        'patience': args.patience,

        # 其他
        'exist_ok': args.exist_ok,
        'deterministic': args.deterministic,
        'verbose': True,
    }

    # 如果是负数批次大小，自动搜索最大批次大小
    if args.batch < 0:
        LOGGER.info('🔍 自动搜索最大批次大小...')
        train_args['batch'] = 'auto'

    # 开始训练
    LOGGER.info('🏋️ 开始训练...')
    LOGGER.info('=' * 80)

    try:
        results = model.train(**train_args)

        LOGGER.info('=' * 80)
        LOGGER.info('✅ 训练完成！')
        LOGGER.info('=' * 80)

        # 打印最佳结果
        if hasattr(results, 'results_dict'):
            LOGGER.info('📊 最佳结果:')
            for metric, value in results.results_dict.items():
                LOGGER.info(f'  {metric}: {value:.4f}')

        return results

    except KeyboardInterrupt:
        LOGGER.info('⚠️ 训练被用户中断')
        LOGGER.info('💡 使用 --resume 参数可以从断点继续训练')
        return None

    except Exception as e:
        LOGGER.error(f'❌ 训练出错: {e}')
        raise


if __name__ == '__main__':
    main()

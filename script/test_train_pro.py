#!/usr/bin/env python3
"""
快速测试脚本 - 验证配置正确性
使用 1 epoch 和小图像尺寸快速测试训练流程
"""

import os
import sys
from pathlib import Path

# 添加项目路径到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from ultralytics.utils import LOGGER


def quick_test():
    """快速测试训练流程"""

    LOGGER.info('=' * 80)
    LOGGER.info('🧪 快速测试模式')
    LOGGER.info('=' * 80)
    LOGGER.info('配置:')
    LOGGER.info('  - 模型: yolo11n.pt (最小模型)')
    LOGGER.info('  - 数据集: coco_custom.yaml')
    LOGGER.info('  - 训练轮次: 1')
    LOGGER.info('  - 批次大小: 32')
    LOGGER.info('  - 图像尺寸: 320')
    LOGGER.info('  - 工作进程: 8')
    LOGGER.info('=' * 80)

    # 加载预训练的最小模型
    model = YOLO('yolo11n.pt')

    # 快速测试训练
    results = model.train(
        data='ultralytics/cfg/datasets/coco_custom.yaml',
        epochs=1,
        batch=32,
        imgsz=320,
        workers=8,
        cache='ram',  # 测试 RAM 缓存
        amp=True,
        project='runs/test',
        name='quick_test',
        exist_ok=True,
        verbose=True,
    )

    LOGGER.info('=' * 80)
    LOGGER.info('✅ 快速测试完成！')
    LOGGER.info('=' * 80)

    # 运行一次验证
    LOGGER.info('🔍 运行验证...')
    metrics = model.val()

    LOGGER.info('📊 验证结果:')
    LOGGER.info(f'  mAP50-95: {metrics.box.map:.4f}')
    LOGGER.info(f'  mAP50: {metrics.box.map50:.4f}')
    LOGGER.info(f'  mAP75: {metrics.box.map75:.4f}')

    LOGGER.info('=' * 80)
    LOGGER.info('✅ 所有测试通过！配置正确，可以开始正式训练。')
    LOGGER.info('=' * 80)


if __name__ == '__main__':
    try:
        quick_test()
    except Exception as e:
        LOGGER.error(f'❌ 测试失败: {e}')
        sys.exit(1)

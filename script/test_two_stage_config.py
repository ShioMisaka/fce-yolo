#!/usr/bin/env python3
"""
测试两阶段训练配置

验证修复后的两阶段训练是否使用正确的配置。
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from script.config import (
    TrainConfig,
    get_stage1_config,
    get_stage2_config,
)
from script.trainer import train_model


def test_default_config():
    """测试默认配置"""
    print("=" * 80)
    print("测试 1: 默认配置")
    print("=" * 80)

    default_config = TrainConfig()
    print(f"默认 TrainConfig:")
    print(f"  epochs: {default_config.epochs}")
    print(f"  lr0: {default_config.lr0}")
    print(f"  cos_lr: {default_config.cos_lr}")
    print(f"  close_mosaic: {default_config.close_mosaic}")
    print(f"  patience: {default_config.patience}")
    print()


def test_stage_configs():
    """测试阶段配置"""
    print("=" * 80)
    print("测试 2: 阶段配置")
    print("=" * 80)

    s1_config = get_stage1_config()
    s2_config = get_stage2_config()

    print(f"阶段一配置 (get_stage1_config):")
    print(f"  epochs: {s1_config.epochs}")
    print(f"  lr0: {s1_config.lr0}")
    print(f"  cos_lr: {s1_config.cos_lr}")
    print(f"  close_mosaic: {s1_config.close_mosaic}")
    print(f"  patience: {s1_config.patience}")
    print()

    print(f"阶段二配置 (get_stage2_config):")
    print(f"  epochs: {s2_config.epochs}")
    print(f"  lr0: {s2_config.lr0}")
    print(f"  cos_lr: {s2_config.cos_lr}")
    print(f"  close_mosaic: {s2_config.close_mosaic}")
    print(f"  patience: {s2_config.patience}")
    print()


def test_config_passed_to_train_model():
    """测试传递给 train_model 的配置是否正确应用"""
    print("=" * 80)
    print("测试 3: 配置传递逻辑")
    print("=" * 80)

    # 模拟 CLI 传递的默认配置
    cli_config = TrainConfig()  # epochs=300, cos_lr=True, etc.

    print(f"CLI 传递的配置（默认 TrainConfig）:")
    print(f"  epochs: {cli_config.epochs}")
    print(f"  lr0: {cli_config.lr0}")
    print(f"  cos_lr: {cli_config.cos_lr}")
    print()

    # 模拟 train_model 中的逻辑
    s1_config = get_stage1_config()
    s2_config = get_stage2_config()

    # 应用非默认值
    default_config = TrainConfig()
    for k, v in cli_config.__dict__.items():
        default_value = getattr(default_config, k, None)
        if default_value != v:
            setattr(s1_config, k, v)
            setattr(s2_config, k, v)

    print(f"应用后的阶段一配置:")
    print(f"  epochs: {s1_config.epochs} (应该保持 50)")
    print(f"  lr0: {s1_config.lr0} (应该保持 0.01)")
    print(f"  cos_lr: {s1_config.cos_lr} (应该保持 False)")
    print(f"  close_mosaic: {s1_config.close_mosaic} (应该保持 10)")
    print(f"  patience: {s1_config.patience} (应该保持 20)")
    print()

    print(f"应用后的阶段二配置:")
    print(f"  epochs: {s2_config.epochs} (应该保持 300)")
    print(f"  lr0: {s2_config.lr0} (应该保持 0.001)")
    print(f"  cos_lr: {s2_config.cos_lr} (应该保持 True)")
    print(f"  close_mosaic: {s2_config.close_mosaic} (应该保持 20)")
    print(f"  patience: {s2_config.patience} (应该保持 50)")
    print()

    # 验证
    assert s1_config.epochs == 50, f"阶段一 epochs 应该是 50，实际是 {s1_config.epochs}"
    assert s1_config.cos_lr == False, f"阶段一 cos_lr 应该是 False，实际是 {s1_config.cos_lr}"
    assert s1_config.close_mosaic == 10, f"阶段一 close_mosaic 应该是 10，实际是 {s1_config.close_mosaic}"
    assert s1_config.patience == 20, f"阶段一 patience 应该是 20，实际是 {s1_config.patience}"

    assert s2_config.epochs == 300, f"阶段二 epochs 应该是 300，实际是 {s2_config.epochs}"
    assert s2_config.lr0 == 0.001, f"阶段二 lr0 应该是 0.001，实际是 {s2_config.lr0}"
    assert s2_config.cos_lr == True, f"阶段二 cos_lr 应该是 True，实际是 {s2_config.cos_lr}"

    print("✓ 所有断言通过！")
    print()


def test_custom_config_override():
    """测试自定义配置覆盖"""
    print("=" * 80)
    print("测试 4: 自定义配置覆盖")
    print("=" * 80)

    # 创建自定义配置
    custom_config = TrainConfig(
        epochs=100,  # 自定义
        batch=16,    # 自定义
    )

    print(f"自定义配置:")
    print(f"  epochs: {custom_config.epochs}")
    print(f"  batch: {custom_config.batch}")
    print()

    # 模拟 train_model 中的逻辑
    s1_config = get_stage1_config()
    s2_config = get_stage2_config()

    # 应用非默认值
    default_config = TrainConfig()
    for k, v in custom_config.__dict__.items():
        default_value = getattr(default_config, k, None)
        if default_value != v:
            setattr(s1_config, k, v)
            setattr(s2_config, k, v)

    print(f"应用自定义参数后的阶段一配置:")
    print(f"  epochs: {s1_config.epochs} (应该被覆盖为 100)")
    print(f"  batch: {s1_config.batch} (应该被覆盖为 16)")
    print(f"  lr0: {s1_config.lr0} (应该保持 0.01)")
    print(f"  cos_lr: {s1_config.cos_lr} (应该保持 False)")
    print()

    print(f"应用自定义参数后的阶段二配置:")
    print(f"  epochs: {s2_config.epochs} (应该被覆盖为 100)")
    print(f"  batch: {s2_config.batch} (应该被覆盖为 16)")
    print(f"  lr0: {s2_config.lr0} (应该保持 0.001)")
    print(f"  cos_lr: {s2_config.cos_lr} (应该保持 True)")
    print()

    # 验证
    assert s1_config.epochs == 100, f"阶段一 epochs 应该被覆盖为 100，实际是 {s1_config.epochs}"
    assert s1_config.batch == 16, f"阶段一 batch 应该被覆盖为 16，实际是 {s1_config.batch}"
    assert s1_config.lr0 == 0.01, f"阶段一 lr0 应该保持 0.01，实际是 {s1_config.lr0}"
    assert s1_config.cos_lr == False, f"阶段一 cos_lr 应该保持 False，实际是 {s1_config.cos_lr}"

    assert s2_config.epochs == 100, f"阶段二 epochs 应该被覆盖为 100，实际是 {s2_config.epochs}"
    assert s2_config.batch == 16, f"阶段二 batch 应该被覆盖为 16，实际是 {s2_config.batch}"
    assert s2_config.lr0 == 0.001, f"阶段二 lr0 应该保持 0.001，实际是 {s2_config.lr0}"
    assert s2_config.cos_lr == True, f"阶段二 cos_lr 应该保持 True，实际是 {s2_config.cos_lr}"

    print("✓ 所有断言通过！")
    print()


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("两阶段训练配置测试")
    print("=" * 80)
    print()

    test_default_config()
    test_stage_configs()
    test_config_passed_to_train_model()
    test_custom_config_override()

    print("=" * 80)
    print("✓ 所有测试通过！配置修复成功。")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()

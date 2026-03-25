#!/usr/bin/env python3
"""
测试工具脚本

提供配置一致性测试等功能。
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from script.config import MODEL_CONFIGS, get_model_config


def test_config_consistency(scale: str = None):
    """测试配置一致性

    验证不同模块中的配置是否一致。
    """
    print("=" * 80)
    print("配置一致性测试")
    print("=" * 80)

    scales = scale or ["n", "s", "m", "l", "x"]
    if isinstance(scales, str):
        scales = [scales]

    all_passed = True

    for model_type in MODEL_CONFIGS.keys():
        print(f"\n模型: {model_type.upper()}")
        print("-" * 80)

        model_cfg = get_model_config(model_type)

        # 基本信息
        print(f"  YAML 路径: {model_cfg.yaml_path}")
        print(f"  两阶段训练: {model_cfg.use_two_stage}")
        print(f"  颜色: {model_cfg.color}")

        # 测试路径生成
        print(f"\n  路径生成测试:")
        for s in scales:
            # 单阶段或两阶段
            if model_cfg.use_two_stage:
                expected = f"{model_type}_{s}_stage2"
            else:
                expected = f"{model_type}_yolo11{s}"

            actual = model_cfg.get_result_path(s)

            match = expected == actual
            symbol = "✓" if match else "✗"
            print(f"    scale={s}: {symbol} {'一致' if match else '不一致'}")
            if not match:
                print(f"      预期: {expected}")
                print(f"      实际: {actual}")
                all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ 所有配置一致性检查通过！")
    else:
        print("✗ 发现配置不一致")
    print("=" * 80)

    return all_passed


def list_models():
    """列出所有可用模型"""
    print("=" * 80)
    print("可用模型列表")
    print("=" * 80)

    for model_type, model_cfg in MODEL_CONFIGS.items():
        print(f"\n{model_type.upper()}:")
        print(f"  YAML: {model_cfg.yaml_path}")
        print(f"  两阶段: {model_cfg.use_two_stage}")
        print(f"  显示名称: {model_cfg.get_display_name('s')}")
        print(f"  颜色: {model_cfg.color}")

    print("=" * 80)


def test_imports():
    """测试模块导入"""
    print("=" * 80)
    print("模块导入测试")
    print("=" * 80)

    try:
        from script.config import MODEL_CONFIGS, TrainConfig, ExperimentConfig
        print("✓ script.config 导入成功")
    except ImportError as e:
        print(f"✗ script.config 导入失败: {e}")
        return False

    try:
        from script.trainer import YOLOv11Trainer, train_model
        print("✓ script.trainer 导入成功")
    except ImportError as e:
        print(f"✗ script.trainer 导入失败: {e}")
        return False

    try:
        from ultralytics import YOLO
        print("✓ ultralytics 导入成功")
    except ImportError as e:
        print(f"✗ ultralytics 导入失败: {e}")
        return False

    print("=" * 80)
    print("✓ 所有模块导入成功！")
    print("=" * 80)
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="测试工具脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试配置一致性
  python script/test.py --config

  # 测试特定尺度的配置
  python script/test.py --config --scale s

  # 列出所有模型
  python script/test.py --list

  # 测试模块导入
  python script/test.py --import
        """
    )

    parser.add_argument(
        "--config",
        action="store_true",
        help="测试配置一致性"
    )
    parser.add_argument(
        "--scale",
        type=str,
        default=None,
        choices=["n", "s", "m", "l", "x"],
        help="测试特定尺度"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用模型"
    )
    parser.add_argument(
        "--import",
        action="store_true",
        dest="test_import",
        help="测试模块导入"
    )

    args = parser.parse_args()

    # 默认运行所有测试
    if not any([args.config, args.list, args.test_import]):
        args.config = True
        args.test_import = True

    success = True

    if args.list:
        list_models()

    if args.config:
        success = test_config_consistency(args.scale) and success

    if args.test_import:
        success = test_imports() and success

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

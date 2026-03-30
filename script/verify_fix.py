#!/usr/bin/env python3
"""
验证模块缩放修复效果

打印不同scale下各模块的实际参数，验证修复是否有效。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ultralytics.nn.modules.fce_block import BiCoordCrossAtt, CoordAtt, CoordCrossAtt
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_module_params():
    """测试不同scale下各模块的实际参数"""
    print_section("测试不同scale下各模块的参数")

    # 定义不同scale的width和对应的基础通道数
    scales = {
        'n': (0.25, 128),  # (width_multiple, base_channels)
        's': (0.50, 256),
        'm': (1.00, 512),
        'l': (1.00, 512),
        'x': (1.50, 512),
    }

    for scale_name, (width, base_ch) in scales.items():
        print(f"\n【{scale_name.upper()} Scale】 width={width}, base_channels={base_ch}")
        print("-" * 80)

        # 实际输入通道（应用width缩放后）
        inp = base_ch

        # 测试 BiCoordCrossAtt（使用默认参数）
        print("\n1. BiCoordCrossAtt (使用默认参数):")
        # 模拟tasks.py中的参数计算逻辑
        if scale_name == 'n':
            reduction = max(8, min(32, int(inp ** 0.5)))
            base_dim = max(8, inp // reduction)
            num_heads = max(1, min(8, inp // 32))
            while num_heads > 1 and base_dim // num_heads < 8:
                num_heads -= 1
        else:
            reduction = max(8, min(32, int(inp ** 0.5)))
            base_dim = max(8, inp // reduction)
            num_heads = max(1, min(8, inp // 32))
            while num_heads > 1 and base_dim // num_heads < 8:
                num_heads -= 1

        dim_head = base_dim // num_heads
        mid_dim = dim_head * num_heads

        print(f"   - inp: {inp}")
        print(f"   - oup: {inp}")
        print(f"   - reduction: {reduction}")
        print(f"   - num_heads: {num_heads}")
        print(f"   - dim_head: {dim_head} {'✓' if dim_head >= 8 else '❌'}")
        print(f"   - mid_dim: {mid_dim}")
        print(f"   - 压缩比: {inp}:{mid_dim} = {inp/mid_dim:.1f}:1")

        # 测试 CoordAtt（使用默认参数）
        print("\n2. CoordAtt (使用默认参数):")
        reduction = max(8, min(32, int(inp ** 0.5)))
        mip = max(8, inp // reduction)
        print(f"   - inp: {inp}")
        print(f"   - oup: {inp}")
        print(f"   - reduction: {reduction}")
        print(f"   - mip (中间通道): {mip}")
        print(f"   - 压缩比: {inp}:{mip} = {inp/mip:.1f}:1")


def test_model_building():
    """测试实际模型构建时的参数"""
    print_section("测试实际模型构建时的参数")

    model_paths = {
        'fce': 'ultralytics/cfg/models/11/yolo11-fce.yaml',
    }

    for model_name, model_path in model_paths.items():
        print(f"\n【{model_name.upper()} 模型】")
        print("-" * 80)

        for scale in ['n', 's', 'm']:
            print(f"\n  {scale.upper()} scale:")
            try:
                # 构建模型（直接传入YAML路径）
                model = DetectionModel(model_path, verbose=False)

                # 查找BiCoordCrossAtt模块
                found_bica = False
                for name, module in model.named_modules():
                    if isinstance(module, BiCoordCrossAtt):
                        if not found_bica:  # 只打印第一个
                            print(f"    BiCoordCrossAtt模块 '{name}':")
                            print(f"      - mid_dim: {module.mid_dim}")
                            print(f"      - dim_head: {module.dim_head}")
                            print(f"      - num_heads: {module.num_heads}")
                            print(f"      - scale: {module.scale:.4f}")
                            status = "✓" if module.dim_head >= 8 else "❌"
                            print(f"      - dim_head >= 8: {status}")
                            found_bica = True
                            break

                if not found_bica:
                    print("    ❌ 未找到BiCoordCrossAtt模块")

            except Exception as e:
                print(f"    ❌ 错误: {e}")


def main():
    """主函数"""
    print_section("模块缩放修复验证工具")
    print("本工具验证BiCoordCrossAtt、CoordAtt等模块在不同scale下的参数")
    print("确保修复后的参数满足每个注意力头至少8个通道的要求")

    # 测试模块参数
    test_module_params()

    # 测试实际模型构建
    test_model_building()

    print_section("验证完成")
    print("\n✓ 如果所有模块的 dim_head >= 8，说明修复成功")
    print("✓ 如果有模块的 dim_head < 8，需要进一步调整参数")


if __name__ == "__main__":
    main()

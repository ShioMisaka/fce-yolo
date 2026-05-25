#!/usr/bin/env python3
"""
测试工具脚本.

验证配置一致性、覆盖逻辑等。
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from script.config import (
    MODEL_CONFIGS,
    TrainConfig,
    apply_overrides,
    get_dataset_preset,
    get_model_config,
    get_quick_test_config,
)


def test_imports():
    """测试模块导入."""
    print("=" * 80)
    print("测试 1: 模块导入")
    print("=" * 80)

    try:
        from script.config import MODEL_CONFIGS, ModelConfig, StageConfig, TrainConfig

        print("  ✓ script.config 导入成功")
    except ImportError as e:
        print(f"  ✗ script.config 导入失败: {e}")
        return False

    try:
        from script.trainer import YOLOv11Trainer

        print("  ✓ script.trainer 导入成功")
    except ImportError as e:
        print(f"  ✗ script.trainer 导入失败: {e}")
        return False

    try:
        from script.analysis import (
            extract_metrics,
            load_results,
            plot_comparison_curves,
            print_comparison_table,
            reorganize_results,
            save_comparison_summary,
        )

        print("  ✓ script.analysis 导入成功")
    except ImportError as e:
        print(f"  ✗ script.analysis 导入失败: {e}")
        return False

    return True


def test_model_configs():
    """测试模型配置."""
    print("\n" + "=" * 80)
    print("测试 2: 模型配置")
    print("=" * 80)

    all_ok = True
    scales = ["n", "s", "m", "l", "x"]

    for model_type in MODEL_CONFIGS:
        model_cfg = get_model_config(model_type)
        print(f"\n  {model_type.upper()}:")
        print(f"    YAML: {model_cfg.yaml_path}")
        print(f"    两阶段: {model_cfg.is_two_stage()}")

        for s in scales:
            path = model_cfg.get_result_path(s)
            print(f"    scale={s}: {path}")

    return all_ok


def test_dataset_presets():
    """测试数据集预设."""
    print("\n" + "=" * 80)
    print("测试 3: 数据集预设")
    print("=" * 80)

    for name in ["default", "coco", "coco_hq"]:
        config = get_dataset_preset(name)
        print(f"  {name}: data={config.data}, imgsz={config.imgsz}, batch={config.batch}, cache={config.cache}")

    return True


def test_override_logic():
    """测试配置覆盖逻辑."""
    print("\n" + "=" * 80)
    print("测试 4: 配置覆盖逻辑")
    print("=" * 80)

    # 4.1 单阶段模型：--epochs 不影响 stage1（因为 stage1=None）
    print("\n  4.1 baseline 单阶段 --epochs 200")
    cfg = get_dataset_preset("default")
    m = get_model_config("baseline")
    cfg = apply_overrides(cfg, m, shared={}, stage2={"epochs": 200}, stage1={})
    assert cfg.stage1 is None, "baseline 应该是单阶段"
    assert cfg.stage2.epochs == 200
    print("    ✓ stage1=None, stage2.epochs=200")

    # 4.2 两阶段模型：--epochs 只改 stage2
    print("\n  4.2 fce 两阶段 --epochs 200")
    cfg = get_dataset_preset("default")
    m = get_model_config("fce")
    cfg = apply_overrides(cfg, m, shared={}, stage2={"epochs": 200}, stage1={})
    assert cfg.stage1.epochs == 50, f"stage1.epochs 应该保持 50，实际 {cfg.stage1.epochs}"
    assert cfg.stage2.epochs == 200
    print("    ✓ stage1.epochs=50, stage2.epochs=200")

    # 4.3 共享参数覆盖两个阶段
    print("\n  4.3 fce --batch 16")
    cfg = get_dataset_preset("default")
    m = get_model_config("fce")
    cfg = apply_overrides(cfg, m, shared={"batch": 16}, stage2={}, stage1={})
    assert cfg.batch == 16
    print("    ✓ batch=16")

    # 4.4 显式 stage1 覆盖
    print("\n  4.4 fce --stage1-epochs 30 --stage1-lr0 0.005")
    cfg = get_dataset_preset("default")
    m = get_model_config("fce")
    cfg = apply_overrides(cfg, m, shared={}, stage2={}, stage1={"epochs": 30, "lr0": 0.005})
    assert cfg.stage1.epochs == 30
    assert cfg.stage1.lr0 == 0.005
    assert cfg.stage2.epochs == 300  # stage2 不受影响
    print("    ✓ stage1.epochs=30, stage1.lr0=0.005, stage2.epochs=300")

    # 4.5 自定义数据集路径
    print("\n  4.5 --data 覆盖")
    cfg = get_dataset_preset("default")
    m = get_model_config("baseline")
    cfg = apply_overrides(cfg, m, shared={"data": "/custom/path.yaml"}, stage2={}, stage1={})
    assert cfg.data == "/custom/path.yaml"
    print("    ✓ data=/custom/path.yaml")

    # 4.6 默认配置不被覆盖
    print("\n  4.6 无 override 时保持模型默认值")
    cfg = get_dataset_preset("default")
    m = get_model_config("fce")
    cfg = apply_overrides(cfg, m, shared={}, stage2={}, stage1={})
    assert cfg.stage1.epochs == 50
    assert cfg.stage2.epochs == 300
    assert cfg.stage1.lr0 == 0.01
    assert cfg.stage2.lr0 == 0.001
    print("    ✓ stage1 和 stage2 保持模型默认值")

    print("\n  ✓ 所有覆盖逻辑测试通过!")
    return True


def test_quick_test_config():
    """测试快速测试配置."""
    print("\n" + "=" * 80)
    print("测试 5: 快速测试配置")
    print("=" * 80)

    cfg = get_quick_test_config()
    assert cfg.stage2.epochs == 1
    assert cfg.batch == 2
    assert cfg.imgsz == 64
    assert cfg.stage2.close_mosaic == 0
    print("  ✓ 快速测试配置正确")

    return True


def test_train_config_to_dict():
    """测试 TrainConfig.to_dict()."""
    print("\n" + "=" * 80)
    print("测试 6: TrainConfig.to_dict()")
    print("=" * 80)

    cfg = TrainConfig(data="test.yaml", batch=16)
    d = cfg.to_dict()
    assert d["data"] == "test.yaml"
    assert d["batch"] == 16
    assert "iou_type" in d
    assert "epochs" not in d  # epochs 在 StageConfig 中
    print("  ✓ to_dict() 包含共享参数，不包含阶段参数")

    return True


def main():
    """主函数."""
    print("\n" + "=" * 80)
    print("Script 模块测试")
    print("=" * 80)

    success = True
    success = test_imports() and success
    success = test_model_configs() and success
    success = test_dataset_presets() and success
    success = test_override_logic() and success
    success = test_quick_test_config() and success
    success = test_train_config_to_dict() and success

    print("\n" + "=" * 80)
    if success:
        print("✓ 所有测试通过!")
    else:
        print("✗ 部分测试失败")
    print("=" * 80 + "\n")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

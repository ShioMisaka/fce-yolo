# 训练策略说明

## 公平对比的重要性

为了确保 Baseline、BiFPN 和 FCE 模型之间的对比是公平的，我们需要确保所有模型都有**等效的完整训练时间**。

## 问题分析

### 原来的配置（不公平）

```
Baseline: 300 epochs 完整训练
BiFPN/FCE:
  - 阶段一: 50 epochs（冻结 backbone，只训练新增模块）
  - 阶段二: 250 epochs（解冻所有层）
  - 总计: 300 epochs

问题：阶段一的 50 epochs 是在冻结状态下训练的，实际效果不如完整训练。
```

### 新的配置（公平）

```
Baseline: 300 epochs 完整训练
BiFPN/FCE:
  - 阶段一: 30 epochs（冻结 backbone，快速预热）
  - 阶段二: 300 epochs（解冻所有层，完整训练）
  - 总计: 330 epochs

优势：阶段二的 300 epochs 与 Baseline 的 300 epochs 相当，确保公平对比。
```

## 详细说明

### Baseline 模型

- **训练方式**: 单阶段训练
- **训练轮次**: 300 epochs
- **学习率**: 0.01 → 余弦退火
- **特点**: 标准 YOLOv11 训练流程

### BiFPN/FCE 模型

#### 阶段一：冻结预热

- **目的**: 让随机初始化的新模块（BiFPN、注意力）快速适应特征尺度
- **冻结层数**: 前 10 层（backbone）
- **训练轮次**: 30 epochs
- **学习率**: 0.01（较大）
- **学习率策略**: 线性衰减
- **关闭 Mosaic**: 前 5 epochs

**为什么缩短阶段一？**
- 阶段一是冻结训练，只对新增模块有作用
- 30 epochs 足够让新模块适应
- 更多时间投入在阶段二的完整训练上

#### 阶段二：全局微调

- **目的**: 端到端完整训练所有层
- **冻结层数**: 无（全层训练）
- **训练轮次**: 300 epochs（与 Baseline 一致）
- **学习率**: 0.001（较小）
- **学习率策略**: 余弦退火
- **关闭 Mosaic**: 最后 20 epochs

**为什么是 300 epochs？**
- 与 Baseline 的 300 epochs 相当
- 确保公平对比
- 给予足够的收敛时间

## 训练时间对比

| 模型 | 阶段一（冻结） | 阶段二（完整） | 总时间 | 有效训练时间 |
|------|---------------|---------------|--------|-------------|
| **Baseline** | - | 300 epochs | 300 epochs | 300 epochs |
| **BiFPN** | 30 epochs | 300 epochs | 330 epochs | 300 epochs |
| **FCE** | 30 epochs | 300 epochs | 330 epochs | 300 epochs |

**结论**: 所有模型的**有效完整训练时间都是 300 epochs**，确保公平对比。

## 输出目录

对比实验的输出目录会反映总训练轮次：

```
runs/detect/
└── baselinevsfce_s_330/       # 总共 330 epochs（30 + 300）
    ├── baseline_yolo11s/      # Baseline: 300 epochs
    ├── fce_s_stage1/          # FCE 阶段一: 30 epochs
    ├── fce_s_stage2/          # FCE 阶段二: 300 epochs
    ├── comparison_curves.png
    └── comparison_summary.txt
```

## 使用建议

### 快速实验（测试）

```bash
# 使用小 epoch 数快速测试
python script/train.py baseline --scale s --epochs 10
python script/train.py fce --scale s --epochs 10
```

### 正式实验

```bash
# 使用默认配置（完整的公平对比）
python script/compare.py --models baseline fce --scale s
```

### 自定义训练轮次

```bash
# 如果需要自定义，确保两阶段模型的阶段二与 Baseline 相同
python script/train.py fce --scale s --epochs 300
```

## 总结

通过调整两阶段训练的 epoch 分配：
- ✅ 确保所有模型都有 300 epochs 的完整训练
- ✅ 保持两阶段训练的优势（保护预训练权重）
- ✅ 实现公平对比
- ✅ 总训练时间增加不多（300 → 330 epochs，+10%）

这样的配置既保证了公平性，又充分利用了两阶段训练的优势。

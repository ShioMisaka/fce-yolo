# 添加新模块指南

本文件详细说明如何向 FCE-YOLOv11 项目添加新的神经网络模块。

## 添加模块的完整流程

### 步骤 1: 创建模块实现

在 `ultralytics/nn/modules/` 目录下创建或编辑模块文件。建议将自定义模块放在 `fce_block.py` 中。

```python
# ultralytics/nn/modules/fce_block.py
import torch
import torch.nn as nn


class YourNewModule(nn.Module):
    """模块描述.

    Args:
        inp: 输入通道数
        oup: 输出通道数
        other_param: 其他参数说明
    """

    def __init__(self, inp: int, oup: int, other_param: int = default_value):
        super().__init__()
        # 模块实现

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向传播
        return x
```

### 步骤 2: 更新模块导出列表

在 `fce_block.py` 的 `__all__` 中添加新模块名：

```python
__all__ = (
    "BiCoordCrossAtt",
    "BiFPN_Concat",
    "CoordAtt",
    "CoordCrossAtt",
    "YourNewModule",  # 添加新模块
)
```

### 步骤 3: 导入到 tasks.py

在 `ultralytics/nn/tasks.py` 的导入列表中添加新模块：

```python

```

### 步骤 4: 添加 YAML 参数解析逻辑

在 `ultralytics/nn/tasks.py` 的 `parse_model()` 方法中添加解析逻辑（约 1635-1665 行）：

```python
elif m is YourNewModule:
    # YourNewModule: inp, oup, other_param=default
    inp = ch[f]
    oup = args[0] if args else inp
    other_param = args[1] if len(args) > 1 else default_value
    c2 = oup
    args = [inp, oup, other_param]
```

### 步骤 5: 在模型 YAML 中使用

```yaml
# 使用默认参数
- [-1, 1, YourNewModule, []]

# 指定输出通道
- [-1, 1, YourNewModule, [256]]

# 指定所有参数
- [-1, 1, YourNewModule, [256, 16]]
```

### 步骤 6: 更新文档

1. 更新 `README.md` 的新增模块表格
2. 更新 `CLAUDE.md` 的模块说明（简要）
3. 如需要，创建测试脚本

## 参数解析模式

根据模块的输入特性选择合适的解析模式：

### 模式 1: 单输入自动检测

适用于大多数模块，自动从上一层获取输入通道数。

```python
elif m is YourModule:
    inp = ch[f]  # 自动获取上一层输出通道
    oup = args[0] if args else inp  # 默认输出=输入
    param = args[1] if len(args) > 1 else default
    c2 = oup
    args = [inp, oup, param]
```

**示例**: CoordAtt, CoordCrossAtt, BiCoordCrossAtt

### 模式 2: 多输入列表

适用于需要融合多个输入的模块（如特征融合）。

```python
elif m is YourMultiInputModule:
    # 从多个输入层提取通道数
    c1 = [ch[x] for x in f] if isinstance(f, list) else [ch[f]]
    c2 = args[0] if args else max(c1)  # 默认取最大通道数
    args = [c1, c2]
```

**示例**: BiFPN_Concat

### 模式 3: 固定输出通道

适用于输出通道固定的模块。

```python
elif m is YourFixedOutputModule:
    inp = ch[f]
    c2 = args[0]  # 直接使用 YAML 中指定的输出通道
    args = [inp, c2]
```

### 模式 4: 多可选参数

适用于有多个可选参数的模块。

```python
elif m is YourComplexModule:
    inp = ch[f]
    oup = args[0] if args else inp
    param1 = args[1] if len(args) > 1 else default1
    param2 = args[2] if len(args) > 2 else default2
    param3 = args[3] if len(args) > 3 else default3
    c2 = oup
    args = [inp, oup, param1, param2, param3]
```

**示例**: CoordCrossAtt (inp, oup, reduction, num_heads)

## 当前模块解析代码位置

`ultralytics/nn/tasks.py` 的 `parse_model()` 方法中（约 1635-1665 行）：

```python
# BiFPN_Concat: 多输入特征融合
elif m is BiFPN_Concat:
    c1 = [ch[x] for x in f] if isinstance(f, list) else [ch[f]]
    c2 = args[0] if args else max(c1)
    args = [c1, c2]

# CoordAtt: 坐标注意力
elif m is CoordAtt:
    inp = ch[f]
    oup = args[0] if args else inp
    reduction = args[1] if len(args) > 1 else 32
    c2 = oup
    args = [inp, oup, reduction]

# CoordCrossAtt: 坐标交叉注意力
elif m is CoordCrossAtt:
    inp = ch[f]
    oup = args[0] if args else inp
    reduction = args[1] if len(args) > 1 else 32
    num_heads = args[2] if len(args) > 2 else 1
    c2 = oup
    args = [inp, oup, reduction, num_heads]

# BiCoordCrossAtt: 双向坐标交叉注意力
elif m is BiCoordCrossAtt:
    inp = ch[f]
    oup = args[0] if args else inp
    reduction = args[1] if len(args) > 1 else 32
    num_heads = args[2] if len(args) > 2 else 4
    c2 = oup
    args = [inp, oup, reduction, num_heads]
```

## YAML 配置文件格式说明

YOLO 模型配置文件使用以下格式：

```yaml
# [from, repeats, module, args]
- [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
- [[-1, 6], 1, BiFPN_Concat, []] # 融合多层
```

- `from`: 输入层索引，-1 表示上一层，列表表示多层输入
- `repeats`: 重复次数
- `module`: 模块类名
- `args: 模块参数列表

## 测试新模块

创建测试脚本验证模块功能：

```python
# my_test/your_module_test.py
from ultralytics import YOLO

# 测试模型构建
model = YOLO("ultralytics/cfg/models/11/yolo11-your-module.yaml")
model.info()

# 测试前向传播
results = model.predict("path/to/test.jpg")
```

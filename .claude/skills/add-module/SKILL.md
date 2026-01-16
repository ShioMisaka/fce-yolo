---
name: add-module
description: 向 FCE-YOLOv11 项目添加新的神经网络模块。当用户请求添加自定义模块时使用此 skill：(1) 创建新的注意力/卷积/特征融合模块，(2) 将模块集成到 YOLO 架构中，(3) 配置 YAML 参数解析，(4) 更新相关文档
---

# 添加新模块

向 FCE-YOLOv11 项目添加自定义神经网络模块。

## 快速开始

核心文件位置：
- 模块实现：`ultralytics/nn/modules/fce_block.py`
- 参数解析：`ultralytics/nn/tasks.py` 的 `parse_model()` 方法
- 模型配置：`ultralytics/cfg/models/11/*.yaml`

## 工作流程

**按顺序执行以下步骤**，不要跳过或重组：

### 1. 创建模块实现

在 `ultralytics/nn/modules/fce_block.py` 中实现模块类：

```python
class YourModule(nn.Module):
    def __init__(self, inp: int, oup: int, ...):
        super().__init__()
        # 模块实现

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
```

**要求**：
- 继承 `nn.Module`
- 类型注解参数和返回值
- Google 风格 docstring（描述 + Args + Returns）

然后更新 `__all__` 导出列表。

### 2. 导入模块

在 `ultralytics/nn/tasks.py` 顶部添加导入：

```python
from ultralytics.nn.modules import (
    # ... 其他模块 ...
    YourModule,
)
```

### 3. 配置参数解析

在 `ultralytics/nn/tasks.py` 的 `parse_model()` 方法中添加解析逻辑（约 1635-1665 行）。

**必须遵循的模式**：
- `ch[f]` 获取上一层输出通道数
- `c2` 设置为输出通道数（用于下一层输入）
- `args` 构造为完整参数列表传递给模块

**详细参数解析模式**：参见 [references/parse_patterns.md](references/parse_patterns.md)

### 4. 在模型 YAML 中使用

```yaml
# [from, repeats, module, args]
- [-1, 1, YourModule, [256, 16]]
```

### 5. 更新文档

更新以下位置（保持格式一致）：
- `README.md` 的模块表格
- `CLAUDE.md` 的模块说明

## 参考资源

- **参数解析模式**：[references/parse_patterns.md](references/parse_patterns.md)
- **当前模块代码**：[references/existing_modules.md](references/existing_modules.md)
- **YAML 配置说明**：[references/yaml_format.md](references/yaml_format.md)

# 参数解析模式

在 `ultralytics/nn/tasks.py` 的 `parse_model()` 方法中配置参数解析逻辑（约 1630-1665 行）。

## 核心变量

| 变量 | 含义 |
|------|------|
| `f` | 输入层索引（-1=上一层，列表=多层） |
| `ch` | 各层输出通道数的字典 |
| `ch[f]` | 输入层的输出通道数（当前层的输入通道） |
| `args` | YAML 中配置的参数列表 |
| `c2` | 当前层的输出通道数（传递给下一层） |
| `width` | 全局宽度缩放系数 |
| `max_channels` | 最大通道数限制 |
| `make_divisible()` | 确保通道数能被 8 整除 |

## 解析模式

### 模式 1: 单输入，输出通道可选

适用于大多数模块，自动从上一层获取输入通道。

```python
elif m is YourModule:
    inp = ch[f]                           # 自动获取上一层输出通道
    oup = args[0] if args else inp        # YAML 未指定则等于输入
    if args:                               # 只有 YAML 明确指定时才应用 width 缩放
        oup = make_divisible(min(oup, max_channels) * width, 8)
    param = args[1] if len(args) > 1 else default
    c2 = oup                               # 设置输出通道
    args = [inp, oup, param]               # 构造完整参数列表
```

**示例**：CoordAtt, CoordCrossAtt, BiCoordCrossAtt

### 模式 2: 多输入列表

适用于融合多个输入的模块（如特征融合）。

```python
elif m is YourMultiInputModule:
    # 从多个输入层提取通道数
    c1 = [ch[x] for x in f] if isinstance(f, list) else [ch[f]]
    c2 = args[0] if args else max(c1)     # 默认取最大通道数
    c2 = make_divisible(min(c2, max_channels) * width, 8)
    args = [c1, c2]
```

**示例**：BiFPN_Concat

### 模式 3: 固定输出通道

适用于输出通道固定的模块。

```python
elif m is YourFixedOutputModule:
    inp = ch[f]
    c2 = args[0]                          # 直接使用 YAML 中指定的值
    args = [inp, c2]
```

### 模式 4: 多可选参数

```python
elif m is YourComplexModule:
    inp = ch[f]
    oup = args[0] if args else inp
    if args:
        oup = make_divisible(min(oup, max_channels) * width, 8)
    param1 = args[1] if len(args) > 1 else default1
    param2 = args[2] if len(args) > 2 else default2
    param3 = args[3] if len(args) > 3 else default3
    c2 = oup
    args = [inp, oup, param1, param2, param3]
```

## YOLO scale 缩放支持

当模块的输出通道需要响应全局 `width` 缩放时：

```python
oup = args[0] if args else inp
if args:  # 只有 YAML 明确指定时才应用 width 缩放
    oup = make_divisible(min(oup, max_channels) * width, 8)
```

**注意**：只在 YAML 明确指定输出通道时应用缩放，默认值（`oup = inp`）不缩放。

## 完整示例

```python
elif m is YourModule:
    # YourModule: inp, oup, reduction=32, num_heads=4
    inp = ch[f]
    oup = args[0] if args else inp
    if args:
        oup = make_divisible(min(oup, max_channels) * width, 8)
    reduction = args[1] if len(args) > 1 else 32
    num_heads = args[2] if len(args) > 2 else 4
    c2 = oup
    args = [inp, oup, reduction, num_heads]
```

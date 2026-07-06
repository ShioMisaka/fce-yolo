# YOLO Detect 头部详解

本文档详细解析 Ultralytics YOLO 系列模型中 `Detect` 类的实现原理和设计思想。

## 目录

1. [概述](#概述)
2. [类结构](#类结构)
3. [初始化详解](#初始化详解)
4. [前向传播](#前向传播)
5. [边界框解码](#边界框解码)
6. [端到端检测](#端到端检测)
7. [后处理](#后处理)
8. [偏置初始化](#偏置初始化)

---

## 概述

`Detect` 类是 YOLO (You Only Look Once) 系列模型的检测头，负责将骨干网络和颈部网络提取的特征图转换为最终的检测结果（边界框坐标和类别概率）。

**文件位置**: `ultralytics/nn/modules/head.py:26`

### 核心功能

1. **边界框回归**: 预测目标的边界框坐标
2. **分类预测**: 预测每个边界框的类别概率
3. **多尺度检测**: 在不同尺度的特征图上进行检测
4. **训练/推理双模式**: 支持训练和推理两种不同的计算流程

### 设计特点

- **双分支结构**: 边界框分支 (cv2) 和 分类分支 (cv3) 独立处理
- **DFL 解码**: 使用分布焦点损失进行边界框回归
- **Anchor-based**: 基于 anchor points 进行预测
- **灵活输出**: 支持 xywh 和 xyxy 两种输出格式

---

## 类结构

### 类继承关系

```
torch.nn.Module
    └── Detect  # 基础检测头
        ├── Segment     # 实例分割头
        ├── Pose        # 姿态估计头
        ├── OBB         # 旋转边界框头
        ├── v10Detect   # YOLOv10 检测头
        ├── WorldDetect # YOLO-World 检测头
        └── YOLOEDetect # YOLOE 检测头
```

### 类属性

```python
class Detect(nn.Module):
    # 类级别属性（所有实例共享）
    dynamic = False  # 强制重建网格
    export = False  # 导出模式标志
    format = None  # 导出格式
    end2end = False  # 端到端检测模式
    max_det = 300  # 每张图像最大检测数
    shape = None  # 输入形状缓存
    anchors = torch.empty(0)  # anchor points 缓存
    strides = torch.empty(0)  # 步长缓存
    legacy = False  # 向后兼容标志（v3/v5/v8/v9）
    xyxy = False  # 输出格式：xyxy 或 xywh
```

### 实例属性

```python
def __init__(self, nc=80, ch=()):
    self.nc = nc  # 类别数
    self.nl = len(ch)  # 检测层数量
    self.reg_max = 16  # DFL 通道数
    self.no = nc + reg_max * 4  # 每个 anchor 的输出数
    self.stride = torch.zeros(self.nl)  # 步长（build 时计算）

    # 边界框分支 (cv2)
    self.cv2 = nn.ModuleList(...)  # 每个检测层一个分支

    # 分类分支 (cv3)
    self.cv3 = nn.ModuleList(...)  # 每个检测层一个分支

    # DFL 模块
    self.dfl = DFL(self.reg_max)  # 分布焦点损失解码器
```

---

## 初始化详解

### 初始化参数

```python
def __init__(self, nc: int = 80, ch: tuple = ()):
    """
    Args:
        nc (int): 类别数量，默认为 80 (COCO 数据集)
        ch (tuple): 来自骨干网络/颈部网络的特征图通道数
        例如: (256, 512, 1024) 表示 3 个尺度的特征图.
    """
```

### 网络结构构建

#### 1. 通道数计算

```python
self.nc = nc  # 类别数
self.nl = len(ch)  # 检测层数量，通常为 3
self.reg_max = 16  # DFL 通道数
self.no = nc + self.reg_max * 4  # 每个位置的总输出数

# 计算中间通道数
# c2: 边界框分支的中间通道数
c2 = max((16, ch[0] // 4, self.reg_max * 4))
# c3: 分类分支的中间通道数
c3 = max(ch[0], min(self.nc, 100))
```

**通道数设计原则**:

- `c2`: 至少为 16，至少为 `ch[0] // 4`，至少为 `reg_max * 4`（即 64）
- `c3`: 至少为 `ch[0]`，但不超过 `min(nc, 100)`

#### 2. 边界框分支 (cv2)

```python
self.cv2 = nn.ModuleList(
    nn.Sequential(
        Conv(x, c2, 3),  # 3x3 卷积
        Conv(c2, c2, 3),  # 3x3 卷积
        nn.Conv2d(c2, 4 * self.reg_max, 1),  # 1x1 卷积输出
    )
    for x in ch
)
```

**输出**: `4 * reg_max = 64` 通道

- 边界框的 4 个边（左、上、右、下）每个用 `reg_max` 个值表示
- 例如：reg_max=16 时，每个边用 16 个值建模其分布

**网络结构示意**:

```
输入特征图 (ch[i])
    ↓
Conv(c2, 3) + BN + SiLU
    ↓
Conv(c2, 3) + BN + SiLU
    ↓
Conv(64, 1)  # 无激活函数
    ↓
输出 (64, H, W)
```

#### 3. 分类分支 (cv3)

分类分支有两种实现方式：

##### Legacy 模式 (v3/v5/v8/v9 兼容)

```python
if self.legacy:
    self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
```

##### 现代 YOLO (v11 等)

```python
else:
    self.cv3 = nn.ModuleList(
        nn.Sequential(
            nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),  # 深度可分离卷积
            nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
            nn.Conv2d(c3, self.nc, 1)
        ) for x in ch
    )
```

**输出**: `nc` 通道（类别数）

**现代模式的优势**:

- 使用深度可分离卷积 (DWConv) 减少参数量
- 更轻量级，适合移动端部署

#### 4. DFL 模块

```python
from .block import DFL

self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
```

**DFL (Distribution Focal Loss)**:

- 将边界框坐标建模为分布而不是单一值
- 提高边界框回归的精度
- 详见 `notes/loss_functions_implementation.md`

#### 5. 端到端检测分支

```python
if self.end2end:
    self.one2one_cv2 = copy.deepcopy(self.cv2)
    self.one2one_cv3 = copy.deepcopy(self.cv3)
```

**端到端检测**:

- 用于 YOLOv10 等模型
- 同时训练 one-to-many 和 one-to-one 匹配
- 推理时只使用 one-to-one 分支，无需 NMS

### 初始化流程图

```
输入: nc=80, ch=(256, 512, 1024)
    ↓
计算通道数
    ├─ c2 = max(16, 64, 64) = 64
    └─ c3 = max(256, min(80, 100)) = 256
    ↓
构建 cv2 (边界框分支)
    ├─ Layer 0: 256 → 64 → 64 → 64
    ├─ Layer 1: 512 → 64 → 64 → 64
    └─ Layer 2: 1024 → 64 → 64 → 64
    ↓
构建 cv3 (分类分支)
    ├─ Layer 0: 256 → 256 → 256 → 80
    ├─ Layer 1: 512 → 256 → 256 → 80
    └─ Layer 2: 1024 → 256 → 256 → 80
    ↓
构建 DFL
    └─ DFL(reg_max=16)
    ↓
输出: Detect 模块
```

---

## 前向传播

### 主前向传播函数

```python
def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
    """前向传播.

    Args:
        x: 来自骨干网络/颈部网络的特征图列表
        例如: [feat1, feat2, feat3]
        feat1: (B, 256, 80, 80) - P3 特征图
        feat2: (B, 512, 40, 40) - P4 特征图
        feat3: (B, 1024, 20, 20) - P5 特征图

    Returns:
        训练模式: list[torch.Tensor] - 原始预测输出
        推理模式: tuple - (处理后结果, 原始输出)
    """
    # 端到端检测模式 (YOLOv10)
    if self.end2end:
        return self.forward_end2end(x)

    # 正常模式
    for i in range(self.nl):
        # 拼接边界框和分类预测
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

    # 训练模式
    if self.training:
        return x

    # 推理模式
    y = self._inference(x)
    return y if self.export else (y, x)
```

### 前向传播流程

```
输入特征图列表: [feat1, feat2, feat3]
    ↓
┌───────────────────────────────────┐
│  for i in range(self.nl):         │
│    x[i] = cat(cv2[i](x[i]),       │
│                cv3[i](x[i]))      │
│  ┌──────────────┬──────────────┐  │
│  │  边界框分支  │   分类分支   │  │
│  │    cv2[i]    │    cv3[i]    │  │
│  │  (B,64,H,W)  │  (B,nc,H,W)  │  │
│  └──────────────┴──────────────┘  │
│       │                   │       │
│       └──────  cat  ──────┘       │
│          (B, 64+nc, H, W)         │
└───────────────────────────────────┘
    ↓
    ├─ 训练模式 → 返回原始预测 (用于损失计算)
    │
    └─ 推理模式 → _inference(x)
                  ├─ 解码边界框
                  ├─ 应用 sigmoid
                  └─ 返回最终结果
```

### 输出张量格式

#### 训练模式输出

```python
# 每个特征图的输出
x[i].shape = (batch_size, 64 + nc, height, width)

# 分解:
# - 64 = 4 * reg_max (边界框分布)
# - nc = 类别数
# - height, width = 特征图尺寸
```

**示例** (nc=80, 3 个检测层):

```
x[0]: (B, 144, 80, 80)  # P3 特征图，大检测量，小感受野
x[1]: (B, 144, 40, 40)  # P4 特征图
x[2]: (B, 144, 20, 20)  # P5 特征图，小检测量，大感受野
```

### 推理模式处理

```python
def _inference(self, x: list[torch.Tensor]) -> torch.Tensor:
    """推理时的处理流程."""
    # 1. 获取输入形状
    shape = x[0].shape  # (B, C, H, W)

    # 2. 展平并拼接所有特征图
    # x_cat: (B, no, total_anchors)
    # total_anchors = H1*W1 + H2*W2 + H3*W3
    x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

    # 3. 动态重建 anchors (如果需要)
    if self.dynamic or self.shape != shape:
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        self.shape = shape

    # 4. 分离边界框和分类预测
    box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

    # 5. 解码边界框
    # DFL 解码 + 距离到边界框转换 + 缩放
    dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

    # 6. 拼接结果: (边界框 + sigmoid(分类得分))
    return torch.cat((dbox, cls.sigmoid()), 1)
```

### 推理输出格式

```python
# 最终输出张量
output.shape = (batch_size, 4 + nc, total_anchors)

# 格式:
# - 前 4 个通道: [x_center, y_center, width, height] 或 [x1, y1, x2, y2]
# - 后 nc 个通道: 类别概率 (已应用 sigmoid)
```

---

## 边界框解码

### 解码流程

```python
def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
    """将预测的分布解码为边界框坐标.

    Args:
        bboxes: (B, 4*reg_max, total_anchors) - 预测的边界框分布
        anchors: (total_anchors, 2) 或 (1, total_anchors, 2) - anchor points
        xywh: 是否输出 xywh 格式 (True) 或 xyxy 格式 (False)

    Returns:
        解码后的边界框坐标
    """
    return dist2bbox(
        bboxes,
        anchors,
        xywh=xywh and not self.end2end and not self.xyxy,
        dim=1,
    )
```

### dist2bbox 函数

**位置**: `ultralytics/utils/tal.py:367`

```python
def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """将距离分布转换为边界框坐标.

    Args:
        distance: (..., 4*reg_max) - 预测的距离分布
        anchor_points: (..., 2) - anchor points (中心点坐标)
        xywh: 输出格式 (True=xywh, False=xyxy)
        dim: 分割维度
        流程:
        1. 将距离分为左、上、右、下四部分
        2. DFL 解码: 将分布转换为单一距离值
        3. 计算边界框坐标
    """
    # 1. 分割为左右两部分
    lt, rb = distance.chunk(2, dim)  # left-top, right-bottom

    # 2. 计算边界框角点
    x1y1 = anchor_points - lt  # 左上角
    x2y2 = anchor_points + rb  # 右下角

    # 3. 根据输出格式返回
    if xywh:
        # 返回中心点坐标和宽高
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)
    return torch.cat((x1y1, x2y2), dim)  # 返回角点坐标
```

### DFL 解码过程

```python
# DFL 类定义 (ultralytics/nn/modules/block.py)


class DFL(nn.Module):
    """Distribution Focal Loss 解码器."""

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.proj = nn.Linear(reg_max, 1)  # 投影层

    def forward(self, x):
        """将分布转换为值.

        Args:
            x: (B, 4*reg_max, anchors) - 边界框分布

        Returns:
            (B, 4, anchors) - 解码后的距离值
        """
        # 1. 重塑: (B, 4, reg_max, anchors)
        b, a, c = x.shape
        x = x.view(b, a, 4, c // 4)

        # 2. Softmax: 转换为概率分布
        x = x.softmax(2)

        # 3. 期望计算: Σ(i * p_i)
        # proj 是 (0, 1, 2, ..., reg_max-1)
        x = x.matmul(self.proj.weight.type(x.dtype).view(1, -1))

        # 4. 返回: (B, 4, anchors)
        return x.squeeze(-1)
```

### 解码示例

```python
# 假设:
# - reg_max = 16
# - anchor_point = (10.5, 20.3)
# - 预测的距离分布经过 DFL 解码后:
#   left_dist = 2.3
#   top_dist = 1.8
#   right_dist = 3.1
#   bottom_dist = 2.9

# xywh 格式:
# x_center = 10.5 + (2.3 - 3.1) / 2 = 9.6
# y_center = 20.3 + (1.8 - 2.9) / 2 = 19.25
# width = 2.3 + 3.1 = 5.4
# height = 1.8 + 2.9 = 4.7

# xyxy 格式:
# x1 = 10.5 - 2.3 = 8.2
# y1 = 20.3 - 1.8 = 18.5
# x2 = 10.5 + 3.1 = 13.6
# y2 = 20.3 + 2.9 = 23.2
```

---

## 端到端检测

端到端检测是 YOLOv10 引入的新特性，通过同时训练 one-to-many 和 one-to-one 匹配，在推理时可以省去 NMS 后处理。

### forward_end2end 函数

```python
def forward_end2end(self, x: list[torch.Tensor]) -> dict | tuple:
    """端到端检测前向传播.

    训练时返回包含 one2many 和 one2one 的字典 推理时使用 one2one 分支并后处理
    """
    # 1. 分离梯度用于 one2one
    x_detach = [xi.detach() for xi in x]

    # 2. one2one 分支 (使用 detached features)
    one2one = [
        torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
    ]

    # 3. one2many 分支 (正常训练)
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

    # 4. 训练模式
    if self.training:
        return {"one2many": x, "one2one": one2one}

    # 5. 推理模式: 只使用 one2one
    y = self._inference(one2one)
    y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
    return y if self.export else (y, {"one2many": x, "one2one": one2one})
```

### 端到端检测优势

| 特性     | 传统 YOLO      | 端到端 YOLO              |
| -------- | -------------- | ------------------------ |
| 训练策略 | 仅 one-to-many | one-to-many + one-to-one |
| 推理流程 | 预测 → NMS     | 预测 (无需 NMS)          |
| 延迟     | NMS 增加延迟   | 更低延迟                 |
| 精度     | 高             | 相当或更高               |

### 后处理函数

```python
@staticmethod
def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80) -> torch.Tensor:
    """端到端检测的后处理.

    Args:
        preds: (batch_size, num_anchors, 4 + nc) - 原始预测
        max_det: 每张图最大检测数
        nc: 类别数

    Returns:
        (batch_size, max_det, 6) - 后处理后的预测
        格式: [x, y, w, h, max_class_prob, class_index]
    """
    batch_size, anchors, _ = preds.shape
    boxes, scores = preds.split([4, nc], dim=-1)

    # 1. 选择 top-k anchors (基于最大类别得分)
    index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)

    # 2. 收集对应的边界框和得分
    boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
    scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))

    # 3. 全局 top-k 选择
    scores, index = scores.flatten(1).topk(min(max_det, anchors))
    i = torch.arange(batch_size)[..., None]

    # 4. 组装最终输出
    return torch.cat(
        [
            boxes[i, index // nc],  # 边界框
            scores[..., None],  # 最大类别得分
            (index % nc)[..., None].float(),  # 类别索引
        ],
        dim=-1,
    )
```

---

## 后处理

### Anchor Points 生成

```python
def make_anchors(feats, strides, grid_cell_offset=0.5):
    """为每个特征图生成 anchor points.

    Args:
        feats: 特征图列表 [(B, C, H1, W1), (B, C, H2, W2), ...]
        strides: 每个特征图的步长 [8, 16, 32]
        grid_cell_offset: grid cell 偏移量

    Returns:
        anchor_points: (total_anchors, 2) - anchor points 坐标
        stride_tensor: (total_anchors, 1) - 对应的步长
    """
    anchor_points, stride_tensor = [], []

    for i, stride in enumerate(strides):
        # 获取特征图尺寸
        h, w = feats[i].shape[2:]

        # 生成网格坐标
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")

        # 创建 anchor points
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

    # 拼接所有层
    return torch.cat(anchor_points), torch.cat(stride_tensor)
```

### Anchor Points 示例

```
特征图 1 (80x80), stride=8:
  anchor_points[0:6400] = [(0.5, 0.5), (1.5, 0.5), ..., (79.5, 79.5)]
  stride_tensor[0:6400] = [8, 8, ..., 8]

特征图 2 (40x40), stride=16:
  anchor_points[6400:8000] = [(0.5, 0.5), (1.5, 0.5), ..., (39.5, 39.5)]
  stride_tensor[6400:8000] = [16, 16, ..., 16]

特征图 3 (20x20), stride=32:
  anchor_points[8000:8400] = [(0.5, 0.5), (1.5, 0.5), ..., (19.5, 19.5)]
  stride_tensor[8000:8400] = [32, 32, ..., 32]

总计: 8400 anchor points
```

---

## 偏置初始化

合理的偏置初始化对模型收敛速度和最终性能至关重要。

```python
def bias_init(self):
    """初始化检测头的偏置.

    初始化策略:
    1. 边界框分支偏置设为 1.0
    2. 分类分支偏置基于类别频率和先验概率设置
    """
    m = self  # self.model[-1]

    # 遍历每个检测层
    for a, b, s in zip(m.cv2, m.cv3, m.stride):
        # 1. 边界框分支偏置设为 1.0
        a[-1].bias.data[:] = 1.0

        # 2. 分类分支偏置
        # 公式: log(5 / nc / (640 / s) ** 2)
        # - 5: 预期每张图约 5 个目标
        # - nc: 类别数
        # - (640/s)**2: 特征图相对于输入图像的缩放因子
        b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)

    # 端到端检测的 one2one 分支也需要初始化
    if self.end2end:
        for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
```

### 偏置初始化原理

#### 分类分支偏置

```python
bias = log(5 / nc / (640 / s) ** 2)
```

**解释**:

- **目标**: 设置初始预测概率，使模型在训练初期就能产生合理的预测
- **5**: 预期每张图约有 5 个目标
- **nc**: 类别数，用于归一化
- **(640/s)²**: 特征图尺寸相对于输入图像的缩放因子

**示例计算** (nc=80, stride=8):

```
scale_factor = (640 / 8) ** 2 = 6400
bias = log(5 / 80 / 6400) = log(0.00000977) ≈ -11.54
```

这意味着初始时每个 anchor 的类别预测概率约为 `sigmoid(-11.54) ≈ 0.00001`，非常低，这是合理的，因为大多数 anchor 不是正样本。

#### 边界框分支偏置

```python
a[-1].bias.data[:] = 1.0
```

设置为 1.0 是经验值，有助于边界框回归的稳定训练。

---

## 完整前向传播示例

### 输入

```python
# 输入特征图 (3 个尺度)
x = [
    torch.randn(1, 256, 80, 80),  # P3: 1x256x80x80
    torch.randn(1, 512, 40, 40),  # P4: 1x512x40x40
    torch.randn(1, 1024, 20, 20),  # P5: 1x1024x20x20
]

# Detect 头部
detect = Detect(nc=80, ch=(256, 512, 1024))
```

### 训练模式

```python
detect.train()
output = detect(x)

# 输出格式
output = [
    tensor_0,  # (1, 144, 80, 80)  - 64+80=144 通道
    tensor_1,  # (1, 144, 40, 40)
    tensor_2,  # (1, 144, 20, 20)
]

# 每个张量的通道分解
tensor_[:, :64, :, :]  # 边界框分布 (4 * reg_max)
tensor_[:, 64:, :, :]  # 分类得分 (80 个类别)
```

### 推理模式

```python
detect.eval()
output, raw = detect(x)

# output.shape = (1, 84, 8400)
# 8400 = 80*80 + 40*40 + 20*20 (总 anchor 数)
# 84 = 4 (边界框) + 80 (类别)

# 边界框格式 (默认 xywh):
output[:, :4, :]  # [x_center, y_center, width, height] (像素坐标)
output[:, 4:, :]  # 80 个类别的概率 (0-1)

# raw: 未处理的原始输出，用于调试或自定义后处理
```

---

## 与其他 Head 的对比

### Detect vs Segment

| 特性       | Detect | Segment     |
| ---------- | ------ | ----------- |
| 边界框预测 | ✓      | ✓           |
| 分类预测   | ✓      | ✓           |
| 掩码预测   | ✗      | ✓           |
| 输出通道   | 64+nc  | 64+nc+nm    |
| 额外组件   | 无     | Proto + cv4 |

### Detect vs Pose

| 特性       | Detect | Pose     |
| ---------- | ------ | -------- |
| 边界框预测 | ✓      | ✓        |
| 分类预测   | ✓      | ✓        |
| 关键点预测 | ✗      | ✓        |
| 输出通道   | 64+nc  | 64+nc+nk |
| 额外组件   | 无     | cv4      |

### Detect vs OBB

| 特性       | Detect   | OBB      |
| ---------- | -------- | -------- |
| 边界框预测 | ✓ (水平) | ✓ (旋转) |
| 角度预测   | ✗        | ✓        |
| 输出通道   | 64+nc    | 64+nc+ne |
| IoU 计算   | CIoU     | ProbIoU  |

---

## 总结

`Detect` 类是 YOLO 检测模型的核心组件，其主要特点包括：

### 设计特点

1. **双分支架构**: 边界框和分类独立处理，提高灵活性
2. **DFL 解码**: 使用分布建模提高边界框精度
3. **多尺度检测**: 在 3 个不同尺度的特征图上进行检测
4. **灵活输出**: 支持 xywh/xyxy 格式，支持端到端模式

### 关键组件

1. **cv2**: 边界框回归分支，输出 64 通道的距离分布
2. **cv3**: 分类分支，输出 nc 个类别的得分
3. **dfl**: DFL 解码器，将分布转换为坐标值
4. **one2one_cv2/cv3**: 端到端检测的 one-to-one 匹配分支

### 计算流程

```
特征图 → cv2/cv3 → 拼接 → 训练/推理分支
                        ├─ 训练: 直接返回
                        └─ 推理: DFL → 解码 → sigmoid
```

### 相关文件

- `ultralytics/nn/modules/head.py`: Detect 类定义
- `ultralytics/nn/modules/block.py`: DFL 类定义
- `ultralytics/utils/tal.py`: dist2bbox, make_anchors 等工具函数
- `ultralytics/utils/loss.py`: 损失函数定义

### 下一步学习

1. **Segment 类**: 了解实例分割头如何扩展 Detect
2. **Pose 类**: 了解姿态估计如何集成关键点预测
3. **OBB 类**: 了解旋转边界框检测的实现
4. **RTDETRDecoder 类**: 了解 Transformer-based 检测头

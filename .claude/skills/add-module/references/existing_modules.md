# 当前模块代码参考

`ultralytics/nn/modules/fce_block.py` 中的现有模块实现。

## BiFPN_Concat

```python
class BiFPN_Concat(nn.Module):
    """Learnable weighted feature fusion and bidirectional cross-scale connectivity Concat"""
    def __init__(self, c1, c2=None):
        super(BiFPN_Concat, self).__init__()
        self.output_ch = c2 if c2 else max(c1)
        self.realign_convs = nn.ModuleList()
        for ch in c1:
            if ch != self.output_ch:
                self.realign_convs.append(Conv(ch, self.output_ch, 1, 1))
            else:
                self.realign_convs.append(nn.Identity())
        self.w = nn.Parameter(torch.ones(len(c1), dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        processed_x = []
        for i, tensor in enumerate(x):
            processed_x.append(self.realign_convs[i](tensor))
        w = torch.relu(self.w)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        result = weight[0] * processed_x[0]
        for i in range(1, len(processed_x)):
            result += weight[i] * processed_x[i]
        return result
```

## CoordAtt

```python
class CoordAtt(nn.Module):
    """Coordinate Attention 模块 (标准版)

    通过分别捕获水平和垂直方向的空间依赖关系来增强特征表达。
    参考: https://arxiv.org/abs/2103.02907
    """
    def __init__(self, inp: int, oup: int, reduction: int = 32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.cv1 = Conv(inp, mip, k=1, s=1, p=0)
        self.cv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.cv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.identity = nn.Conv2d(inp, oup, 1) if inp != oup else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = self.cv1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.cv_h(x_h).sigmoid()
        a_w = self.cv_w(x_w).sigmoid()
        return self.identity(x) * a_h * a_w
```

## CoordCrossAtt

```python
class CoordCrossAtt(nn.Module):
    """Coordinate Cross Attention 模块

    在 CoordAtt 基础上引入 Cross-Attention 机制，让水平和垂直特征之间
    进行更深入的交互。
    """
    def __init__(self, inp: int, oup: int, reduction: int = 32, num_heads = 1):
        super().__init__()
        self.mip = max(8, inp // reduction)
        self.num_heads = num_heads
        self.scale = (self.mip // num_heads) ** -0.5
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.cv1 = nn.Conv2d(inp, self.mip, kernel_size=1)
        self.q_conv = nn.Conv2d(self.mip, self.mip, 1)
        self.k_conv = nn.Conv2d(self.mip, self.mip, 1)
        self.v_conv = nn.Conv2d(self.mip, self.mip, 1)
        self.proj = nn.Conv2d(self.mip, oup, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = self.cv1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        q = self.q_conv(x_h).view(n, self.num_heads, -1, h).permute(0, 1, 3, 2)
        k = self.k_conv(x_w).view(n, self.num_heads, -1, w)
        v = self.v_conv(x_w).view(n, self.num_heads, -1, w).permute(0, 1, 3, 2)
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        z = (attn @ v).permute(0, 1, 3, 2).contiguous().view(n, self.mip, h, 1)
        y_att = self.gate(self.proj(z))
        return x * y_att
```

## BiCoordCrossAtt

```python
class BiCoordCrossAtt(nn.Module):
    """Bidirectional Coordinate Cross Attention

    特点：
    1. 对称结构：同时计算 H->W 和 W->H 的注意力
    2. 效率优化：直接对池化后的特征进行投影
    """
    def __init__(self, inp: int, oup: int, reduction: int = 32, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = max(8, inp // reduction) // num_heads
        self.mid_dim = self.dim_head * num_heads
        self.scale = self.dim_head ** -0.5
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        # Branch H
        self.proj_q_h = nn.Conv2d(inp, self.mid_dim, 1)
        self.proj_k_h = nn.Conv2d(inp, self.mid_dim, 1)
        self.proj_v_h = nn.Conv2d(inp, self.mid_dim, 1)
        self.out_h = nn.Conv2d(self.mid_dim, oup, 1)
        # Branch W
        self.proj_q_w = nn.Conv2d(inp, self.mid_dim, 1)
        self.proj_k_w = nn.Conv2d(inp, self.mid_dim, 1)
        self.proj_v_w = nn.Conv2d(inp, self.mid_dim, 1)
        self.out_w = nn.Conv2d(self.mid_dim, oup, 1)
        self.activation = nn.Sigmoid()
        self.identity = nn.Conv2d(inp, oup, 1) if inp != oup else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        # Branch H
        q_h = self.proj_q_h(x_h).view(n, self.num_heads, self.dim_head, h).permute(0, 1, 3, 2)
        k_h = self.proj_k_h(x_w).view(n, self.num_heads, self.dim_head, w)
        v_h = self.proj_v_h(x_w).view(n, self.num_heads, self.dim_head, w).permute(0, 1, 3, 2)
        attn_h = (q_h @ k_h) * self.scale
        attn_h = attn_h.softmax(dim=-1)
        y_h = (attn_h @ v_h).permute(0, 1, 3, 2).reshape(n, self.mid_dim, h, 1)
        weight_h = self.activation(self.out_h(y_h))
        # Branch W
        q_w = self.proj_q_w(x_w).view(n, self.num_heads, self.dim_head, w).permute(0, 1, 3, 2)
        k_w = self.proj_k_w(x_h).view(n, self.num_heads, self.dim_head, h)
        v_w = self.proj_v_w(x_h).view(n, self.num_heads, self.dim_head, h).permute(0, 1, 3, 2)
        attn_w = (q_w @ k_w) * self.scale
        attn_w = attn_w.softmax(dim=-1)
        y_w = (attn_w @ v_w).permute(0, 1, 3, 2).reshape(n, self.mid_dim, 1, w)
        weight_w = self.activation(self.out_w(y_w))
        return self.identity(x) * weight_h * weight_w
```

## 模块模板

```python
class YourModule(nn.Module):
    """模块描述

    Args:
        inp: 输入通道数
        oup: 输出通道数
        param1: 参数1说明
        param2: 参数2说明
    """
    def __init__(self, inp: int, oup: int, param1: int = default1, param2: int = default2):
        super().__init__()
        # 模块实现

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向传播
        return x
```

import torch
import torch.nn as nn

from .conv import Conv

__all__ = (
    "BiFPN_Concat",
    "CoordAtt",
    "CoordCrossAtt"
)

class BiFPN_Concat(nn.Module):
    """Learnable weighted feature fusion and bidirectional cross-scale connectivity Concat"""
    def __init__(self, c1, c2=None):
        """Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super(BiFPN_Concat, self).__init__()
        
        # 确定最终输出的通道数
        self.output_ch = c2 if c2 else max(c1)
        
        # 为通道数不等于 output_ch 的输入创建 1x1 卷积进行对齐
        self.realign_convs = nn.ModuleList()
        for ch in c1:
            if ch != self.output_ch:
                # 使用 1x1 卷积改变通道，不改变特征图大小
                self.realign_convs.append(Conv(ch, self.output_ch, 1, 1))
            else:
                self.realign_convs.append(nn.Identity())

        # 设置可学习权重
        self.w = nn.Parameter(torch.ones(len(c1), dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4

    def forward(self, x: list[torch.Tensor])-> torch.Tensor:
        """Perform a forward pass of the BiFPN block.

        Args:
            x (torch.Tensor): Input tensor list.

        Returns:
            (torch.Tensor): Output tensor.
        """
        # 1. 通道对齐
        processed_x = []
        for i, tensor in enumerate(x):
            processed_x.append(self.realign_convs[i](tensor))
            
        # 2. 归一化权重
        w = torch.relu(self.w)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        
        # 3. 加权融合
        result = weight[0] * processed_x[0]
        for i in range(1, len(processed_x)):
            result += weight[i] * processed_x[i]
            
        return result
    
class CoordAtt(nn.Module):
    """
    Coordinate Attention 模块 (标准版)

    通过分别捕获水平和垂直方向的空间依赖关系来增强特征表达。
    参考: https://arxiv.org/abs/2103.02907

    Args:
        inp: 输入通道数
        oup: 输出通道数
        reduction: 缩减比例，用于构建 Bottleneck 结构
    """

    def __init__(self, inp: int, oup: int, reduction: int = 32):
        super().__init__()
        # 1. 空间池化：分别聚合水平和垂直信息
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 计算中间层的压缩通道数 (Bottleneck)
        mip = max(8, inp // reduction)

        # 2. 核心融合层：将 H 和 W 信息拼接后进行通道压缩和非线性激活
        self.cv1 = Conv(inp, mip, k=1, s=1, p=0)

        # 3. 恢复层：将压缩的通道 mip 恢复到输出通道 oup
        self.cv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.cv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        # 4. 如果输入输出通道不一致，需要一个 shortcut 变换
        self.identity = nn.Conv2d(inp, oup, 1) if inp != oup else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()

        # --- 信息嵌入 (Embedding) ---
        x_h = self.pool_h(x)  # [N, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [N, C, 1, W] -> [N, C, W, 1]

        # --- 协同注意力生成 (Generation) ---
        y = self.cv1(torch.cat([x_h, x_w], dim=2))  # [N, mip, H+W, 1]

        # 拆分回两个方向
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # [N, mip, 1, W]

        # 生成 H 和 W 方向的权重 (Sigmoid 激活)
        a_h = self.cv_h(x_h).sigmoid()  # [N, oup, H, 1]
        a_w = self.cv_w(x_w).sigmoid()  # [N, oup, 1, W]

        # --- 重加权 (Reweight) ---
        return self.identity(x) * a_h * a_w


class CoordCrossAtt(nn.Module):
    """
    Coordinate Cross Attention 模块

    在 CoordAtt 的基础上引入 Cross-Attention 机制，让水平和垂直特征之间
    进行更深入的交互，捕获跨方向的空间依赖关系。

    Args:
        inp: 输入通道数
        oup: 输出通道数
        reduction: 缩减比例
        num_heads: 多头注意力的头数
    """

    def __init__(self, inp: int, oup: int, reduction: int=32, num_heads=1):
        super().__init__()
        self.mip = max(8, inp // reduction)
        self.num_heads = num_heads
        self.scale = (self.mip // num_heads) ** -0.5

        # 空间池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 融合层
        self.cv1 = nn.Conv2d(inp, self.mip, kernel_size=1)

        # Cross-Attention 的线性映射
        self.q_conv = nn.Conv2d(self.mip, self.mip, 1)
        self.k_conv = nn.Conv2d(self.mip, self.mip, 1)
        self.v_conv = nn.Conv2d(self.mip, self.mip, 1)

        # 输出投影
        self.proj = nn.Conv2d(self.mip, oup, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()

        # 1. 嵌入与压缩
        x_h = self.pool_h(x)  # [N, mip, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [N, mip, W, 1]

        y = self.cv1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(y, [h, w], dim=2)

        # 2. Cross-Attention 核心逻辑 (让 H 与 W 交互)
        q = self.q_conv(x_h).view(n, self.num_heads, -1, h).permute(0, 1, 3, 2)
        k = self.k_conv(x_w).view(n, self.num_heads, -1, w)
        v = self.v_conv(x_w).view(n, self.num_heads, -1, w).permute(0, 1, 3, 2)

        # 计算相关性矩阵 [N, head, H, W]
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)

        # 聚合信息
        z = (attn @ v).permute(0, 1, 3, 2).contiguous().view(n, self.mip, h, 1)

        # 3. 施加注意力
        y_att = self.gate(self.proj(z))  # [N, oup, H, 1]

        return x * y_att

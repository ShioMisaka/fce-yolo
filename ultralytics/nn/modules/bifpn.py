import torch
import torch.nn as nn

from .conv import Conv

class BiFPN_Concat(nn.Module):
    def __init__(self, c1, c2=None, **kwargs):
        """
        c1: 输入通道列表，例如 [128, 256]
        c2: 输出通道数。如果不指定，默认取 c1 中的最大值
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
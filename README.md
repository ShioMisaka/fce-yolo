# FCE-YOLOv11

> 基于 [Ultralytics](https://github.com/ultralytics/ultralytics) 的 YOLOv11 改进版本，集成了多种特征增强和注意力机制模块。

## 简介

本项目是从 Ultralytics fork 而来的深度学习目标检测框架，在 YOLOv11 的基础上添加了自定义的特征增强模块（Feature Enhancement Modules），构建了 **FCE-YOLOv11** 模型。

### 新增模块

| 模块                | 描述                                                 | 论文/参考                                        |
| ------------------- | ---------------------------------------------------- | ------------------------------------------------ |
| **BiFPN_Concat**    | 可学习的加权特征融合，支持双向跨尺度连接             | [EfficientDet](https://arxiv.org/abs/1911.09070) |
| **CoordAtt**        | 坐标注意力机制，分别捕获水平和垂直方向的空间依赖关系 | [CoordAtt](https://arxiv.org/abs/2103.02907)     |
| **CoordCrossAtt**   | 增强版坐标注意力，引入交叉注意力机制进行跨方向交互   | 基于 CoordAtt 改进                               |
| **BiCoordCrossAtt** | 双向坐标交叉注意力，对称的 H<->W 双向注意力机制      | 基于 CoordAtt 改进                               |

### 模型架构

- **Backbone**: YOLOv11 主干网络（集成 CoordAtt 模块）
- **Neck**: BiFPN 特征金字塔网络
- **Head**: YOLOv11 检测头

## 安装

```bash
# 克隆仓库
git clone https://github.com/ShioMisaka/ultralytics.git
cd ultralytics

# 创建 conda 环境
conda create -n ultralytics python=3.10
conda activate ultralytics

# 安装依赖
pip install -e .
```

## 快速开始

### 模型训练

```bash
# 使用 FCE-YOLOv11 模型训练
yolo detect train data=coco8.yaml model=ultralytics/cfg/models/11/yolo11-fce.yaml epochs=100

# 使用 BiFPN-YOLOv11 模型训练
yolo detect train data=coco8.yaml model=ultralytics/cfg/models/11/yolo11-bifpn.yaml epochs=100
```

### 模型验证

```bash
yolo detect val model=ultralytics/cfg/models/11/yolo11-fce.yaml data=coco8.yaml
```

### 模型预测

```bash
yolo detect predict model=path/to/best.pt source=path/to/images
```

### 自定义模型配置

在 YAML 配置文件中使用新模块：

```yaml
# CoordAtt: 输出通道默认等于输入通道，reduction=32
- [-1, 1, CoordAtt, []]

# CoordAtt: 指定输出通道和 reduction
- [-1, 1, CoordAtt, [256, 16]]

# BiFPN_Concat: 自动检测输入通道，输出通道取最大值
- [[-1, 6], 1, BiFPN_Concat, []]

# BiFPN_Concat: 指定输出通道
- [[-1, 6], 1, BiFPN_Concat, [256]]

# CoordCrossAtt: 使用默认参数 (num_heads=1)
- [-1, 1, CoordCrossAtt, []]

# CoordCrossAtt: 指定所有参数
- [-1, 1, CoordCrossAtt, [256, 16, 2]]

# BiCoordCrossAtt: 使用默认参数 (num_heads=4)
- [-1, 1, BiCoordCrossAtt, []]

# BiCoordCrossAtt: 指定输出通道
- [-1, 1, BiCoordCrossAtt, [512]]

# BiCoordCrossAtt: 指定所有参数
- [-1, 1, BiCoordCrossAtt, [512, 16, 8]]
```

#### 模块参数说明

**CoordAtt** - 坐标注意力

- 参数：`[output_channels, reduction]`
- 默认值：`output_channels = input_channels`, `reduction = 32`

**BiFPN_Concat** - 可学习加权特征融合

- 参数：`[output_channels]`
- 默认值：`output_channels = max(input_channels)`
- 支持多输入：`[[layer1, layer2, ...], 1, BiFPN_Concat, []]`

**CoordCrossAtt** - 坐标交叉注意力

- 参数：`[output_channels, reduction, num_heads]`
- 默认值：`output_channels = input_channels`, `reduction = 32`, `num_heads = 1`

**BiCoordCrossAtt** - 双向坐标交叉注意力

- 参数：`[output_channels, reduction, num_heads]`
- 默认值：`output_channels = input_channels`, `reduction = 32`, `num_heads = 4`
- 特点：对称的双向注意力（H<->W），比 CoordCrossAtt 更强的特征交互

## 测试

运行测试脚本验证模块可用性：

```bash
# 测试 FCE-YOLOv11 模型
python my_test/fce_test.py

# 测试 BiFPN-YOLOv11 模型
python my_test/bifpn_test.py
```

## 项目结构

```
ultralytics/
├── ultralytics/
│   ├── nn/
│   │   ├── modules/
│   │   │   └── fce_block.py      # 自定义模块实现
│   │   └── tasks.py              # 模型解析逻辑
│   └── cfg/
│       └── models/
│           └── 11/
│               ├── yolo11-fce.yaml      # FCE-YOLOv11 配置
│               └── yolo11-bifpn.yaml    # BiFPN-YOLOv11 配置
└── my_test/
    ├── fce_test.py               # FCE 模块测试
    └── bifpn_test.py             # BiFPN 模块测试
```

## 模型说明

### yolo11-fce.yaml

在 Backbone 和 Neck 中集成了 CoordAtt 和 BiFPN_Concat 模块：

- Backbone: 第 5、8 层添加 CoordAtt
- Neck: 使用 BiFPN_Concat 进行多尺度特征融合

### yolo11-bifpn.yaml

在 Neck 中使用 BiFPN_Concat 替代标准 Concat：

- 更高效的特征融合
- 可学习的权重参数

## 开发计划

- [ ] 添加更多注意力机制模块（SE、ECA、CBAM 等）
- [ ] 添加更多特征融合模块（ASFF、PAFPN 等）
- [ ] 发布预训练权重
- [ ] 性能对比实验

## 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) - 优秀的 YOLO 实现
- CoordAtt 论文: [Coordinate Attention for Efficient Mobile Network Design](https://arxiv.org/abs/2103.02907)
- BiFPN 论文: [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)

## License

本项目遵循 AGPL-3.0 许可证。详见 [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 文件。

## Citation

如果您使用了本项目，请引用：

```bibtex
@software{fce_yolov11,
  author = {ShioMisaka},
  title = {FCE-YOLOv11: Feature Enhancement Modules for YOLOv11},
  year = {2025},
  url = {https://github.com/ShioMisaka/ultralytics}
}
```

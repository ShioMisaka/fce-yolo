from ultralytics import YOLO

# 1. 加载模型与迁移学习 (非常关键)
# 既然你修改了网络结构 (BiFPN, CoordAtt等)，你需要用你的 custom yaml，
# 但同时一定要加载官方的 .pt 权重来获取骨干网的预训练知识！

# custom_yaml_path = "yolo11n-fce.yaml" # 替换为你修改了网络结构的 yaml 文件名
custom_yaml_path = "yolo11n.yaml"
model = YOLO(custom_yaml_path).load("yolo11n.pt") 

# 2. 启动极致性能训练
model.train(
    data="/mnt/ssd1/Dataset/haixi_jixieshou/yolo_dataset/data.yaml",
    
    # --- 训练轮次与早停 ---
    epochs=300,              # 增加 epoch 上限
    patience=50,             # 如果连续 50 个 epoch 验证集 mAP 没有提升，则提前停止，防止过拟合
    
    # --- 硬件性能榨取 ---
    imgsz=1280,
    batch=32,                # RTX 5090 显存极大，32起步。如果爆显存（基本不可能），再降回16；如果没满，可以尝试 batch=64 或直接使用 batch=-1 (自动计算最大batch)
    device="0",
    amp=True,                # 【必开】激活 RTX 5090 Tensor Cores，速度狂飙
    workers=16,              # 充分利用 Ryzen 9 9950X3D 的多线程性能进行数据预处理
    cache=True,              # 只有 700 多张图，直接全缓存到 RAM 里，训练速度起飞
    
    # --- 优化器与学习率 ---
    optimizer='AdamW',       # 对于修改了结构、且数据集较小的情况，AdamW 通常比 SGD 收敛更稳、泛化更好
    cos_lr=True,             # 开启余弦退火学习率，后期寻找全局最优解时更平滑
    
    # --- 增强策略调整 (配合你的离线增强) ---
    close_mosaic=20,         # 在最后 20 个 epoch 关闭 Mosaic 增强，让模型在真实的图像分布上“收心”
    mixup=0.0,               # 坚决保持为 0，符合你“不能有物体相互遮挡”的真实工况要求
    degrees=10.0,            # 因为你离线做过大角度旋转，这里在线扰动设小一点即可
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, # 保持一定的色彩和亮度扰动，有助于应对真空辐射环境的光照变化
    
    deterministic=False,
    name="yolo11n-fce"
)
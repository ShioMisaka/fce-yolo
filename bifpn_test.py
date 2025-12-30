import torch
import torch.nn as nn
from ultralytics import YOLO

# 1. 加载模型
model_path = "yolo11-bifpn.yaml"
model = YOLO(model_path)

# 2. 获取底层的 PyTorch 模型并设为训练模式
network = model.model
network.train()  # ⭐ 关键修改：必须设为 train 模式，否则梯度会被 Detect 层阻断

# 3. 找到 BiFPN 模块并记录初始权重
bifpn_layers = []
for name, m in network.named_modules():
    if "BiFPN_Concat" in str(type(m)):
        # 记录引用和初始权重的副本
        bifpn_layers.append({
            "name": name,
            "module": m,
            "orig_w": m.w.detach().clone()
        })

if not bifpn_layers:
    print("❌ 未在模型中找到 BiFPN_Concat 模块。")
else:
    print(f"✅ 找到 {len(bifpn_layers)} 个 BiFPN 模块，准备进行梯度测试。")

# 4. 模拟前馈传播 (Forward Pass)
dummy_input = torch.randn(1, 3, 640, 640)

try:
    # 训练模式下，YOLO 输出通常是一个列表，包含不同尺度的特征图
    results = network(dummy_input)
    print("✅ 前馈传播测试通过！数据流正常。")
except Exception as e:
    print(f"❌ 前馈传播失败: {e}")
    exit()

# 5. 模拟反向传播 (Backward Pass)
print("\n=== 开始参数更新测试 ===")
optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
optimizer.zero_grad()

# ⭐ 改进的 Loss 计算：确保所有输出尺度的特征都参与反向传播
if isinstance(results, (list, tuple)):
    # 叠加所有尺度的 Loss，确保覆盖所有 Neck 分支
    loss = 0
    for res in results:
        if isinstance(res, torch.Tensor):
            loss += res.sum()
        elif isinstance(res, (list, tuple)): # 针对某些版本输出的 [cls, box] 结构
            loss += sum(x.sum() for x in res if isinstance(x, torch.Tensor))
else:
    loss = results.sum()

loss.backward()
optimizer.step()

# 6. 验证权重变化与梯度
print(f"{'模块名称':<15} | {'梯度(Grad)':<12} | {'更新状态':<8} | {'权重变化值':<15}")
print("-" * 65)



updated_count = 0
for layer in bifpn_layers:
    m = layer["module"]
    orig_w = layer["orig_w"]
    curr_w = m.w.detach()
    
    # 检查梯度是否存在
    grad_val = m.w.grad.abs().sum().item() if m.w.grad is not None else 0
    
    # 计算权重差异
    diff = torch.abs(orig_w - curr_w).sum().item()
    is_updated = diff > 0
    
    status = "✅ YES" if is_updated else "❌ NO"
    print(f"{layer['name']:<15} | {grad_val:<12.6f} | {status:<8} | {diff:.8f}")
    
    if is_updated:
        updated_count += 1

if updated_count == len(bifpn_layers):
    print("\n结论: 🚀 所有 BiFPN 模块均已成功参与梯度更新，模块完全可用！")
else:
    print("\n结论: ⚠️ 仍有模块未更新。")
    print("提示: 请检查 YAML 中 Detect 层的输入索引是否包含了未更新的层（如 [15, 18, 21]）。")
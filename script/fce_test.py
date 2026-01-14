import torch
import torch.nn as nn
from ultralytics import YOLO

# 1. 加载模型
model_path = "yolo11n-fce.yaml"
model = YOLO(model_path)

# 2. 获取底层的 PyTorch 模型并设为训练模式
network = model.model
network.train()  # 必须设为 train 模式，否则梯度会被 Detect 层阻断

# 3. 找到 CoordAtt 模块并记录初始权重
coordatt_layers = []
for name, m in network.named_modules():
    if "CoordAtt" in str(type(m)):
        # 记录引用和初始参数的副本
        coordatt_layers.append({
            "name": name,
            "module": m,
            # 记录 cv1 的初始权重
            "orig_weight": m.cv1.conv.weight.detach().clone() if hasattr(m.cv1, 'conv') else m.cv1.weight.detach().clone()
        })

# 4. 找到 BiFPN_Concat 模块并记录初始权重
bifpn_layers = []
for name, m in network.named_modules():
    if "BiFPN_Concat" in str(type(m)):
        bifpn_layers.append({
            "name": name,
            "module": m,
            "orig_w": m.w.detach().clone()
        })

# 打印找到的模块
print(f"✅ 找到 {len(coordatt_layers)} 个 CoordAtt 模块")
print(f"✅ 找到 {len(bifpn_layers)} 个 BiFPN_Concat 模块")

if not coordatt_layers and not bifpn_layers:
    print("❌ 未在模型中找到 CoordAtt 或 BiFPN_Concat 模块。")
    exit()

# 5. 模拟前馈传播 (Forward Pass)
dummy_input = torch.randn(1, 3, 640, 640)

try:
    results = network(dummy_input)
    print("✅ 前馈传播测试通过！数据流正常。")
except Exception as e:
    print(f"❌ 前馈传播失败: {e}")
    exit()

# 6. 模拟反向传播 (Backward Pass)
print("\n=== 开始参数更新测试 ===")
optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
optimizer.zero_grad()

# 改进的 Loss 计算：确保所有输出尺度的特征都参与反向传播
if isinstance(results, (list, tuple)):
    loss = 0
    for res in results:
        if isinstance(res, torch.Tensor):
            loss += res.sum()
        elif isinstance(res, (list, tuple)):  # 针对某些版本输出的 [cls, box] 结构
            loss += sum(x.sum() for x in res if isinstance(x, torch.Tensor))
else:
    loss = results.sum()

loss.backward()
optimizer.step()

# 7. 验证 CoordAtt 权重变化与梯度
print(f"\n{'='*70}")
print(f"CoordAtt 模块测试结果:")
print(f"{'='*70}")
print(f"{'模块名称':<25} | {'梯度(Grad)':<12} | {'更新状态':<8} | {'权重变化值':<15}")
print("-" * 70)

coordatt_updated_count = 0
for layer in coordatt_layers:
    m = layer["module"]
    orig_weight = layer["orig_weight"]

    # 获取 cv1 的当前权重
    curr_weight = m.cv1.conv.weight.detach() if hasattr(m.cv1, 'conv') else m.cv1.weight.detach()

    # 检查梯度是否存在
    cv1_weight = m.cv1.conv.weight if hasattr(m.cv1, 'conv') else m.cv1.weight
    grad_val = cv1_weight.grad.abs().sum().item() if cv1_weight.grad is not None else 0

    # 计算权重差异
    diff = torch.abs(orig_weight - curr_weight).sum().item()
    is_updated = diff > 0

    status = "✅ YES" if is_updated else "❌ NO"
    print(f"{layer['name']:<25} | {grad_val:<12.6f} | {status:<8} | {diff:.8f}")

    if is_updated:
        coordatt_updated_count += 1

# 8. 验证 BiFPN_Concat 权重变化与梯度
print(f"\n{'='*70}")
print(f"BiFPN_Concat 模块测试结果:")
print(f"{'='*70}")
print(f"{'模块名称':<25} | {'梯度(Grad)':<12} | {'更新状态':<8} | {'权重变化值':<15}")
print("-" * 70)

bifpn_updated_count = 0
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
    print(f"{layer['name']:<25} | {grad_val:<12.6f} | {status:<8} | {diff:.8f}")

    if is_updated:
        bifpn_updated_count += 1

# 9. 输出结论
print(f"\n{'='*70}")
print(f"测试总结:")
print(f"{'='*70}")
print(f"CoordAtt:    {coordatt_updated_count}/{len(coordatt_layers)} 个模块已更新")
print(f"BiFPN_Concat: {bifpn_updated_count}/{len(bifpn_layers)} 个模块已更新")

if coordatt_updated_count == len(coordatt_layers) and bifpn_updated_count == len(bifpn_layers):
    print("\n结论: 🚀 所有模块均已成功参与梯度更新，模型完全可用！")
else:
    print("\n结论: ⚠️ 仍有模块未更新，请检查模型配置。")

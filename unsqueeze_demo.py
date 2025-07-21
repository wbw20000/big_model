import torch
import numpy as np

print("=== PyTorch unsqueeze() 函数详解 ===\n")

# 1. 基本概念
print("1. 基本概念:")
print("unsqueeze() 函数用于在指定维度上增加一个大小为1的维度")
print("语法: tensor.unsqueeze(dim) 或 torch.unsqueeze(tensor, dim)\n")

# 2. 创建示例张量
print("2. 创建示例张量:")
# 一维张量
tensor_1d = torch.tensor([1, 2, 3, 4, 5])
print(f"原始一维张量: {tensor_1d}")
print(f"形状: {tensor_1d.shape}\n")

# 3. 使用 unsqueeze(1) - 在第1维增加维度
print("3. 使用 unsqueeze(1):")
tensor_unsqueezed = tensor_1d.unsqueeze(1)
print(f"unsqueeze(1) 后: \n{tensor_unsqueezed}")
print(f"形状: {tensor_unsqueezed.shape}")
print("解释: 从 [5] 变成 [5, 1]\n")

# 4. 不同维度的 unsqueeze 示例
print("4. 不同维度的 unsqueeze 示例:")
print(f"原始张量形状: {tensor_1d.shape}")
print(f"unsqueeze(0): {tensor_1d.unsqueeze(0).shape} - 在第0维增加")
print(f"unsqueeze(1): {tensor_1d.unsqueeze(1).shape} - 在第1维增加")
print(f"unsqueeze(-1): {tensor_1d.unsqueeze(-1).shape} - 在最后一维增加\n")

# 5. 二维张量示例
print("5. 二维张量示例:")
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"原始二维张量: \n{tensor_2d}")
print(f"形状: {tensor_2d.shape}")

tensor_2d_unsqueezed = tensor_2d.unsqueeze(1)
print(f"\nunsqueeze(1) 后: \n{tensor_2d_unsqueezed}")
print(f"形状: {tensor_2d.shape} -> {tensor_2d_unsqueezed.shape}\n")

# 6. 实际应用场景
print("6. 实际应用场景:")

# 场景1: 批处理维度
print("\n场景1: 为单个样本添加批处理维度")
single_image = torch.randn(3, 224, 224)  # 单张图片 [C, H, W]
print(f"单张图片形状: {single_image.shape}")
batch_image = single_image.unsqueeze(0)  # 添加批处理维度
print(f"添加批处理维度后: {batch_image.shape} - [batch, C, H, W]")

# 场景2: 广播运算
print("\n场景2: 广播运算")
a = torch.tensor([1, 2, 3])
b = torch.tensor([10, 20])
print(f"a 形状: {a.shape}")
print(f"b 形状: {b.shape}")
# 需要将 b 扩展维度以进行广播
b_expanded = b.unsqueeze(1)
print(f"b.unsqueeze(1) 形状: {b_expanded.shape}")
result = a + b_expanded
print(f"广播运算结果: \n{result}")

# 7. unsqueeze vs reshape
print("\n7. unsqueeze vs reshape 的区别:")
tensor = torch.tensor([1, 2, 3, 4])
print(f"原始张量: {tensor}, 形状: {tensor.shape}")
print(f"unsqueeze(1): {tensor.unsqueeze(1).shape} - 增加新维度")
print(f"reshape(-1, 1): {tensor.reshape(-1, 1).shape} - 重塑形状")
print("虽然结果形状相同，但 unsqueeze 更明确地表示增加维度的意图")

# 8. 多次 unsqueeze
print("\n8. 连续使用 unsqueeze:")
tensor = torch.tensor([1, 2, 3])
print(f"原始: {tensor.shape}")
tensor = tensor.unsqueeze(0)
print(f"第一次 unsqueeze(0): {tensor.shape}")
tensor = tensor.unsqueeze(2)
print(f"第二次 unsqueeze(2): {tensor.shape}")

# 9. squeeze 函数 - unsqueeze 的反操作
print("\n9. squeeze 函数 - unsqueeze 的反操作:")
tensor = torch.tensor([[[1], [2], [3]]])
print(f"原始张量形状: {tensor.shape}")
squeezed = tensor.squeeze()
print(f"squeeze() 后: {squeezed.shape} - 移除所有大小为1的维度")
squeezed_1 = tensor.squeeze(2)
print(f"squeeze(2) 后: {squeezed_1.shape} - 只移除第2维")

# 10. 注意事项
print("\n10. 注意事项:")
print("- dim 参数可以是负数，-1 表示最后一维")
print("- dim 的范围是 [-tensor.dim()-1, tensor.dim()]")
print("- unsqueeze 不会改变原张量，返回新张量")
print("- 常用于准备数据以匹配模型输入要求")
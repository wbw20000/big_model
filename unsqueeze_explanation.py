"""
PyTorch中 .unsqueeze(1) 函数详解

unsqueeze() 是 PyTorch 中用于增加张量维度的函数。
它在指定位置插入一个大小为 1 的新维度。

基本语法：
    tensor.unsqueeze(dim)
    或
    torch.unsqueeze(tensor, dim)

参数说明：
    dim: 要插入新维度的位置（可以是负数，-1表示最后一维）

示例说明：
"""

# 示例 1: 一维张量使用 unsqueeze(1)
"""
原始张量: [1, 2, 3, 4, 5]
形状: torch.Size([5])

使用 unsqueeze(1) 后:
[[1],
 [2],
 [3],
 [4],
 [5]]
形状: torch.Size([5, 1])

解释: 在第1维（索引为1的位置）增加了一个大小为1的维度
"""

# 示例 2: 不同维度的 unsqueeze
"""
假设有张量 x = [1, 2, 3]，形状为 [3]

x.unsqueeze(0) -> [[1, 2, 3]]        形状: [1, 3]
x.unsqueeze(1) -> [[1], [2], [3]]    形状: [3, 1]
x.unsqueeze(-1) -> [[1], [2], [3]]   形状: [3, 1] (与unsqueeze(1)相同)
"""

# 示例 3: 二维张量使用 unsqueeze(1)
"""
原始张量:
[[1, 2, 3],
 [4, 5, 6]]
形状: [2, 3]

使用 unsqueeze(1) 后:
[[[1, 2, 3]],
 [[4, 5, 6]]]
形状: [2, 1, 3]

解释: 在第1维位置插入了新维度
"""

# 实际应用场景

# 1. 为单个样本添加批处理维度
"""
深度学习模型通常期望输入有批处理维度。
如果你有一张图片，形状为 [3, 224, 224] (通道, 高度, 宽度)
使用 image.unsqueeze(0) 可以得到 [1, 3, 224, 224] (批次, 通道, 高度, 宽度)
"""

# 2. 广播运算
"""
当需要对不同形状的张量进行运算时，unsqueeze 可以帮助调整维度。
例如：
a 形状: [3]     值: [1, 2, 3]
b 形状: [2]     值: [10, 20]

为了让 a 和 b 相加，可以：
b.unsqueeze(1) -> [[10], [20]]  形状: [2, 1]

然后广播运算：
a + b.unsqueeze(1) = [[11, 12, 13],
                      [21, 22, 23]]
"""

# 3. 匹配网络层的输入要求
"""
某些网络层可能需要特定的输入维度。
例如，LSTM 期望输入形状为 [batch, seq_len, features]
如果你的数据是 [seq_len, features]，可以使用 unsqueeze(0) 添加批次维度
"""

# unsqueeze vs reshape 的区别
"""
虽然有时 unsqueeze 和 reshape 能达到相同的效果，但：
- unsqueeze 明确表示"增加一个新维度"
- reshape 表示"重新组织现有元素"

例如，对于张量 [1, 2, 3, 4]:
- tensor.unsqueeze(1) -> [[1], [2], [3], [4]]  # 明确增加维度
- tensor.reshape(-1, 1) -> [[1], [2], [3], [4]]  # 重新组织形状
"""

# squeeze 函数 - unsqueeze 的反操作
"""
squeeze() 移除所有大小为 1 的维度
squeeze(dim) 移除指定位置的大小为 1 的维度

例如：
张量形状 [1, 3, 1, 4]
squeeze() -> [3, 4]
squeeze(0) -> [3, 1, 4]
squeeze(2) -> [1, 3, 4]
"""

# 注意事项
"""
1. dim 参数的有效范围是 [-tensor.dim()-1, tensor.dim()]
2. unsqueeze 不会改变原张量，而是返回一个新张量
3. 常用于数据预处理，特别是准备模型输入
4. 在处理批次数据时特别有用
"""

# 代码示例（伪代码）
print("PyTorch unsqueeze 使用示例：")
print("\n# 导入 PyTorch")
print("import torch")
print("\n# 创建一维张量")
print("x = torch.tensor([1, 2, 3, 4, 5])")
print("print(x.shape)  # 输出: torch.Size([5])")
print("\n# 使用 unsqueeze(1)")
print("x_unsqueezed = x.unsqueeze(1)")
print("print(x_unsqueezed.shape)  # 输出: torch.Size([5, 1])")
print("\n# 查看结果")
print("print(x_unsqueezed)")
print("# 输出:")
print("# tensor([[1],")
print("#         [2],")
print("#         [3],")
print("#         [4],")
print("#         [5]])")
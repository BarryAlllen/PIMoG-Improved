import torch
import torch.nn as nn

# 创建一个示例输入张量
input_tensor = torch.randn(1, 3, 4, 4)  # 假设输入是一个 3 通道的 4x4 图像

# 使用 nn.Upsample 进行上采样操作
upsample = nn.Upsample(scale_factor=2, mode='bilinear')  # 使用双线性插值
output_tensor = upsample(input_tensor)

# 输出上采样后的张量形状
print("输出张量的形状:", output_tensor.shape)


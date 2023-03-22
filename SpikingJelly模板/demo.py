import torch
from spikingjelly.clock_driven import encoding, functional, neuron, layer

# x = torch.rand(2, 3, 3)     # [batch_size, w, h]
# y = encoding.WeightedPhaseEncoder(10)
# for i in range(10):
#     print(y(x))

# x = neuron.LIFNode(tau=1.1)
# xx = torch.rand(10, 3, 5, 5) * 3
# # for i in range(10):           # 单步模式
# #     print(x(xx[i]))
# x = layer.MultiStepContainer(x)
# print(x(xx))                    # 多步模式

a = torch.rand(10, 10, 3)
b = layer.Dropout2d(0.5)
print(b(a))
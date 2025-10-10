import torch.nn as nn
import torch
# 文献来源：EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation, CVPR2024
# 论文链接：https://arxiv.org/pdf/2405.06880

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # 构造激活函数层，根据传入的名称返回相应激活模块
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('未实现的激活函数层 [%s]' % act)
    return layer

class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int=16, kernel_size=3, groups=1, activation='relu'):
        super(LGAG, self).__init__()

        # 若卷积核尺寸为1，则强制设置组数为1以保持计算一致性
        if kernel_size == 1:
            groups = 1
        # 构建处理全局信息的卷积及归一化模块
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        # 构建处理局部信息的卷积及归一化模块
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        # 定义计算注意力权重的模块，由1x1卷积、批归一化和Sigmoid激活构成
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        # 利用前面定义的函数构造激活层
        self.activation = act_layer(activation, inplace=True)
        # ai缝合大王

    def forward(self, g, x):
        # 对全局输入进行卷积处理
        g1 = self.W_g(g)
        # 对局部输入进行卷积处理
        x1 = self.W_x(x)
        # 将全局和局部特征相加后，先通过激活函数再计算注意力权重
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        # 返回局部特征与计算出的注意力权重逐元素相乘的结果
        return x * psi

if __name__ == '__main__':
    # 生成示例数据：全局特征张量
    g = torch.randn(1, 32, 64, 64)
    # 生成示例数据：局部特征张量
    x = torch.randn(1, 64, 64, 64)

    # 实例化LGAG模块
    lgag = LGAG(F_g=32, F_l=64)

    # 输出示例数据的尺寸信息
    print("g输入尺寸:", g.shape)
    # ai缝合大王
    print("x输入尺寸:", x.shape)

    # 执行前向传播，并打印输出结果的尺寸
    output = lgag(g, x)
    print("输出尺寸:", output.shape)
    # ai缝合大王

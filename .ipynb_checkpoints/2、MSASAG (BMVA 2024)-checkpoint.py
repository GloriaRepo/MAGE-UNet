import torch
import torch.nn as nn
import torch.nn.functional as F

"""
论文链接：https://openaccess.thecvf.com/content/CVPR2024/papers/Fan_Bilateral_Event_Mining_and_Complementary_for_Event_Stream_Super-Resolution_CVPR_2024_paper.pdf
论文题目：Bilateral Event Mining and Complementary for Event Stream Super-Resolution (CVPR 2024)
"""

class GlobalExtraction(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        # 使用全局平均池化提取通道平均特征
        self.avgpool = self.globalavgchannelpool
        # 使用全局最大池化提取通道极值特征
        self.maxpool = self.globalmaxchannelpool
        # 利用1x1卷积和批归一化融合两种池化特征
        self.proj = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1)
        )

    def globalavgchannelpool(self, x):
        return x.mean(1, keepdim=True)  # 计算通道平均值

    def globalmaxchannelpool(self, x):
        return x.max(dim=1, keepdim=True)[0]  # 计算通道最大值

    def forward(self, x):
        x_clone = x.clone()
        x_avg = self.avgpool(x)    # 平均池化
        x_max = self.maxpool(x_clone)  # 最大池化
        # 拼接平均和最大池化结果
        cat = torch.cat((x_avg, x_max), dim=1)
        # 通过投影层融合特征
        proj = self.proj(cat)
        return proj

class ContextExtraction(nn.Module):
    def __init__(self, dim, reduction=None):
        super().__init__()
        # 若未指定降维比例，则不做降维，否则使用2
        self.reduction = 1 if reduction is None else 2
        self.dconv = self.DepthWiseConv2dx2(dim)
        self.proj = self.Proj(dim)

    def DepthWiseConv2dx2(self, dim):
        dconv = nn.Sequential(
            # 第一次深度可分离卷积：3x3卷积保持尺寸不变
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            # 第二次空洞卷积：3x3卷积，扩张率为2以扩大感受野
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        return dconv

    def Proj(self, dim):
        proj = nn.Sequential(
            # 1x1卷积降维，降低计算量
            nn.Conv2d(in_channels=dim, out_channels=dim // self.reduction, kernel_size=1),
            nn.BatchNorm2d(dim // self.reduction)
        )
        return proj

    def forward(self, x):
        x = self.dconv(x)  # 提取局部上下文特征
        x = self.proj(x)   # 降维处理
        return x

class MultiscaleFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 使用局部和全局两种信息提取模块
        self.local = ContextExtraction(dim)
        self.global_ = GlobalExtraction()
        self.bn = nn.BatchNorm2d(dim)  # 融合后的标准化层

    def forward(self, x, g):
        x_local = self.local(x)    # 局部上下文特征
        g_global = self.global_(g)  # 全局通道注意力特征
        fuse = self.bn(x_local + g_global)  # 融合并标准化
        return fuse

class MultiScaleGatedAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.multi = MultiscaleFusion(dim)
        # 自适应特征选择层，将通道映射为2个通道的权重
        self.selection = nn.Conv2d(dim, 2, 1)
        # 投影层用于调整通道数
        self.proj = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.bn_2 = nn.BatchNorm2d(dim)
        # 简单的卷积块用于进一步特征提取
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1)
        )

    def forward(self, x, g):
        # 保存原始输入的副本
        x_orig = x.clone()
        g_orig = g.clone()  # ai缝合大王

        # 第一阶段：利用多尺度融合提取局部与全局特征（粉色部分）
        multi = self.multi(x, g)
        # 第二阶段：自适应特征选择（黄色部分）
        multi = self.selection(multi)
        attention_weights = F.softmax(multi, dim=1)
        A, B = attention_weights.split(1, dim=1)  # 分离两个注意力通道
        x_att = A.expand_as(x_orig) * x_orig
        g_att = B.expand_as(g_orig) * g_orig
        x_att = x_att + x_orig
        g_att = g_att + g_orig
        # 第三阶段：特征交互与增强（蓝色部分）
        x_sig = torch.sigmoid(x_att)
        g_att_mod = x_sig * g_att
        g_sig = torch.sigmoid(g_att)
        x_att_mod = g_sig * x_att
        interaction = x_att_mod * g_att_mod
        # 第四阶段：特征重校准（紫色部分）
        projected = torch.sigmoid(self.bn(self.proj(interaction)))
        weighted = projected * x_orig
        y = self.conv_block(weighted)
        y = self.bn_2(y)
        return y

if __name__ == '__main__':
    x1 = torch.randn(1, 64, 32, 32)
    x2 = torch.randn(1, 64, 32, 32)
    Model = MultiScaleGatedAttn(dim=64)
    out = Model(x1, x2)
    print(out.shape)
    print("Input size:", x1.size(), x2.size())
    print("Output size:", out.size())
    # ai缝合大王

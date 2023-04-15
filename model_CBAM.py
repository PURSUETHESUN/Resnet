import torch.nn as nn
import torch
from self import self


#通道注意力模块实现
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, rotio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // rotio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // rotio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

#空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self,channel):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channel)
        self.sp = SpatialAttention()

    def forward(self,x):
        x = self.ca(x)*x
        x = self.sp(x)*x
        return x


class BasicBlock(nn.Module):#定义18层、34层残差结构
    #expansion表示在残差结构中，每一个残差快输出层的channel是输入层channel的几倍
    expansion = 1
    #out_channel是每一个残差快中，第1、2个卷积核的个数
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.cbam = CBAM(out_channel)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        #判断是否经过虚线下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        #主支线上正向传播
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        #加入CBAM模块
        out = self.cbam(out)

        #主线上的卷积结果+捷径分支的输出，再经过激活函数
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,#训练集的分类个数
                 include_top=True,#为了方便以后搭建更加复杂的网络
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64#表格中通过maxpooling之后所得到的深度

        self.groups = groups
        self.width_per_group = width_per_group

        #定义第一层卷积层，输入RGB图像：3
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        #定义最大池化层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #初始化4层残差卷积网络
        self.layer1 = self._make_layer(block, 64, blocks_num[0])#conv2_x
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)#conv3_x
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)#conv4_x
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)#conv5_x
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():#卷积层初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

     #channel指的每一层残差网络的第一个残差快中第一层卷积核个数，block_num指残差快个数
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        #第一个残差结构压入layers[0]
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        #将剩余一系列实线残差结构压入layers
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,#输入的特征矩阵深度
                                channel,#每一个残差快的第一个卷积核个数（深度）
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)#conv2_x
        x = self.layer2(x)#conv3_x
        x = self.layer3(x)#conv4_x
        x = self.layer4(x)#conv5_x

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34_CBAM(num_classes=5, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


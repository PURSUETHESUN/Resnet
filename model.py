import torch.nn as nn
import torch
import torchvision.models.resnet

#首先定义残差块
class BasicBlock(nn.Module):#定义18层、34层残差结构
    #expansion表示在残差结构中，每一个残差快输出层的channel是输入层channel的几倍
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        #判断是否经过虚线下采样，高宽减半，通道翻倍，只有conv3_*,conv4_*,conv5_*的第一个残差快才进行下采样，conv2_*都是实线
        if self.downsample is not None:
            identity = self.downsample(x)

        #主支线上正向传播
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        #主线上的卷积结果+捷径分支的输出，再经过激活函数
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):#定义50、101、152层残差结构
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvid
    ia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4
    #out_channel是指残差快中第1个卷积层的卷积核个数，乘以拓展因子为残差快真实输出
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups
        # width是指一通计算以后第1,2层的深度,理论上应该与out_channel相等

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        #如果为none，对应实线残差结构，如果不为none，对应虚线残差结构
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 主线上的卷积结果+捷径分支的输出，再经过激活函数
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,#BasicBlock，即所对应的残差结构
                 blocks_num,#[3,4,6,3]
                 num_classes=1000,#训练集的分类个数
                 include_top=True,#为了方便以后搭建更加复杂的网络
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64#表格中通过maxpooling之后所得到的深度都是64

        self.groups = groups
        self.width_per_group = width_per_group

        #定义第一层卷积层，输入RGB图像：3
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        #定义最大池化层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #初始化conv2_*,conv3_*,conv4_*,conv5_*4层残差卷积网络
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
        if stride != 1 or self.in_channel != channel * block.expansion:#判断是否需要虚线下采样
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


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)

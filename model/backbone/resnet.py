import torch
from torch import nn

"""
    # in_channel:输入block之前的通道数
    # channel:在block中间处理的时候的通道数（这个值是输出维度的1/4)
    # channel * block.expansion:输出的维度
"""

#对应18、34层残差结构，其中stride可调整
class BasicBlock(nn.Module):
    #用于调节残差结构主分支上卷积核个数
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU6()
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        '''在18、34层的结构中
           在conv2_x这个阶段对应的是实线残差结构
           conv3_x及之后阶段的第一个残差块都是虚线残差结构，每阶段的第一个残差结构中第一层分辨率会降为上层的一半，
           如conv2_x对应输出56*56，conv3_x需要输出28*28大小特征图，则对应虚线残差结构中捷径分支的1*1卷积层stride=2即downsample操作
        '''
        self.downsample = downsample

    def forward(self, x):
        identity = x
        #对应虚线残差结构
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out += identity

        out = self.relu(out)
        return out

#对应50、101、152层残差结构
class BottleNeck(nn.Module):
    # 用于调节残差结构主分支上卷积核个数，第三层个数为4倍
    expansion = 4

    #stride=2对应虚线残差结构，=1则为实线残差结构
    def __init__(self, in_channel, channel, stride=1, downsample=None):
    # resnext:区别在于中间卷积层使用分组卷积，groups=32, width_per_group=4)
    # def __init__(self, in_channel, channel, stride=1, downsample=None, groups=1, width_per_group=64):
        super().__init__()
        #width：根据传参，决定resnet和resnext残差块中第一、二个卷积层卷积核个数，groups=32，则width=128
        #width = int(channenl*(wodth_per_group / 64))* groups

        #resnext中输出通道对应width=128
        self.conv1 = nn.Conv2d(in_channel, channel, kernel_size=1, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)      #传入width

        #resnext中输入、输出通道对应width=128，另外传入参数groups=32
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3,  stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)      #传入width

        #resnext中输入width
        self.conv3 = nn.Conv2d(channel, channel * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channel * self.expansion)

        self.relu = nn.ReLU6(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))  # bs,c,h,w
        out = self.relu(self.bn2(self.conv2(out)))  # bs,c,h,w
        out = self.relu(self.bn3(self.conv3(out)))  # bs,4c,h,w

        if (self.downsample != None):
            residual = self.downsample(residual)

        out += residual
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, groups=1, width_per_group=64):
        super().__init__()
        # 定义输入模块的维度，最大池化之后对应通道数
        self.in_channel = 64
        self.groups=groups
        self.width_per_group=width_per_group

        ### stem layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        ### 构建每个阶段的残差块
        #对应conv2_x，步长默认为1，实线残差结构
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 对应conv3_x，步长为2，虚线残差结构
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512 * block.expansion, num_classes)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        ##stem layer
        out = self.relu(self.bn1(self.conv1(x)))  # bs,112,112,64
        out = self.maxpool(out)  # bs,56,56,64

        ##layers:
        out = self.layer1(out)  # bs,56,56,64*4
        out = self.layer2(out)  # bs,28,28,128*4
        out = self.layer3(out)  # bs,14,14,256*4
        out = self.layer4(out)  # bs,7,7,512*4

        ##classifier
        out = self.avgpool(out)  # bs,1,1,512*4
        out = out.reshape(out.shape[0], -1)  # bs,512*4
        out = self.classifier(out)  # bs,1000
        out = self.softmax(out)

        return out

    # channel为每块中第一个卷积核个数 ，stride=1、2对应实线和虚线残差结构
    def _make_layer(self, block, channel, blocks, stride=1):

        # downsample 主要用来处理H(x)=F(x)+x中F(x)和x的channel维度不匹配问题，即对残差结构的输入进行升维，在做残差相加的时候，必须保证残差的纬度与真正的输出维度（宽、高、以及深度）相同
        # 比如步长！=1 或者 in_channel!=channel&self.expansion
        downsample = None
        #对于50、101、152中conv2_x只调节通道数为原来4倍，不改变长宽
        #从conv3_x开始，不管多深都需要一次下采样，传入stride=2
        if (stride != 1 or self.in_channel != channel * block.expansion):
            downsample = nn.Conv2d(self.in_channel, channel * block.expansion, stride=stride, kernel_size=1,
                                        bias=False)
        layers = []
        '''block()实例化了实线与虚线残差结构
           例：18、34对应的conv2_x 大小与通道均不变，而50及以后的，需要调整通道数为初始的4倍
        '''
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        #经过虚线残差结构后得到的通道数
        self.in_channel = channel * block.expansion
        #对应构建重复的残差结构，除了开始的虚线后续重复的都是实线残差结构
        for _ in range(1, blocks):
            layers.append(block(self.in_channel, channel, groups=self.groups, width_per_group=self.width_per_group))
        return nn.Sequential(*layers)


def ResNet50(num_classes=1000):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=1000):
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=1000):
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes=num_classes)


if __name__ == '__main__':
    input = torch.randn(50, 3, 224, 224)
    resnet50 = ResNet50(1000)
    # resnet101=ResNet101(1000)
    # resnet152=ResNet152(1000)
    out = resnet50(input)
    print(out.shape)

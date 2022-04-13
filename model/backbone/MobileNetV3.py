import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable, List
from functools import partial

def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

class ConvBnActivation(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size = 3, stride = 1, groups = 1, norm_layer = None, activation_layer = None ):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBnActivation, self).__init__(nn.Conv2d(in_channels=in_channel,
                                                         out_channels=out_channel,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_channel),
                                               activation_layer(inplace=True))

class Se(nn.Module):
    def __init__(self, in_channel, ratio):
        super(Se, self).__init__()
        squeeze_channel = _make_divisible(in_channel // ratio, 8 )
        self.fc1 = nn.Conv2d(in_channel, squeeze_channel, 1)
        self.fc2 = nn.Conv2d(squeeze_channel, in_channel, 1)
    def forward(self, x):
        scale = nn.AdaptiveAvgPool2d((1,1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x

#MobielNetV3中每个block的参数配置
class InvertedResidualConfig:
    def __init__(self,
                 input_c: int,
                 kernel: int,
                 expanded_c: int,
                 out_c: int,
                 use_se: bool,
                 activation: str,
                 stride: int,
                 width_multi: float):   #调节卷积核个数的倍率因子
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        #对应模块中1*1卷积核个数
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"  # whether using h-swish activation
        self.stride = stride

    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)

class InvertedResidual(nn.Module):
    def __init__(self, cnf : InvertedResidualConfig, norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()
        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        #expand
        if cnf.expanded_c != cnf.input_c:
            layers.append(ConvBnActivation(cnf.input_c, cnf.expanded_c, kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer))

        #depthwise
        layers.append(ConvBnActivation(cnf.input_c, cnf.expanded_c, kernel_size=cnf.kernel, groups=cnf.expanded_c, norm_layer=norm_layer, activation_layer=activation_layer))

        if cnf.use_se:
            layers.append(Se(cnf.expanded_c))

        layers.append(ConvBnActivation(cnf.input_c, cnf.expanded_c, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result += x

        return  x

class MobileNetV3(nn.Module):
    def __init__(self,
                 inverted_residual_setting: List[InvertedResidualConfig],
                 last_channel,
                 num_classes = 1000,
                 block = None,
                 norm_layer = None):
        super(MobileNetV3, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps= 0.001, momentum=0.01)

        layers = []
        firstconv_channel = inverted_residual_setting[0].input_c
        layers.append(ConvBnActivation(3, firstconv_channel, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.Hardswish))

        #构建block块
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        #构建最后几层卷积层
        lastconv_input_c = inverted_residual_setting[-1].out_c
        lastconv_output_C = 6 * lastconv_input_c
        layers.append(ConvBnActivation(lastconv_input_c, lastconv_output_C, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_C, last_channel),
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(last_channel, num_classes))

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out









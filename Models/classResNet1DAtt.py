import torch
import torch.nn as nn

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict
from Models.classSimpleModel import Generator

class Conv1dAuto(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2,)  # dynamic add padding based on the kernel_size


conv3x3 = partial(Conv1dAuto, kernel_size=3, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()
        self.activation = None

        # print('residual block')

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activation(x)

        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, conv, expansion=1, downsampling=1, *args, **kwargs):
        # print('ResNetResudualblock')
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
            {
                'conv': nn.Conv1d(self.in_channels, self.expanded_channels, kernel_size=1,
                                  stride=self.downsampling, bias=False),
                'bn': nn.BatchNorm1d(self.expanded_channels)

            })) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        # print('ici expanded channel : out = {}, expension = {} expended channel = {}'.format(self.out_channels,
        #                                                                                      self.expansion,
        #                                                                                      self.out_channels * self.expansion))
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                                      'bn': nn.BatchNorm1d(out_channels)}))

class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels, conv, *args, **kwargs)
        self.activation = activation(inplace=True)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(inplace=True),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )



class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, conv=conv3x3, *args, **kwargs):
        # print('ResNetBottleneck')
        super().__init__(in_channels, out_channels, conv, expansion=4, *args, **kwargs)
        self.activation = activation(inplace=True)
        self.blocks = nn.Sequential(
           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation(inplace=True),
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             activation(inplace=True),
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        # print('ResNet Layer : in = {}, out = {}'.format(in_channels, out_channels))
        #
        # print('downsamplig = {}'.format(downsampling))
        # block_1 = block(in_channels, out_channels, activation, *args, **kwargs, downsampling=downsampling)
        # block_suite = block(out_channels * block.expansion,
        #             out_channels, activation, downsampling=1, *args, **kwargs)
        self.blocks = nn.Sequential(
            block(in_channels, out_channels, activation, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, activation, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )
        # print('block suite : in channel = {}, expended = {}'.format(out_channels * block.expansion, block_suite.out_channels * block_suite.expansion))

    def forward(self, x):
        x = self.blocks(x)
        return x


class Gate(nn.Module):
    def __init__(self, in_channels, blocks_sizes):
        super(Gate, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, blocks_sizes[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(blocks_sizes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x0):
        # print('size x0 = {}'.format(x0.size()))
        x0 = self.conv1(x0)
        # print('size x0 = {}'.format(x0.size()))
        x0 = self.bn1(x0)
        # print('size x0 = {}'.format(x0.size()))
        x0 = self.relu(x0)
        # print('size x0 = {}'.format(x0.size()))
        x0 = self.maxpool(x0)
        # print('size after block x0 = {}'.format(x0.size()))

        return x0

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """

    def __init__(self, in_channels=1, blocks_sizes=[64, 128, 256, 512], deepths=[2, 2, 2, 2],
                 activation=nn.ReLU, block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes

        self.gate = Gate(in_channels, blocks_sizes)

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        # print(self.in_out_block_sizes)
        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation,
                        block=block, *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]
        ])

        self.avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # print(self.gate)
        x = self.gate(x)
        # print(self.blocks)
        for block in self.blocks:
            x = block(x)

        x = self.avg(x)
        x = x.view(x.size(0), -1)
        return x


class ResNet(nn.Module):

    def __init__(self, in_channels, output_type, output_shape, *args, **kwargs):
        super().__init__()

        self.m_type = 'ResNet'

        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.generator = Generator(self.m_type, output_type, None, self.encoder.blocks[-1].blocks[-1].expanded_channels, output_shape)

    def forward(self, x):
        x = self.encoder(x)
        x = self.generator(x)
        return x


def resnet18_little(in_channels, output_type, output_shape):
    return ResNet(in_channels, output_type, output_shape, block=ResNetBasicBlock, blocks_sizes=[16, 32, 64, 128],
                  deepths=[2, 2, 2, 2])

def resnet18(in_channels, output_type, output_shape):
    return ResNet(in_channels, output_type, output_shape, block=ResNetBasicBlock, deepths=[2, 2, 2, 2])


def resnet34(in_channels, output_type, output_shape):
    return ResNet(in_channels, output_type, output_shape, block=ResNetBasicBlock, deepths=[3, 4, 6, 3])


def resnet50(in_channels, output_type, output_shape):
    return ResNet(in_channels, output_type, output_shape, block=ResNetBottleNeckBlock, deepths=[3, 4, 6, 3])

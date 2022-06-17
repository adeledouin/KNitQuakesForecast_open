import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
import torch
from Models.classSimpleModel import Generator, TimeDistributed

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, bias=False)

def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)

def conv_output_size(l_in, kernel_size, stride=None, padding=None, dilation=None):
    if stride is None:
        stride = 1
    if padding is None:
        padding = 0
    if dilation is None:
        dilation = 1

    l_out = int((l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    # print('dans ce conv : lin = {}, channel = {}, kernel = {} '.format(l_in, self.output_channel, self.maxpoll2d_kernel_size))
    # print('lout = {}, avec padding = {}'.format(l_out, int((kernel_size - 1)/2)))

    return l_out

class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        ## BasicBlock
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        #activation
        out = self.relu(out)

        return out


class BasicBlock5x5(nn.Module):
    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()
        self.conv1 = conv5x5(inplanes5, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        ## BasicBlock
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # d = residual.shape[2] - out.shape[2]
        # out1 = residual[:,:,0:-d] + out

        # activation
        out1 = self.relu(out)
        # out += residual

        return out1


class BasicBlock7x7(nn.Module):
    expansion = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        super(BasicBlock7x7, self).__init__()
        self.conv1 = conv7x7(inplanes7, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv7x7(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        ## BasicBlock
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # d = residual.shape[2] - out.shape[2]
        # out1 = residual[:, :, 0:-d] + out

        ## activation
        out1 = self.relu(out)
        # out += residual

        return out1


class Gate(nn.Module):
    def __init__(self, input_channel):

        super(Gate, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

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

class MSResNet(nn.Module):
    def __init__(self, input_channel, output_type, output_shape, blocks_sizes=[32, 64, 128, 256], deepths=[1, 1, 1, 1]):
        self.inplanes3 = blocks_sizes[0]
        self.inplanes5 = blocks_sizes[0]
        self.inplanes7 = blocks_sizes[0]

        super(MSResNet, self).__init__()

        self.m_type = 'MSResNet'

        self.gate = Gate(input_channel)

        ## ResNetEncoder pour 3x3
        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, blocks_sizes[0], deepths[0], stride=2)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, blocks_sizes[1], deepths[1], stride=2)
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3, blocks_sizes[2], deepths[2], stride=2)
        self.layer3x3_4 = self._make_layer3(BasicBlock3x3, blocks_sizes[3], deepths[3], stride=2)

        # maxplooing kernel size: 16, 11, 6
        self.maxpool3 = nn.AvgPool2d(kernel_size=16, stride=1, padding=0)

        ## ResNetEncoder pour 5x5
        self.layer5x5_1 = self._make_layer5(BasicBlock5x5, blocks_sizes[0], deepths[0], stride=2)
        self.layer5x5_2 = self._make_layer5(BasicBlock5x5, blocks_sizes[1], deepths[1], stride=2)
        self.layer5x5_3 = self._make_layer5(BasicBlock5x5, blocks_sizes[2], deepths[2], stride=2)
        self.layer5x5_4 = self._make_layer5(BasicBlock5x5, blocks_sizes[3], deepths[3], stride=2)
        self.maxpool5 = nn.AvgPool2d(kernel_size=16, stride=1, padding=0)

        ## ResNetEncoder pour 7x7
        self.layer7x7_1 = self._make_layer7(BasicBlock7x7, blocks_sizes[0], deepths[0], stride=2)
        self.layer7x7_2 = self._make_layer7(BasicBlock7x7, blocks_sizes[1], deepths[1], stride=2)
        self.layer7x7_3 = self._make_layer7(BasicBlock7x7, blocks_sizes[2], deepths[2], stride=2)
        self.layer7x7_4 = self._make_layer7(BasicBlock7x7, blocks_sizes[3], deepths[3], stride=2)
        self.maxpool7 = nn.AvgPool2d(kernel_size=16, stride=1, padding=0)

        # self.drop = nn.Dropout(p=0.2)
        # self.fc = nn.Linear(256*3, num_classes)
        self.generator = Generator(self.m_type, output_type, None, 256*3, output_shape)

        # todo: modify the initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        ### equivalant to shortcut
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        ### ResNetLayer pour 3x3
        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        # print('layers {}'.format(np.size(layers)))
        # print(layers)

        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        ### equivalant to shortcut
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        ### ResNetLayer pour 5x5
        layers = []
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)


    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        ### equivalant to shortcut
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        ### ResNetLayer pour 7x7
        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)

    def forward(self, x0):
        x0 = self.gate(x0)
        # print('size x0 = {}'.format(x0.size()))
        x = self.layer3x3_1(x0)
        # print('size x = {}'.format(x.size()))
        x = self.layer3x3_2(x)
        # print('size x = {}'.format(x.size()))
        x = self.layer3x3_3(x)
        # print('size x = {}'.format(x.size()))
        # x = self.layer3x3_4(x)
        # print('size x = {}'.format(x.size()))
        x = self.maxpool3(x)
        # print('size x = {}'.format(x.size()))

        # print('size x0 = {}'.format(x0.size()))
        y = self.layer5x5_1(x0)
        # print('size y = {}'.format(y.size()))
        y = self.layer5x5_2(y)
        # print('size y = {}'.format(y.size()))
        y = self.layer5x5_3(y)
        # print('size y = {}'.format(y.size()))
        # y = self.layer5x5_4(y)
        # print('size y = {}'.format(y.size()))
        y = self.maxpool5(y)
        # print('size y = {}'.format(y.size()))

        # print('size x0 = {}'.format(x0.size()))
        z = self.layer7x7_1(x0)
        # print('size x0 = {}'.format(x0.size()))
        z = self.layer7x7_2(z)
        # print('size z = {}'.format(z.size()))
        z = self.layer7x7_3(z)
        # print('size z = {}'.format(z.size()))
        # z = self.layer7x7_4(z)
        # print('size z = {}'.format(z.size()))
        z = self.maxpool7(z)
        # print('size z = {}'.format(z.size()))

        out = torch.cat([x, y, z], dim=1)
        # print('size out = {}'.format(out.size()))

        out = out.squeeze()
        # print('size out = {}'.format(out.size()))
        # out = self.drop(out)
        # out1 = self.generator(out)
        # print('size out1 = {}'.format(out1.size()))

        out = self.generator(out)

        return out


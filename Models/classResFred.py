import torch
import torch.nn as nn
from functools import partial
from Models.classSimpleModel import Generator
from collections import OrderedDict


class Conv1dAuto(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (int(self.dilation[0] * (self.kernel_size[0] - 1) / 2),)  # dynamic add padding based on the kernel_size


conv3x3 = partial(Conv1dAuto, kernel_size=3, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()
        self.activation = None
        self.dropout = None

        # print('residual block')

    def forward(self, x):
        residual = x
        # print('x enter size = {}'.format(x.size()))
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
            # print('residual size = {}'.format(residual.size()))
        x = self.blocks(x)
        # print('block size = {}'.format(x.size()))
        x += residual
        # print('after sum = {}'.format(x.size()))
        x = self.activation(x)
        # x = self.dropout(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, conv, dilatation=1, *args, **kwargs):
        # print('ResNetResudualblock')
        super().__init__(in_channels, out_channels)
        self.dilatation, self.conv = dilatation, conv
        self.shortcut = nn.Sequential(OrderedDict(
            {
                'conv': nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1,
                                  dilation=self.dilatation, bias=True),
                'bn': nn.BatchNorm1d(self.out_channels)

            })) if self.should_apply_shortcut else None
        # print(self.shortcut)

def conv_bn(in_channels, out_channels, conv, cardinality=1, *args, **kwargs):
    if cardinality == 1:
        return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                                          'bn': nn.BatchNorm1d(out_channels)}))
    assert not in_channels % cardinality
    _d = in_channels // cardinality

    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, groups=_d, *args, **kwargs),
                                      'bn': nn.BatchNorm1d(out_channels)}))


class ResNetBottleNeckBlock(ResNetResidualBlock):

    def __init__(self, in_channels, inter_channels, out_channels, cardinality, dilatation_rate=1, activation=nn.ReLU,
                 conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels, conv, dilatation_rate, *args, **kwargs)
        self.activation = activation(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, inter_channels, self.conv, kernel_size=1),
            activation(inplace=True),
            conv_bn(inter_channels, inter_channels, self.conv, cardinality, kernel_size=3, dilation=dilatation_rate),
            activation(inplace=True),
            conv_bn(inter_channels, self.out_channels, self.conv, kernel_size=1),
        )
        # print(self.blocks)


class Gate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Gate, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, kernel_size=7, stride=2, bias=True)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x0):
        # print('size x0 = {}'.format(x0.size()))
        x0 = self.conv1(x0)
        # print('size x0 = {}'.format(x0.size()))
        x0 = self.bn1(x0)
        # print('size x0 = {}'.format(x0.size()))
        x0 = self.relu(x0)
        # print('size en gate = {}'.format(x0.size()))

        return x0


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """

    def __init__(self, in_channels, cardinality=1, activation=nn.ReLU, block=ResNetBottleNeckBlock, *args, **kwargs):
        super().__init__()

        self.gate = Gate(in_channels, 256)

        self.blocks = nn.ModuleList([block(256, 256, 512, cardinality, dilatation_rate=1,
                                             activation=activation, *args, **kwargs),
                                    *[block(512, 256, 512, cardinality, dilatation_rate=2 ** i,
                                             activation=activation, *args, **kwargs) for i
                                       in range(1, 3)],
                                     *[block(512, 512, 512, cardinality, dilatation_rate=2 ** i,
                                             activation=activation, *args, **kwargs) for i in
                                       range(3)]])
        # print(self.blocks)

    def forward(self, x):
        # print(self.gate)
        x = self.gate(x)
        for block in self.blocks:
            # print(x.size())
            x = block(x)

        return x


class ResNetDecoder(nn.Module):
    """
    ResNet decoder composed by increasing different layers with increasing features.
    """

    def __init__(self, into_lstm=False):
        super().__init__()

        self.into_lstm = into_lstm
        self.activation = nn.ReLU()
        if into_lstm:
            self.lstm = nn.LSTM(
                input_size=512,
                hidden_size=256,
                num_layers=1,
                batch_first=True,
            bidirectional=True)
            'bla'
        else:
            self.avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        if self.into_lstm:
            r_in = torch.transpose(x, 1, 2)
            r_out, (h_n, c_n) = self.lstm(r_in)
            r_out = self.activation(r_out)
            x = r_out[:, -1, :]
            x = x.view(x.size(0), -1)
        else:
            x = self.avg(x)
            x = x.view(x.size(0), -1)

        return x


class ResNet(nn.Module):

    def __init__(self, in_channels, output_type, output_shape, *args, **kwargs):
        super().__init__()

        self.m_type = 'ResNetFred'

        self.encoder = ResNetEncoder(in_channels, 32, *args, **kwargs)
        self.decoder = ResNetDecoder(into_lstm=False)
        self.generator = Generator(self.m_type, output_type, None, 512, output_shape)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.generator(x)
        return x
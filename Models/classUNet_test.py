import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict
from Models.classSimpleModel import Generator
import numpy as np


try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None


# basic blocks pour resnet encoder
class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2,)  # dynamic add padding based on the kernel_size


conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

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
                'conv': nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                                  stride=self.downsampling, bias=False),
                'bn': nn.BatchNorm2d(self.expanded_channels)

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
                                      'bn': nn.BatchNorm2d(out_channels)}))


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


# basic blocks pour decoder
class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class Attention(nn.Module):

    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == 'scse':
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


# Decoder
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
            *args, **kwargs
    ):
        super().__init__()

        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None, ratioh=2, ratiow=2):

        x = F.interpolate(x, scale_factor=(ratioh, ratiow), mode="nearest")
        #print('size x decoder block  = {}'.format(x.size()))

        if skip is not None:
            # print('size skip  = {}'.format(skip.size()))
            x = torch.cat([x, skip], dim=1)
            # print('size x decoder block  = {}'.format(x.size()))
            x = self.attention1(x)
            # print('size x decoder block  = {}'.format(x.size()))
        x = self.conv1(x)
        # print('size x decoder block  = {}'.format(x.size()))
        x = self.conv2(x)
        # print('size x decoder block  = {}'.format(x.size()))
        x = self.attention2(x)
        # print('size x decoder block  = {}'.format(x.size()))
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, *args, **kwargs):
        conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels=[256, 128, 64, 32, 16], n_blocks=5,
            use_batchnorm=True, attention_type=None, center=False, *args, **kwargs
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        self.decoder_channels = decoder_channels
        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        #print('decoder unet : ')
        #print('head channel = {}'.format(head_channels))
        #print('in channel = {}'.format(in_channels))
        #print('skip channel = {}'.format(skip_channels))
        #print('out channel = {}'.format(out_channels))

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        # print('size x  = {}'.format(np.shape(features)))
        # for i in range(np.size(features)):
        #     print('shape of feature {} ={}'.format(i, features[i].size()))

        # features = features[1:]    # remove first skip with same spatial resolution
        # print('shape x  = {}'.format(np.shape(features)))
        features = features[::-1]  # reverse channels to start from head of encoder
        # print('shape x  = {}'.format(np.shape(features)))
        # for i in range(np.size(features)):
        #     print('shape of feature {} ={}'.format(i, features[i].size()))

        head = features[0]
        # print('size head  = {}'.format(head.size()))
        skips = features[1:]
        # for i in range(np.size(skips)):
            #print('shape of skips {} ={}'.format(i, skips[i].size()))

        x = self.center(head)
        #print('size x  = {}'.format(x.size()))
        for i, decoder_block in enumerate(self.blocks):
            # #print('skips jusqu a i = {}'.format(len(skips)))
            skip = skips[i] if i < len(skips) else None
            # if skip is None:
                #print('dans block {} no skip anymore !'.format(i))
            ratioh = skip.size(-2)/x.size(-2) if skip is not None else 2
            ratiow = skip.size(-1)/x.size(-1) if skip is not None else 2
            x = decoder_block(x, skip, ratioh, ratiow)
            #print('size blocks  = {}'.format(x.size()))

        return x


# Encoder Resnet
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

        self.conv1 = nn.Conv2d(in_channels, blocks_sizes[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(blocks_sizes[0])
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

        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # print(self.gate)
        ftrs = []

        x = self.gate(x)
        ftrs.append(x)
        #print('size x  gate = {}'.format(x.size()))
        # print(self.blocks)
        for block in self.blocks:
            x = block(x)
            ftrs.append(x)
            #print('size x block = {}'.format(x.size()))

        x = self.avg(x)
        # print('size x  encoder = {}'.format(x.size()))
        return ftrs


# Unet
class UNet(nn.Module):
    def __init__(self, in_channels, output_type, output_shape, seq_size, h_input, w_input, *args, **kwargs):
        super().__init__()
        self.m_type = 'UNet'

        self.seq_size = seq_size
        self.in_channels = in_channels
        self.h = h_input
        self.w = w_input

        self.encoder = ResNetEncoder(in_channels*seq_size, *args, **kwargs)

        #print(self.encoder)
        # print('in Unet : encoder channels', [in_channels, self.encoder.blocks_sizes[0]] + self.encoder.blocks_sizes)

        self.decoder = UnetDecoder(
            encoder_channels=[in_channels, self.encoder.blocks_sizes[0]] + self.encoder.blocks_sizes,
            center=False,
            attention_type=None,
            *args, **kwargs
        )

        #print(self.decoder)

        self.generator = Generator(self.m_type, output_type, None, self.decoder.decoder_channels[-1],
                                   output_shape, lin=False)

        #print(self.generator)


    def forward(self, x):
        #print('size x UNet = {}'.format(x.size()))
        x = x.view(-1, self.seq_size * self.in_channels, self.h, self.w)
        #print('size x UNet = {}'.format(x.size()))

        features = self.encoder(x)

        out = self.decoder(*features)
        #print('size x Unet = {}'.format(out.size()))

        out = self.generator(out, upsampling=[70/36, 65/34])
        #print('size x Unet = {}'.format(out.size()))

        return out


def unet_basic(in_channels, output_type, output_shape, seq_size, h_input, w_input):
    return UNet(in_channels, output_type, output_shape, seq_size, h_input, w_input)

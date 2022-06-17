import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict
from Models.classSimpleModel import Generator


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Encoder(nn.Module):
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        print('size x  encoder = {}'.format(x.size()))
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            print('size x  encoder = {}'.format(x.size()))
            ftrs.append(x)
            print('size x  encoder = {}'.format(x.size()))
            x = self.pool(x)
            print('size x  encoder = {}'.format(x.size()))
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, in_channels, output_type, output_shape, seq_size, h_input, w_input, enc_chs, dec_chs,
                 num_class=1, retain_dim=False,
                 out_sz=(572, 572)):
        super().__init__()
        self.m_type = 'UNet'

        self.seq_size = seq_size
        self.in_channels = in_channels
        self.h = h_input
        self.w = w_input

        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim = retain_dim
        self.out_sz = out_sz

    def forward(self, x):
        print('size x  UNet = {}'.format(x.size()))
        x = x.view(-1, self.seq_size * self.in_channels, self.h, self.w)
        print('size x  UNet = {}'.format(x.size()))
        enc_ftrs = self.encoder(x)
        print('size x  UNet = {}'.format(x.size()))

        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out


def unet_basic(in_channels, output_type, output_shape, seq_size, h_input, w_input):
    enc_chs = (in_channels * seq_size, 64, 128, 256, 512)
    dec_chs = (512, 256, 128, 64)
    return UNet(in_channels, output_type, output_shape, seq_size, h_input, w_input, enc_chs, dec_chs)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# ------- Commun blocs models ------- #
class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, m_type, output_type, layers_param, input_size, output_size, lin=True):
        super(Generator, self).__init__()

        self.layer_param = layers_param
        self.output_type = output_type
        self.input_type = input_size
        self.output_size = output_size
        self.m_type = m_type

        if lin:
            self.predict = torch.nn.Linear(input_size, output_size)  # output layer
        else:
            self.predict = nn.Conv2d(input_size, output_size, kernel_size=3,
                               padding=self.padding_same(3))

    def padding_same(self, kernel_size):
        # print(int((kernel_size - 1)/2))
        return [int((kernel_size - 1) / 2), int((kernel_size - 1) / 2)]

    def forward(self, x, upsampling=[1, 1]):

        # print(x.shape)
        if upsampling != [1, 1]:
            x = F.interpolate(x, scale_factor=upsampling, mode='bilinear', align_corners=True)
        x = self.predict(x)

        if self.output_type == 'class':
            # if self.m_type == 'conv_intolstm':
            #     output = F.softmax(x, dim=1)
            # else:
            #     output = F.softmax(x)
            output = x
        else:
            output = x
            # output = F.tanh(x)

        return output


class Dense1d(nn.Module):
    def __init__(self, output_type, channel_input, l_input, output_size, layers_param):
        super(Dense1d, self).__init__()

        self.m_type = 'dense1d'

        self.output_type = output_type
        self.dropout = layers_param['dropout']
        self.batch_norm = layers_param['batch_norm']

        self.dense1 = torch.nn.Linear(channel_input * l_input, layers_param['dense1'])  # hidden layer
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(layers_param['dense1'])
        self.dense2 = torch.nn.Linear(layers_param['dense1'], layers_param['dense2'])  # hidden layer
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(layers_param['dense2'])

        self.generator = Generator(self.m_type, output_type, layers_param, layers_param['dense2'], output_size)

        if self.dropout is not None:
            self.dense_drop = nn.Dropout(self.dropout)

    def forward(self, x):
        x = F.relu(self.dense1(x.view(x.size(0), -1)))  # activation function for hidden layer
        if self.batch_norm:
            x = self.batch_norm1(x)
        if self.dropout is not None:
            x = self.dense_drop(x)
        x = F.relu(self.dense2(x))  # activation function for hidden layer
        if self.batch_norm:
            x = self.batch_norm2(x)
        if self.dropout is not None:
            x = self.dense_drop(x)

        x = self.generator(x)

        return x


class Dense1d_multi(nn.Module):
    def __init__(self, output_type, channel_input, l_input, output_size, layers_param):
        super(Dense1d_multi, self).__init__()

        self.m_type = 'dense1d_multi'

        self.output_type = output_type
        self.dropout = layers_param['dropout']
        self.batch_norm = layers_param['batch_norm']

        self.dense1 = torch.nn.Linear(channel_input * l_input, layers_param['dense1'])  # hidden layer
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(layers_param['dense1'])

        self.dense2 = torch.nn.Linear(layers_param['dense1'], layers_param['dense2'])  # hidden layer
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(layers_param['dense2'])

        self.dense3 = torch.nn.Linear(layers_param['dense2'], layers_param['dense3'])  # hidden layer
        if self.batch_norm:
            self.batch_norm3 = nn.BatchNorm1d(layers_param['dense3'])

        self.generator = Generator(self.m_type, output_type, layers_param, layers_param['dense3'], output_size)

        if self.dropout is not None:
            self.dense_drop = nn.Dropout(self.dropout)

    def forward(self, x):
        x = F.relu(self.dense1(x.view(x.size(0), -1)))  # activation function for hidden layer
        if self.batch_norm:
            x = self.batch_norm1(x)
        if self.dropout is not None:
            x = self.dense_drop(x)
        x = F.relu(self.dense2(x))  # activation function for hidden layer
        if self.batch_norm:
            x = self.batch_norm2(x)
        x = F.relu(self.dense3(x))  # activation function for hidden layer
        if self.batch_norm:
            x = self.batch_norm3(x)
        if self.dropout is not None:
            x = self.dense_drop(x)

        return x


# ------- Img blocs models ------- #
class TimeDistributed(nn.Module):
    "Applies a module over tdim identically for each step"

    def __init__(self, module, low_mem=False, tdim=1):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.low_mem = low_mem
        self.tdim = tdim

    def forward(self, *args, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        if self.low_mem or self.tdim != 1:
            return self.low_mem_forward(*args)
        else:
            # only support tdim=1
            inp_shape = args[0].shape
            bs, seq_len = inp_shape[0], inp_shape[1]
            # print('on verifie les shape batch = {} et seq = {}'.format(inp_shape[0], inp_shape[1]))
            out = self.module(*[x.view(bs * seq_len, *x.shape[2:]) for x in args], **kwargs)
            out_shape = out.shape
            # print('look at output shape : {}'.format(out_shape))
            return out.view(bs, seq_len, *out_shape[1:])

    def low_mem_forward(self, *args, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        tlen = args[0].shape[self.tdim]
        args_split = [torch.unbind(x, dim=self.tdim) for x in args]
        out = []
        for i in range(tlen):
            out.append(self.module(*[args[i] for args in args_split]), **kwargs)
        return torch.stack(out, dim=self.tdim)

    def __repr__(self):
        return f'TimeDistributed({self.module})'


class CNN2DDoubleMaxpool(nn.Module):
    def __init__(self, input_channel, h_input, w_input, output_channel, cnn_kernel_size, maxpool_kernel_size,
                 dropout=0.5,
                 batch_norm=False, flatten=False):
        self.output_channel = output_channel
        self.maxpoll2d_kernel_size = maxpool_kernel_size
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.flatten = flatten

        super(CNN2DDoubleMaxpool, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=cnn_kernel_size,
                               padding=self.padding_same(cnn_kernel_size))
        h_out, w_out = self.conv_output_size(h_input, w_input, kernel_size=cnn_kernel_size,
                                             padding=self.padding_same(cnn_kernel_size))
        if batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(output_channel)

        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=cnn_kernel_size,
                               padding=self.padding_same(cnn_kernel_size))
        h_out, w_out = self.conv_output_size(h_out, w_out, kernel_size=cnn_kernel_size,
                                             padding=self.padding_same(cnn_kernel_size))
        if batch_norm:
            self.batch_norm2 = nn.BatchNorm2d(output_channel)
        # print('dans init : {} {}'.format(h_out, w_out))
        if self.maxpoll2d_kernel_size is not None:
            self.h_out, self.w_out = self.maxpool2d_output_size(h_out, w_out, kernel_size=maxpool_kernel_size,
                                                                padding=self.padding_same(cnn_kernel_size))
        else:
            self.h_out = h_out
            self.w_out = w_out
        # print('dans init : {} {}'.format(h_out, w_out))
        if dropout is not None:
            self.conv2_drop = nn.Dropout2d()

    def padding_same(self, kernel_size):
        # print(int((kernel_size - 1)/2))
        return [int((kernel_size - 1) / 2), int((kernel_size - 1) / 2)]

    def conv_output_size(self, h_in, w_in, kernel_size, stride=None, padding=None, dilation=None):
        if stride is None:
            stride = [1, 1]
        if padding is None:
            padding = [0, 0]
        if dilation is None:
            dilation = [1, 1]

        # print(type(h_in), type(w_in))
        h_out = int(((h_in + 2 * padding[0] - dilation[0] * (kernel_size - 1) - 1) / stride[0] + 1))
        w_out = int(((w_in + 2 * padding[1] - dilation[1] * (kernel_size - 1) - 1) / stride[1] + 1))
        return h_out, w_out

    def maxpool2d_output_size(self, h_in, w_in, kernel_size, stride=None, padding=None, dilation=None):
        # print(h_in, w_in)
        if stride is None:
            stride = [kernel_size, kernel_size]
        if padding is None:
            padding = [0, 0]
        if dilation is None:
            dilation = [1, 1]

        # print(kernel_size)
        h_out = int((h_in + 2 * padding[0] - dilation[0] * (kernel_size - 1) - 1) / stride[0] + 1) - 1
        w_out = int((w_in + 2 * padding[1] - dilation[1] * (kernel_size - 1) - 1) / stride[1] + 1) - 1

        # print('dans ce conv : lin = {}, channel = {}, kernel = {} '.format(h_in, self.output_channel,
        # self.maxpoll2d_kernel_size))
        # print('lout = {}, avec padding = {}'.format(h_out, padding))

        # print(h_out, w_out)
        return h_out, w_out

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print('size inter conv 1 = {}'.format(x.size()))
        if self.batch_norm:
            x = self.batch_norm1(x)
        x = F.relu(self.conv2(x))
        if self.batch_norm:
            x = self.batch_norm2(x)
        if self.maxpoll2d_kernel_size is not None:
            x = F.max_pool2d(x, self.maxpoll2d_kernel_size)
        # print('size inter maxpool = {}'.format(x.size()))
        if self.dropout is not None:
            x = self.conv2_drop(x)
        # print('size inter maxpool = {}'.format(x.size()))
        if self.flatten:
            # print('into h_out : {} w_out : {}, output channel : {}'.format(self.h_out, self.w_out, self.output_channel))
            x = x.view(-1, self.output_channel * self.h_out * self.w_out)
            # print('into flatten size = {}'.format(x.size()))
        return x


class CNN2D3Scales(nn.Module):
    def __init__(self, seq_size, input_channel, h_input, w_input, output_channel, kernels,
                 dropout=0.5,
                 batch_norm=False, flatten=False):

        self.seq_size = seq_size
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.flatten = flatten

        self.h = h_input
        self.w = w_input

        super(CNN2D3Scales, self).__init__()

        self.conv1 = nn.Conv2d(input_channel*seq_size, output_channel, kernel_size=kernels[0],
                               padding=self.padding_same(kernels[0]))
        h_out, w_out = self.conv_output_size(h_input, w_input, kernel_size=kernels[0],
                                             padding=self.padding_same(kernels[0]))
        if batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(output_channel)

        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=kernels[1],
                               padding=self.padding_same(kernels[1]))
        h_out, w_out = self.conv_output_size(h_out, w_out, kernel_size=kernels[1],
                                             padding=self.padding_same(kernels[1]))
        if batch_norm:
            self.batch_norm2 = nn.BatchNorm2d(output_channel)
        # print('dans init : {} {}'.format(h_out, w_out))

        self.conv3 = nn.Conv2d(output_channel, output_channel, kernel_size=kernels[2],
                               padding=self.padding_same(kernels[2]))
        h_out, w_out = self.conv_output_size(h_out, w_out, kernel_size=kernels[2],
                                             padding=self.padding_same(kernels[2]))
        if batch_norm:
            self.batch_norm3 = nn.BatchNorm2d(output_channel)
        # print('dans init : {} {}'.format(h_out, w_out))

        self.h_out = h_out
        self.w_out = w_out

        # print('dans init : {} {}'.format(h_out, w_out))
        if dropout is not None:
            self.conv2_drop = nn.Dropout2d()

    def padding_same(self, kernel_size):
        # print(int((kernel_size - 1)/2))
        return [int((kernel_size - 1) / 2), int((kernel_size - 1) / 2)]

    def conv_output_size(self, h_in, w_in, kernel_size, stride=None, padding=None, dilation=None):
        if stride is None:
            stride = [1, 1]
        if padding is None:
            padding = [0, 0]
        if dilation is None:
            dilation = [1, 1]

        # print(type(h_in), type(w_in))
        h_out = int(((h_in + 2 * padding[0] - dilation[0] * (kernel_size - 1) - 1) / stride[0] + 1))
        w_out = int(((w_in + 2 * padding[1] - dilation[1] * (kernel_size - 1) - 1) / stride[1] + 1))
        return h_out, w_out

    def forward(self, x):

        x = x.view(-1, self.seq_size * self.input_channel, self.h, self.w)

        x1 = F.relu(self.conv1(x))
        # print('size inter conv 1 = {}'.format(x.size()))
        if self.batch_norm:
            x1 = self.batch_norm1(x1)
        if self.dropout is not None:
            x1 = self.conv2_drop(x1)

        x2 = F.relu(self.conv2(x))
        # print('size inter conv 1 = {}'.format(x.size()))
        if self.batch_norm:
            x2 = self.batch_norm2(x2)
        if self.dropout is not None:
            x2 = self.conv2_drop(x2)

        x3 = F.relu(self.conv3(x))
        # print('size inter conv 1 = {}'.format(x.size()))
        if self.batch_norm:
            x3 = self.batch_norm3(x3)
        if self.dropout is not None:
            x3 = self.conv2_drop(x3)

        x = torch.cat((x1, x2, x3), dim=1)

        return x


class Conv2d(nn.Module):
    def __init__(self, output_type, seq_size, input_channel, h_input, w_input, output_size, layers_param):
        super(Conv2d, self).__init__()

        self.m_type = 'conv2d'

        self.seq_size = seq_size
        self.input_channel = input_channel
        self.output_type = output_type
        self.dropout = layers_param['dropout']
        self.batch_norm = layers_param['batch_norm']
        self.h = h_input
        self.w = w_input

        self.cnn_1 = CNN2DDoubleMaxpool(input_channel*seq_size, h_input, w_input,
                                        layers_param['cnn1'],
                                        layers_param['kernel_1'],
                                        layers_param['maxpool_1'],
                                        self.dropout,
                                        self.batch_norm)
        self.cnn_2 = CNN2DDoubleMaxpool(self.cnn_1.output_channel, self.cnn_1.h_out, self.cnn_1.w_out,
                                        layers_param['cnn2'],
                                        layers_param['kernel_2'],
                                        layers_param['maxpool_2'],
                                        self.dropout,
                                        self.batch_norm)
        self.cnn_3 = CNN2DDoubleMaxpool(self.cnn_2.output_channel, self.cnn_2.h_out, self.cnn_2.w_out,
                                        layers_param['cnn3'],
                                        layers_param['kernel_3'],
                                        layers_param['maxpool_3'],
                                        self.dropout,
                                        self.batch_norm)

        self.generator = Generator(self.m_type, output_type, layers_param,
                                   self.cnn_3.output_channel,
                                   output_size, lin=False)

    def forward(self, x):
        # print('x size = {}'.format(x.size()))
        c_in = x.view(-1, self.seq_size * self.input_channel, self.h, self.w)
        c_out = self.cnn_1(c_in)
        # print(c_out.size())
        c_out = self.cnn_2(c_out)
        # print(c_out.size())
        c_out = self.cnn_3(c_out)
        # print(c_out.size())

        x = self.generator(c_out)
        return x


class Conv2dIntoDense(nn.Module):
    def __init__(self, output_type, time_size, channel_input, h_input, w_input, output_size, layers_param):
        super(Conv2dIntoDense, self).__init__()

        self.m_type = 'conv2d_into_dense'

        self.output_type = output_type
        self.dropout = layers_param['dropout']
        self.batch_norm = layers_param['batch_norm']

        self.cnn_1 = CNN2DDoubleMaxpool(channel_input, h_input, w_input,
                                        layers_param['cnn1'],
                                        layers_param['kernel_1'],
                                        layers_param['maxpool_1'],
                                        self.dropout,
                                        self.batch_norm)
        self.cnn_2 = CNN2DDoubleMaxpool(self.cnn_1.output_channel, self.cnn_1.h_out, self.cnn_1.w_out,
                                        layers_param['cnn2'],
                                        layers_param['kernel_2'],
                                        layers_param['maxpool_2'],
                                        self.dropout,
                                        self.batch_norm)
        self.cnn_3 = CNN2DDoubleMaxpool(self.cnn_2.output_channel, self.cnn_2.h_out, self.cnn_2.w_out,
                                        layers_param['cnn3'],
                                        layers_param['kernel_3'],
                                        layers_param['maxpool_3'],
                                        self.dropout,
                                        self.batch_norm,
                                        flatten=True)

        self.dense1 = torch.nn.Linear(self.cnn_3.output_channel * self.cnn_3.h_out * self.cnn_3.w_out,
                                      layers_param['dense1'])  # hidden layer
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(time_size)
        self.dense2 = torch.nn.Linear(layers_param['dense1'], layers_param['dense2'])  # hidden layer
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(time_size)
        self.generator = Generator(self.m_type, output_type, layers_param, layers_param['dense2'], output_size)

        if self.dropout is not None:
            self.dense_drop = nn.Dropout(self.dropout)

    def forward(self, x):
        # print('x size = {}'.format(x.size()))
        c_in = x
        c_out = TimeDistributed(self.cnn_1, tdim=1)(c_in)
        # print(c_out.size())
        c_out = TimeDistributed(self.cnn_2, tdim=1)(c_out)
        # print(c_out.size())
        c_out = TimeDistributed(self.cnn_3, tdim=1)(c_out)
        # print(c_out.size())

        x = F.relu(self.dense1(c_out))  # activation function for hidden layer
        # print(x.size())
        if self.batch_norm:
            x = self.batch_norm1(x)
        if self.dropout is not None:
            x = self.dense_drop(x)
        x = F.relu(self.dense2(x))  # activation function for hidden layer
        if self.batch_norm:
            x = self.batch_norm2(x)
        if self.dropout is not None:
            x = self.dense_drop(x)

        x = self.generator(x)
        return x


class Conv2dIntoLSTM(nn.Module):
    def __init__(self, output_type, channel_input, h_input, w_input, output_size, layers_param):
        super(Conv2dIntoLSTM, self).__init__()

        self.m_type = 'conv2d_into_lstm'

        self.output_type = output_type
        self.dropout = layers_param['dropout']
        self.batch_norm = layers_param['batch_norm']

        self.cnn_1 = CNN2DDoubleMaxpool(channel_input, h_input, w_input,
                                        layers_param['cnn1'],
                                        layers_param['kernel_1'],
                                        layers_param['maxpool_1'],
                                        self.dropout,
                                        self.batch_norm)
        self.cnn_output_channel = self.cnn_1.output_channel

        if layers_param['cnn2'] is not None:
            self.cnn_2 = CNN2DDoubleMaxpool(self.cnn_1.output_channel, self.cnn_1.h_out, self.cnn_1.w_out,
                                            layers_param['cnn2'],
                                            layers_param['kernel_2'],
                                            layers_param['maxpool_2'],
                                            self.dropout,
                                            self.batch_norm)
            self.cnn_output_channel = self.cnn_2.output_channel
        if layers_param['cnn3'] is not None:
            self.cnn_3 = CNN2DDoubleMaxpool(self.cnn_2.output_channel, self.cnn_2.h_out, self.cnn_2.w_out,
                                            layers_param['cnn3'],
                                            layers_param['kernel_3'],
                                            layers_param['maxpool_3'],
                                            self.dropout,
                                            self.batch_norm,
                                            flatten=True)
            self.cnn_output_channel = self.cnn_3.output_channel

        self.rnn = nn.LSTM(
            input_size=self.self.cnn_output_channel,
            hidden_size=layers_param['lstm_hidden_size'],
            num_layers=layers_param['nb_lstm'],
            batch_first=True)

        self.generator = Generator(self.m_type, output_type, layers_param, layers_param['lstm_hidden_size'],
                                   output_size)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x
        c_out = TimeDistributed(self.cnn_1, tdim=1)(c_in)
        if self.layers_param['cnn2'] is not None:
            c_out = TimeDistributed(self.cnn_2, tdim=1)(c_out)
        if self.layers_param['cnn3'] is not None:
            c_out = TimeDistributed(self.cnn_3, tdim=1)(c_out)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.rnn(r_in)
        x = self.generator(r_out[:, -1, :])

        return x


# ------- Scalar blocs models ------- #


class CNN1DDoubleMaxpool(nn.Module):
    def __init__(self, input_channel, l_input, output_channel, cnn_kernel_size, maxpool_kernel_size, dropout=0.5,
                 batch_norm=False, flatten=False):
        self.output_channel = output_channel
        self.maxpoll1d_kernel_size = maxpool_kernel_size
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.flatten = flatten

        super(CNN1DDoubleMaxpool, self).__init__()

        self.conv1 = nn.Conv1d(input_channel, output_channel, kernel_size=cnn_kernel_size,
                               padding=self.padding_same(cnn_kernel_size))
        l_out = self.conv_output_size(l_input, kernel_size=cnn_kernel_size, padding=self.padding_same(cnn_kernel_size))
        # print(l_out)
        if batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(output_channel)

        self.conv2 = nn.Conv1d(output_channel, output_channel, kernel_size=cnn_kernel_size,
                               padding=self.padding_same(cnn_kernel_size))
        l_out = self.conv_output_size(l_out, kernel_size=cnn_kernel_size, padding=self.padding_same(cnn_kernel_size))
        # print(l_out)
        if batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(output_channel)

        if self.maxpoll1d_kernel_size is not None:
            self.l_out = self.maxpool1d_output_size(l_out, kernel_size=maxpool_kernel_size,
                                                    padding=self.padding_same(maxpool_kernel_size))
            # print(self.l_out)
        else:
            self.l_out = l_out

        if dropout is not None:
            self.conv2_drop = nn.Dropout(dropout)

    def padding_same(self, kernel_size):
        return int((kernel_size - 1) / 2)

    def conv_output_size(self, l_in, kernel_size, stride=None, padding=None, dilation=None):
        if stride is None:
            stride = 1
        if padding is None:
            padding = 0
        if dilation is None:
            dilation = 1

        l_out = int((l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
        # print('dans ce conv : lin = {}, channel = {}, kernel = {} '.format(l_in, self.output_channel, self.maxpoll1d_kernel_size))
        # print('lout = {}, avec padding = {}'.format(l_out, int((kernel_size - 1)/2)))

        return l_out

    def maxpool1d_output_size(self, l_in, kernel_size, stride=None, padding=None, dilation=None):
        if stride is None:
            stride = kernel_size
        if padding is None:
            padding = 0
        if dilation is None:
            dilation = 1

        l_out = int(((l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
        # print('dans ce conv : lin = {}, channel = {}, kernel = {} '.format(l_in, self.output_channel,
        # self.maxpoll1d_kernel_size))
        # print('lout = {}, avec padding = {}'.format(l_out, padding))

        return l_out

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print('size inter conv 1 = {}'.format(x.size()))
        if self.batch_norm:
            x = self.batch_norm1(x)
        x = F.relu(self.conv2(x))
        if self.batch_norm:
            x = self.batch_norm2(x)
        # print('size inter conv 2 = {}'.format(x.size()))
        if self.maxpoll1d_kernel_size is not None:
            x = F.max_pool1d(x, self.maxpoll1d_kernel_size)
        if self.dropout is not None:
            x = self.conv2_drop(x)
        # print('size inter maxpool = {}'.format(x.size()))
        if self.flatten:
            # print('into l_out : {}, output channel : {}'.format(self.l_out, self.output_channel))
            x = x.view(-1, self.l_out * self.output_channel)
            # print('flatten size = {}'.format(x.size()))
        return x


class Conv1dIntoDense(nn.Module):
    def __init__(self, output_type, channel_input, l_input, output_size, layers_param):
        super(Conv1dIntoDense, self).__init__()

        self.m_type = 'conv1d_into_dense'

        self.output_type = output_type
        self.dropout = layers_param['dropout']
        self.batch_norm = layers_param['batch_norm']

        self.cnn_1 = CNN1DDoubleMaxpool(channel_input, l_input,
                                        layers_param['cnn1'],
                                        layers_param['kernel_1'],
                                        layers_param['maxpool_1'],
                                        self.dropout,
                                        self.batch_norm)
        self.cnn_2 = CNN1DDoubleMaxpool(self.cnn_1.output_channel, self.cnn_1.l_out,
                                        layers_param['cnn2'],
                                        layers_param['kernel_2'],
                                        layers_param['maxpool_2'],
                                        self.dropout,
                                        self.batch_norm)
        self.cnn_3 = CNN1DDoubleMaxpool(self.cnn_2.output_channel, self.cnn_2.l_out,
                                        layers_param['cnn3'],
                                        layers_param['kernel_3'],
                                        layers_param['maxpool_3'],
                                        self.dropout,
                                        self.batch_norm,
                                        flatten=True)

        self.dense1 = torch.nn.Linear(self.cnn_3.output_channel * self.cnn_3.l_out,
                                      layers_param['dense1'])  # hidden layer
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(layers_param['dense1'])
        self.dense2 = torch.nn.Linear(layers_param['dense1'], layers_param['dense2'])  # hidden layer
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(layers_param['dense2'])
        self.generator = Generator(self.m_type, output_type, layers_param, layers_param['dense2'], output_size)

        if self.dropout is not None:
            self.dense_drop = nn.Dropout(self.dropout)

    def forward(self, x):
        # print('x size = {}'.format(x.size()))
        c_in = x
        c_out = self.cnn_1(c_in)
        # print(c_out.size())
        c_out = self.cnn_2(c_out)
        # print(c_out.size())
        c_out = self.cnn_3(c_out)
        # print(c_out.size())

        x = F.relu(self.dense1(c_out))  # activation function for hidden layer
        if self.batch_norm:
            x = self.batch_norm1(x)
        if self.dropout is not None:
            x = self.dense_drop(x)
        x = F.relu(self.dense2(x))  # activation function for hidden layer
        if self.batch_norm:
            x = self.batch_norm2(x)
        if self.dropout is not None:
            x = self.dense_drop(x)

        x = self.generator(x)
        return x


class Conv1dIntoLSTM(nn.Module):
    def __init__(self, output_type, channel_input, l_input, output_size, layers_param):
        super(Conv1dIntoLSTM, self).__init__()

        self.m_type = 'conv1d_into_lstm'
        self.layers_param = layers_param
        self.dropout = layers_param['dropout']
        self.batch_norm = layers_param['batch_norm']

        self.cnn_1 = CNN1DDoubleMaxpool(channel_input, l_input,
                                        layers_param['cnn1'],
                                        layers_param['kernel_1'],
                                        layers_param['maxpool_1'],
                                        self.dropout,
                                        self.batch_norm)
        self.cnn_output_channel = self.cnn_1.output_channel

        if layers_param['cnn2'] is not None:
            self.cnn_2 = CNN1DDoubleMaxpool(self.cnn_1.output_channel, self.cnn_1.l_out,
                                            layers_param['cnn2'],
                                            layers_param['kernel_2'],
                                            layers_param['maxpool_2'],
                                            self.dropout,
                                            self.batch_norm)
            self.cnn_output_channel = self.cnn_2.output_channel

        if layers_param['cnn3'] is not None:
            self.cnn_3 = CNN1DDoubleMaxpool(self.cnn_2.output_channel, self.cnn_2.l_out,
                                            layers_param['cnn3'],
                                            layers_param['kernel_3'],
                                            layers_param['maxpool_3'],
                                            self.dropout,
                                            self.batch_norm)
            self.cnn_output_channel = self.cnn_3.output_channel

        self.rnn = nn.LSTM(
            input_size=self.cnn_output_channel,
            hidden_size=layers_param['lstm_hidden_size'],
            num_layers=layers_param['nb_lstm'],
            batch_first=True)

        self.generator = Generator(self.m_type, output_type, layers_param, layers_param['lstm_hidden_size'],
                                   output_size)

    def forward(self, x):
        # print('x size = {}'.format(x.size()))
        c_in = x
        c_out = self.cnn_1(c_in)
        # print('cnn1 out size = {}'.format(c_out.size()))
        if self.layers_param['cnn2'] is not None:
            c_out = self.cnn_2(c_out)
        # print('cnn2 out size = {}'.format(c_out.size()))
        if self.layers_param['cnn3'] is not None:
            c_out = self.cnn_3(c_out)
        # print('cnn3 out size = {}'.format(c_out.size()))
        r_in = torch.transpose(c_out, 1, 2)
        r_out, (h_n, c_n) = self.rnn(r_in)
        # print('r out size = {}'.format(r_out.size()))

        x = self.generator(r_out[:, -1, :])
        return x

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()

class M4ELoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(M4ELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        if torch.sum(torch.isnan(input)) != 0:
            print('WARNING nan in pred')
        if torch.sum(torch.isnan(target)) != 0:
            print('WARNING nan in target')
        # print(np.shape(target), np.shape(input))
        return torch.mean((target[:, 0] - input[:, 0]) ** 4)

class M6ELoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(M6ELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        if torch.sum(torch.isnan(input)) != 0:
            print('WARNING nan in pred')
        if torch.sum(torch.isnan(target)) != 0:
            print('WARNING nan in target')
        return torch.mean((target[:, 0] - input[:, 0]) ** 4)

class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion_param, criterion, opti_param, opt):
        self.generator = generator
        self.criterion_param = criterion_param
        self.criterion = criterion
        self.opti_param = opti_param
        self.optimizer = opt

    def __call__(self, out, targets, phase):

        # if self.generator is not None:
        #     out = self.generator(out)

        # print(x.shape)

        # print('shape out = {} and shape target = {}'.format(np.shape(out), np.shape(targets)))
        # print(type(targets))
        # print(targets.size())
        loss = self.criterion(out, targets)
        # print(loss.dtype)

        if phase == 'train':
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return out, loss


class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."

    def __init__(self, generator, criterion_param, criterion, devices, opti_param, opt, chunk_size=32):
        # Send out to different gpus.
        self.generator = generator
        self.criterion_param = criterion_param
        self.criterion = nn.parallel.replicate(criterion,
                                               devices=devices)
        self.opti_param = opti_param
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, phase):
        total = 0.0
        generator = nn.parallel.replicate(self.generator,
                                          devices=self.devices)
        # print('out shape in input = {}'.format(np.shape(out)))
        out = torch.transpose(out, 0, 1)
        # print('out shape after transpose = {}'.format(np.shape(out)))
        out_scatter = nn.parallel.scatter(out,
                                          target_gpus=self.devices,
                                          dim=0)
        # print('out_scatter size = {}'.format(np.size(out_scatter)))
        # print('out_scatter shape = {}'.format(np.shape(out_scatter[0])))

        out_grad = [[] for _ in out_scatter]
        out_pred = [[] for _ in out_scatter]
        # print('out_grad size = {}'.format(np.size(out_grad)))

        # print('target shape in input = {}'.format(np.shape(targets)))
        targets = nn.parallel.scatter(targets,
                                      target_gpus=self.devices,
                                      dim=0)
        # print('targets size = {}'.format(np.size(targets)))
        # print('targets shape = {}'.format(np.shape(targets[0])))

        # #reshape for model
        # for i in range(np.size(device)):
        #     out_scatter[i] = torch.transpose(out_scatter[i], 0, 1)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        count = 0
        for i in range(0, out_scatter[0].size(0), chunk_size):
            # Predict distributions

            out_column = [[Variable(torch.transpose(o[i:i + chunk_size, :], 0, 1).data,
                                    requires_grad=self.opt is not None)]
                          for o in out_scatter]

            # out_column = [[Variable(o[:, i:i + chunk_size].data,
            #                         requires_grad=self.opt is not None)]
            #               for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)
            # print('prediction hape = {}'.format(np.shape(gen[0])))

            # Compute loss.
            if self.criterion_param['name_criterion'] == 'cross_entropy_loss':
                # for g, t in zip(gen, targets):
                #     print('ble', (g, t[i:i + chunk_size], phase))
                y = [(g, t[i:i + chunk_size]) for g, t in zip(gen, targets)]
                loss = nn.parallel.parallel_apply(self.criterion, y)
            elif self.criterion_param['name_criterion'] == 'MSELoss':
                y = [(g, t[i:i + chunk_size]) for g, t in zip(gen, targets)]
                loss = nn.parallel.parallel_apply(self.criterion, y)
            else:
                print('warning: non code pour criterion {}'.format(self.criterion_param['name_criterion']))

            # Sum and normalize loss
            # print('total = {}'.format(total))
            # print('loss at {} = {} '.format(i, loss))
            l = nn.parallel.gather(loss,
                                   target_device=self.devices[0])
            l = l.sum()/np.size(self.devices)
            total += l.data
            count += 1
            # print('l data = {} et total = {}'.format(l.data, total))

            # Backprop loss to output of transformer
            if phase == 'train':
                l.backward()
                for j, l in enumerate(loss):
                    # print('j = {} et l = {}'.format(j, l))
                    # print('shape de out_column at j = {}'.format(np.shape(out_column[j][0])))
                    # out_grad[j].append(torch.transpose(out_column[j][0], 0, 1).grad.data.clone())
                    out_grad[j].append(out_column[j][0].grad.data.clone())

            # prediction
            for j, l in enumerate(loss):
                # print('shape de pred at j = {}'.format(np.shape(gen[j])))
                out_pred[j].append(gen[j].data)

        # print('out grade shape = {}'.format(np.shape(out_grad)))
        # print('out grade shape = {}'.format(np.shape(out_grad[0])))

        # print('pred shape = {}'.format(np.shape(out_pred)))
        # print('pred shape = {}'.format(np.shape(out_pred[0])))

        # Backprop all loss through transformer.
        if phase == 'train':
            # out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            out_grad = [Variable(torch.transpose(torch.cat(og, dim=1), 0, 1)) for og in out_grad]
            # print('out grade shape = {}'.format(np.shape(out_grad)))
            # print('out grade shape = {}'.format(np.shape(out_grad[0])))
            o1 = torch.transpose(out, 0, 1)
            # print('o1 grade shape = {}'.format(np.shape(o1)))
            o2 = nn.parallel.gather(out_grad,
                                    target_device=self.devices[0])
            o2 = torch.transpose(o2, 0, 1)
            # print('o2 grade shape = {}'.format(np.shape(o2)))
            o1.backward(gradient=o2)
            self.opt.step()
            if self.opti_param['name_opti'] == 'noam':
                self.opt.optimizer.zero_grad()

        # print('out pred shape = {}'.format(np.shape(out_pred)))
        # print('out pred shape = {}'.format(np.shape(out_pred[0])))

        out_pred = [Variable(torch.cat(p, dim=0)) for p in out_pred]
        # print('out pred shape = {}'.format(np.shape(out_pred)))
        # print('out pred shape = {}'.format(np.shape(out_pred[0])))
        prediction = nn.parallel.gather(out_pred,
                                target_device=self.devices[0])
        # print(total/count)
        return torch.transpose(prediction, 0, 1), total/count


class MultiGPULossComputeHarvard:
    "A multi-gpu loss compute and train function."

    def __init__(self, generator, criterion_param, criterion, devices, opti_param, opt, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion_param = criterion_param
        self.criterion = nn.parallel.replicate(criterion,
                                               devices=devices)
        self.opti_param = opti_param
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, phase):
        total = 0.0
        generator = nn.parallel.replicate(self.generator,
                                          devices=self.devices)
        out_scatter = nn.parallel.scatter(out,
                                          target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets,
                                      target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i + chunk_size].data,
                                    requires_grad=self.opt is not None)]
                          for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss.
            if self.criterion_param['name_criterion'] == 'cross_entropy_loss':
                # for g, t in zip(gen, targets):
                #     print('ble', (g, t[i:i + chunk_size], phase))
                y = [(g, t[i:i + chunk_size]) for g, t in zip(gen, targets)]
                loss = nn.parallel.parallel_apply(self.criterion, y)
            else:
                print('warning: non code pour criterion {}'.format(self.criterion_param['name_criterion']))

            # Sum and normalize loss
            print('loss shape ', loss)
            l = nn.parallel.gather(loss,
                                   target_device=self.devices[0])
            l = l.sum()/np.size(self.devices)
            print('l ', l)
            total += l.data
            print('l ', l.data, total)

            # Backprop loss to output of transformer
            if phase == 'train':
                l.backward()
                for j, l in enumerate(loss):
                    print(j, l)
                    print(np.shape(out_column[j]))
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.
        if phase == 'train':
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad,
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            if self.opti_param['name_opti'] == 'noam':
                self.opt.optimizer.zero_grad()

        print(total/(i+1))
        return total/(i+1)

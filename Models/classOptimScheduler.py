import torch
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import warnings
from torch.autograd import Variable

def look_at_lr(training_size, bsz, nb_epochs, d_model, factor, warmup):
    nb_batch = int(np.round(training_size/bsz, 0))
    nb_batch_tot = nb_batch * nb_epochs
    batch = np.arange(1, nb_batch_tot+1)
    epochs = np.arange(nb_batch, nb_batch_tot + nb_batch, nb_batch)

    print('pour un trainsize de {} avec {} epochs et un bsz = {}'.format(training_size, nb_epochs, bsz))
    print('par epoch : {} batchs ; soit un tot de {} batchs'.format(nb_batch, nb_batch_tot))
    # print('batch = {}'.format(batch))
    # print('epochs = {}'.format(epochs))

    lr_batch = factor*(d_model**(-0.5)*np.minimum(batch**(-0.5), batch*warmup**(-1.5)))
    lr_epoch = factor*(d_model**(-0.5)*np.minimum(epochs**(-0.5), epochs*warmup**(-1.5)))

    return lr_batch, lr_epoch, batch, epochs


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


class GoogleTransformerLR(_LRScheduler):
    """
    """

    def __init__(self, optimizer, d_model, step_num, warmup_step, last_epoch=-1, verbose=False):
        self.d_model = d_model
        self.step_num = step_num
        self.warmup_step = warmup_step
        super(GoogleTransformerLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch

from Models.summary_perso import summary
from Models.classSimpleModel import Conv1dIntoLSTM, Conv2dIntoLSTM, Dense1d, Conv1dIntoDense, Dense1d_multi, Conv2d, Conv2dIntoDense
# from Models.classTransformer import TransformerModel
from Models.classOptimScheduler import NoamOpt, LabelSmoothing
from torch._utils import (
    _get_available_device_type)
from Models.classLoss import M4ELoss, M6ELoss
from Models.classMultiScale1DResNet import MSResNet
from Models.classMultiScale2DResNet_propre import ResNet2D_multiscale
from Models.classResNet1D import resnet18, resnet34, resnet50
from Models.classResNet2D import resnet2d18, resnet2d34, resnet2d50
from Models.classResNext import resnext50_32x4d
from Models.classResFred import ResNet
from Models.classUNet_test import unet_basic

# ------- Def optimizer ------- #
def optim_rms(param, lr, momentum):
    return optim.RMSprop(param, lr=lr, momentum=momentum)


def optim_adam(param, lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):
    return optim.Adam(param, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)


def optim_SGD(param, lr, momentum, dampening, weight_decay, nesterov):
    if momentum is None:
        momentum = 0
    if dampening is None:
        dampening = 0
    if weight_decay is None:
        weight_decay = 0
    if nesterov is None:
        nesterov = False

    return optim.SGD(param, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)


def optim_Adadelta(param, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0):
    return optim.Adadelta(param, lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)


def optim_Noam(d_model, factor, warmup, param):
    if factor is None:
        factor = 2
    if warmup is None:
        warmup = 4000
    return NoamOpt(d_model, factor, warmup,
                   torch.optim.Adam(param, lr=0, betas=(0.9, 0.98), eps=1e-9))


# ------- Def scheduler ------- #
def stepLR(optimizer, step_size, gamma):
    if gamma is None:
        gamma = 0.1
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def multistepLR(optimizer, gamma=None, milestones=None, last_epoch=None):
    if milestones is None:
        milestones = [50, 100, 150, 200, 250, 300]
    if gamma is None:
        gamma = 0.1
    if last_epoch is None:
        last_epoch = -1

    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma, last_epoch=last_epoch)


def reduceLRplateau(optimizer, mode, factor, patience, threshold, cooldown, verbose):
    if mode is None:
        mode = 'min'
    if factor is None:
        factor = 0.1
    if patience is None:
        patience = 10
    if threshold is None:
        threshold = 0.0001
    if cooldown is None:
        cooldown = 0
    if verbose is None:
        verbose = False

    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience,
                                                threshold=threshold, cooldown=cooldown, verbose=verbose)


def SGDR(optimizer, T_0, T_mul=1, eta_min=0, last_epoch=-1):
    return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=T_mul, eta_min=eta_min,
                                                          last_epoch=last_epoch)


# ------- Def criterions ------- #
def criterion_cross_entropy_loss(weight):
    return nn.CrossEntropyLoss(weight)


def criterion_mean_average_loss():
    return torch.nn.L1Loss()


def criterion_mean_squared_loss():
    return torch.nn.MSELoss()


def criterion_mean_fourth_loss():
    return M4ELoss()


def criterion_mean_six_loss():
    return M6ELoss()

def criterion_neg_log_likelihood_loss():
    return torch.nn.NLLLoss(torch.Tensor([0.25, 0.75]))


def criterion_label_smoothing(tgt_ntoken, padding_idx):
    # channel == src_ntokens pour l'instant
    # ici ce devrait plutot etre tgt_ntoken mais bon...
    return LabelSmoothing(tgt_ntoken, padding_idx, smoothing=0.0)


# ------- Def models ------- #
class Model():
    '''la classe modele contient elle même une copie des datas, ca peut etre nécessaire pour les reshape etc
    si superflu bah on peut supprimer'''

    def __init__(self, args, device, m_type, output_type, layers_param, opti_param, criterion_param,
                 batch_size, seq_size, channel, sw, sc, output_shape):

        # ------------- devices
        self.args = args
        self.device = device

        # ------------- data
        self.output_type = output_type
        self.batch_size = batch_size
        self.seq_size = seq_size
        self.channel = channel
        self.sw = sw
        self.sc = sc
        self.output_shape = output_shape

        # ------------- nom du NN
        self.m_type = m_type

        # ------------- param
        self.layers_param = layers_param
        self.opti_param = opti_param
        self.criterion_param = criterion_param

    # --------------------------------------------------------#
    def NN_model_summary(self, model):

        if self.args.pred == 'field':
            summary(model, (self.seq_size, self.channel, self.sw, self.sc), batch_size=self.batch_size,
                    device=self.device)
        else:
            if self.m_type == 'transformer':
                summary(model, (self.seq_size,), batch_size=self.batch_size,
                        device=self.device)  ## for now without msk or tgt
            else:
                summary(model, (self.channel, self.seq_size), batch_size=self.batch_size, device=self.device)

    # --------------------------------------------------------#
    def NN_model(self):
        '''defines model of the given type; many if not most explicit choices are made here :
        kernel size, number of lstm cells, nb of filters, etc'''

        # ------------- models
        print(self.m_type)
        if self.m_type == 'dense1d':
            model = Dense1d(self.output_type, self.channel, self.seq_size, self.output_shape, self.layers_param)
        elif self.m_type == 'dense1d_multi':
            model = Dense1d_multi(self.output_type, self.channel, self.seq_size, self.output_shape, self.layers_param)
        elif self.m_type == 'conv1d_into_dense':
            model = Conv1dIntoDense(self.output_type, self.channel, self.seq_size, self.output_shape, self.layers_param)
        elif self.m_type == 'conv1d_into_lstm':
            model = Conv1dIntoLSTM(self.output_type, self.channel, self.seq_size, self.output_shape, self.layers_param)
        # elif self.m_type == 'transformer':
        #     # channel == src_ntokens pour l'instant
        #     model = TransformerModel(self.channel, None, self.criterion_param['name_criterion'],
        #                              self.layers_param,
        #                              src_size=self.seq_size,
        #                              output_size=self.output_shape)
        elif self.m_type == 'MSResNet':
            model = MSResNet(self.channel, self.output_type, output_shape=self.output_shape,
                             layers=self.layers_param['layers'])
        elif self.m_type == 'ResNet':
            if self.layers_param['num_res_net'] == '18':
                model = resnet18(self.channel, self.output_type, self.output_shape)
            if self.layers_param['num_res_net'] == '18_little':
                model = resnet18(self.channel, self.output_type, self.output_shape)
            elif self.layers_param['num_res_net'] == '34':
                model = resnet34(self.channel, self.output_type, self.output_shape)
            elif self.layers_param['num_res_net'] == '50':
                model = resnet50(self.channel, self.output_type, self.output_shape)
        elif self.m_type == 'ResNeXt':
            model = resnext50_32x4d(self.channel, self.output_type, self.output_shape)
        elif self.m_type == 'ResNetFred':
            model = ResNet(self.channel, self.output_type, self.output_shape)

        elif self.m_type == 'conv2d':
            model = Conv2d(self.output_type, self.seq_size, self.channel, self.sw, self.sc, self.output_shape, self.layers_param)
        elif self.m_type == 'ResNet2D_multiscale':
            model = ResNet2D_multiscale(self.channel, self.output_type, self.output_shape, self.seq_size, self.sw, self.sc)
        elif self.m_type == 'ResNet2D':
            model = resnet2d18(self.channel, self.output_type, self.output_shape, self.seq_size, self.sw,
                                        self.sc)
        elif self.m_type == 'UNet_basic':
            model = unet_basic(self.channel, self.output_type, self.output_shape, self.seq_size, self.sw, self.sc)

        # ------------- model summary
        self.NN_model_summary(model)


        # ------------- use GPU if needed
        if self.args.cuda:
            if type(self.args.cuda_device).__name__ == 'str':
                model = model.to(self.device)
            else:
                device_type = _get_available_device_type()
                src_device_obj = torch.device(device_type, self.device[0])
                model.to(src_device_obj)

        return model

    # --------------------------------------------------------#
    def NN_opti_loss_scheduler(self, model):
        # ------------- optimiseurs
        # print(self.opti_param['name_opti'])
        if self.opti_param['name_opti'] == 'rms':
            optimizer = optim_rms(model.parameters(), self.opti_param['lr'], self.opti_param['momentum'])
        elif self.opti_param['name_opti'] == 'adam':
            optimizer = optim_adam(model.parameters(), self.opti_param['lr'], weight_decay=self.opti_param['weight_decay'])
        elif self.opti_param['name_opti'] == 'noam':
            optimizer = optim_Noam(model.d_model, self.opti_param['factor'], self.opti_param['warmup'],
                                   model.parameters())
        elif self.opti_param['name_opti'] == 'SGD':
            optimizer = optim_SGD(model.parameters(), self.opti_param['lr'], self.opti_param['momentum'],
                                  self.opti_param['dampening'], self.opti_param['weight_decay'],
                                  self.opti_param['nesterov'])
        elif self.opti_param['name_opti'] == 'adadelta':
            optimizer = optim_Adadelta(model.parameters(), self.opti_param['lr'],
                                       weight_decay=self.opti_param['weight_decay'])

        # ------------- lr scheduler
        if self.opti_param['scheduler'] is None:
            # print('je suis none')
            scheduler = None
        elif self.opti_param['scheduler'] == 'stepLR':
            scheduler = stepLR(optimizer, self.opti_param['stepsize'], self.opti_param['gamma'])
        elif self.opti_param['scheduler'] == 'multistepLR':
            scheduler = multistepLR(optimizer, self.opti_param['gamma'], self.opti_param['milestones'])
        elif self.opti_param['scheduler'] == 'reduceLRplateau':
            scheduler = reduceLRplateau(optimizer, self.opti_param['mode'], self.opti_param['factor'],
                                        self.opti_param['patience'], self.opti_param['threshold'],
                                        self.opti_param['cooldown'], self.opti_param['verbose'])
        elif self.opti_param['scheduler'] == 'SGDR':
            scheduler = SGDR(optimizer, self.opti_param['T_0'], self.opti_param['T_mult'], self.opti_param['T_0'])

        # ------------- criterion
        if self.criterion_param['name_criterion'] == 'cross_entropy_loss':
            criterion = criterion_cross_entropy_loss(
                torch.Tensor(self.criterion_param['weight']) if self.criterion_param['weight'] is not None else None)
        elif self.criterion_param['name_criterion'] == 'MAELoss':
            criterion = criterion_mean_average_loss()
        elif self.criterion_param['name_criterion'] == 'MSELoss':
            criterion = criterion_mean_squared_loss()
        elif self.criterion_param['name_criterion'] == 'M4ELoss':
            criterion = criterion_mean_fourth_loss()
        elif self.criterion_param['name_criterion'] == 'M6ELoss':
            criterion = criterion_mean_six_loss()
        elif self.criterion_param['name_criterion'] == 'NLLLoss':
            criterion = criterion_neg_log_likelihood_loss()

        # ------------- use GPU if needed
        if self.args.cuda:
            if type(self.args.cuda_device).__name__ == 'str':
                criterion = criterion.to(self.device)
            else:
                device_type = _get_available_device_type()
                src_device_obj = torch.device(device_type, self.device[0])
                criterion.to(src_device_obj)

        return optimizer, scheduler, criterion

    # --------------------------------------------------------#
    def recup_from_checkpoint(self, args, model, optimizer, num_test, load_best_acc, load_best_loss, checkpoint):

        if checkpoint == 'acc':
            check = torch.load(load_best_acc)
        elif checkpoint == 'loss':
            check = torch.load(load_best_loss)

        pretrained_dict = check['model_state_dict']

        if args.transfert == 'switch_class_reg' or args.transfert == 'inter_classes':
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               (k != 'generator.predict.weight' and k != 'generator.predict.bias')}

        if type(self.args.cuda_device).__name__ == 'str':
            model.load_state_dict(pretrained_dict, strict=False)
        else:
            model.module.load_state_dict(pretrained_dict, strict=False)

        if args.transfert == 'switch_class_reg':
            for name, param in model.named_parameters():
                if (name != 'generator.predict.weight' and name != 'generator.predict.bias'):
                    param.requires_grad = False

        if not args.train:
            if self.opti_param['name_opti'] == 'noam':
                optimizer.optimizer.load_state_dict(check['optimizer_state_dict'])
            else:
                optimizer.load_state_dict(check['optimizer_state_dict'])
            print('on recup model {} '.format(num_test))

        epoch = check['epoch']
        loss = check['loss']
        acc = check['acc']
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        print('| Recup model on \tepoch: {} |\tLoss: {:.6f} |\tAcc: {:.6f} | lr {:.5f} '.format(epoch, loss, acc, lr))

        return model, optimizer



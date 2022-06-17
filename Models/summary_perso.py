import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def summary(model, input_size, batch_size=-1, device="cuda", info=False, src_msk=None, tgt=None, tgt_msk=None,
            transformer=False):

    if model.m_type == 'transformer':
        transformer = True

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            if info:
                print('----- hook -----')
                print('class name : {}'.format(class_name))
                print('output type : {}'.format(type(output)))

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            # print('on verif input shape enregistreé : {}'.format(summary[m_key]["input_shape"]))
            # summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                gost = output[0]
                summary[m_key]["output_shape"] = list(gost.size())
                # print('on verif output shape enregistreé : {}'.format(summary[m_key]["output_shape"]))
                # summary[m_key]["output_shape"][0] = batch_size
            else:
                summary[m_key]["output_shape"] = list(output.size())
                # print('on verif output shape enregistreé : {}'.format(summary[m_key]["output_shape"]))
                # summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if class_name == 'MultiheadAttention':
                for parameter in module.parameters():
                    # list_size.append(parameter.shape)
                    params += torch.prod(torch.LongTensor(list(parameter.shape)))
                summary[m_key]["trainable"] = params

            elif class_name == 'LSTM':
                # print(module._all_weights)
                allweight = module._all_weights[0]
                if hasattr(module, "weight_ih_l0") and hasattr(module.weight_ih_l0, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight_ih_l0.size())))
                    summary[m_key]["trainable"] = module.weight_ih_l0.requires_grad
                    # print('taille lstm 4*input*outpu : {} '.format(module.weight_ih_l0.size()))
                if hasattr(module, "weight_hh_l0") and hasattr(module.weight_hh_l0, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight_hh_l0.size())))
                    summary[m_key]["trainable"] = module.weight_hh_l0.requires_grad
                if hasattr(module, "bias_ih_l0") and hasattr(module.bias_ih_l0, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias_ih_l0.size())))
                if hasattr(module, "bias_hh_l0") and hasattr(module.bias_hh_l0, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias_hh_l0.size())))
                if hasattr(module, "weight_ih_l1") and hasattr(module.weight_ih_l1, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight_ih_l1.size())))
                    summary[m_key]["trainable"] = module.weight_ih_l1.requires_grad
                    # print('taille lstm 4*input*outpu : {} '.format(module.weight_ih_l1.size()))
                if hasattr(module, "weight_hh_l1") and hasattr(module.weight_hh_l1, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight_hh_l1.size())))
                    summary[m_key]["trainable"] = module.weight_hh_l1.requires_grad
                if hasattr(module, "bias_ih_l1") and hasattr(module.bias_ih_l1, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias_ih_l1.size())))
                if hasattr(module, "bias_hh_l1") and hasattr(module.bias_hh_l1, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias_hh_l1.size())))
            else:
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    # print('weight here !')
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    if type(device).__name__ == 'list':
        device_num = 2
        device = "cuda"
    elif type(device) == torch.device:
        device_num = device
        device = "cuda"
    else:
        device_num = None
        device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    # def input type
    if device == "cuda" and torch.cuda.is_available():
        if transformer:
            if model.embedding_type == 'nn':
                dtype = torch.cuda.LongTensor
            else:
                dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.cuda.FloatTensor
    else:
        if transformer:
            if model.embedding_type == 'nn':
                dtype = torch.LongTensor
            else:
                dtype = torch.FloatTensor
        else:
            dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    x = torch.rand(batch_size, *input_size[0] if np.size(input_size[0]) != 1 else input_size[0]).type(dtype)

    if transformer:
        x = torch.transpose(x, 0, 1)  # dans transformer batchsize en dim=1

        if src_msk is not None:
            src_msk = model.generate_square_subsequent_mask(src_msk)

        if tgt is not None:
            tgt = torch.rand(batch_size, *tgt[0] if np.size(tgt[0]) != 1 else tgt[0]).type(dtype)
            tgt = torch.transpose(tgt, 0, 1)  # dans transformer batchsize en dim=1

        if tgt_msk is not None:
            tgt_msk = model.generate_square_subsequent_mask(tgt_msk)

        if model.embedding_type != 'nn':
            x = x.unsqueeze(-1)
            if tgt is not None:
                tgt = tgt.unsqueeze(-1)

        print('into summary : size input = ({}, {})'.format(x.shape[0], x.shape[1]))
        if info:
            print('src size = {}'.format(x.shape))
            print('src msk size = {}'.format(src_msk.shape))
            print('tgt size = {}'.format(tgt.shape))
            print('tgt msk size = {}'.format(tgt_msk.shape))

    print('into summary : size input = ({})'.format(x.shape))
    if device_num is not None:
        model = model.to(device_num)
        x = x.to(device_num)
        if tgt is not None:
            tgt = tgt.to(device_num)
        if src_msk is not None:
            src_msk = src_msk.to(device_num)
        if tgt_msk is not None:
            tgt_msk = tgt_msk.to(device_num)

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    if transformer:
        output = model(x, tgt, src_msk, tgt_msk)
        model.generator(output)

    else:
        output = model(x)
        #model.generator(output)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Input",
        str(list(x.size())),
        "{0:,}".format(0)
    )
    print(line_new)
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])

        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary

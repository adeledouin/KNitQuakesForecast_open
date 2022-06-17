import numpy as np
import torch
import psutil
import warnings
from torch.utils.data import DataLoader
import tqdm
import timeit
import numpy as np
import torch
import time
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from time import sleep
# from livelossplot import PlotLosses
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import torch.nn as nn
from torchcontrib.optim import SWA
import matplotlib.patches as mpatches
import argparse

warnings.filterwarnings("ignore")

from classConfig import ConfigData, ConfigPred
from Models.classModel import Model
import Config_data
import Config_pred #_retro as Config_pred
from classPlot import ClassPlot
from Datas.classStat import Histo
from Datas.classSignal import SignalForce
from Datas.classEvent import ForceEvent
from Datas.classLoader import DataSetTIMESCALAR
from Datas.sub_createdata import create_generator_false_data
from Datas.classDataParallel import DataParallel
from utils_functions_learning import init_callback, AverageMeter, RecupTensor, which_model_to_recup, run_epoch_scalar, train_model, test_model
from Models.classLoss import MultiGPULossCompute, SimpleLossCompute
from Datas.classStat import pearson_corr_coef

################### Parser ##################################

parser = argparse.ArgumentParser(description="run data analyse")
parser.add_argument("-r", "--remote", action="store_true", help="n'affiche pas les plots")
parser.add_argument("-c", "--cuda", action="store_true", help="run sur gpu")
parser.add_argument("p", type=str, help="data path_from_root")
parser.add_argument("d", type=str, help="date")
# parser.add_argument("t", type=str, default='knit005_', help="référence du tricot")
# parser.add_argument("n", type=str, default='mix_', help="référence de l'expérience")
parser.add_argument("v", type=str, default='v2', help="version sous laquelle on travaille")
parser.add_argument("vpred", type=str, help="version pred sous laquelle on travaille")
parser.add_argument("subv", type=str, help="décline version sub sous laquelle on travaille")
parser.add_argument("model", type=int, help="num du model a train")

parser.add_argument("--train", action="store_true", help="train le model")
parser.add_argument("--reverse", action="store_true", help="use reverse classes")
parser.add_argument("--samplec", action="store_true", help="sample value on 2 classes")
parser.add_argument("--s1", type=int, default=None, help="classes 1 to sample")
parser.add_argument("--s2", type=int, default=None, help="classes 2 to sample")
parser.add_argument("--mask", type=int, default=None, help="add mask on input")
parser.add_argument("--checkpoint", type=str, default='', help="load checkpoint model")

parser.add_argument("--transfert", type=str, default=None, help="transfert type")
parser.add_argument("--transfert_date", type=str, default='', help="date du training a transferer")
parser.add_argument("--transfert_vpred", type=str, default='', help="version a transferer")
parser.add_argument("--transfert_subv", type=str, default='', help="subv a transferer")
parser.add_argument("--transfert_trainsize", type=str, default='', help="size du training a transfere")

parser.add_argument("--epoch", type=int, default=300, help="nb d epochs")
parser.add_argument("--trainsize", default=1000000, type=int, help="nb d epochs")
parser.add_argument("--cuda_device", type=str, default=None, help="cuda device")

parser.add_argument("--pred", type=str, default='scalar', help="scalar ou field prediction")
parser.add_argument("--verif_stats_output", action="store_true", help="look at stats des output")

args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()

################### Args ##################################

args.t = 'knit005_'
args.n = 'mix_'

if args.cuda:
    if type(args.cuda_device).__name__ == 'str':
        device = torch.device(args.cuda_device)
    else:
        device = args.cuda_device
else:
    device = "cpu"

NAME_EXP = args.t + args.n + args.v
config_data = ConfigData(args.path_from_root, Config_data.exp[NAME_EXP])

NAME_EXP = args.t + args.vpred
version_exp = Config_pred.exp_scalar[NAME_EXP]
config_pred = ConfigPred(Config_pred.exp_scalar[NAME_EXP], config_data)

config_pred.set_model_attribute(args.model)
model_spec = config_pred.spec_model_scalar()

histo = Histo(config_data)
plot = ClassPlot(args.remote, histo)

if config_pred.channel != 1:
    multi_features = True
else:
    multi_features = False
args.with_clip = True
args.use_SWA = False
random = False
if args.samplec:
    sample_classes = np.array([args.s1, args.s2])
else:
    sample_classes = None

# %% ################### Create sequences ##################################

print('------ create data sequences ------')
if not config_pred.false_data:
    signal_flu = [SignalForce(config_data, 'flu_rsc', NN) for NN in config_pred.NN_data]
    signalevent = [ForceEvent(config_data, signal_flu[i].f, signal_flu[i].ext, signal_flu[i].t,
                              'flu_rsc', config_pred.NN_data[i], Sm=False) for i in
                   range(np.size(config_pred.NN_data))]

    nb_seq = [args.trainsize, int(20 / 100 * args.trainsize), None]

    train_dataset = DataSetTIMESCALAR(config_data, config_pred, args.remote, plot, histo, nb_seq.copy(), signal_flu,
                                      signalevent, 0, multi_features=multi_features, reverse_classes=args.reverse,
                                      which_to_keep=sample_classes, mask_size=args.mask)
    # print('avant val nb seq = {}'.format(nb_seq))
    val_dataset = DataSetTIMESCALAR(config_data, config_pred, args.remote, plot, histo, nb_seq.copy(), signal_flu,
                                    signalevent,
                                    1, multi_features=multi_features, reverse_classes=args.reverse,
                                    which_to_keep=sample_classes, mask_size=args.mask)
    # print('avant test nb seq = {}'.format(nb_seq))
    test_dataset = DataSetTIMESCALAR(config_data, config_pred, args.remote, plot, histo, nb_seq.copy(), signal_flu,
                                     signalevent,
                                     2, multi_features=multi_features, reverse_classes=args.reverse,
                                     which_to_keep=sample_classes, mask_size=args.mask)

    if sample_classes is not None:
        config_pred.set_output_shape(np.size(sample_classes))
        config_pred.set_weight_classes(1 / train_dataset.prop[sample_classes])
        print(train_dataset.classes_edges)
        print(train_dataset.prop, 1 / train_dataset.prop)
    elif config_pred.output_type == 'class':
        config_pred.set_weight_classes(1 / train_dataset.prop)
        print(train_dataset.classes_edges)
        print(train_dataset.prop, 1 / train_dataset.prop)

else:
    # tests on false data
    from Datas import fakesin

    X, Y = fakesin.getxy()
    nb_seq = [X.size(), 0, Y.size()]


# %% ################### Data generator ##################################

print('------ create data generator ------')
if not config_pred.false_data:
    kwargs = {'num_workers': 5, 'pin_memory': True} if args.cuda else {}

    train_loader = DataLoader(train_dataset, batch_size=config_pred.batch_size, shuffle=True, **kwargs)

    val_loader = DataLoader(val_dataset, batch_size=config_pred.batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(test_dataset, batch_size=config_pred.batch_size, shuffle=False, **kwargs)

else:
    # test on false data
    train_loader, val_loader, test_loader = create_generator_false_data(args, config_pred, X, Y)



# %% ################### Informations ##################################

if args.reverse:
    num_test = ('{}_{}_m{}_{}seq{}_reverse').format(args.d, args.vpred, model_spec['magic_number'], nb_seq[0],
                                                    args.subv)
else:
    num_test = ('{}_{}_m{}_{}seq{}').format(args.d, args.vpred, model_spec['magic_number'], nb_seq[0], args.subv)

save_best_acc = config_pred.global_path_save + 'net/{}/best_acc_{}.tar'.format(args.pred, num_test)
save_best_loss = config_pred.global_path_save + 'net/{}/best_loss_{}.tar'.format(args.pred, num_test)
save_dict_version = config_pred.global_path_save + 'net/{}/config_pred_{}'.format(args.pred, num_test)
save_dict_model = config_pred.global_path_save + 'net/{}/model_{}'.format(args.pred, num_test)

save_callback = config_pred.global_path_save + 'callback/{}/callback_{}.npy'.format(args.pred, num_test)
save_test_callback = config_pred.global_path_save + 'callback/{}/test_callback_{}.npy'.format(args.pred, num_test)

save_loss = config_pred.global_path_save + 'callback/{}/loss_{}'.format(args.pred, num_test)
save_acc = config_pred.global_path_save + 'callback/{}/acc_{}'.format(args.pred, num_test)


save_metric = config_pred.global_path_save + 'callback/{}/metric_{}'.format(args.pred, num_test)

print('------ {} on model {} ------'.format(NAME_EXP, model_spec['magic_number']))
print('run on cuda : {} ; cuda device : {}'.format(args.cuda, device))
if args.train:
    print('on va train {} on {} epochs with training size {}'.format(model_spec['model_name'], args.epoch, nb_seq[0]))
else:
    print('on va test {}'.format(model_spec['model_name']))
print('pred with {} past on futur window of {}'.format(config_pred.seq_size, config_pred.futur))
print('avec {} seqs dans val, {} seqs dans test'.format(nb_seq[1], nb_seq[2]))


# %% ################### NN model ##################################

print('------ create model ------')
np.save(save_dict_version, version_exp)
np.save(save_dict_model, model_spec)

def_nn = Model(args, device, model_spec['model_name'], config_pred.output_type,
               model_spec['layers_param'], model_spec['opti_param'], model_spec['criterion_param'],
               config_pred.batch_size, config_pred.seq_size, config_pred.channel, None, None,
               config_pred.output_shape)

model = def_nn.NN_model()

optimizer, scheduler, criterion = def_nn.NN_opti_loss_scheduler(model)

if args.transfert is not None:
    num_test, load_best_acc, load_best_loss, checkpoint = which_model_to_recup(args, config_pred, model_spec, num_test,
                                                                   save_best_acc,
                                                                   save_best_loss)

    model, optimizer = def_nn.recup_from_checkpoint(args, model, optimizer, num_test, load_best_acc, load_best_loss, checkpoint)

if type(args.cuda_device).__name__ == 'list':
    model = DataParallel(model, device_ids=device)

if args.use_SWA:
    optimizer = SWA(optimizer)

# creat Loss compute
if type(args.cuda_device).__name__ == 'str':
    loss_compute = SimpleLossCompute(model.generator,
                                     model_spec['criterion_param'], criterion, model_spec['opti_param'], optimizer)
else:
    loss_compute = MultiGPULossCompute(model.module.generator,
                                       model_spec['criterion_param'], criterion,
                                       device, model_spec['opti_param'], optimizer.optimizer)

import time
tic = time.time()

if args.train:
    train_model(args, config_pred, model, loss_compute, optimizer, train_loader, val_loader,
                save_best_acc, save_best_loss, save_callback, scheduler, device)

print('t = {}'.format(time.time() - tic))

if random:
    model = model
    optimizer = optimizer
    test_model(args, config_pred, model, loss_compute, test_loader, save_test_callback,
               scheduler, device)
else:
    if args.checkpoint == '':
        if config_pred.output_type == 'class':
            print('Best acc model :')
            model, optimizer = def_nn.recup_from_checkpoint(args, model, optimizer, num_test, save_best_acc,
                                                            save_best_loss, 'acc')
            loss_compute = SimpleLossCompute(model.generator,
                                             model_spec['criterion_param'], criterion, model_spec['opti_param'],
                                             optimizer)
            test_model(args, config_pred, model, loss_compute, test_loader, save_test_callback,
                       scheduler, device)
        else:
            print('Best loss model :')
            model, optimizer = def_nn.recup_from_checkpoint(args, model, optimizer, num_test, save_best_acc,
                                                            save_best_loss,
                                                            'loss')
            loss_compute = SimpleLossCompute(model.generator,
                                             model_spec['criterion_param'], criterion, model_spec['opti_param'],
                                             optimizer)
            test_model(args, config_pred, model, loss_compute, test_loader, save_test_callback,
                       scheduler, device)
    else:
        print('Best {}} model :'.format(args.checkpoint))
        model, optimizer = def_nn.recup_from_checkpoint(args, model, optimizer, num_test, save_best_acc, save_best_loss,
                                                        args.checkpoint)
        loss_compute = SimpleLossCompute(model.generator,
                                         model_spec['criterion_param'], criterion, model_spec['opti_param'], optimizer)
        test_model(args, config_pred, model, loss_compute, test_loader, save_test_callback,
                   scheduler, device)



# %% ################### plot callback ##################################
from pathlib import Path

fileName = save_callback
fileObj = Path(fileName)
is_fileObj = fileObj.is_file()
if is_fileObj:
    callback = np.load(save_callback, allow_pickle=True).flat[0]

    # ------- loss and acc
    val_loss = callback['val_loss']
    val_acc = callback['val_acc']
    loss = callback['loss']
    acc = callback['acc']

    fig, ax = plot.belleFigure('${}$'.format('epoch'), '${}$'.format('loss'), nfigure=None)
    ax.plot(np.arange(np.size(loss)), loss, color='#1f77b4', marker='*', linestyle="None", label='train')
    ax.plot(np.arange(np.size(val_loss)), val_loss, color='#ff7f0e', marker='*', linestyle="None", label='val')
    save = save_loss
    plot.fioritures(ax, fig, title='loss for  {}'.format(num_test), label=True,
                    grid=None, save=save)

    fig, ax = plot.belleFigure('${}$'.format('epochs'), '${}$'.format('acc'), nfigure=None)
    ax.plot(np.arange(np.size(acc)), acc, color='#1f77b4', marker='*', linestyle="None", label='train')
    ax.plot(np.arange(np.size(val_acc)), val_acc, color='#ff7f0e', marker='*', linestyle="None", label='val')
    save = save_acc
    plot.fioritures(ax, fig, title='acc for  {}'.format(num_test), label=True, grid=None,
                    save=save)

# ------- stat on pred
callback = np.load(save_test_callback, allow_pickle=True).flat[0]
y_value = callback['y_value'].cpu().data.numpy()
y_target = callback['y_target'].cpu().data.numpy()
y_pred = callback['y_pred'].cpu().data.numpy()

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import torch.nn.functional as F


def classes_scores(config_pred, y_target, y_pred, repport=False):
    target = np.zeros((np.size(y_target), config_pred.output_shape))
    pred = np.zeros((np.size(y_target), config_pred.output_shape))

    for i in range(np.size(y_target)):
        target[i, int(y_target[i])] = 1
        pred[i, int(y_pred[i])] = 1

    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_target, y_pred)))

    print('Micro Precision: {:.2f}'.format(precision_score(y_target, y_pred, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(y_target, y_pred, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(y_target, y_pred, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(y_target, y_pred, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(y_target, y_pred, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(y_target, y_pred, average='macro')))

    print('Weighted Precision: {:.2f}'.format(precision_score(y_target, y_pred, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(y_target, y_pred, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(y_target, y_pred, average='weighted')))

    if repport:
        print('\nClassification Report\n')
        print(classification_report(y_target, y_pred,
                                    target_names=['Class {}'.format(i) for i in range(config_pred.output_shape)]))


def make_grid_classes(classes_edges):
    grid = np.zeros(np.size(classes_edges))
    for i in range(np.size(classes_edges)):
        grid[i] = classes_edges[i]
    return grid


def accuracy_value_dispach(y_value, y_target, y_pred):
    g_value = []
    w_value = []

    for i in range(np.size(y_value)):
        if y_pred[i] == y_target[i]:
            g_value.append(y_value[i])
        else:
            w_value.append(y_value[i])

    return np.asarray(g_value), np.asarray(w_value)



if config_pred.output_type == 'class':
    from Datas.classStat import Stat

    nbclasses = config_pred.output_shape
    print('#-------------classes--------------#')
    log_classes = train_dataset.classes_edges

    # ----------- Class learning
    if nbclasses == 2:
        x_conf = np.array([5e-4, 3e-1])
    elif nbclasses == 3:
        x_conf = np.array([5e-4, 3e-2, 2e0])
    elif nbclasses == 4:
        x_conf = np.array([5e-4, 1e-2, 1e-1, 2e0])
    else:
        x_conf = [3e-3, 1e-2, 1e-1, 1e0, 5e0]
    class_grid = np.concatenate((np.array([1e-4]), test_dataset.classes_edges[1::]))

    # ----------- Class physiques
    decade = np.array([0, 5e-3, 3e-2, 3e-1, 3e0, np.max(test_dataset.dict_all_value['train'])])
    name_decade = ['decade_{}'.format(i) for i in range(decade.size - 1)]
    decade_nzero = np.array([5e-3, 3e-2, 3e-1, 3e0, np.max(np.max(y_value))])
    x_count = [3e-3, 1e-2, 1e-1, 1e0, 5e0]
    decade_grid = decade_nzero
    major = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, np.max(test_dataset.dict_all_value['train'])]

    # ----------- Targets
    y_Pdf, x_Pdf = plot.histo.my_histo(y_value, 1e-4, np.max(y_value),
                                                 'log', 'log', density=2, binwidth=None, nbbin=70)
    yname = config_pred.label_name
    ysave = config_pred.label_save
    title = True
    label = True
    grid = decade_grid
    fig, ax = plot.belleFigure('$log10({})$'.format(yname), '${}({})$'.format('Pdf_{log10}', yname), nfigure=None)
    ax.plot(x_Pdf, y_Pdf, '.')
    plot.plt.xscale('log')
    plot.plt.yscale('log')
    for i in range(nbclasses + 1):
        plot.plt.axvline(x=class_grid[i], color='k')
    save = save_metric + '_Pdf_values'
    plot.fioritures(ax, fig, title='Pdf sur {}'.format(ysave) if title else None, label=label, grid=grid, save=save)


    value_good_classe, value_wrong_classe = accuracy_value_dispach(y_value, y_target, y_pred)

    grid = make_grid_classes(log_classes)
    fig, ax = plot.belleFigure('${}$'.format('bla'), '${}$'.format('values'), nfigure=None)
    ax.plot(np.arange(np.size(value_good_classe)), value_good_classe, '.', label='good class')
    ax.plot(np.arange(np.size(value_wrong_classe)), value_wrong_classe, 'r.', label='wrong class')
    ax.set_yticks(grid, minor=False)
    ax.yaxis.grid(True)
    save = save_metric + '_output_values'
    plot.fioritures(ax, fig, title=None, label=None, grid=None, save=None)

    value_wrong_classe = value_wrong_classe[value_wrong_classe != 0]
    value_good_classe = value_good_classe[value_good_classe != 0]

    stats_good_v = Stat(config_data, value_good_classe)
    stats_wrong_v = Stat(config_data, value_wrong_classe)

    good_y_Pdf, good_x_Pdf = plot.histo.my_histo(value_good_classe, 1e-4, stats_good_v.max,
                                                 'log', 'log', density=2, binwidth=None, nbbin=70)
    wrong_y_Pdf, wrong_x_Pdf = plot.histo.my_histo(value_wrong_classe, 1e-4, stats_good_v.max,
                                                   'log', 'log', density=2, binwidth=None, nbbin=70)

    yname = config_pred.label_name
    ysave = config_pred.label_save
    title = True
    label = True
    grid = decade_grid
    fig, ax = plot.belleFigure('$log10({})$'.format(yname), '${}({})$'.format('Pdf_{log10}', yname), nfigure=None)
    ax.plot(good_x_Pdf, good_y_Pdf, '.', label='good pred')
    ax.plot(wrong_x_Pdf, wrong_y_Pdf, 'r.', label='wrong pred')
    plot.plt.xscale('log')
    plot.plt.yscale('log')
    for i in range(nbclasses + 1):
        plot.plt.axvline(x=class_grid[i], color='k')
    save = save_metric + '_Pdf_output_values_density2'
    plot.fioritures(ax, fig, title='Pdf sur {}'.format(ysave) if title else None, label=label, grid=grid, save=save)

    X_good = [0 for i in range(nbclasses)]
    Y_good = [0 for i in range(nbclasses)]

    X_wrong = [0 for i in range(nbclasses)]
    Y_wrong = [0 for i in range(nbclasses)]

    for i in range(nbclasses):
        df_seuil_good = signalevent[0].df_seuil_fast(value_good_classe, log_classes[i], log_classes[i + 1])
        df_seuil_wrong = signalevent[0].df_seuil_fast(value_wrong_classe, log_classes[i],
                                                      log_classes[i + 1])

        if np.size(df_seuil_good) != 0:
            stats_good_v = Stat(config_data, df_seuil_good)
            Y_good[i], X_good[i] = histo.my_histo(df_seuil_good, 1e-4, stats_good_v.max, 'log',
                                                  'log', density=1, binwidth=None, nbbin=70)
        else:
            Y_good[i] = None
            X_good[i] = None

        if np.size(df_seuil_wrong) != 0:
            stats_wrong_v = Stat(config_data, df_seuil_wrong)

            Y_wrong[i], X_wrong[i] = histo.my_histo(df_seuil_wrong, 1e-4, stats_good_v.max, 'log',
                                                    'log', density=1, binwidth=None, nbbin=70)
        else:
            Y_wrong[i] = None
            X_wrong[i] = None

    fig, ax = plot.belleFigure('$log10({})$'.format(yname), '${}({})$'.format('Count_{log10}', yname), nfigure=None)
    for i in range(nbclasses):
        if X_good[i] is not None:
            ax.plot(X_good[i], Y_good[i], color='#1f77b4', marker='.', linestyle="None",
                    label='good ; seuil =' + str(log_classes[i]) + 'N')
        if X_wrong[i] is not None:
            ax.plot(X_wrong[i], Y_wrong[i], 'r.',
                    label='wrong; seuil =' + str(log_classes[i]) + 'N')
    plot.plt.xscale('log')
    plot.plt.yscale('log')
    for i in range(nbclasses + 1):
        plot.plt.axvline(x=class_grid[i], color='k')
    save = save_metric + '_Pdf_output_values_density1'
    plot.fioritures(ax, fig, title='renorm count sur {}'.format(ysave) if title else None, label=None, grid=grid, save=save)

    classes_scores(config_pred, y_target, y_pred, repport=True)

    conf = np.asarray(confusion_matrix(y_target, y_pred))

    colorpred = ['C0', 'C1', 'C2', 'C3', 'C4']
    X_conf = np.arange(nbclasses)
    yname = 'qt-pred'
    ysave = 'qt_pred'
    fig, ax = plot.belleFigure('${}$'.format('classe-pred'), '${}$'.format(yname), nfigure=None)
    norm = np.asarray([np.sum(conf[:, i]) for i in range(nbclasses)])
    for i in range(nbclasses):
        for k in range(nbclasses):
            ax.plot(X_conf[k], (conf[i, k] / norm[k]) * 100, color=colorpred[i], marker='.', linestyle="None",label='classe {}'.format(i))
            plot.plt.plot([X_conf[k]-0.5, X_conf[k]+0.5], [(conf[i, k] / norm[k]) * 100, (conf[i, k] / norm[k]) * 100], color=colorpred[i])
    # for i in range(nbclasses + 1):
    #     plot.plt.axvline(x=i-0.5, color='k')
    # plot.plt.xscale('log')
    # plot.plt.yscale('log')
    save = save_metric + '_qt_pred'
    patchs = [0 for _ in range(nbclasses)]
    plot.fioritures(ax, fig, title='{}'.format(ysave) if title else None, label=False, grid=None, save=save)
    for i in range(nbclasses):
        patchs[i] = mpatches.Patch(color=colorpred[i], label='classe-target {}'.format(i))
    plot.plt.legend(handles=patchs)
    plot.plt.show()

    # print('\nClassification Report\n')
    # print(classification_report(y_target, y_pred,
    #                             target_names=['Class {}'.format(i) for i in range(config_pred.output_shape)]))


    conf_norm = np.zeros_like(conf, dtype=float)
    for i in range(nbclasses):
        conf_norm[i, :] = (conf[i, :] / norm) * 100
    #
    # classedges = log_classes
    # prop = train_dataset.prop
    # plot.Pdf_loglog(y_value, np.min(y_value[y_value != 0]), np.max(y_value),
    #                 'log', 'log', save=None, binwidth=None, nbbin=150)
    #
    # x_Pdf, y_Pdf = histo.my_histo(y_value, np.min(y_value[y_value != 0]), np.max(y_value), 'log',
    #                                       'log', density=2, binwidth=None, nbbin=70)
    #
    # linx = x_Pdf[y_Pdf != 0]
    # liny = y_Pdf[y_Pdf != 0]
    #
    # coef_distri, x, y = plot.histo.regression(linx, liny, None, None, x_axis='log', y_axis='log')
    #
    # polynomial = np.poly1d(coef_distri)
    # ys = polynomial(x)
    #
    # fig, ax = plot.belleFigure('${}$'.format('value'), '${}({})$'.format('Pdf_{log10}', 'value'), nfigure=None)
    # ax.plot(x, y, '.')
    # ax.plot(x, ys, 'r-', label='coeff polifit = {}'.format(coef_distri[0]))
    # save = None
    # plot.fioritures(ax, fig, title='pdf value', label=True, grid=None,
    #                 save=save)
else:
    MAE = np.mean(np.abs(y_target - y_pred))
    MSE = np.mean((y_target - y_pred) ** 2)
    M4E = np.mean((y_target - y_pred) ** 4)
    M6E = np.mean((y_target - y_pred) ** 6)

    print(
        '\tMAE: {:.6f} |\tMSE: {:.6f} |\tM4E: {:.6f} |\tM6E: {:.6f}'.format(
            MAE, MSE, M4E, M6E))

    fig, ax = plot.belleFigure('${}$'.format('pts'), '${}$'.format('y'), nfigure=None)
    ax.plot(np.arange(y_target.size), y_target, 'b.', label='target')
    ax.plot(np.arange(y_pred.size), y_pred, 'r.', label='pred')
    save = save_metric + '_y_time_plot'
    plot.fioritures(ax, fig, title=None, label=True, grid=None, save=save)

    fig, ax = plot.belleFigure('${}$'.format('y_{target}'), '${}$'.format('y_{pred}'), nfigure=None)
    ax.plot(y_target, y_pred, '.', label='')
    save = save_metric + '_y_target_vs_pred'
    plot.fioritures(ax, fig, title=None, label=None, grid=None, save=save)

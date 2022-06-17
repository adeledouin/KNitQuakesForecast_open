import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")

from classConfig import ConfigData, ConfigPred
from Models.classModel import Model
import Config_data
import Config_pred
from classPlot import ClassPlot
from Datas.classStat import Histo
from utils_functions_learning import train_model, test_model
from Datas.sub_createdata import create_sequences_scalar, create_sequences_field, create_generator_scalar, \
    create_generator_field, create_generator_false_data
from Datas.classDataParallel import DataParallel
from Models.classLoss import SimpleLossCompute, MultiGPULossCompute

from Datas.classSignal import VariationsScalar, SignalForce
from Datas.classEvent import ForceEvent

################### Args ##################################
remote = False

date = '210519'
ref_tricot = 'knit12_'
n_exp = 'mix_'
version_data = 'v2'
version_pred = 'v1'
sub_version = ''
model = 12

epochs = 100
trainsize = None
cuda_device = "cuda:1"


class Args:
    def __init__(self):
        self.remote = remote
        self.cuda = True
        self.vpred = version_pred
        self.subv = sub_version
        self.pred = 'scalar'
        self.train = True
        self.verif_equipro = False
        self.checkpoint = ''
        self.epoch = epochs
        self.trainsize = trainsize
        self.cuda_device = cuda_device


args = Args()

if args.cuda:
    if type(args.cuda_device).__name__ == 'str':
        device = torch.device(args.cuda_device)
    else:
        device = args.cuda_device
else:
    device = "cpu"

NAME_EXP = ref_tricot + n_exp + version_data
config_data = ConfigData(Config_data.exp[NAME_EXP])

NAME_EXP = ref_tricot + version_pred
if args.pred == 'scalar':
    version_exp = Config_pred.exp_scalar[NAME_EXP]
    config_pred = ConfigPred(Config_pred.exp_scalar[NAME_EXP], config_data)
else:
    version_exp = Config_pred.exp_img[NAME_EXP]
    config_pred = ConfigPred(Config_pred.exp_img[NAME_EXP], config_data)

config_pred.set_model_attribute(model)
if args.pred == 'scalar':
    model_spec = config_pred.spec_model_scalar()
else:
    model_spec = config_pred.spec_model_img()

histo = Histo(config_data)
plot = ClassPlot(args.remote, histo)


if args.pred == 'scalar':
    num_test = ('{}_{}_m{}_{}seq{}').format(date, version_pred, model_spec['magic_number'], nb_seq[0], sub_version)
else:
    num_test = ('{}_{}_m{}_{}seq{}').format(date, version_pred, model_spec['magic_number'], nb_seq[0], sub_version)

save_best_acc = config_pred.global_path_save + 'net/{}/best_acc_{}.tar'.format(args.pred, num_test)
save_best_loss = config_pred.global_path_save + 'net/{}/best_loss_{}.tar'.format(args.pred, num_test)
save_dict_version = config_pred.global_path_save + 'net/{}/config_pred_{}'.format(args.pred, num_test)
save_dict_model = config_pred.global_path_save + 'net/{}/model_{}'.format(args.pred, num_test)

save_callback = config_pred.global_path_save + 'callback/{}/callback_{}.npy'.format(args.pred, num_test)
save_test_callback = config_pred.global_path_save + 'callback/{}/test_callback_{}.npy'.format(args.pred, num_test)

save_loss = config_pred.global_path_save + 'callback/{}/loss_{}'.format(args.pred, num_test)
save_acc = config_pred.global_path_save + 'callback/{}/acc_{}'.format(args.pred, num_test)

print('------ {} on model {} ------'.format(NAME_EXP, model_spec['magic_number']))
print('run on cuda : {} ; cuda device : {}'.format(args.cuda, device))

signal_flu = SignalForce(config_data, 'flu_rsc', 'train')
signalevent = ForceEvent(config_data, signal_flu.f, signal_flu.ext, signal_flu.t,
                              'flu_rsc', 'train', Sm=False)

def threshold_f(f, threshold):
    print('befor : min = {} max = {}'.format(np.min(f), np.max(f)))
    f_cut = np.round(f, threshold)
    print('after : min = {} max = {}'.format(np.min(f_cut), np.max(f_cut)))

    return f_cut

f_cut = threshold_f(signalevent.f, 3)
binwidth = 0.01
variations_flu = VariationsScalar(config_data, pourcentage=5, f=signalevent.f, ext=signalevent.ext, t=signalevent.t,
                                          index=None, number=None, directsignal=True,
                                          signaltype='flu_rsc', NN_data='train', ftype='force', fname=signalevent.fname,
                                          stats=False)

plot.Pdf_linlin(variations_flu.ndim_to_1dim, variations_flu.stats_f.min, variations_flu.stats_f.max,
                         'F', '{}'.format(signalevent.fname), None,
                    label='de moyenne {} N et de variance {} N'.format(np.round(variations_flu.stats_f.mean, 4),
                                                               np.round(variations_flu.stats_f.var, 4)))

variations_flu = VariationsScalar(config_data, pourcentage=5, f=f_cut, ext=signalevent.ext, t=signalevent.t,
                                          index=None, number=None, directsignal=True,
                                          signaltype='flu_rsc', NN_data='train', ftype='force', fname=signalevent.fname,
                                          stats=False)

plot.Pdf_linlin(variations_flu.ndim_to_1dim, variations_flu.stats_f.min, variations_flu.stats_f.max,
                         'F', '{}'.format(signalevent.fname), None, nbbin=None, binwidth=binwidth,
                    label='de moyenne {} N et de variance {} N'.format(np.round(variations_flu.stats_f.mean, 4),
                                                               np.round(variations_flu.stats_f.var, 4)))


event = ForceEvent(config_data, f_cut, signalevent.ext, signalevent.t, 'flu_rsc', 'train', Sm=False, saving_step=False)

variations = [0, 0]

variations[0] = VariationsScalar(config_data, pourcentage=5, f=event.df_tt, ext=signalevent.ext, t=signalevent.t,
                                    index=event.index_df_tt, number=event.number_df_tt, directsignal=False,
                                    signaltype='flu_rsc', NN_data='train', ftype='force',
                                    fname='df' + event.savename_df_tt,
                                    stats=False, multi_seuils=True)
variations[1] = VariationsScalar(config_data, pourcentage=5, f=signalevent.df_tt, ext=signalevent.ext, t=signalevent.t,
                                    index=event.index_df_tt, number=event.number_df_tt, directsignal=False,
                                    signaltype='flu_rsc', NN_data='train', ftype='force',
                                    fname='df' + event.savename_df_tt,
                                    stats=False, multi_seuils=True)

plot.plot_pdf_loglog_multiseuil(2, variations, '\Delta \delta f', 'df' + event.savename_df_tt, None,
                                ['None', '-5'], ['None', '-5'])




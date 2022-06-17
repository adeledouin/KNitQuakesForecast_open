import numpy as np
import torch
import warnings
from torch.utils.data import DataLoader
from pathlib import Path
import scipy as sp

from classConfig import ConfigData, ConfigPred
import Config_data
import Config_pred
from classPlot import ClassPlot
from Datas.classStat import Histo, Stat
from Datas.classSignal import SignalForce, VariationsScalar
from Datas.classEvent import ForceEvent
from utils_functions_learning import train_model, test_model
from Datas.classData import find_nbseq, CreateDataScalar
from Datas.classLoader import DataSetTIMESCALAR, DataSetTIMEFIELD, DataSetFalseData
from Datas.sub_createdata import create_sequences_field, \
    create_generator_field, create_generator_false_data
from Datas.classDataParallel import DataParallel
from Models.classLoss import SimpleLossCompute, MultiGPULossCompute
from dictdata import dictdata
from Datas.classSignal import def_names
from Datas.classSignal import Shape
from Datas.classStat import Stat, Histo
from Datas.classCell import Cell
import numpy as np
import torch
from torch.utils.data import Dataset
from Datas.classData import find_nbseq, CreateDataScalar

import logging
import time

logging.basicConfig(format='| %(levelname)s | %(asctime)s | %(message)s', level=logging.INFO)

# %% ################### Args ##################################
remote = False

path_from_root = '/path_from_root/'

date = '220524'
ref_tricot = 'knit005_'
n_exp = 'mix_'
version_data = 'v2'
version_pred = 'v1_5_3'
sub_version = '_stats'
model = 23
num_target = 2
num_tau = 4

epochs = 1
trainsize = 1000000
cuda_device = "cuda:0"
reverse = False

class Args:
    def __init__(self):
        self.remote = remote
        self.cuda = True
        self.vpred = version_pred
        self.subv = sub_version
        self.pred = 'scalar'
        self.train = True
        self.verif_stats_output = False
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
config_data = ConfigData(path_from_root, Config_data.exp[NAME_EXP])

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

# %% ################### Create sequences ##################################

print('------ create data sequences ------')
signal_flu = [SignalForce(config_data, 'flu_rsc', NN) for NN in config_pred.NN_data]
signalevent = [ForceEvent(config_data, signal_flu[i].f, signal_flu[i].ext, signal_flu[i].t,
                          'flu_rsc', config_pred.NN_data[i], Sm=False) for i in
               range(np.size(config_pred.NN_data))]

nb_seq = [1000000, 200000, None]

import scipy as sp

def gaussian_distrib(x, mu, sigma):
    return np.exp(-(x - mu) ** 2 / (2 * sigma**2))

def normale_distrib(x, mu, sigma):
    return 1 / (np.sqrt(2 * np.pi * sigma)) * np.exp(-(x - mu) ** 2 / (2 * sigma))

def levy_distrib(x, mu, c):
    return 1/(x-mu)**(3/2)*np.exp(-c/(2*(x-mu)))

class CreateDataScalar:
    def __init__(self, config_data, config_pred, remote, plot, histo, cycle_sample=None):
        self.config_data = config_data
        self.config_pred = config_pred
        self.remote = remote
        self.plot = plot
        self.histo = histo

        self.NN = config_pred.NN_data
        self.label_name = config_pred.label_name
        self.label_save = config_pred.label_save

        self.seq_size = config_pred.seq_size
        self.overlap_step = config_pred.overlap_step
        self.output_shape = config_pred.output_shape
        self.futur = config_pred.futur
        self.cycle_sample = cycle_sample

    # ------------------------------------------
    def save_single(self, path_signal, data, name, extension='npy', nbfichier=None):

        if data is not None:
            if extension == 'npy':
                to_save = path_signal + name
                # print(to_save)
                np.save(to_save, data)
            elif extension == 'cell':
                Cell(path_signal + name, nbfichier, data=data, extension='cell')
            else:
                Cell(path_signal + name, nbfichier, data=data, extension='csv')

    # ------------------------- Sequences
    # ------------------------------------------
    def random_indices(self, f, size):

        pickrandom_lin = np.random.choice(f.shape[0] * f.shape[1], size, replace=False)
        pickrandom_mat = np.unravel_index(pickrandom_lin, (f.shape[0], f.shape[1]))

        return pickrandom_mat[0], pickrandom_mat[1]

    # ------------------------------------------
    def all_indices(self, f):

        lin = np.arange(f.shape[0] * f.shape[1])
        mat = np.unravel_index(lin, (f.shape[0], f.shape[1]))

        return mat[0], mat[1]

    # ------------------------------------------
    def cycles_sample_indices(self, f, cycle):
        shape = Shape(f)

        tps_to_keep = np.arange(0, shape.tps - (self.seq_size + self.futur) + 1, self.overlap_step)
        cycle_to_keep = np.ones_like(tps_to_keep) * cycle

        print('indice to keep shape = {}'.format(np.shape(tps_to_keep)))
        indices = np.vstack((cycle_to_keep, tps_to_keep)).transpose()

        return indices

    # ------------------------------------------
    def seq_indices(self, f):
        shape = Shape(f)

        print('shape de f : {}, {}'.format(shape.cycles, shape.tps))

        tps_to_keep = np.arange(0, shape.tps - (self.seq_size + self.futur) + 1, self.overlap_step)
        indice_to_keep = np.zeros((shape.cycles, tps_to_keep.size))

        print(
            'indice to keep shape = {} et size = {}'.format(np.shape(indice_to_keep), shape.cycles * tps_to_keep.size))
        cycles, tps = self.all_indices(indice_to_keep)
        indices = np.vstack((cycles, tps_to_keep[tps])).transpose()

        return indices

    # ------------------------------------------
    def __NN_sequences__(self, signal_flu):

        fileName = self.config_pred.global_path_load + 'sequence_NN/' + 'dict_sequences_{}_{}seqsize_{}step_{}futur.npy'.format(
            self.config_pred.input_data, self.seq_size, self.overlap_step, self.futur)
        fileObj = Path(fileName)
        is_fileObj = fileObj.is_file()

        if not is_fileObj:
            dict_sequences = dictdata()
            for i in range(np.size(self.NN)):
                NN = self.NN[i]

                indices = self.seq_indices(signal_flu[i].f)

                dict_sequences.add(NN, indices)

            self.save_single(fileName, dict_sequences, '')

        else:
            dict_sequences = np.load(fileName, allow_pickle=True).flat[0]

        self.dict_all_sequence = dict_sequences

    # ------------------------- Values
    # ------------------------------------------
    def def_sub_df(self, df, index, number, cycle, t0, from_X=True):

        if from_X:
            seq_size = self.seq_size
        else:
            seq_size = 1

        sub_index = index[cycle, t0 + seq_size:t0 + seq_size + self.futur]
        sub_number = number[cycle, t0 + seq_size:t0 + seq_size + self.futur]

        numbers = sub_number[sub_index == 1].astype(int)

        sub_df = np.zeros_like(sub_index)
        sub_df[sub_index == 1] = df[numbers]

        return sub_df

    # ------------------------------------------
    def def_value_for_classes(self, sub_df, conv=None):

        if self.label_save == 'max_df':
            countdf = np.max(sub_df)
        elif self.label_save == 'sum_df':
            countdf = np.sum(sub_df)
        else:
            countdf = np.sum(sub_df * conv)
        return countdf

    # ------------------------------------------
    def def_value_for_reg(self, sub_df):

        if self.label_save == 'expdf' or self.label_save == 'expdf_logrsc':
            exp = np.exp(-np.arange(0, np.size(sub_df)) / self.config_pred.tau)
            countdf = np.sum(sub_df * exp)
        else:
            print('warning non codé')
            countdf = None

        return countdf

    # ------------------------------------------
    def __all_value__(self, signalevent):
        if self.config_pred.output_type == 'class':
            fileName = self.config_pred.global_path_load + 'sequence_NN/' + 'dict_all_value_for_class_{}_seqsize_{}_futur_{}.npy'.format(
                self.label_save, self.seq_size, self.futur)
        else:
            fileName = self.config_pred.global_path_load + 'sequence_NN/' + 'dict_all_value_for_reg_{}_futur_{}.npy'.format(
                self.label_save, self.futur)

        fileObj = Path(fileName)
        is_fileObj = fileObj.is_file()
        if not is_fileObj:
            dict_value = dictdata()

            dict_sub_df = self.__all_sub_df_value__(signalevent)

            if self.label_save == 'max_df' or self.label_save == 'sum_df':
                conv = None
            elif self.label_save == 'sum_expdf':
                conv = np.exp(-np.arange(0, self.futur) / self.config_pred.tau)
            elif self.label_save == 'sum_gammadf':
                gamma = sp.stats.gamma(self.config_pred.k, loc=0, scale=self.config_pred.theta)
                conv = gamma.pdf(np.arange(0, self.futur))
            else:
                conv = gaussian_distrib(np.arange(0, self.futur), self.config_pred.mu, self.config_pred.sigma)

            for i in range(np.size(self.NN)):
                NN = self.NN[i]
                df = dict_sub_df[NN]
                nb_value = self.dict_all_sequence[NN].shape[0]
                value = np.zeros(nb_value)
                for j in range(nb_value):
                    logging.info("calcul value {}".format(j))
                    if self.config_pred.output_type == 'class':
                        value[j] = self.def_value_for_classes(df[j, :], conv=conv)
                    else:
                        value[j] = self.def_value_for_reg(df[j, :])

                dict_value.add(NN, value)

                logging.info("save into {}".format(fileName))
                self.save_single(fileName, dict_value, '')
        else:
            logging.info("load from {}".format(fileName))
            dict_value = np.load(fileName, allow_pickle=True).flat[0]

        self.dict_all_value = dict_value

    # ------------------------------------------
    def __all_sub_df_value__(self, signalevent):

        dict_sub_df = dictdata()
        for i in range(np.size(self.NN)):
            NN = self.NN[i]
            logging.info("in {}".format(NN))

            fileName = self.config_pred.global_path_load + 'sequence_NN/' + 'all_sub_df_value_seqsize_{}_futur_{}_{}.npy'.format(
                    self.seq_size, self.futur, NN)

            fileObj = Path(fileName)
            is_fileObj = fileObj.is_file()
            if not is_fileObj:

                df = signalevent[i].df_tt
                index = signalevent[i].index_df_tt
                number = signalevent[i].number_df_tt

                nb_value = self.dict_all_sequence[NN].shape[0]
                value = np.ones((nb_value, self.futur)) * np.NaN
                for j in range(nb_value):
                    cycle = self.dict_all_sequence[NN][j, 0]
                    t0 = self.dict_all_sequence[NN][j, 1]
                    value[j, :] = self.def_sub_df(df, index, number, cycle, t0)

                    if j % 10/100*nb_value == 0:
                        logging.info("calcul {}%".format(j/nb_value*100))
                logging.info("save into {}".format(fileName))
                self.save_single(fileName, value, '')
                dict_sub_df.add(NN, value)
            else:
                logging.info("load from {}".format(fileName))
                value = np.load(fileName)
                dict_sub_df.add(NN, value)

        return dict_sub_df

    # ------------------------------------------
    def __cycle_value__(self, signalevent):
        if self.config_pred.output_type == 'class':
            fileName = self.config_pred.global_path_load + 'sequence_NN/' + 'dict_cycle{}_value_for_class_{}_futur_{}.npy'.format(
                self.cycle_sample, self.label_save, self.futur)
        else:
            fileName = self.config_pred.global_path_load + 'sequence_NN/' + 'dict_cycle{}_value_for_reg_{}_futur_{}.npy'.format(
                self.cycle_sample, self.label_save, self.futur)

        fileObj = Path(fileName)
        is_fileObj = fileObj.is_file()
        if not is_fileObj:
            dict_value = dictdata()
            dict_sub_df = self.__all_sub_df_value__(signalevent)

            if self.label_save == 'max_df' or self.label_save == 'sum_df':
                conv = None
            elif self.label_save == 'sum_expdf':
                conv = np.exp(-np.arange(0, self.futur) / self.config_pred.tau)
            elif self.label_save == 'sum_gammadf':
                gamma = sp.stats.gamma(self.config_pred.k, loc=0, scale=self.config_pred.theta)
                conv = gamma.pdf(np.arange(0, self.futur))
            else:
                conv = gaussian_distrib(np.arange(0, self.futur), self.config_pred.mu, self.config_pred.sigma)

            for i in range(np.size(self.NN)):
                NN = self.NN[i]
                df = dict_sub_df[NN]
                nb_value = self.dict_all_sequence[NN].shape[0]
                value = np.zeros(nb_value)
                for j in range(nb_value):
                    logging.info("calcul value {}".format(j))
                    if self.config_pred.output_type == 'class':
                        value[j] = self.def_value_for_classes(df[j, :], conv=conv)
                    else:
                        value[j] = self.def_value_for_reg(df[j, :])

                dict_value.add(NN, value)

                logging.info("save into {}".format(fileName))
                self.save_single(fileName, dict_value, '')
        else:
            logging.info("load from {}".format(fileName))
            dict_value = np.load(fileName, allow_pickle=True).flat[0]

    # ------------------------- Info Class Edges
    # ------------------------------------------
    def classes(self, labels):

        print('min value_on_futur = {} et max value_on_futur = {}'.format(np.min(labels), np.max(labels)))

        if self.output_shape == 2:
            classes_edges = np.array([np.min(labels), 5e-3, np.max(labels)])
        elif self.output_shape == 3:
            classes_edges = np.array([np.min(labels), 5e-3, 3e-1, np.max(labels)])
        elif self.output_shape == 4:
            classes_edges = np.array([np.min(labels), 5e-3, 3e-2, 3e-1, np.max(labels)])
        elif self.output_shape == 5:
            classes_edges = np.array([np.min(labels), 5e-3, 3e-2, 3e-1, 3e0, np.max(labels)])
        else:
            supp = np.logspace(np.log10(5e-3), np.log10(3e0), self.output_shape-1)
            classes_edges = np.concatenate((np.array([np.min(labels)]), supp, np.array([np.max(labels)])))
        return classes_edges

    # ------------------------------------------
    def classes_reversed(self, labels):

        print('min value_on_futur = {} et max value_on_futur = {}'.format(np.min(labels), np.max(labels)))

        if self.output_shape == 2:
            classes_edges = np.array([np.min(labels), 3e0, np.max(labels)])
        elif self.output_shape == 3:
            classes_edges = np.array([np.min(labels), 3e-1, 3e0, np.max(labels)])
        elif self.output_shape == 4:
            classes_edges = np.array([np.min(labels), 5e-3, 3e-1, 3e0, np.max(labels)])
        elif self.output_shape == 5:
            classes_edges = np.array([np.min(labels), 5e-3, 3e-2, 3e-1, 3e0, np.max(labels)])
        else:
            supp = np.logspace(np.log10(5e-3), np.log10(3e0), self.output_shape-1)
            classes_edges = np.concatenate((np.array([np.min(labels)]), supp, np.array([np.max(labels)])))
        return classes_edges

        return classes_edges

    # ------------------------------------------
    def nonequipro_classes(self, labels):

        print('min value_on_futur = {} et max value_on_futur = {}'.format(np.min(labels), np.max(labels)))

        label_still_to_cut = labels[labels >= 5e-3]

        slice = np.round(label_still_to_cut.size / (self.output_shape - 1))
        sorted_labels = np.sort(label_still_to_cut, axis=0)

        classes_edges = np.zeros(self.output_shape - 1)
        for i in range(self.output_shape - 1):
            # print(i)
            where = np.where(label_still_to_cut >= sorted_labels[int(i * slice)])[0]
            classes_edges[i] = np.min(label_still_to_cut[where])
            # print(classes_edges[i])

        classes_edges = np.hstack((np.array([0]), classes_edges))
        classes_edges = np.hstack((classes_edges, np.max(label_still_to_cut)))

        hist, bin_edges = np.histogram(labels, density=False, bins=classes_edges)
        proportion = np.round(np.asarray(hist) / np.sum(hist), 3)

        if proportion.sum() != 1:
            proportion[0] = 1 - proportion[1::].sum()

        classes_edges = np.vstack((classes_edges, np.hstack((proportion, np.array([0])))))

        return classes_edges

    # ------------------------------------------
    def equipro_classes(self, labels):

        print('min value_on_futur = {} et max value_on_futur = {}'.format(np.min(labels), np.max(labels)))

        classes_edges = np.zeros(self.output_shape + 1)
        slice = np.round(labels.size / self.output_shape)
        sorted_labels = np.sort(labels, axis=0)

        for i in range(self.output_shape):
            where = np.where(labels >= sorted_labels[int(i * slice)])[0]
            classes_edges[i] = np.min(labels[where])
        classes_edges[-1] = np.max(labels)
        # print(classes_edges[-1])

        hist, bin_edges = np.histogram(labels, density=False, bins=classes_edges)
        proportion = np.round(np.asarray(hist) / np.sum(hist), 3)

        if proportion.sum() != 1:
            proportion[0] = 1 - proportion[1::].sum()

        classes_edges = np.vstack((classes_edges, np.hstack((proportion, np.array([0])))))

        return classes_edges

    # ------------------------------------------
    def verif_prop_on_value(self, value, NN):

        hist, bin_edges = np.histogram(value, density=False, bins=self.classes_edges)
        proportion = np.round(np.asarray(hist) / np.sum(hist), 3)

        if proportion.sum() != 1:
            proportion[0] = 1 - proportion[1::].sum()

        print('prop on Y {} is {}'.format(NN, proportion))
        print('prop on all {} is {}'.format(NN, self.prop))

    # ------------------------- Info Reg rsc
    # ------------------------------------------
    def verif_pdf_on_value(self, value, all_value, min, NN):
        print('zeros values on {}| on value [{:.1f}%] | on all [{:.1f}%]'.format(NN,
                                                                                 100 * np.size(
                                                                                     value[value == 0]) / np.size(
                                                                                     value),
                                                                                 100 * np.size(
                                                                                     all_value[
                                                                                         all_value == 0]) / np.size(
                                                                                     all_value)))

        Y_all_df, X_all_df = self.histo.my_histo(all_value, min, np.max(all_value), 'log',
                                                 'log', density=2, binwidth=None, nbbin=70)
        Y_df, X_df = self.histo.my_histo(value, min, np.max(value), 'log',
                                         'log', density=2, binwidth=None, nbbin=70)

        fig, ax = self.plot.belleFigure('$\Delta \delta f$', '$P(\Delta \delta f)$', nfigure=None)
        ax.plot(X_all_df, Y_all_df, 'r.')
        ax.plot(X_df, Y_df, 'b.')
        self.plot.plt.xscale('log')
        self.plot.plt.yscale('log')
        save = None
        self.plot.fioritures(ax, fig, title='value on all set vs split', label=None, grid=None, save=save)

    # ------------------------------------------
    def log_rsc(self, plot, labels, min):

        labels_nzeros = labels[labels != 0]
        plot.Pdf_loglog(labels_nzeros, np.min(labels_nzeros), np.max(labels_nzeros),
                        'labels_{nzero}', 'labels_nzero',
                        save=None,
                        nbbin=70)

        log_labels_nzeros = np.log(labels_nzeros)
        plot.Pdf_linlin(log_labels_nzeros, np.min(log_labels_nzeros), np.max(log_labels_nzeros),
                        'log labels_{nzero}', 'log_labels_nzeros', save=None, nbbin=150)

        if self.config_pred.label_rsc == 'cutsmall' and min is not None:
            label_cutsmall = labels[labels > min]
            log_label_cutsmall = np.log(label_cutsmall)
            plot.Pdf_linlin(log_label_cutsmall, np.min(log_label_cutsmall), np.max(log_label_cutsmall),
                            'log labels_{cutsmall}', 'log_labels_cutsmall', save=None, nbbin=150)

            print('relou values enlevées | on nzeros [{:.1f}%] | on cutsmall [{:.1f}%]'.format(
                100 * (np.size(labels) - np.size(labels_nzeros)) / np.size(labels),
                100 * (np.size(labels) - np.size(label_cutsmall)) / np.size(labels)))
            print('sur {} values'.format(np.size(labels)))

            mean_cutsmall = np.mean(log_label_cutsmall)
            dev_cutsmall = np.sqrt(np.var(log_label_cutsmall))
            print('stats on log labels cutsmall : mean = {} | dev = {}'.format(mean_cutsmall, dev_cutsmall))

            rsc_log_label_cutsmall = (log_label_cutsmall - mean_cutsmall) / dev_cutsmall
            plot.Pdf_linlin(rsc_log_label_cutsmall, np.min(rsc_log_label_cutsmall), np.max(rsc_log_label_cutsmall),
                            'rsc log labels_{cutsmall}', 'rsc_log_labels_cutsmall', save=None, nbbin=150)

            stats = np.array([mean_cutsmall, dev_cutsmall])
        elif min is not None:
            label_relocc = labels
            label_relocc[labels < min] = min
            plot.Pdf_loglog(label_relocc, np.min(label_relocc), np.max(label_relocc),
                            'labels', 'labels',
                            save=None, nbbin=150)

            log_label_relocc = np.log(label_relocc)
            plot.Pdf_linlin(log_label_relocc, np.min(log_label_relocc), np.max(log_label_relocc),
                            'log labels', 'log_labels',
                            save=None, nbbin=150)

            mean_relocc = np.mean(log_label_relocc)
            dev_relocc = np.sqrt(np.var(log_label_relocc))
            print('stats on log labels relocc : mean = {} | dev = {}'.format(mean_relocc, dev_relocc))

            rsc_log_label_relocc = (log_label_relocc - mean_relocc) / dev_relocc
            plot.Pdf_linlin(rsc_log_label_relocc, np.min(rsc_log_label_relocc), np.max(rsc_log_label_relocc),
                            'log labels rsc', 'log_labels_rsc',
                            save=None, nbbin=150)

            stats = np.array([mean_relocc, dev_relocc])
        else:
            stats = None

        return stats

    # ------------------------- Info stats
    # ------------------------------------------
    def __info_stats__(self, plot, min, reverse_classes=False):

        if self.config_pred.output_type == 'class':
            value = self.dict_value['train']
            if reverse_classes:
                fileName = self.config_pred.global_path_load + 'sequence_NN/' + '{}_reversed2_classes_edges_{}_futur_{}.npy'.format(
                    self.config_pred.output_shape, self.label_save,
                    self.futur)
            else:
                fileName = self.config_pred.global_path_load + 'sequence_NN/' + '{}_classes_edges_{}_futur_{}.npy'.format(
                    self.config_pred.output_shape, self.label_save,
                    self.futur)
            fileObj = Path(fileName)
            is_fileObj = fileObj.is_file()
            if not is_fileObj:
                if reverse_classes:
                    classes = self.classes_reversed(value)
                else:
                    classes = self.classes(value)
                logging.info("writting into {}".format(fileName))
                self.save_single(fileName, classes, '')
            else:
                logging.info("Load from {}".format(fileName))
                classes = np.load(fileName) #[0, :]
            hist, bin_edges = np.histogram(value, density=False, bins=classes)
            proportion = np.round(np.asarray(hist) / np.sum(hist), 3)

            if proportion.sum() != 1:
                proportion[0] = 1 - proportion[1::].sum()

            info_stats = np.vstack((classes, np.hstack((proportion, np.array([0])))))
        else:
            value = self.dict_value['train']
            if self.config_pred.label_save == 'expdf_logrsc':
                fileName = self.config_pred.global_path_load + 'sequence_NN/' + 'stats_{}_{}_futur_{}.npy'.format(
                    self.label_save, min, self.futur)
                fileObj = Path(fileName)
                is_fileObj = fileObj.is_file()
                if not is_fileObj:
                    info_stats = self.log_rsc(plot, value, min)
                    self.save_single(fileName, info_stats, '')

                else:
                    info_stats = np.load(fileName)
            else:
                info_stats = None

        self.info_stats = info_stats
        if self.config_pred.output_type == 'class':
            self.classes_edges = info_stats[0, :]
            self.prop = info_stats[1, :-1]
        else:
            self.rsc_stat = info_stats

        return info_stats

    # ------------------------- Features supp
    # ------------------------------------------
    def add_features(self, f, df, index, number, NN, syncro):

        fileName_df = self.config_pred.global_path_load + 'sequence_NN/' + 'df_feature_{}futur_{}_{}.npy'.format(
            self.futur, NN, syncro)
        fileName_value = self.config_pred.global_path_load + 'sequence_NN/' + 'value_feature_{}_{}futur_{}_{}.npy'.format(
            self.label_save, self.futur, NN, syncro)

        fileObj_df = Path(fileName_df)
        is_fileObj_df = fileObj_df.is_file()
        fileObj_value = Path(fileName_value)
        is_fileObj_value = fileObj_value.is_file()

        tic = time.time()

        if self.label_save == 'max_df':
            conv = None
        elif self.label_save == 'sum_df':
            conv = np.ones(self.futur)
        elif self.label_save == 'sum_expdf':
            conv = np.exp(-np.arange(0, self.futur) / self.config_pred.tau)
        elif self.label_save == 'sum_gammadf':
            gamma = sp.stats.gamma(self.config_pred.k, loc=0, scale=self.config_pred.theta)
            conv = gamma.pdf(np.arange(0, self.futur))
        else:
            conv = gaussian_distrib(np.arange(0, self.futur), self.config_pred.mu, self.config_pred.sigma)

        if not is_fileObj_df:
            new_df = np.zeros_like(f)
            new_value = np.ones_like(f) * -1

            for i in range(f.shape[0]):
                logging.info("cycle : {}".format(i))
                where = np.where(index[i, :] == 1)[0]
                new_df[i, where] = df[number[i, where].astype(int)]

                if self.label_save == 'max_df':
                    for j in range(1, f.shape[1]):
                        if j + self.futur <= f.shape[1]:
                            if self.config_pred.output_type == 'class':
                                value = np.max(new_df[i, j:j + self.futur])

                                new_value[i, j + self.futur - 1] = value
                else:
                    conv = conv[::-1]
                    values = np.convolve(new_df[i, 1::], conv, mode='valid')
                    logging.debug('conv shape = {}, values shape {}'.format(conv.shape, values.shape))
                    new_value[i, self.futur::] = values

            logging.info("Writting into: {}".format(fileName_df))
            self.save_single(fileName_df, new_df, '')
            self.save_single(fileName_value, new_value, '')

        elif not is_fileObj_value:
            logging.info("Load from: {}".format(fileName_df))
            new_df = np.load(fileName_df)
            new_value = np.ones_like(f) * -1

            for i in range(f.shape[0]):
                logging.info("cycle : {}".format(i))

                if self.label_save == 'max_df':
                    for j in range(1, f.shape[1]):
                        if j + self.futur <= f.shape[1]:
                            if self.config_pred.output_type == 'class':
                                value = np.max(new_df[i, j:j + self.futur])

                                new_value[i, j + self.futur - 1] = value
                else:
                    conv = conv[::-1]
                    values = np.convolve(new_df[i, 1::], conv, mode='valid')
                    logging.debug('conv shape = {}, values shape {}'.format(conv.shape, values.shape))
                    new_value[i, self.futur::] = values

            logging.info("Writting into: {}".format(fileName_value))
            self.save_single(fileName_value, new_value, '')

        else:
            logging.info("Load from: {} and from {}".format(fileName_df, fileName_value))
            new_df = np.load(fileName_df)
            new_value = np.load(fileName_value)

        logging.info("time  : {}".format(time.time() - tic))
        logging.info("inew df s Nan : {}".format(np.sum(np.isnan(new_df))))
        logging.info("inew value s Nan : {}".format(np.sum(np.isnan(new_value))))

        return new_df, new_value

    # ------------------------------------------
    def multifeature_f(self, f, df, index, number, NN):

        new_df, new_value = self.add_features(f, df, index, number, NN, syncro='not_syncro')

        if self.config_pred.channel == 2:
            new_f = np.concatenate((np.expand_dims(f, axis=0), np.expand_dims(new_df, axis=0)), axis=0)
        else:
            new_f = np.concatenate(
                (np.expand_dims(f, axis=0), np.expand_dims(new_df, axis=0), np.expand_dims(new_value, axis=0)), axis=0)

        logging.info("----- new_f shape : {}".format(np.shape(new_f)))
        return new_f

    # ------------------------- Target
    # ------------------------------------------
    def get_class_label(self, value):

        if value < self.classes_edges[0]:
            self.classes_edges[0] = value
        elif value > self.classes_edges[-1]:
            self.classes_edges[-1] = value + 1

        classe_value, _ = np.histogram(value, bins=self.classes_edges)
        classe_label = np.where(classe_value == 1)[0][0]

        return classe_label

    # ------------------------------------------
    def rsc_reg_label(self, value, min):
        mean = self.rsc_stat[0]
        dev = self.rsc_stat[1]
        if value < min:
            value = min
        value = np.log(value)
        label = (value - mean) / dev

        return label

    # ------------------------------------------
    def def_target(self, index, min=None):
        if self.config_pred.output_type == 'class':
            value = self.list_value[index]
            target = self.get_class_label(value)
        else:
            value = self.list_value[index]
            if self.config_pred.label_save == 'expdf_logrsc':
                target = self.rsc_reg_label(value, min)
            else:
                target = value

        return target, value

    # ------------------------- Sub Sample dans split
    # ------------------------------------------
    def __sub_sample_split__(self, min, max, nb_seq):
        dict_sequences = dictdata()
        dict_value = dictdata()
        for i in range(np.size(self.NN)):
            NN = self.NN[i]
            size = nb_seq[i]

            seq = self.dict_all_sequence[NN]
            val = self.dict_all_value[NN]

            if min is not None:
                to_keep = np.where(val > min)[0]
                seq = seq[to_keep]
                val = val[to_keep]

            if max is not None:
                to_keep = np.where(val < max)[0]
                seq = seq[to_keep]
                val = val[to_keep]

            if size is not None:
                if size < val.size:
                    to_keep = np.random.choice(val.size, size, replace=False)
                else:
                    to_keep = np.random.choice(val.size, val.size, replace=False)
                seq = seq[to_keep]
                val = val[to_keep]

            dict_sequences.add(NN, seq)
            dict_value.add(NN, val)

        self.dict_sequence = dict_sequences
        self.dict_value = dict_value

    # ------------------------------------------
    def __sub_sample_ratio__(self, classes, nb_seq, real_test):
        dict_sequences = dictdata()
        dict_value = dictdata()

        for i in range(np.size(self.NN)):
            NN = self.NN[i]
            size = nb_seq[i]
            logging.info("sub sample ratio in {}".format(NN))
            logging.info("classes are {}".format(classes))
            seq = self.dict_all_sequence[NN]
            val = self.dict_all_value[NN]

            hist, _ = np.histogram(val, bins=classes, density=False)
            nb_to_keep = np.min(hist)
            logging.info("nb_to_keep = {}".format(nb_to_keep))

            sub_seq = np.array([[0, 0]])
            sub_val = np.array([0])
            if size is not None:
                sub_size = int(size / (classes.size - 1))
            else:
                sub_size = val.size

            if NN != 'test' or not real_test:
                logging.info("sub_size = {}".format(sub_size))
                for j in range(classes.size - 1):
                    if j != classes.size - 2:
                        where = np.where((val >= classes[j]) & (val < classes[j + 1]))[0]
                    else:
                        where = np.where((val >= classes[j]) & (val <= classes[j + 1]))[0]
                    logging.debug(
                        "il y a {} elment dans classe {} - {}  {}".format(where.size, j, classes[j], classes[j + 1]))
                    bla = val[where]
                    bli = seq[where]
                    if sub_size < nb_to_keep:
                        logging.debug('reduce from {} to {}'.format(where.size, sub_size))
                        to_keep = np.random.choice(bla.size, sub_size, replace=False)
                        logging.debug(to_keep.size)
                    else:
                        logging.debug('reduce from {} to {}'.format(where.size, nb_to_keep))
                        to_keep = np.random.choice(bla.size, nb_to_keep, replace=False)
                    bli = bli[to_keep]
                    bla = bla[to_keep]

                    logging.debug('shape seq = {}, shape val = {}'.format(sub_seq.shape, sub_val.shape))
                    logging.debug('shape bli = {}, shape bla = {}'.format(bli.shape, bla.shape))
                    sub_seq = np.concatenate((sub_seq, bli))
                    sub_val = np.concatenate((sub_val, bla))

                val = sub_val[1:]
                seq = sub_seq[1:, :]

            dict_sequences.add(NN, seq)
            dict_value.add(NN, val)

            logging.info('----- {} taille finale = {}'.format(NN, val.size))
            nb_seq[i] = dict_value[NN].size

        self.dict_sequence = dict_sequences
        self.dict_value = dict_value

        return nb_seq

    # ------------------------------------------
    def __set_classes_edges__(self, classes):
        new_classe_edges = self.classes_edges[np.array([classes[0], classes[1], classes[1] + 1])]
        self.classes_edges = new_classe_edges

    # ------------------------------------------
    def __sub_sample_classes__(self, classes, nb_seq, ):
        dict_sequences = dictdata()
        dict_value = dictdata()

        for i in range(np.size(self.NN)):
            NN = self.NN[i]
            size = nb_seq[i]
            logging.info("sub sample ration in {}".format(NN))
            logging.info("classes to keep are {}".format(classes))
            seq = self.dict_all_sequence[NN]
            val = self.dict_all_value[NN]


            hist, _ = np.histogram(val, bins=self.classes_edges, density=False)
            nb_to_keep = np.min(hist)
            logging.info("nb_to_keep = {}".format(nb_to_keep))

            sub_seq = np.array([[0, 0]])
            sub_val = np.array([0])
            if size is not None:
                sub_size = int(size / (self.classes_edges.size - 1))
            else:
                sub_size = np.sum(hist[classes])


            logging.info("sub_size = {}".format(sub_size))
            for j in classes:
                if j !=4:
                    where = np.where((val >= self.classes_edges[j]) & (val < self.classes_edges[j + 1]))[0]
                else:
                    where = np.where((val >= self.classes_edges[j]) & (val <= self.classes_edges[j + 1]))[0]
                logging.debug(
                    "il y a {} elment dans classe {} - {}  {}".format(where.size, j, self.classes_edges[j], self.classes_edges[j + 1]))
                bla = val[where]
                bli = seq[where]
                if sub_size < nb_to_keep:
                    logging.debug('reduce from {} to {}'.format(where.size, sub_size))
                    to_keep = np.random.choice(bla.size, sub_size, replace=False)
                    logging.debug(to_keep.size)
                else:
                    logging.debug('reduce from {} to {}'.format(where.size, nb_to_keep))
                    to_keep = np.random.choice(bla.size, nb_to_keep, replace=False)
                if size is None:
                    logging.debug('reduce from {} to {}'.format(where.size, hist[j]))
                    to_keep = np.random.choice(bla.size, hist[j], replace=False)
                bli = bli[to_keep]
                bla = bla[to_keep]

                logging.debug('shape seq = {}, shape val = {}'.format(sub_seq.shape, sub_val.shape))
                logging.debug('shape bli = {}, shape bla = {}'.format(bli.shape, bla.shape))
                sub_seq = np.concatenate((sub_seq, bli))
                sub_val = np.concatenate((sub_val, bla))

            val = sub_val[1:]
            seq = sub_seq[1:, :]

            dict_sequences.add(NN, seq)
            dict_value.add(NN, val)

            logging.info('----- {} taille finale = {}'.format(NN, val.size))
            nb_seq[i] = dict_value[NN].size

        self.dict_sequence = dict_sequences
        self.dict_value = dict_value

        return nb_seq

    # ------------------------------------------
    def __cycles_NN_sequences__(self, signal_flu):

        fileName = self.config_pred.global_path_load + 'sequence_NN/' + 'dict_sequences_{}_cycle{}_{}seqsize_{}step_{}futur.npy'.format(
            self.config_pred.input_data, self.cycle_sample, self.seq_size, self.overlap_step,
            self.futur)
        fileObj = Path(fileName)
        is_fileObj = fileObj.is_file()

        if not is_fileObj:
            dict_sequences = dictdata()
            for i in range(np.size(self.NN)):
                NN = self.NN[i]

                indices = self.cycles_sample_indices(signal_flu[i].f, self.cycle_sample)
                dict_sequences.add(NN, indices)

            self.save_single(fileName, dict_sequences, '')

        else:
            dict_sequences = np.load(fileName, allow_pickle=True).flat[0]

        self.dict_sequence = dict_sequences

create_data = CreateDataScalar(config_data, config_pred, remote, plot, histo)
create_data.__NN_sequences__(signal_flu)
create_data.__all_value__(signalevent)

create_data.dict_sequence = create_data.dict_all_sequence.copy()
create_data.dict_value = create_data.dict_all_value.copy()

NN = 'train'
split = signalevent[0]
all_value = create_data.dict_all_value[NN]
print('zeros values on {} on all [{:.0f}%]'.format(NN, 100 * np.size(all_value[all_value == 0]) / np.size(
    all_value)))

NN = 'val'
split = signalevent[1]
all_value = create_data.dict_all_value[NN]
print('zeros values on {} on all [{:.0f}%]'.format(NN, 100 * np.size(all_value[all_value == 0]) / np.size(
    all_value)))

NN = 'test'
split = signalevent[2]
all_value = create_data.dict_all_value[NN]
print('zeros values on {} on all [{:.0f}%]'.format(NN, 100 * np.size(all_value[all_value == 0]) / np.size(
    all_value)))

# %% ################### Info stats on all value du train ##################################

min = None
create_data.__info_stats__(plot, min, reverse_classes=reverse)
print('class edges are {} of prop = {}'.format(create_data.classes_edges, create_data.prop))
print('on train : min = {} | max = {}'.format(np.min(create_data.dict_all_value['train']),
                                              np.max(create_data.dict_all_value['train'])))
print('on val : min = {} | max = {}'.format(np.min(create_data.dict_all_value['val']),
                                            np.max(create_data.dict_all_value['val'])))
print('on test : min = {} | max = {}'.format(np.min(create_data.dict_all_value['test']),
                                             np.max(create_data.dict_all_value['test'])))

print('nb sequences :', create_data.dict_all_value['train'].size)
print('nb sequences :', create_data.dict_all_value['val'].size)
print('nb sequences :', create_data.dict_all_value['test'].size)

# %% ################### Stats on delta f ##################################
major = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, np.max(create_data.dict_all_value['train'])]
decade = np.array([0, 5e-3, 3e-2, 3e-1, 3e0, np.max(create_data.dict_all_value['train'])])
decade_grid = np.array([5e-3, 3e-2, 3e-1, 3e0,  np.max(create_data.dict_all_value['train'])])
class_grid = np.concatenate((np.array([1e-4]), create_data.classes_edges[1::]))

nbclasses = config_pred.output_shape
log_classes = create_data.classes_edges

signal_flu = SignalForce(config_data, 'flu_rsc', 'train')
signalevent = ForceEvent(config_data, signal_flu.f, signal_flu.ext, signal_flu.t,
                         'flu_rsc', 'train', Sm=False)

y_value = create_data.dict_all_value['test']
# %% # ----------- Targets

path_Tex = config_pred.global_path_load + '/representation_apprentissage_classification/'
num_TeX = '{}_{}_{}'.format(num_target, config_pred.output_shape, num_tau)

title = True
label = True
grid = decade_grid
fig, ax = plot.belleFigure('$log10$', '${}$'.format('Pdf_{log10}'), nfigure=None)
y_Pdf, x_Pdf = plot.histo.my_histo(signalevent.df_tt, np.min(signalevent.df_tt),
                                   np.max(signalevent.df_tt),
                                   'log', 'log', density=2, binwidth=None, nbbin=70)

ax.plot(x_Pdf, y_Pdf, '.', label='$\delta f$')
y_Pdf, x_Pdf = plot.histo.my_histo(create_data.dict_all_value['test'],
                                   np.min(create_data.dict_all_value['test'][create_data.dict_all_value['test'] !=0]),
                                   np.max(create_data.dict_all_value['test']),
                                   'log', 'log', density=2, binwidth=None, nbbin=70)

ax.plot(x_Pdf, y_Pdf, '.', label='${}$'.format(create_data.label_name))
plot.plt.xscale('log')
plot.plt.yscale('log')
save = None
plot.fioritures(ax, fig, title=None, label=label, grid=grid, save=save, major=major)


yname = '\delta f'
ysave = 'df'
title = True
label = True
grid = decade_grid
fig, ax = plot.belleFigure('$log10({})$'.format(yname), '${}({})$'.format('Pdf_{log10}', yname), nfigure=None)
y_Pdf, x_Pdf = plot.histo.my_histo(signalevent.df_tt, 1e-4,
                                   np.max(signalevent.df_tt),
                                   'log', 'log', density=2, binwidth=None, nbbin=70)

ax.plot(x_Pdf, y_Pdf, '.')
plot.plt.xscale('log')
plot.plt.yscale('log')
save = path_Tex + '{}'.format('pdf_df_all')
plot.fioritures(ax, fig, title=None, label=None, grid=decade_grid, save=save, major=None)


title = True
label = True
grid = decade_grid
fig, ax = plot.belleFigure('$log10$', '${}$'.format('Pdf_{log10}'), nfigure=None)
y_Pdf, x_Pdf = plot.histo.my_histo(signalevent.df_tt, 1e-4,
                                   np.max(signalevent.df_tt),
                                   'log', 'log', density=2, binwidth=None, nbbin=70)

ax.plot(x_Pdf, y_Pdf, '.', label='$\delta f$')
y_Pdf, x_Pdf = plot.histo.my_histo(create_data.dict_all_value['test'], 1e-4,
                                   np.max(create_data.dict_all_value['test']),
                                   'log', 'log', density=2, binwidth=None, nbbin=70)

ax.plot(x_Pdf, y_Pdf, '.', label='${}$'.format(create_data.label_name))
plot.plt.xscale('log')
plot.plt.yscale('log')
save = path_Tex + '{}'.format('pdf_df_target_') + '{}'.format(num_target)
plot.fioritures(ax, fig, title=None, label=label, grid=grid, save=save, major=major)

yname = create_data.label_name
ysave = create_data.label_save
title = True
label = True
grid = decade_grid
fig, ax = plot.belleFigure('$log10({})$'.format(yname), '${}({})$'.format('Count_{log10}', yname), nfigure=None)
for i in range(nbclasses):
    where = np.where((y_value >= log_classes[i]) & (y_value <= log_classes[i + 1]))
    y_Pdf, x_Pdf = plot.histo.my_histo(y_value[where], log_classes[i] if log_classes[i] != 0 else 1e-4,
                                       np.max(y_value[where]),
                                       'log', 'log', density=1, binwidth=None, nbbin=70)

    ax.plot(x_Pdf, y_Pdf, '.', label='class {}'.format(i))
plot.plt.xscale('log')
plot.plt.yscale('log')
for i in range(nbclasses + 1):
    plot.plt.axvline(x=class_grid[i], color='k')
save = path_Tex + '{}'.format('pdf_target_') + num_TeX
plot.fioritures(ax, fig, title=None, label=label, grid=grid, save=save, major=major)

yname = '\delta f'
ysave = 'df'
title = True
label = True
grid = decade_grid
fig, ax = plot.belleFigure('$log10({})$'.format(yname), '${}({})$'.format('Count_{log10}', yname), nfigure=None)
for i in range(nbclasses):
    where = np.where((signalevent.df_tt >= log_classes[i]) & (signalevent.df_tt <= log_classes[i + 1]))
    y_Pdf, x_Pdf = plot.histo.my_histo(signalevent.df_tt[where], log_classes[i] if log_classes[i] != 0 else 1e-4,
                                       np.max(signalevent.df_tt[where]),
                                       'log', 'log', density=1, binwidth=None, nbbin=70)

    ax.plot(x_Pdf, y_Pdf, '.', label='class {}'.format(i))
plot.plt.xscale('log')
plot.plt.yscale('log')
for i in range(nbclasses + 1):
    plot.plt.axvline(x=class_grid[i], color='k')
save = path_Tex + '{}'.format('pdf_df_') + num_TeX
plot.fioritures(ax, fig, title=None, label=label, grid=grid, save=save, major=major)

# %% ################### sub sampled on classes ##################################
classes = create_data.info_stats[0, :]
nb_seq = [trainsize, int(20 / 100 * trainsize), None]
nb_seq = create_data.__sub_sample_ratio__(classes, nb_seq, real_test=True)

create_data.__info_stats__(plot, None, reverse_classes=reverse)
print('in {}_dataset classes_edges are : {}'.format('train', create_data.classes_edges))

hist, _ = np.histogram(signalevent.df_tt, bins=decade, density=False)
print('prop_on_decade of df_tt in all train = {}'.format(np.round(hist / signalevent.df_tt.size * 100, 3)))

df_tab = signalevent.df_tab()

hist, _ = np.histogram(df_tab.reshape(df_tab.size), bins=decade, density=False)
print('prop_on_decade of df_tab in all train = {}'.format(np.round(hist / df_tab.size * 100, 2)))

signal_flu = SignalForce(config_data, 'flu_rsc', 'val')
signalevent = ForceEvent(config_data, signal_flu.f, signal_flu.ext, signal_flu.t,
                         'flu_rsc', 'val', Sm=False)
hist, _ = np.histogram(signalevent.df_tt, bins=decade, density=False)
print('prop_on_decade of df_tt in val = {}'.format(np.round(hist / signalevent.df_tt.size * 100, 1)))

signal_flu = SignalForce(config_data, 'flu_rsc', 'test')
signalevent = ForceEvent(config_data, signal_flu.f, signal_flu.ext, signal_flu.t,
                         'flu_rsc', 'test', Sm=False)
hist, _ = np.histogram(signalevent.df_tt, bins=decade, density=False)
print('prop_on_decade of df_tt in test = {}'.format(np.round(hist / signalevent.df_tt.size * 100, 1)))

df_tab = signalevent.df_tab()

# hist, _ = np.histogram(df_tab, bins=create_data.classes_edges, density=False)
# print('prop_on_classes of df_tab in test = {}'.format(np.round(hist / df_tab.size * 100, 2)))

print('proportion in train for learning : {}'.format(create_data.prop*100))

hist, _ = np.histogram(create_data.dict_all_value['train'], bins=create_data.classes_edges, density=False)
print('prop_on_classe value in all train = {}'.format(np.round(hist / create_data.dict_all_value['train'].size * 100, 3)))

hist, _ = np.histogram(create_data.dict_all_value['train'], bins=decade, density=False)
print('prop_on_decade value in all train = {}'.format(np.round(hist / create_data.dict_all_value['train'].size * 100, 3)))

count_on_train, _ = np.histogram(create_data.dict_value['train'], bins=create_data.classes_edges, density=False)
print('count on learning train  = {} seq'.format(np.sum(count_on_train)))

hist, _ = np.histogram(create_data.dict_all_value['val'], bins=create_data.classes_edges, density=False)
print('prop_on_classe value in val = {}'.format(np.round(hist / create_data.dict_all_value['val'].size * 100, 1).astype(int)))

hist, _ = np.histogram(create_data.dict_all_value['val'], bins=decade, density=False)
print('prop_on_decade value in val = {}'.format(np.round(hist / create_data.dict_all_value['val'].size * 100, 1)))

hist, _ = np.histogram(create_data.dict_all_value['test'], bins=create_data.classes_edges, density=False)
print('prop_on_classe value in test = {}'.format(np.round(hist / create_data.dict_all_value['test'].size * 100, 1)))

hist, _ = np.histogram(create_data.dict_all_value['test'], bins=decade, density=False)
print('prop_on_decade value in test = {}'.format(np.round(hist / create_data.dict_all_value['test'].size * 100, 1)))

import numpy as np
import torch
from torch.utils.data import Dataset
from Datas.classData import find_nbseq, CreateDataScalar, CreateDataField, def_names
import logging

class DataSetTIMEFIELD(Dataset, CreateDataField):

    def __init__(self, config_data, config_pred, remote, plot, histo, nb_seq, signal_img, signalevent, num_split,
                 sw, sc, cycle_sample=None, multi_features=False, min=None, max=None, reverse_classes=False,
                 which_to_keep=None, pred_img='last'):

        CreateDataField.__init__(self, config_data, config_pred, remote, plot, histo, sw, sc, cycle_sample)

        self.split = self.NN[num_split]
        self.nb_seq = nb_seq
        self.which_img = -1 if pred_img == 'last' else pred_img
        seq_signaltype, self.seq_fname, seq_savename = def_names('sequence', None, self.split)

        print('------ all sequences on {} ------'.format(self.split))
        self.__NN_sequences__(signal_img)
        self.__all_value__()

        self.dict_sequence = self.dict_all_sequence.copy()
        self.dict_value = self.dict_all_value.copy()

        if cycle_sample:
            self.__cycles_NN_sequences__(signal_img)
            self.__cycle_value__()

        self.min = min
        self.max = max

        if config_pred.output_type == 'img_vort' or config_pred.output_type == 'digit_img_vort':
            print('------ target is vort img ------')
            print('------ sub sample slpit------')
            self.__sub_sample_split__(self.min, self.max, nb_seq)

        self.list_indices = self.dict_sequence[self.split]
        self.list_value = self.dict_value[self.split]
        self.model_spec = config_pred.spec_model_scalar()

        self.mean_stitches = 7.6e-6
        self.tanh_std = config_pred.tanh_std

    def __len__(self):
        return self.list_indices.shape[0]

    def __getitem__(self, index):

        all_seq = self.recup_sequence(self.seq_fname, index)
        cycle = self.list_indices[index, 0]
        t0 = self.list_indices[index, 1]

        timeseq = all_seq[:-1, :, :, :]

        label, value = self.def_target(index, all_seq, self.which_img, self.min)

        timeseq = self.transform_seq(timeseq)
        label = self.transform_label(label)

        return timeseq, label

    # ------------------------------------------
    def transform_seq(self, seq):
        seq = torch.from_numpy(seq)
        seq = seq.permute(0, 3, 1, 2)
        seq = np.tanh(((seq - self.mean_stitches) / self.tanh_std))
        # seq = seq.unsqueeze(0)

        return seq.float()

    # ------------------------------------------
    def transform_label(self, label):
        label = torch.from_numpy(label)
        label = np.tanh(((label - self.mean_stitches) / self.tanh_std))

        return label.float() if self.config_pred.output_type == 'img_vort' else label.long()


class DataSetTIMESCALAR(Dataset, CreateDataScalar):

    def __init__(self, config_data, config_pred, remote, plot, histo, nb_seq, signal_flu, signalevent, num_split,
                 cycle_sample=None, multi_features=False, min=None, max=None, reverse_classes=False, which_to_keep=None,
                 real_test=True, mask_size=None):

        CreateDataScalar.__init__(self, config_data, config_pred, remote, plot, histo, cycle_sample)

        self.split = self.NN[num_split]
        self.nb_seq = nb_seq
        self.mask_size = mask_size

        print('------ all sequences on {} ------'.format(self.split))
        self.__NN_sequences__(signal_flu)
        self.__all_value__(signalevent)

        self.dict_sequence = self.dict_all_sequence.copy()
        self.dict_value = self.dict_all_value.copy()

        if cycle_sample:
            self.__cycles_NN_sequences__(signal_flu)
            self.__cycle_value__(signalevent)

        print('------ info stats on {} ------'.format(self.split))
        if config_pred.output_type == 'class':
            self.min = min
            self.max = max
        else:
            if self.config_pred.label_rsc == 'cutsmall' or self.config_pred.label_rsc == 'relocc':
                self.min = 5e-3
            else:
                self.min = min
            # self.__sub_sample_split__(self.min, None, [None, None, None])
        self.__info_stats__(plot, self.min, reverse_classes)

        if config_pred.output_type == 'class':
            if which_to_keep is not None:
                print('------ sub sample two classes ------')
                self.__sub_sample_classes__(which_to_keep, nb_seq)
                self.__info_stats__(plot, self.min, reverse_classes)
                print('in {}_dataset classes_edges are : {}'.format(self.split, self.classes_edges))
                print('of proportion : {}'.format(self.prop, 1 / self.prop))
                self.config_pred.set_weight_classes(1 / self.prop)
            elif config_pred.equipro:
                print('------ sub sample ratio------')
                self.__sub_sample_ratio__(self.classes_edges, nb_seq, real_test=real_test)
                self.__info_stats__(plot, self.min, reverse_classes)
                print('in {}_dataset classes_edges are : {}'.format(self.split, self.classes_edges))
                print('of proportion : {}'.format(self.prop, 1 / self.prop))
                self.config_pred.set_weight_classes(1 / self.prop)
            else:
                print('------ sub sample slpit------')
                self.__sub_sample_split__(self.min, self.max, nb_seq)
                self.config_pred.set_weight_classes(1 / self.prop)
        else:
            if self.config_pred.label_rsc == 'cutsmall':
                print('------ sub sample ------')
                self.__sub_sample_split__(self.min, None, nb_seq)

                print('zeros values | on {} [{:.0f}%] | on all [{:.0f}%]'.format(self.split,
                                                                                 100 * np.size(
                                                                                     self.dict_value[self.split][
                                                                                         self.dict_value[
                                                                                             self.split] == 0]) / np.size(
                                                                                     self.dict_value[self.split]),
                                                                                 100 * np.size(
                                                                                     self.dict_all_value[self.split][
                                                                                         self.dict_all_value[
                                                                                             self.split] == 0]) / np.size(
                                                                                     self.dict_all_value[self.split])))
            elif self.config_pred.label_rsc == 'relocc':
                print('------ sub sample ------')
                self.__sub_sample_split__(None, None, nb_seq)

                print('zeros values | on {} [{:.0f}%] that will be changed to {}'
                      '| on all [{:.0f}%]'.format(self.split,
                                                  100 * np.size(
                                                      self.dict_value[self.split][
                                                          self.dict_value[
                                                              self.split] == 0]) / np.size(
                                                      self.dict_value[self.split]),
                                                  self.min,
                                                  100 * np.size(
                                                      self.dict_all_value[self.split][
                                                          self.dict_all_value[
                                                              self.split] == 0]) / np.size(
                                                      self.dict_all_value[self.split])))
            else:
                self.min = None

        self.multi_features = multi_features
        if not multi_features:
            self.f = signal_flu[num_split].f
        else:
            print('------ add features ------')
            self.f = self.multifeature_f(signal_flu[num_split].f, signalevent[num_split].df_tt,
                                         signalevent[num_split].index_df_tt, signalevent[num_split].number_df_tt,
                                         self.split)

        if config_pred.output_type == 'class':
            if which_to_keep is not None:
                self.__set_classes_edges__(which_to_keep)
        self.list_indices = self.dict_sequence[self.split]
        self.list_value = self.dict_value[self.split]
        self.model_spec = config_pred.spec_model_scalar()

    def __len__(self):
        return self.list_indices.shape[0]

    def __getitem__(self, index):

        cycle = self.list_indices[index, 0]
        t0 = self.list_indices[index, 1]

        if not self.multi_features:
            timeseq = self.f[cycle, t0:t0 + self.seq_size]
        else:
            timeseq = self.f[:, cycle, t0:t0 + self.seq_size]
            # timeseq[2, -self.futur::] = -1 ## if syncr else no

        label, value = self.def_target(index, self.min)

        timeseq = self.transform_seq(timeseq)
        label = self.transform_label(label)

        if self.mask_size is not None:
            mask = torch.ones_like(timeseq) * False
            mask[:, -self.mask_size::] = True
            timeseq[mask.bool()] = 0
            # print(mask, timeseq)

        return timeseq, label, value

    # ------------------------------------------
    def transform_seq(self, seq):
        seq = torch.from_numpy(seq)
        if self.model_spec['model_name'] == 'transformer':
            seq = seq.unsqueeze(-1)
        else:
            if not self.multi_features:
                seq = seq.unsqueeze(0)

        return seq.float()

    # ------------------------------------------
    def transform_label(self, label):
        if self.config_pred.output_type == 'reg':
            label = np.float32(label)

        return label


class DataSetFalseData(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return np.shape(self.X)[0]

    def __getitem__(self, index):
        seq = torch.from_numpy(self.X[index, :]).float().unsqueeze(-1)
        target = self.Y[index]

        return seq, target

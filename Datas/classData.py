import numpy as np
import timeit
from pathlib import Path
import pandas as pd
from skimage import measure
from dictdata import dictdata
from Datas.classSignal import def_names
from Datas.classSignal import Shape
from Datas.classStat import Stat, Histo
from Datas.classCell import Cell
import logging
import time
import scipy as sp

def find_nbseq(config_pred, signal_flu, seq_size, futur, overlap_step, trainsize=None):
    shape = Shape(signal_flu.f)

    if config_pred.label_save == 'int_fexp':
        tps_to_keep = np.arange(0, shape.tps - (seq_size + 5 * futur) + 1, overlap_step)
    else:
        tps_to_keep = np.arange(0, shape.tps - (seq_size + futur) + 1, overlap_step)

    nb_seq_possible = shape.cycles * tps_to_keep.size

    if trainsize is not None or trainsize != 20000000:
        if nb_seq_possible > 20 / 100 * trainsize:
            nb_seq_possible = int(20 / 100 * trainsize)

    return nb_seq_possible

def gaussian_distrib(x, mu, sigma):
    return np.exp(-(x - mu) ** 2 / (2 * sigma**2))


# ---------------------------------------------------------------------------------------------------------------------#
class InfoField():
    """
    Classe qui permet de charger signal des event en force et ses dépendances.

    Attributes:
        config (class) : config associée à la l'analyse

        nb_area (int) : nombre de region dans l'img
        num_area (1D array) : numéro des regions
        size_area (1D array) : taille en nombre de mailles de chaque region
        sum_field (1D array) : sum des valeurs du champs sur les mailles d'une region, pour toutes les regions
        size_area_img (int) :  taille en nombre de mailles compté sur toutes les regions
        size_field_img (int) : sum des valeurs de champs sur toutes les regions
        conncomp (array) : Labeled array, where all connected regions are assigned the same integer value.

    """

    # ---------------------------------------------------------#
    def __init__(self, field, normfield, field_seuil, seuil, fname, signe, fault_analyse=True,
                 debug=False):
        """
        The constructor for InfoField.

        Parameters:
            config (class) : config associée à la l'analyse

            field (array) : une img de champ
            field_seuil (array) : labeled array, where all regions supp to seuil are assigned 1.
            seuil (int) : valeur utilisée pour seuiller les évents
            fault_analyse (bol) : est ce que l'analyse est pour étudier les fault
            debug (bol) : permet d'afficher plot des img des region pour debuguer

        """

        self.fname = fname
        self.signe = signe

        self.field = field
        self.normfield = normfield
        self.field_seuil = field_seuil
        self.seuil = seuil

        self.nb_area, self.num_area, self.size_area, self.center, self.orientation, self.sum_field, self.sum_normfield, \
        self.size_area_img, self.sum_field_img, self.sum_normfield_img, self.conncomp = self.info(fault_analyse,
                                                                                                  debug)

    # ------------------------------------------
    def info(self, fault_analyse, debug):

        info = dictdata()
        nb_area, num_area, size_area, center_area, orientation_area, sum_field, sum_normfield, \
        size_area_img, sum_field_img, sum_normfield_img, conncomp = self.analyse_info(
            fault_analyse,
            debug)
        info.add('nb_area', nb_area)
        info.add('num_area', num_area)
        info.add('size_area', size_area)
        info.add('center_area', center_area)
        info.add('orientation_area', orientation_area)
        info.add('sum_field', sum_field)
        info.add('sum_normfield', sum_normfield)
        info.add('size_area_img', size_area_img)
        info.add('sum_field_img', sum_field_img)
        info.add('sum_normfield_img', sum_normfield_img)
        info.add('conncomp', conncomp)

        return nb_area, num_area, size_area, center_area, orientation_area, sum_field, sum_normfield, \
               size_area_img, sum_field_img, sum_normfield_img, conncomp

    # ------------------------------------------
    def analyse_info(self, fault_analyse, debug):

        conncomp, Nobj = measure.label(self.field_seuil, return_num=True)

        Reg = measure.regionprops(conncomp)

        num_area = np.arange(1, Nobj + 1)
        Area = np.zeros(Nobj)
        Center = np.zeros((Nobj, 2))
        Orient = np.zeros(Nobj)

        for i in range(Nobj):
            Area[i] = Reg[i].area
            if Area[i] <= 1:
                pixels = np.nonzero(conncomp == num_area[i])
                conncomp[pixels] = 0
            else:
                Center[i, :] = Reg[i].centroid
                Orient[i] = Reg[i].orientation

        num_area = num_area[np.where(Area > 1)[0]]
        center = Center[np.where(Area > 1)[0], :]
        orientation = Orient[np.where(Area > 1)[0]]
        Area = Area[np.where(Area > 1)[0]]

        Nobj = np.size(Area)

        sum_field = np.zeros(Nobj)
        sum_normfield = np.zeros(Nobj)

        for i in range(Nobj):
            # print('vs field shape = {}'.format(np.shape(self.field)))
            # print('vs field_seuil shape = {}'.format(np.shape(self.field_seuil)))
            # print('conncomp shape = {}'.format(np.shape(conncomp)))
            to_sum_field = np.zeros_like(self.field_seuil)
            to_sum_normfield = np.zeros_like(self.field_seuil)

            pixels = np.nonzero(conncomp == num_area[i])

            to_sum_field[pixels] = self.field[pixels].copy()
            to_sum_normfield[pixels] = self.normfield[pixels].copy()

            sum_field[i] = np.sum(to_sum_field)
            sum_normfield[i] = np.sum(to_sum_normfield)
            if sum_field[i] < self.seuil:
                print('Proooooooooooooooobleeeeeeeeeeeeeme')
                print('sum = {} vs field {} '.format(sum_normfield[i], self.seuil))

        if not fault_analyse:
            conncomp = None

        return Nobj, num_area, Area, center, orientation, sum_field, sum_normfield, np.sum(Area), np.sum(
            sum_field), np.sum(sum_normfield), conncomp


class CreateDataField:
    def __init__(self, config_data, config_pred, remote, plot, histo, sw, sc, cycle_sample=None):
        self.config_data = config_data
        self.config_pred = config_pred
        self.remote = remote
        self.plot = plot
        self.histo = histo

        ##jusqu'à nouvelle ordre
        self.seuil = 3e-3

        self.NN = config_pred.NN_data
        self.label_name = config_pred.label_name
        self.label_save = config_pred.label_save

        label_type = pd.Series(self.label_save)
        if label_type.str.contains('sub').all():
            self.sub = True
        else:
            self.sub = False

        self.seq_size = config_pred.seq_size
        self.overlap_step = config_pred.overlap_step

        self.output_shape = config_pred.output_shape
        self.futur = config_pred.futur
        self.channel = config_pred.channel
        self.cycle_sample = cycle_sample

        self.sw = int(sw)
        self.sc = int(sc)

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

    # ------------------------------------------
    def stack_channels(self, fields):

        sequence = np.zeros((self.seq_size + self.futur, self.sw, self.sc, self.channel))
        logging.debug("sequence shape {} ".format(sequence.shape))

        for i in range(self.seq_size + self.futur):
            input = np.zeros((self.sw, self.sc, self.channel))

            for j in range(np.size(self.config_pred.fields)):
                logging.debug("fields {} ".format(self.config_pred.fields[j]))
                field = fields[j]

                J = 1  ### car red sinon 2

                if self.config_pred.fields[j] == 'vit_x':
                    logging.debug(
                        "field {} of shape {} ".format(self.config_pred.fields[j], field[1: -1, J: -J, i].shape))
                    input[:, :, 0] = field[J: -J, J: -J, i]

                elif self.config_pred.fields[j] == 'vit_x_sub':
                    logging.debug(
                        "field {} of shape {} ".format(self.config_pred.fields[j], field[1: -1, J: -J, i].shape))
                    if (field.shape[0] % 2 == 0) and (field.shape[1] % 2 == 0):
                        input[:, :, 0] = field[1::, 1::, i]
                    elif (field.shape[0] % 2 == 0):
                        input[:, :, 0] = field[1::, 1:-1, i]
                    elif (field.shape[1] % 2 == 0):
                        input[:, :, 0] = field[1:-1, 1::, i]

                elif self.config_pred.fields[j] == 'vit_y':
                    logging.debug(
                        "field {} of shape {} ".format(self.config_pred.fields[j], field[1: -1, J: -J, i].shape))
                    input[:, :, 1] = field[J: -J, J: -J, i]

                elif self.config_pred.fields[j] == 'vit_y_sub':
                    logging.debug(
                        "field {} of shape {} ".format(self.config_pred.fields[j], field[1: -1, J: -J, i].shape))

                    if (field.shape[0] % 2 == 0) and (field.shape[1] % 2 == 0):
                        input[:, :, 1] = field[1::, 1::, i]
                    elif (field.shape[0] % 2 == 0):
                        input[:, :, 1] = field[1::, 1:-1, i]
                    elif (field.shape[1] % 2 == 0):
                        input[:, :, 1] = field[1:-1, 1::, i]

                else:
                    logging.info("field {} of shape {} ".format(self.config_pred.fields[j], field[:, :, i].shape))
                    input[:, :, 2] = field[:, :, i]

            sequence[i, :, :, :] = input

        return sequence

    # ------------------------------------------
    def cycles_sample_indices(self, number, signal_img, num_set, cycle):
        logging.debug("number shape is  {}".format(number.shape))
        tps_to_keep = np.arange(0, number.size - (self.seq_size + self.futur) + 1, self.overlap_step)
        set = np.ones_like(tps_to_keep) * num_set
        set_cycle = np.ones_like(tps_to_keep) * signal_img.NN_sub_cycles[num_set][cycle]
        cycle_to_keep = np.ones_like(tps_to_keep) * cycle

        logging.debug("indice to keep shape = {}".format(np.shape(tps_to_keep)))
        indices = np.vstack((set, set_cycle, cycle_to_keep, number[tps_to_keep])).transpose()

        return indices

    # ------------------------------------------
    def create_seq_img(self, indices_cycle_k, fields, seq_fname, num_seq):

        num_and_indice = None
        num_array = None
        for i in range(indices_cycle_k.shape[0]):
            t0 = indices_cycle_k[i, 3]

            if not self.sub:
                fileName = self.config_pred.global_path_load + 'pict_event_sequence_NN/' + '{}seqsize_{}step_{}futur/{}_{}.npy'.format(
                    self.seq_size, self.overlap_step, self.futur, seq_fname, num_seq)
            else:
                fileName = self.config_pred.global_path_load + 'pict_event_sequence_NN/' + 'sub_{}seqsize_{}step_{}futur/{}_{}.npy'.format(
                    self.seq_size, self.overlap_step, self.futur, seq_fname, num_seq)
            fileObj = Path(fileName)
            is_fileObj = fileObj.is_file()

            logging.info("file {} is {}".format(fileName, is_fileObj))

            if not is_fileObj:
                sub_field = [fields[m][:, :, t0: t0 + self.seq_size + self.futur] for m in
                             range(np.size(self.config_pred.fields))]

                logging.info("field are {}".format(self.config_pred.fields))

                sequence = self.stack_channels(sub_field)

                logging.debug("writting into {}".format(fileName))
                self.save_single(fileName, sequence, '')

            else:
                logging.debug("done : {}".format(fileName))

            if num_and_indice is None:
                logging.debug(
                    "num shape {} | indice shape = {} ".format(np.array([num_seq]).shape, indices_cycle_k.shape))
                num_and_indice = np.hstack((np.array([num_seq]), indices_cycle_k[i, :]))
            else:
                num_and_indice = np.vstack(
                    (num_and_indice, np.hstack((np.array([num_seq]), indices_cycle_k[i, :]))))
            num_seq = num_seq + 1

        return num_and_indice, num_seq

    # ------------------------------------------
    def all_seq_img(self, signal_img, NN):

        logging.info("in {}".format(NN))
        seq_signaltype, seq_fname, seq_savename = def_names('sequence', None, NN)

        num_seq = 0
        num_and_indice = None

        for j in range(self.config_data.nb_set):
            logging.info("set nb {}".format(j))
            index = signal_img.index_picture[signal_img.NN_sub_cycles[j]]
            number = signal_img.number_picture[signal_img.NN_sub_cycles[j]]

            fields = [signal_img.import_field(name_field, num_set=j) for name_field in
                      self.config_pred.fields]

            logging.info("field are {}".format(self.config_pred.fields))

            for k in range(np.size(signal_img.NN_sub_cycles[j])):
                logging.debug("cycle nb {}".format(k))
                sub_number = number[k, index[k, :] == 1].astype(int)

                indices_cycle_k = self.cycles_sample_indices(sub_number, signal_img, num_set=j, cycle=k)
                to_add_dict, num_seq = self.create_seq_img(indices_cycle_k, fields, seq_fname, num_seq)

                if num_and_indice is None:
                    num_and_indice = to_add_dict
                else:
                    num_and_indice = np.vstack((num_and_indice, to_add_dict))

        return num_and_indice

    # ------------------------------------------
    def recup_sequence(self, seq_fname, num_seq):

        if not self.sub:
            fileName = self.config_pred.global_path_load + 'pict_event_sequence_NN/' + '{}seqsize_{}step_{}futur/{}_{}.npy'.format(
                self.seq_size, self.overlap_step, self.futur, seq_fname, num_seq)
        else:
            fileName = self.config_pred.global_path_load + 'pict_event_sequence_NN/' + 'sub_{}seqsize_{}step_{}futur/{}_{}.npy'.format(
                self.seq_size, self.overlap_step, self.futur, seq_fname, num_seq)

        seq = np.load(fileName)

        return seq

    # ------------------------------------------
    def __NN_sequences__(self, signal_img):

        if not self.sub:
            fileName = self.config_pred.global_path_load + 'pict_event_sequence_NN/' + 'dict_sequences_{}_{}seqsize_{}step_{}futur.npy'.format(
                self.config_pred.input_data, self.seq_size, self.overlap_step,
                self.futur)
        else:
            fileName = self.config_pred.global_path_load + 'pict_event_sequence_NN/' + 'dict_sequences_{}_sub_{}seqsize_{}step_{}futur.npy'.format(
                self.config_pred.input_data, self.seq_size, self.overlap_step,
                self.futur)
        fileObj = Path(fileName)
        is_fileObj = fileObj.is_file()

        if not is_fileObj:
            dict_sequences = dictdata()
            for i in range(np.size(self.NN)):
                NN = self.NN[i]

                list_num_seq = self.all_seq_img(signal_img[i], NN)

                dict_sequences.add(NN, list_num_seq)

            logging.info("writting into {}".format(fileName))
            self.save_single(fileName, dict_sequences, '')

        else:
            logging.info("load from {}".format(fileName))
            dict_sequences = np.load(fileName, allow_pickle=True).flat[0]

        self.dict_all_sequence = dict_sequences
        print('bouhuuuh')

    # ------------------------- Values
    # ------------------------------------------
    def recup_target_vort(self, seq_fname, num_seq):

        if not self.sub:
            fileName = self.config_pred.global_path_load + 'pict_event_sequence_NN/' + '{}seqsize_{}step_{}futur/{}_{}.npy'.format(
                self.seq_size, self.overlap_step, self.futur, seq_fname, num_seq)
        else:
            fileName = self.config_pred.global_path_load + 'pict_event_sequence_NN/' + 'sub_{}seqsize_{}step_{}futur/{}_{}.npy'.format(
                self.seq_size, self.overlap_step, self.futur, seq_fname, num_seq)
        seq = np.load(fileName)

        return seq[-1, :, :, 2]

    # ------------------------------------------
    def recup_target_img(self, seq_fname, num_seq):

        if not self.sub:
            fileName = self.config_pred.global_path_load + 'pict_event_sequence_NN/' + '{}seqsize_{}step_{}futur/{}_{}.npy'.format(
                self.seq_size, self.overlap_step, self.futur, seq_fname, num_seq)
        else:
            fileName = self.config_pred.global_path_load + 'pict_event_sequence_NN/' + 'sub_{}seqsize_{}step_{}futur/{}_{}.npy'.format(
                self.seq_size, self.overlap_step, self.futur, seq_fname, num_seq)

        seq = np.load(fileName)

        return seq[-1, :, :, :]

    # ------------------------------------------
    def def_value_for_classes(self, info_p, info_n, how):

        if how == 'max':
            max_p = np.max(info_p) if info_p.size != 0 else 0
            max_n = np.max(info_n) if info_n.size != 0 else 0
            value = np.max(np.array([max_p, max_n]))
        elif how == 'mean':
            value_p = info_p if info_p.size != 0 else np.array([0])
            value_n = info_n if info_n.size != 0 else np.array([0])
            value = np.sum(np.concatenate((value_p, value_n)))/(value_n.size+value_n.size)
        else:
            value_p = info_p if info_p.size != 0 else np.array([0])
            value_n = info_n if info_n.size != 0 else np.array([0])
            value = np.sum(np.concatenate((value_p, value_n)))

        return value

    # ------------------------------------------
    def find_field_seuil(self, field, seuil, signe):

        field_seuil = np.zeros_like(field)
        if signe == '_n':
            field = -field

        where = np.where(field > seuil)
        field_seuil[where[0], where[1]] = 1

        return field_seuil

    # ------------------------------------------
    def __all_sub_field_seuil_value__(self, sub_field, seuil, signe, NN):

        logging.info("in {}".format(NN))

        fileName = self.config_pred.global_path_load + 'pict_event_sequence_NN/' + 'all_sub_field_seuil_value_seuil_{}_signe{}_seqsize_{}_futur_{}_{}.npy'.format(
            seuil, signe, self.seq_size, self.futur, NN)

        fileObj = Path(fileName)
        is_fileObj = fileObj.is_file()
        if not is_fileObj:
            nb_value = self.dict_all_sequence[NN].shape[0]
            sub_field_seuil = np.zeros((nb_value, self.sw, self.sc))
            for j in range(nb_value):
                sub_field_seuil[j, :, :] = self.find_field_seuil(sub_field[j], seuil, signe)
                if j % 10 / 100 * nb_value == 0:
                    logging.info("calcul {}%".format(j / nb_value * 100))
            logging.info("save into {}".format(fileName))
            self.save_single(fileName, sub_field_seuil, '')

        else:
            logging.info("load from {}".format(fileName))
            sub_field_seuil = np.load(fileName)

        return sub_field_seuil

    # ------------------------------------------
    def __all_sub_info_value__(self, dict_sub_field, fname, seuil, signe):

        dict_sub_info = dictdata()
        for i in range(np.size(self.NN)):
            NN = self.NN[i]
            logging.info("in {}".format(NN))

            fileName = self.config_pred.global_path_load + 'pict_event_sequence_NN/' + 'all_sub_info_value_seuil_{}_signe{}_seqsize_{}_futur_{}_{}.npy'.format(
                seuil, signe, self.seq_size, self.futur, NN)

            fileObj = Path(fileName)
            is_fileObj = fileObj.is_file()

            if not is_fileObj:
                sub_field = dict_sub_field[NN]
                sub_field_seuil = self.__all_sub_field_seuil_value__(sub_field, self.seuil, signe=signe, NN=NN)
                nb_value = self.dict_all_sequence[NN].shape[0]
                value = dictdata()
                for j in range(nb_value):
                    v = sub_field[j, :, :] if signe == '_p' else -sub_field[j, :, :]
                    nv = np.abs(sub_field[j, :, :])
                    vs = sub_field_seuil[j, :, :]
                    info = InfoField(v, nv, vs, seuil, fname, signe, fault_analyse=True)
                    value.add('info_{}'.format(j), info.sum_field)
                    if j % (10 / 100 * nb_value) == 0:
                        logging.info("calcul {}%".format(j / nb_value * 100))
                logging.info("save into {}".format(fileName))
                self.save_single(fileName, value, '')

                dict_sub_info.add(NN, value)
            else:
                logging.info("load from {}".format(fileName))
                value = np.load(fileName, allow_pickle=True).flat[0]
                dict_sub_info.add(NN, value)

        return dict_sub_info

    # ------------------------------------------
    def __all_value__(self):
        '''dans value on ne tiens pas compte du fait qu'on peut vouloir entrainer à predir
        des images appartenant à l' input
        mais on a pas besoin de alla value pour le learning à proprement parle, c'est plus
        necessaire pour faire des stats et surtout dans le cas ou on ne predit pas une img !!! '''

        fileName = self.config_pred.global_path_load + 'pict_event_sequence_NN/' + 'dict_all_value_for_target_{}_seqsize_{}_futur_{}.npy'.format(
            self.label_save, self.seq_size, self.futur)

        fileObj = Path(fileName)
        is_fileObj = fileObj.is_file()
        if not is_fileObj:
            dict_value = dictdata()
            label_type = pd.Series(self.label_save)

            if label_type.str.contains('S_f').all():
                logging.info("load from {}".format(
                    self.config_pred.global_path_load + 'pict_event_sequence_NN/' + 'dict_all_value_for_target_{}_seqsize_{}_futur_{}.npy'.format(
                        'vort', self.seq_size, self.futur)))
                dict_sub_field = np.load(
                    self.config_pred.global_path_load + 'pict_event_sequence_NN/' + 'dict_all_value_for_target_{}_seqsize_{}_futur_{}.npy'.format(
                        'vort', self.seq_size, self.futur), allow_pickle=True).flat[0]
                dict_value = dictdata()
                dict_sub_info_p = self.__all_sub_info_value__(dict_sub_field, 'vort',
                                                              self.seuil, signe='_p')
                dict_sub_info_n = self.__all_sub_info_value__(dict_sub_field, 'vort',
                                                              self.seuil, signe='_n')
                for i in range(np.size(self.NN)):
                    NN = self.NN[i]
                    logging.info("calcul S_f on {}".format(NN))
                    nb_value = self.dict_all_sequence[NN].shape[0]
                    value = np.zeros(nb_value)
                    info_p = dict_sub_info_p[NN]
                    info_n = dict_sub_info_n[NN]
                    for j in range(nb_value):
                        logging.info("calcul value {}".format(j))
                        if self.config_pred.output_type == 'class':
                            if label_type.str.contains('sum').all():
                                value[j] = self.def_value_for_classes(info_p['info_{}'.format(j)],
                                                                      info_n['info_{}'.format(j)], 'sum')
                            elif label_type.str.contains('sum').all():
                                value[j] = self.def_value_for_classes(info_p['info_{}'.format(j)],
                                                                      info_n['info_{}'.format(j)], 'max')
                            else:
                                value[j] = self.def_value_for_classes(info_p['info_{}'.format(j)],
                                                                      info_n['info_{}'.format(j)], 'mean')
                        else:
                            value[j] = self.def_value_for_reg()

                    dict_value.add(NN, value)

            else:
                for i in range(np.size(self.NN)):

                    NN = self.NN[i]
                    logging.info("in all value for {}".format(NN))
                    nb_value = self.dict_all_sequence[NN].shape[0]
                    # cas target is vort field
                    if label_type.str.contains('vort').all():
                        value = np.zeros((nb_value, self.sw, self.sc))
                        for j in range(nb_value):
                            logging.debug("calcul value {}".format(j))
                            seq_signaltype, seq_fname, seq_savename = def_names('sequence', None, NN)
                            value[j, :, :] = self.recup_target_vort(seq_fname, j)
                    # cas target is img field
                    elif label_type.str.contains('img').all():
                        value = np.zeros((nb_value, 3, self.sw, self.sc))
                        for j in range(nb_value):
                            logging.debug("calcul value {}".format(j))
                            seq_signaltype, seq_fname, seq_savename = def_names('sequence', None, NN)
                            value[j, :, :, :] = self.recup_target_img(seq_fname, j)

                    dict_value.add(NN, value)

            logging.info("save into {}".format(fileName))
            self.save_single(fileName, dict_value, '')
        else:
            logging.info("load from {}".format(fileName))
            dict_value = np.load(fileName, allow_pickle=True).flat[0]

        self.dict_all_value = dict_value

    # ------------------------------------------
    def digit_img(self, img):
        seuil = 3e-3
        new_img = np.zeros_like(img)
        where_supp_seuil = np.where(img > seuil)
        where_inf_seuil = np.where(img < -seuil)
        new_img[where_inf_seuil[0], where_inf_seuil[1]] = 1
        new_img[where_supp_seuil[0], where_supp_seuil[1]] = 1

        return new_img

    # ------------------------- Target
    # ------------------------------------------
    def def_target(self, index, sequence, which_img=-1, min=None):
        if self.config_pred.output_type == 'img':
            value = None
            target = sequence[which_img, :, :, :]
            target = np.transpose(target, (2, 0, 1))
        elif self.config_pred.output_type == 'img_vort':
            value = None
            target = sequence[which_img, :, :, 2]
            target = np.expand_dims(target, axis=0)
        elif self.config_pred.output_type == 'digit_img':
            value = None
            target = self.digit_img(sequence[which_img, :, :, :])
            target = np.transpose(target, (2, 0, 1))
        elif self.config_pred.output_type == 'digit_img_vort':
            value = None
            target = self.digit_img(sequence[which_img, :, :, 2])
            target = np.expand_dims(target, axis=0)
        elif self.config_pred.output_type == 'class':
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

            logging.info(" new size is {}".format(size))

            seq = self.dict_all_sequence[NN]
            logging.info(" seq shape is {}".format(seq.shape))
            val = self.dict_all_value[NN]
            logging.info(" seq shape is {}".format(seq.shape))
            #
            # if min is not None:
            #     to_keep = np.where(val > min)[0]
            #     seq = seq[to_keep]
            #     val = val[to_keep]
            #
            # if max is not None:
            #     to_keep = np.where(val < max)[0]
            #     seq = seq[to_keep]
            #     val = val[to_keep]

            if size is not None:
                if size < val.size:
                    to_keep = np.random.choice(val.shape[0], size, replace=False)
                    logging.info(" so we keep  {}".format(to_keep.size))
                else:
                    to_keep = np.random.choice(val.shape[0], val.size, replace=False)
                seq = seq[to_keep]
                val = val[to_keep]

            dict_sequences.add(NN, seq)
            dict_value.add(NN, val)

            logging.info(" new seq qhape is {}".format(seq.shape))

        self.dict_sequence = dict_sequences
        self.dict_value = dict_value


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
            new_value = np.zeros_like(f)

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
            new_value = np.zeros_like(f)

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

# class CreateDataScalar:
#     def __init__(self, config_data, config_pred, remote, plot, histo, cycle_sample=None):
#         self.config_data = config_data
#         self.config_pred = config_pred
#         self.remote = remote
#         self.plot = plot
#         self.histo = histo
#
#         self.NN = config_pred.NN_data
#         self.label_name = config_pred.label_name
#         self.label_save = config_pred.label_save
#
#         self.seq_size = config_pred.seq_size
#         self.overlap_step = config_pred.overlap_step
#         self.output_shape = config_pred.output_shape
#         self.futur = config_pred.futur
#         self.cycle_sample = cycle_sample
#
#     # ------------------------------------------
#     def save_single(self, path_signal, data, name, extension='npy', nbfichier=None):
#
#         if data is not None:
#             if extension == 'npy':
#                 to_save = path_signal + name
#                 # print(to_save)
#                 np.save(to_save, data)
#             elif extension == 'cell':
#                 Cell(path_signal + name, nbfichier, data=data, extension='cell')
#             else:
#                 Cell(path_signal + name, nbfichier, data=data, extension='csv')
#
#     # ------------------------- Sequences
#     # ------------------------------------------
#     def random_indices(self, f, size):
#
#         pickrandom_lin = np.random.choice(f.shape[0] * f.shape[1], size, replace=False)
#         pickrandom_mat = np.unravel_index(pickrandom_lin, (f.shape[0], f.shape[1]))
#
#         return pickrandom_mat[0], pickrandom_mat[1]
#
#     # ------------------------------------------
#     def all_indices(self, f):
#
#         lin = np.arange(f.shape[0] * f.shape[1])
#         mat = np.unravel_index(lin, (f.shape[0], f.shape[1]))
#
#         return mat[0], mat[1]
#
#     # ------------------------------------------
#     def cycles_sample_indices(self, f, cycle):
#         shape = Shape(f)
#
#         tps_to_keep = np.arange(0, shape.tps - (self.seq_size + self.futur) + 1, self.overlap_step)
#         cycle_to_keep = np.ones_like(tps_to_keep) * cycle
#
#         print('indice to keep shape = {}'.format(np.shape(tps_to_keep)))
#         indices = np.vstack((cycle_to_keep, tps_to_keep)).transpose()
#
#         return indices
#
#     # ------------------------------------------
#     def seq_indices(self, f):
#         shape = Shape(f)
#
#         tps_to_keep = np.arange(0, shape.tps - (self.seq_size + self.futur) + 1, self.overlap_step)
#         indice_to_keep = np.zeros((shape.cycles, tps_to_keep.size))
#
#         print(
#             'indice to keep shape = {} et size = {}'.format(np.shape(indice_to_keep), shape.cycles * tps_to_keep.size))
#         cycles, tps = self.all_indices(indice_to_keep)
#         indices = np.vstack((cycles, tps_to_keep[tps])).transpose()
#
#         return indices
#
#     # ------------------------------------------
#     def __NN_sequences__(self, signal_flu):
#
#         fileName = self.config_pred.global_path_load + 'sequence_NN/' + 'dict_sequences_{}_{}seqsize_{}step_{}futur.npy'.format(
#             self.config_pred.input_data, self.seq_size, self.overlap_step, self.futur)
#         fileObj = Path(fileName)
#         is_fileObj = fileObj.is_file()
#
#         if not is_fileObj:
#             dict_sequences = dictdata()
#             for i in range(np.size(self.NN)):
#                 NN = self.NN[i]
#
#                 indices = self.seq_indices(signal_flu[i].f)
#
#                 dict_sequences.add(NN, indices)
#
#             self.save_single(fileName, dict_sequences, '')
#
#         else:
#             dict_sequences = np.load(fileName, allow_pickle=True).flat[0]
#
#         self.dict_all_sequence = dict_sequences
#
#     # ------------------------- Values
#     # ------------------------------------------
#     def def_value_for_classes(self, df, index, number, cycle, t0, from_X=True):
#
#         if from_X:
#             seq_size = self.seq_size
#         else:
#             seq_size = 1
#
#         sub_index = index[cycle, t0 + seq_size:t0 + seq_size + self.futur]
#         sub_number = number[cycle, t0 + seq_size:t0 + seq_size + self.futur]
#
#         numbers = sub_number[sub_index == 1].astype(int)
#
#         if np.size(numbers) == 0:
#             countdf = 0
#         else:
#             if self.label_save == 'max_df':
#                 countdf = np.max(df[numbers])
#             elif self.label_save == 'sum_df':
#                 countdf = np.sum(df[numbers])
#             elif self.label_save == 'sum_expdf':
#                 sub_df = np.zeros_like(sub_index)
#                 sub_df[sub_index == 1] = df[numbers]
#                 exp = np.exp(-np.arange(0, np.size(sub_df)) / self.config_pred.tau)
#                 countdf = np.sum(sub_df * exp)
#             else:
#                 sub_df = np.zeros_like(sub_index)
#                 sub_df[sub_index == 1] = df[numbers]
#                 gamma = sp.stats.gamma(self.config_pred.k, loc=0, scale=self.config_pred.theta)
#                 pdf = gamma.pdf(np.arange(0, np.size(sub_df)))
#                 countdf = np.sum(sub_df * pdf)
#
#         return countdf
#
#     # ------------------------------------------
#     def def_value_for_reg(self, df, index, number, cycle, t0, from_X=True):
#
#         if from_X:
#             seq_size = self.seq_size
#         else:
#             seq_size = 1
#
#         sub_index = np.asarray(index[cycle, t0 + seq_size: t0 + seq_size + self.futur])
#         sub_number = number[cycle, t0 + seq_size: t0 + seq_size + self.futur]
#         if np.size(sub_index) != np.size(sub_number):
#             print('error')
#             print('sub index size = {} vs sub_number size = {}'.format(np.size(sub_index), np.size(sub_number)))
#         numbers = sub_number[sub_index == 1].astype(int)
#
#         sub_df = np.zeros_like(sub_index)
#         sub_df[sub_index == 1] = df[numbers]
#
#         if self.label_save == 'expdf' or self.label_save == 'expdf_logrsc':
#             exp = np.exp(-np.arange(0, np.size(sub_df)) / self.config_pred.tau)
#             countdf = np.sum(sub_df * exp)
#         elif self.label_save == 'test_new_renorm':
#             countdf = sub_df
#         else:
#             print('warning non codé')
#             countdf = None
#
#         return countdf
#
#     # ------------------------------------------
#     def __all_value__(self, signalevent):
#         if self.config_pred.output_type == 'class':
#             fileName = self.config_pred.global_path_load + 'sequence_NN/' + 'dict_all_value_for_class_{}_futur_{}.npy'.format(
#                 self.label_save, self.futur)
#         else:
#             fileName = self.config_pred.global_path_load + 'sequence_NN/' + 'dict_all_value_for_reg_{}_futur_{}.npy'.format(
#                 self.label_save, self.futur)
#
#         fileObj = Path(fileName)
#         is_fileObj = fileObj.is_file()
#         if not is_fileObj:
#             dict_value = dictdata()
#             for i in range(np.size(self.NN)):
#                 NN = self.NN[i]
#
#                 df = signalevent[i].df_tt
#                 index = signalevent[i].index_df_tt
#                 number = signalevent[i].number_df_tt
#
#                 nb_value = self.dict_all_sequence[NN].shape[0]
#                 value = np.zeros(nb_value)
#                 for j in range(nb_value):
#                     cycle = self.dict_all_sequence[NN][j, 0]
#                     t0 = self.dict_all_sequence[NN][j, 1]
#                     if self.config_pred.output_type == 'class':
#                         value[j] = self.def_value_for_classes(df, index, number, cycle, t0)
#                     else:
#                         value[j] = self.def_value_for_reg(df, index, number, cycle, t0)
#
#                 dict_value.add(NN, value)
#
#                 self.save_single(fileName, dict_value, '')
#         else:
#             dict_value = np.load(fileName, allow_pickle=True).flat[0]
#
#         self.dict_all_value = dict_value
#
#     # ------------------------------------------
#     def __cycle_value__(self, signalevent):
#         if self.config_pred.output_type == 'class':
#             fileName = self.config_pred.global_path_load + 'sequence_NN/' + 'dict_cycle{}_value_for_class_{}_futur_{}.npy'.format(
#                 self.cycle_sample, self.label_save, self.futur)
#         else:
#             fileName = self.config_pred.global_path_load + 'sequence_NN/' + 'dict_cycle{}_value_for_reg_{}_futur_{}.npy'.format(
#                 self.cycle_sample, self.label_save, self.futur)
#
#         fileObj = Path(fileName)
#         is_fileObj = fileObj.is_file()
#         if not is_fileObj:
#             dict_value = dictdata()
#             for i in range(np.size(self.NN)):
#                 NN = self.NN[i]
#
#                 df = signalevent[i].df_tt
#                 index = signalevent[i].index_df_tt
#                 number = signalevent[i].number_df_tt
#
#                 nb_value = self.dict_sequence[NN].shape[0]
#                 value = np.zeros(nb_value)
#                 for j in range(nb_value):
#                     cycle = self.dict_all_sequence[NN][j, 0]
#                     t0 = self.dict_all_sequence[NN][j, 1]
#                     if self.config_pred.output_type == 'class':
#                         value[j] = self.def_value_for_classes(df, index, number, cycle, t0)
#                     else:
#                         value[j] = self.def_value_for_reg(df, index, number, cycle, t0)
#
#                 dict_value.add(NN, value)
#
#                 self.save_single(fileName, dict_value, '')
#         else:
#             dict_value = np.load(fileName, allow_pickle=True).flat[0]
#
#         self.dict_value = dict_value
#
#     # ------------------------- Info Class Edges
#     # ------------------------------------------
#     def classes(self, labels):
#
#         print('min value_on_futur = {} et max value_on_futur = {}'.format(np.min(labels), np.max(labels)))
#
#         if self.output_shape == 2:
#             classes_edges = np.array([np.min(labels), 5e-3, np.max(labels)])
#         elif self.output_shape == 3:
#             classes_edges = np.array([np.min(labels), 5e-3, 3e-1, np.max(labels)])
#         elif self.output_shape == 4:
#             classes_edges = np.array([np.min(labels), 5e-3, 3e-2, 3e-1, np.max(labels)])
#         elif self.output_shape == 5:
#             classes_edges = np.array([np.min(labels), 5e-3, 3e-2, 3e-1, 3e0, np.max(labels)])
#
#         return classes_edges
#
#     # ------------------------------------------
#     def nonequipro_classes(self, labels):
#
#         print('min value_on_futur = {} et max value_on_futur = {}'.format(np.min(labels), np.max(labels)))
#
#         label_still_to_cut = labels[labels >= 5e-3]
#
#         slice = np.round(label_still_to_cut.size / (self.output_shape - 1))
#         sorted_labels = np.sort(label_still_to_cut, axis=0)
#
#         classes_edges = np.zeros(self.output_shape - 1)
#         for i in range(self.output_shape - 1):
#             # print(i)
#             where = np.where(label_still_to_cut >= sorted_labels[int(i * slice)])[0]
#             classes_edges[i] = np.min(label_still_to_cut[where])
#             # print(classes_edges[i])
#
#         classes_edges = np.hstack((np.array([0]), classes_edges))
#         classes_edges = np.hstack((classes_edges, np.max(label_still_to_cut)))
#
#         hist, bin_edges = np.histogram(labels, density=False, bins=classes_edges)
#         proportion = np.round(np.asarray(hist) / np.sum(hist), 3)
#
#         if proportion.sum() != 1:
#             proportion[0] = 1 - proportion[1::].sum()
#
#         classes_edges = np.vstack((classes_edges, np.hstack((proportion, np.array([0])))))
#
#         return classes_edges
#
#     # ------------------------------------------
#     def equipro_classes(self, labels):
#
#         print('min value_on_futur = {} et max value_on_futur = {}'.format(np.min(labels), np.max(labels)))
#
#         classes_edges = np.zeros(self.output_shape + 1)
#         slice = np.round(labels.size / self.output_shape)
#         sorted_labels = np.sort(labels, axis=0)
#
#         for i in range(self.output_shape):
#             where = np.where(labels >= sorted_labels[int(i * slice)])[0]
#             classes_edges[i] = np.min(labels[where])
#         classes_edges[-1] = np.max(labels)
#         # print(classes_edges[-1])
#
#         hist, bin_edges = np.histogram(labels, density=False, bins=classes_edges)
#         proportion = np.round(np.asarray(hist) / np.sum(hist), 3)
#
#         if proportion.sum() != 1:
#             proportion[0] = 1 - proportion[1::].sum()
#
#         classes_edges = np.vstack((classes_edges, np.hstack((proportion, np.array([0])))))
#
#         return classes_edges
#
#     # ------------------------------------------
#     def verif_prop_on_value(self, value, NN):
#
#         hist, bin_edges = np.histogram(value, density=False, bins=self.classes_edges)
#         proportion = np.round(np.asarray(hist) / np.sum(hist), 3)
#
#         if proportion.sum() != 1:
#             proportion[0] = 1 - proportion[1::].sum()
#
#         print('prop on Y {} is {}'.format(NN, proportion))
#         print('prop on all {} is {}'.format(NN, self.prop))
#
#     # ------------------------- Info Reg rsc
#     # ------------------------------------------
#     def verif_pdf_on_value(self, value, all_value, min, NN):
#         print('zeros values on {}| on value [{:.1f}%] | on all [{:.1f}%]'.format(NN,
#                                                                          100 * np.size(value[value == 0]) / np.size(
#                                                                              value),
#                                                                          100 * np.size(
#                                                                              all_value[all_value == 0]) / np.size(
#                                                                              all_value)))
#
#         Y_all_df, X_all_df = self.histo.my_histo(all_value, min, np.max(all_value), 'log',
#                                                  'log', density=2, binwidth=None, nbbin=70)
#         Y_df, X_df = self.histo.my_histo(value, min, np.max(value), 'log',
#                                          'log', density=2, binwidth=None, nbbin=70)
#
#         fig, ax = self.plot.belleFigure('$\Delta \delta f$', '$P(\Delta \delta f)$', nfigure=None)
#         ax.plot(X_all_df, Y_all_df, 'r.')
#         ax.plot(X_df, Y_df, 'b.')
#         self.plot.plt.xscale('log')
#         self.plot.plt.yscale('log')
#         save = None
#         self.plot.fioritures(ax, fig, title='value on all set vs split', label=None, grid=None, save=save)
#
#     # ------------------------------------------
#     def log_rsc(self, plot, labels, min):
#
#         labels_nzeros = labels[labels != 0]
#         plot.Pdf_loglog(labels_nzeros, np.min(labels_nzeros), np.max(labels_nzeros),
#                         'labels_{nzero}', 'labels_nzero',
#                         save=None,
#                         nbbin=70)
#
#         log_labels_nzeros = np.log(labels_nzeros)
#         plot.Pdf_linlin(log_labels_nzeros, np.min(log_labels_nzeros), np.max(log_labels_nzeros),
#                         'log labels_{nzero}', 'log_labels_nzeros', save=None, nbbin=150)
#
#         if self.config_pred.label_rsc == 'cutsmall' and min is not None:
#             label_cutsmall = labels[labels > min]
#             log_label_cutsmall = np.log(label_cutsmall)
#             plot.Pdf_linlin(log_label_cutsmall, np.min(log_label_cutsmall), np.max(log_label_cutsmall),
#                             'log labels_{cutsmall}', 'log_labels_cutsmall', save=None, nbbin=150)
#
#             print('relou values enlevées | on nzeros [{:.1f}%] | on cutsmall [{:.1f}%]'.format(
#                 100 * (np.size(labels) - np.size(labels_nzeros)) / np.size(labels),
#                 100 * (np.size(labels) - np.size(label_cutsmall)) / np.size(labels)))
#             print('sur {} values'.format(np.size(labels)))
#
#             mean_cutsmall = np.mean(log_label_cutsmall)
#             dev_cutsmall = np.sqrt(np.var(log_label_cutsmall))
#             print('stats on log labels cutsmall : mean = {} | dev = {}'.format(mean_cutsmall, dev_cutsmall))
#
#             rsc_log_label_cutsmall = (log_label_cutsmall - mean_cutsmall) / dev_cutsmall
#             plot.Pdf_linlin(rsc_log_label_cutsmall, np.min(rsc_log_label_cutsmall), np.max(rsc_log_label_cutsmall),
#                             'rsc log labels_{cutsmall}', 'rsc_log_labels_cutsmall', save=None, nbbin=150)
#
#             stats = np.array([mean_cutsmall, dev_cutsmall])
#         elif min is not None:
#             label_relocc = labels
#             label_relocc[labels < min] = min
#             plot.Pdf_loglog(label_relocc, np.min(label_relocc), np.max(label_relocc),
#                             'labels', 'labels',
#                             save=None, nbbin=150)
#
#             log_label_relocc = np.log(label_relocc)
#             plot.Pdf_linlin(log_label_relocc, np.min(log_label_relocc), np.max(log_label_relocc),
#                             'log labels', 'log_labels',
#                             save=None, nbbin=150)
#
#             mean_relocc = np.mean(log_label_relocc)
#             dev_relocc = np.sqrt(np.var(log_label_relocc))
#             print('stats on log labels relocc : mean = {} | dev = {}'.format(mean_relocc, dev_relocc))
#
#             rsc_log_label_relocc = (log_label_relocc - mean_relocc) / dev_relocc
#             plot.Pdf_linlin(rsc_log_label_relocc, np.min(rsc_log_label_relocc), np.max(rsc_log_label_relocc),
#                             'log labels rsc', 'log_labels_rsc',
#                             save=None, nbbin=150)
#
#             stats = np.array([mean_relocc, dev_relocc])
#         else:
#             stats = None
#
#         return stats
#
#     # ------------------------- Info stats
#     # ------------------------------------------
#     def __info_stats__(self, plot, min):
#
#         if self.config_pred.output_type == 'class':
#             value = self.dict_value['train']
#             fileName = self.config_pred.global_path_load + 'sequence_NN/' + '{}_classes_edges_{}_futur_{}.npy'.format(
#                     self.config_pred.output_shape, self.label_save,
#                     self.futur)
#             fileObj = Path(fileName)
#             is_fileObj = fileObj.is_file()
#             if not is_fileObj:
#                 classes = self.classes(value)
#                 logging.info("writting into {}".format(fileName))
#                 self.save_single(fileName, classes, '')
#             else:
#                 logging.info("Load from {}".format(fileName))
#                 classes = np.load(fileName)#[0, :]
#             hist, bin_edges = np.histogram(value, density=False, bins=classes)
#             proportion = np.round(np.asarray(hist) / np.sum(hist), 3)
#             test_value = self.dict_value['test']
#             hist_test, bin_edges = np.histogram(test_value, density=False, bins=classes)
#             test_proportion = np.round(np.asarray(hist_test) / np.sum(hist_test), 3)
#             self.test_prop = test_proportion
#
#             if proportion.sum() != 1:
#                 proportion[0] = 1 - proportion[1::].sum()
#
#             info_stats = np.vstack((classes, np.hstack((proportion, np.array([0])))))
#         else:
#             value = self.dict_value['train']
#             if self.config_pred.label_save == 'expdf_logrsc':
#                 fileName = self.config_pred.global_path_load + 'sequence_NN/' + 'stats_{}_{}_futur_{}.npy'.format(
#                     self.label_save, min, self.futur)
#                 fileObj = Path(fileName)
#                 is_fileObj = fileObj.is_file()
#                 if not is_fileObj:
#                     info_stats = self.log_rsc(plot, value, min)
#                     self.save_single(fileName, info_stats, '')
#
#                 else:
#                     info_stats = np.load(fileName)
#             else:
#                 info_stats = None
#
#         self.info_stats = info_stats
#         if self.config_pred.output_type == 'class':
#             self.classes_edges = info_stats[0, :]
#             self.prop = info_stats[1, :-1]
#         else:
#             self.rsc_stat = info_stats
#
#         return info_stats
#
#     # ------------------------- Features supp
#     # ------------------------------------------
#     def timeseq_df(self, df, index, number, cycle, t0):
#         new_df = np.zeros(self.seq_size)
#
#         sub_index = index[cycle, t0:t0 + self.seq_size]
#         sub_number = number[cycle, t0:t0 + self.seq_size]
#
#         numbers = sub_number[sub_index == 1].astype(int)
#
#         if not np.size(numbers) == 0:
#             new_df[sub_index == 1] = df[numbers]
#
#         return new_df
#
#     # ------------------------------------------
#     def add_features(self, f, df, index, number, NN, syncro):
#
#         fileName_df = self.config_pred.global_path_load + 'sequence_NN/' + 'df_feature_{}futur_{}_{}.npy'.format(
#             self.futur, NN, syncro)
#         fileName_value = self.config_pred.global_path_load + 'sequence_NN/' + 'value_feature_{}futur_{}_{}.npy'.format(
#             self.futur, NN, syncro)
#
#         fileObj_df = Path(fileName_df)
#         is_fileObj_df = fileObj_df.is_file()
#
#         tic = time.time()
#
#         if not is_fileObj_df:
#             new_df = np.zeros_like(f)
#             new_value = np.ones_like(f) * -1
#
#             for i in range(f.shape[0]):
#                 logging.info("cycle : {}".format(i))
#                 where = np.where(index[i, :] == 1)[0]
#                 new_df[i, where] = df[number[i, where].astype(int)]
#
#                 for j in range(f.shape[1]):
#
#                     if j + 1 + self.futur < f.shape[1]:
#                         if self.config_pred.output_type == 'class':
#                             value = self.def_value_for_classes(df, index, number, cycle=i, t0=j, from_X=False)
#
#                         if syncro == 'syncro':
#                             new_value[i, j] = value
#                         else:
#                             new_value[i, j + 1 + self.futur] = value
#
#             logging.info("Writting into: {}".format(fileName_df))
#             self.save_single(fileName_df, new_df, '')
#             self.save_single(fileName_value, new_value, '')
#         else:
#             logging.info("Load from: {}".format(fileName_df))
#             new_df = np.load(fileName_df)
#             new_value = np.load(fileName_value)
#
#         logging.info("time  : {}".format(time.time() - tic))
#         logging.info("inew df s Nan : {}".format(np.sum(np.isnan(new_df))))
#         logging.info("inew value s Nan : {}".format(np.sum(np.isnan(new_value))))
#
#         return new_df, new_value
#
#     # ------------------------------------------
#     def multifeature_f(self, f, df, index, number, NN):
#
#         new_df, new_value = self.add_features(f, df, index, number, NN, syncro='not_syncro')
#
#         if self.config_pred.channel == 2:
#             new_f = np.concatenate((np.expand_dims(f, axis=0), np.expand_dims(new_df, axis=0)), axis=0)
#         else:
#             new_f = np.concatenate(
#                 (np.expand_dims(f, axis=0), np.expand_dims(new_df, axis=0), np.expand_dims(new_value, axis=0)), axis=0)
#
#         logging.info("new_f shape : {}".format(np.shape(new_f)))
#         return new_f
#
#     # ------------------------- Target
#     # ------------------------------------------
#     def get_class_label(self, value):
#
#         if value < self.classes_edges[0]:
#             self.classes_edges[0] = value
#         elif value > self.classes_edges[-1]:
#             self.classes_edges[-1] = value + 1
#
#         classe_value, _ = np.histogram(value, bins=self.classes_edges)
#         classe_label = np.where(classe_value == 1)[0][0]
#
#         return classe_label
#
#     # ------------------------------------------
#     def rsc_reg_label(self, value, min):
#         mean = self.rsc_stat[0]
#         dev = self.rsc_stat[1]
#         if min is not None and value < min:
#             value = min
#         value = np.log(value)
#         label = (value - mean) / dev
#
#         return label
#
#     # ------------------------------------------
#     def def_target(self, index, min=None):
#         if self.config_pred.output_type == 'class':
#             value = self.list_value[index]
#             target = self.get_class_label(value)
#         else:
#             value = self.list_value[index]
#             if self.config_pred.label_save == 'expdf_logrsc':
#                 target = self.rsc_reg_label(value, min)
#             else:
#                 target = value
#
#         return target, value
#
#     # ------------------------- Sub Sample dans split
#     # ------------------------------------------
#     def __sub_sample_split__(self, min, max, nb_seq):
#         dict_sequences = dictdata()
#         dict_value = dictdata()
#         for i in range(np.size(self.NN)):
#             NN = self.NN[i]
#             size = nb_seq[i]
#
#             seq = self.dict_all_sequence[NN]
#             val = self.dict_all_value[NN]
#
#             if min is not None:
#                 to_keep = np.where(val > min)[0]
#                 seq = seq[to_keep]
#                 val = val[to_keep]
#
#             if max is not None:
#                 to_keep = np.where(val < max)[0]
#                 seq = seq[to_keep]
#                 val = val[to_keep]
#
#             if size is not None:
#                 if size < val.size:
#                     to_keep = np.random.choice(val.size, size, replace=False)
#                 else:
#                     to_keep = np.random.choice(val.size, val.size, replace=False)
#                 seq = seq[to_keep]
#                 val = val[to_keep]
#
#             dict_sequences.add(NN, seq)
#             dict_value.add(NN, val)
#
#         self.dict_sequence = dict_sequences
#         self.dict_value = dict_value
#
#     def __sub_sample_ratio__(self, classes, nb_seq):
#         dict_sequences = dictdata()
#         dict_value = dictdata()
#
#         for i in range(np.size(self.NN)):
#             NN = self.NN[i]
#             size = nb_seq[i]
#             logging.debug("sub sample ration in {}".format(NN))
#             logging.debug("classes are {}".format(classes))
#             seq = self.dict_all_sequence[NN]
#             val = self.dict_all_value[NN]
#
#             hist, _ = np.histogram(val, bins=classes, density=False)
#             nb_to_keep = np.min(hist)
#             logging.debug("nb_to_keep = {}".format(nb_to_keep))
#
#             sub_seq = np.array([[0, 0]])
#             sub_val = np.array([0])
#             if size is not None:
#                 sub_size = int(size / (classes.size-1))
#             else:
#                 sub_size = val.size
#
#             if NN != 'test':
#                 logging.debug("sub_size = {}".format(sub_size))
#                 for j in range(classes.size - 1):
#                     if j != classes.size - 2:
#                         where = np.where((val >= classes[j]) & (val < classes[j + 1]))[0]
#                     else:
#                         where = np.where((val >= classes[j]) & (val <= classes[j + 1]))[0]
#                     logging.debug("il y a {} elment dans classe {} - {}  {}".format(where.size, j, classes[j], classes[j+1]))
#                     bla = val[where]
#                     bli = seq[where]
#                     if sub_size < nb_to_keep:
#                         logging.debug('reduce from {} to {}'.format(where.size, sub_size))
#                         to_keep = np.random.choice(bla.size, sub_size, replace=False)
#                         logging.debug(to_keep.size)
#                     else:
#                         logging.debug('reduce from {} to {}'.format(where.size, nb_to_keep))
#                         to_keep = np.random.choice(bla.size, nb_to_keep, replace=False)
#                     bli = bli[to_keep]
#                     bla = bla[to_keep]
#
#                     logging.debug('shape seq = {}, shape val = {}'.format(sub_seq.shape, sub_val.shape))
#                     logging.debug('shape bli = {}, shape bla = {}'.format(bli.shape, bla.shape))
#                     sub_seq = np.concatenate((sub_seq, bli))
#                     sub_val = np.concatenate((sub_val, bla))
#
#                 val = sub_val[1:]
#                 seq = sub_seq[1:, :]
#
#             dict_sequences.add(NN, seq)
#             dict_value.add(NN, val)
#
#             logging.debug('{} taille finale = {}'.format(NN, val.size))
#             nb_seq[i] = dict_value[NN].size
#
#         self.dict_sequence = dict_sequences
#         self.dict_value = dict_value
#
#         return nb_seq
#     # ------------------------------------------
#     def __cycles_NN_sequences__(self, signal_flu):
#
#         fileName = self.config_pred.global_path_load + 'sequence_NN/' + 'dict_sequences_{}_cycle{}_{}seqsize_{}step_{}futur.npy'.format(
#             self.config_pred.input_data, self.cycle_sample, self.seq_size, self.overlap_step,
#             self.futur)
#         fileObj = Path(fileName)
#         is_fileObj = fileObj.is_file()
#
#         if not is_fileObj:
#             dict_sequences = dictdata()
#             for i in range(np.size(self.NN)):
#                 NN = self.NN[i]
#
#                 indices = self.cycles_sample_indices(signal_flu[i].f, self.cycle_sample)
#                 dict_sequences.add(NN, indices)
#
#             self.save_single(fileName, dict_sequences, '')
#
#         else:
#             dict_sequences = np.load(fileName, allow_pickle=True).flat[0]
#
#         self.dict_sequence = dict_sequences

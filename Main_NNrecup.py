import numpy as np

from Datas.classSignal import SignalForce, SignalImg, VariationsField
from classConfig import ConfigData
from Datas.classEvent import ForceEvent
import Config_data
from Datas.classCell import Cell
from classPlot import ClassPlot
from Datas.classStat import Histo

import logging

logging.basicConfig(format='| %(levelname)s | %(asctime)s | %(message)s', level=logging.info)

################### Def param ##################################

remote = False
path_from_root = '/path_from_root/'
ref_tricot = 'knit005_'
n_exp = 'mix_'
version_data = 'v1'

NAME_EXP = ref_tricot + n_exp + version_data
config_data = ConfigData(path_from_root, Config_data.exp[NAME_EXP])

print(NAME_EXP)

histo = Histo(config_data)
plot = ClassPlot(remote, histo)

################### Recup Data ##################################

class RecupData :
    def __init__(self, config_data, remote):

        self.config_data = config_data
        self.remote = remote

    # ------------------------------------------
    def transfert(self, config, to_save, signal):

        tosave_size = to_save + signal.fname + '_size'
        tosave_sub_cycles = to_save + signal.fname + '_NN_sub_cycles'
        tosave_index = to_save + 'index_picture_' + signal.fname
        tosave_number = to_save + 'number_picture_' + signal.fname
        tosave_numbertot = to_save + 'numbertot_picture_' + signal.fname
        tosave_nb_index = to_save + 'nb_index_picture_' + signal.fname

        np.save(tosave_size, signal.nbcycle)
        np.save(tosave_index, signal.index_picture)
        np.save(tosave_number, signal.number_picture)
        np.save(tosave_numbertot, signal.numbertot_picture)
        np.save(tosave_nb_index, signal.nb_index_picture)

        # print(signal.NN_sub_cycles[0])
        # print(signal.NN_sub_cycles[1])
        Cell(tosave_sub_cycles, config.nb_set, signal.NN_sub_cycles, 'cell')

    # ------------------------------------------
    def recup_knitanalyse_img(self, signaltype):
        NN_data = ['train', 'val', 'test']

        signal_img = [SignalImg(self.config_data, signaltype, NN_data[i]) for i in
                      range(np.size(NN_data))]

        for i in range(np.size(NN_data)):
            NN = NN_data[i]

            to_save = self.config_data.global_path_save + 'pict_event_{}_NN/'.format(signaltype)
            sum_vort = np.array([])
            sum_vort_nrsc = np.array([])

            for j in range(self.config_data.nb_set):
                for k in range(np.size(self.config_data.fields)):
                    name_field = self.config_data.fields[k]
                    logging.info('import field {}'.format(name_field))

                    field = signal_img[i].import_field(name_field, num_set=j)

                    # print('for field {} : min = {}, max = {}, mean = {}, var = {}'.format(name_field,
                    #                                                                       variations_mix[j][k].stats_f.min,
                    #                                                                       variations_mix[j][k].stats_f.max, variations_mix[j][k].stats_f.mean,
                    #                                                                       variations_mix[j][k].stats_f.var))
                    field_rsc = field ##variations_mix[j][k].rsc(field)

                    logging.info(
                        'save info : {} | {} | '.format(to_save, name_field + signal_img[i].savename + '_{}'.format(j)))
                    signal_img[i].save_single(to_save, field_rsc, name_field + signal_img[i].savename + '_{}'.format(j))

                    shape = np.asarray(np.shape(field_rsc))
                    signal_img[i].save_single(to_save, shape, name_field + signal_img[i].savename + '_shape_{}'.format(j))

                    if NN == 'train' and name_field == 'vort':
                        sum_vort_nrsc = np.hstack((sum_vort_nrsc, signal_img[i].sum_field(field, abs=True)))
                        sum_vort = np.hstack((sum_vort, signal_img[i].sum_field(field_rsc, abs=True)))
                        print('min sum vort = {} et max sum vort = {}'.format(np.min(sum_vort), np.max(sum_vort)))
                        signal_img[i].save_single(to_save, sum_vort,
                                                  'sum_vort' + signal_img[i].savename)
                        sw_sc = shape[0: 2]
                        signal_img[i].save_single(to_save, sw_sc,
                                                  'sw_sc')

            self.transfert(self.config_data, to_save, signal_img[i])

            # return variations_mix, sum_vort, sum_vort_nrsc

    # ------------------------------------------
    def recup_knitanalyse_img_sub(self, signaltype):
        NN_data = ['train', 'val', 'test']

        signal_img = [SignalImg(self.config_data, signaltype, NN_data[i]) for i in
                      range(np.size(NN_data))]

        for i in range(np.size(NN_data)):
            NN = NN_data[i]

            to_save = self.config_data.global_path_save + 'pict_event_{}_NN/'.format(signaltype)
            sum_vort = np.array([])
            sum_vort_nrsc = np.array([])

            for j in range(self.config_data.nb_set):
                for k in range(np.size(self.config_data.fields)):
                    name_field = self.config_data.fields[k] + '_sub'
                    logging.info('import field {}'.format(name_field))

                    field = signal_img[i].import_field(name_field, num_set=j)

                    # print('for field {} : min = {}, max = {}, mean = {}, var = {}'.format(name_field,
                    #                                                                       variations_mix[j][k].stats_f.min,
                    #                                                                       variations_mix[j][k].stats_f.max, variations_mix[j][k].stats_f.mean,
                    #                                                                       variations_mix[j][k].stats_f.var))
                    field_rsc = field ##variations_mix[j][k].rsc(field)

                    logging.info('save info : {} | {} | '.format(to_save,  name_field + signal_img[i].savename + '_{}'.format(j)))
                    signal_img[i].save_single(to_save, field_rsc, name_field + signal_img[i].savename + '_{}'.format(j))

                    shape = np.asarray(np.shape(field_rsc))
                    signal_img[i].save_single(to_save, shape, name_field + signal_img[i].savename + '_shape_{}'.format(j))

                    if NN == 'train' and name_field == 'vort':
                        sum_vort_nrsc = np.hstack((sum_vort_nrsc, signal_img[i].sum_field(field, abs=True)))
                        sum_vort = np.hstack((sum_vort, signal_img[i].sum_field(field_rsc, abs=True)))
                        print('min sum vort = {} et max sum vort = {}'.format(np.min(sum_vort), np.max(sum_vort)))
                        signal_img[i].save_single(to_save, sum_vort,
                                                  'sum_vort' + signal_img[i].savename)
                        sw_sc = shape[0: 2]
                        signal_img[i].save_single(to_save, sw_sc,
                                                  'sw_sc_sub')

    # ------------------------------------------
    def recup_knitanalyse_scalar(self, signaltype):
        NN_data = ['train', 'val', 'test']

        signal_flu = [SignalForce(self.config_data, signaltype, NN_data[i]) for i in
                      range(np.size(NN_data))]

        for i in range(np.size(NN_data)):
            NN = NN_data[i]

            to_save = self.config_data.global_path_save + '{}_NN/'.format(signaltype)

            signal_flu[i].save_single(to_save, signal_flu[i].f, signal_flu[i].fname)
            signal_flu[i].save_single(to_save, signal_flu[i].ext, 'ext_' + signal_flu[i].fname)
            signal_flu[i].save_single(to_save, signal_flu[i].t, 't_' + signal_flu[i].fname)

            self.transfert(self.config_data, to_save, signal_flu[i])

            signalevent = ForceEvent(self.config_data, signal_flu[i].f, signal_flu[i].ext, signal_flu[i].t, signaltype,
                                     NN, Sm=False)

recup_data = RecupData(config_data, remote)

if config_data.img:
    recup_data.recup_knitanalyse_img('flu_rsc')
    recup_data.recup_knitanalyse_img_sub('flu_rsc')
recup_data.recup_knitanalyse_scalar('flu_rsc')

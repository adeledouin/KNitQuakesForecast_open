import numpy as np
from functools import partial
from multiprocessing import Pool, Array
import ctypes
import timeit
from pathlib import Path
from skimage import measure

from Datas.classFindPeak import Derivee, FindPeak
from Datas.classCell import Cell
from Datas.classStat import Stat, Shape
from Datas import memory


def def_names(signaltype, fname, NN_data, Sm=None):
    """
    Function to define folders and files names

    Parameters:
        signaltype (str) : 'flu', 'flu_rsc', 'rsc_flu_rsc', 'sequence'
        fname (str) : None if pas différent de signaltype
        NN_data (str) : '', 'train', 'val', 'test'
        Sm (bol): rsc par Sm (2nd moment) du signal des events

    Returns:
        output (str) : nom du dossier, nom du fichier force avec extension NN, extention Sm pour df_tt, extension Sm pour df_seuil
    """

    if NN_data != '':
        if fname is None:
            fname = signaltype + '_' + NN_data
        else:
            fname = fname + '_' + NN_data
        signaltype = signaltype + '_NN'
        savename = '_' + NN_data
    else:
        if fname is None:
            fname = signaltype
        else:
            fname = fname
        signaltype = signaltype
        savename = ''
    if Sm is not None:
        if Sm:
            savename_df_tt = '_Sm_tt' + savename
            savename_df_seuil = '_Sm_seuil' + savename
        else:
            savename_df_tt = '_tt' + savename
            savename_df_seuil = '_seuil' + savename
    else:
        savename_df_tt = None
        savename_df_seuil = None


    return signaltype, fname, savename, savename_df_tt, savename_df_seuil

def def_nbcycle(config, path_signal, fname, NN_data):
    """
    Function to define number of cycle on the actual signal

    Parameters:
        config (class) : config associée à la l'analyse
        path_signal (str) : path to folder
        fname (str) : nomd du fichier
        NN_data (str) : '', 'train', 'val', 'test'

    Returns:
        output : nbcycle, sub_nbcycles, cycles, sub_cycles_NN, NN_sub_cycles
        nbcycle (int) : nombre de cycles dans le signal
        sub_cycles (list[liste]) : liste des cycles par set comptés sur nombre total de cycles dans l'analyse
        cycles (list): liste des cycles dans cette extention NN (not None seulement quand NN et single set)
        sub_cycles_NN (list[liste]) : liste des cycles par set dans ce NN comptés sur nombre total de cycles dans analyse
        NN_sub_cycles (list[liste]) : liste des cycles par set dans ce NN comptés sur nombre de cycles total dans NN
    """

    if NN_data == '' and not config.mix_set:
        nbcycle = config.nbcycle
        sub_cycles = None
        cycles = None
        sub_cycles_NN = None
        NN_sub_cycles = None
    elif NN_data == '' and config.mix_set:
        nbcycle = np.sum(config.nbcycle)
        sub_cycles = config.sub_cycles
        cycles = None
        sub_cycles_NN = None
        NN_sub_cycles = None
    elif NN_data != '' and not config.mix_set:
        nbcycle = np.load(path_signal + fname + '_size.npy')
        sub_cycles = None
        cycles = np.load(path_signal + fname + '_cycles.npy')
        sub_cycles_NN = None
        NN_sub_cycles = None
    else:
        nbcycle = np.load(path_signal + fname + '_size.npy')
        sub_cycles = config.sub_cycles
        cycles = None
        sub_cycles_NN = None
        recup_sub_cycles = Cell(path_signal + fname + '_NN_sub_cycles', config.nb_set)
        NN_sub_cycles = recup_sub_cycles.reco_cell()

    return nbcycle, cycles, sub_cycles, sub_cycles_NN, NN_sub_cycles

def fname_to_fsave(fname, sep_posneg, signe):
    """
    Function to compute class ForceEvent on the right field

    Parameters:


    Returns:
        output :
    """

    if fname == 'dev':
        fsave = '_dev'
    elif fname == 'vort' and not sep_posneg:
        fsave ='_vort'
    elif fname == 'vort' and sep_posneg:
        if signe == 'pos':
            fsave = '_vort_p'
        elif signe == 'neg':
            fsave = '_vort_n'
        else:
            fsave = '_vort_pn'
    else:
        if signe == 'pos':
            fsave = '_slip_p'
        elif signe == 'neg':
            fsave = '_slip_n'
        else:
            fsave = '_slip_pn'

    return fsave

def fsave_to_fname(fsave):
    """
    Function to compute class ForceEvent on the right field

    Parameters:


    Returns:
        output :
    """

    # print(fsave)
    if fsave == '_dev':
        fname = 'dev'
        sep_posneg = False
        signe =''
    elif fsave == '_vort':
        fname = 'vort'
        sep_posneg = False
        signe = ''
    elif fsave == '_vort_p':
        fname = 'vort'
        sep_posneg = True
        signe = 'pos'
    elif fsave == '_vort_n':
        fname = 'vort'
        sep_posneg = True
        signe = 'neg'
    elif fsave == '_vort_pn':
        fname = 'vort'
        sep_posneg = True
        signe = 'cumul'
    elif fsave == '_slip_p':
        fname = 'slip_Y'
        sep_posneg = False
        signe = 'pos'
    elif fsave == '_slip_n':
        fname = 'slip_Y'
        sep_posneg = False
        signe = 'neg'
    elif fsave == '_slip_pn':
        fname = 'slip_Y'
        sep_posneg = False
        signe = 'cumul'

    return fname, sep_posneg, signe

def imgevent_cumul_pn(data_p, data_n, concat=False):
    """
    Function to compute class ForceEvent on the right field

    Parameters:


    Returns:
        output :
    """

    if not concat:
        data = np.zeros_like(data_p)
        for i in range(np.size(data)):
            data[i] = data_p[i] + data_n[i]
    else:
        data = np.concatenate((data_p, data_n))

    return data


def find_img_events(config, signal_img, fsave, fname, seuil, save_seuil, signaltype, NN_data, sep_posneg, signe):
    """
    Function to compute class ForceEvent on the right field

    Parameters:


    Returns:
        output :
    """

    if fsave == '_dev':
        event = ImgEvent(config, signal_img.dev, seuil, save_seuil,
                         signaltype=signaltype, NN_data=NN_data, fname=fname, sep_posneg=sep_posneg)

        nb_area_img = event.nb_area_img
        sum_S_a = event.sum_S_a
        sum_S_f = event.sum_S_f
        S_a = event.S_a
        S_f = event.S_f
        pict_S = event.pict_S

    elif fsave == '_vort':
        event = ImgEvent(config, signal_img.vort, seuil, save_seuil,
                         signaltype=signaltype, NN_data=NN_data, fname=fname, sep_posneg=sep_posneg)

        nb_area_img = event.nb_area_img
        sum_S_a = event.sum_S_a
        sum_S_f = event.sum_S_f
        S_a = event.S_a
        S_f = event.S_f
        pict_S = event.pict_S

    elif fname == 'vort' and sep_posneg:
        event = ImgEvent(config, signal_img.vort, seuil, save_seuil,
                         signaltype=signaltype, NN_data=NN_data, fname=fname, sep_posneg=sep_posneg)

        if signe == 'pos':
            nb_area_img = event.nb_area_img_p
            sum_S_a = event.sum_S_a_p
            sum_S_f = event.sum_S_f_p
            S_a = event.S_a_p
            S_f = event.S_f_p
            pict_S = event.pict_S

        elif signe == 'neg':
            nb_area_img = event.nb_area_img_n
            sum_S_a = event.sum_S_a_n
            sum_S_f = event.sum_S_f_n
            S_a = event.S_a_n
            S_f = event.S_f_n
            pict_S = event.pict_S

        else:
            nb_area_img = imgevent_cumul_pn(event.nb_area_img_p, event.nb_area_img_n)
            sum_S_a = imgevent_cumul_pn(event.sum_S_a_p, event.sum_S_a_n)
            sum_S_f = imgevent_cumul_pn(event.sum_S_f_p, event.sum_S_f_n)
            S_a = imgevent_cumul_pn(event.S_a_p, event.S_a_n, concat=True)
            S_f = imgevent_cumul_pn(event.S_f_p, event.S_f_n, concat=True)
            pict_S = imgevent_cumul_pn(event.pict_S_p, event.pict_S_n, concat=True)

    else:
        event_p = ImgEvent(config, signal_img.slip_Y, seuil, save_seuil,
                           signaltype=signaltype, NN_data=NN_data, fname=fname, sep_posneg=sep_posneg)

        event_n = ImgEvent(config, signal_img.slip_X, seuil, save_seuil,
                           signaltype=signaltype, NN_data=NN_data, fname=fname, sep_posneg=sep_posneg)

        if signe == 'pos':
            nb_area_img = event_p.nb_area_img
            sum_S_a = event_p.sum_S_a
            sum_S_f = event_p.sum_S_f
            S_a = event_p.S_a
            S_f = event_p.S_f
            pict_S = event_p.pict_S

        elif signe == 'neg':
            nb_area_img = event_n.nb_area_img
            sum_S_a = event_n.sum_S_a
            sum_S_f = event_n.sum_S_f
            S_a = event_n.S_a
            S_f = event_n.S_f
            pict_S = event_n.pict_S

        else:
            nb_area_img = imgevent_cumul_pn(event_p.nb_area_img, event_n.nb_area_img)
            sum_S_a = imgevent_cumul_pn(event_p.sum_S_a, event_n.sum_S_a)
            sum_S_f = imgevent_cumul_pn(event_p.sum_S_f, event_n.sum_S_f)
            S_a = imgevent_cumul_pn(event_p.S_a, event_n.S_a, concat=True)
            S_f = imgevent_cumul_pn(event_p.S_f, event_n.S_f, concat=True)
            pict_S = imgevent_cumul_pn(event_p.pict_S, event_n.pict_S, concat=True)

    return nb_area_img, sum_S_a, sum_S_f, S_a, S_f, pict_S


# ---------------------------------------------------------------------------------------------------------------------#
class ForceEvent():
    """
    Classe qui permet de trouver les events dans le signal en force.

    Attributes:
        config (class) : config associée à la l'analyse

        signaltype (str): 'flu', 'flu_rsc', 'rsc_flu_rsc', 'sequence' nom du dossier
        fname (str) : nom du fichier du sigal en force - None if pas différent de signaltype
        NN_data (str) : '', 'train', 'val', 'test' extention NN
        Sm (bol) : rsc par Sm (2nd moment) du signal des events
        savename_df_tt (str) : extension de rsc de df tt
        savename_df_seuil : extension de rsc de df_seuil

        nb_process (int) : no de process a utiliser pour le mutliprocess
        saving step (bol) : pêrmet de sauver les étapes dans cas de analyse signel set, si multi set alors ne sauve pas
        display_figure (bol) : affiche les figure pendant recherche des events si besoin

        path_signal (str) : chemin du dossier associé à ce signal
        to_save_fig (str) : chemin associé pour save fig

        nbcycle (int) : nombre de cycles dans le signal
        sub_cycles (list[liste]) : liste des cycles par set comptés sur nombre total de cycles dans l'analyse
        cycles (list): liste des cycles dans cette extention NN (not None seulement quand NN et single set)
        sub_cycles_NN (list[liste]) : liste des cycles par set dans ce NN comptés sur nombre total de cycles dans analyse
        NN_sub_cycles (list[liste]) : liste des cycles par set dans ce NN comptés sur nombre de cycles total dans NN

        f (array) : signal force
        t (array) : tps associé au signal
        ext (array) : extension associé au signal
        f_size (int) : taille d'un cycle

        df_tt (array) : array 1D des events
        dext_tt (array) : array 1D des extension associé aux events
        dt_tt (array) : array 1D  du tps associé aux events
        index_tt (array) : index des events associées au signal
        number_tt (array) : numero des events associées au signal
        nb_index_tt (int ou array) : nombre total d'events associées au signal
    """

    # ---------------------------------------------------------#
    def __init__(self, config, f, ext, t, signaltype, NN_data, fname=None, Sm=False, display_figure_debug=False,
                 saving_step=True):
        """
        The constructor for ForceEvent.

        Parameters:
            config (class) : config associée à la l'analyse

            f (array) : signal force
            t (array) : tps associé au signal
            ext (array) : extension associé au signal

            signaltype (str): 'flu', 'flu_rsc', 'rsc_flu_rsc', 'sequence' type du signal en force
            NN_data (str) : '', 'train', 'val', 'test' extension NN
            fname (str) : nom du signal - None if pas différent de signaltype

            Sm (bol) : rsc par Sm (2nd moment) du signal des events

            display_figure_debug (bol) : affiche les figure pendant recherche des events si besoin
            saving step (bol) : pêrmet de sauver les étapes dans cas de analyse signel set, si multi set alors ne sauve pas
        """

        ## Config
        self.config = config

        self.f = f
        self.ext = ext
        self.t = t

        self.NN_data = NN_data
        self.Sm = Sm
        self.signaltype, self.fname, self.savename, self.savename_df_tt, self.savename_df_seuil = def_names(signaltype, fname, NN_data, Sm)

        self.nb_process = config.nb_process
        self.display_figure = display_figure_debug
        self.saving_step = saving_step

        self.path_signal = self.config.global_path_save + '/' + self.signaltype + '/'
        self.to_save_fig = self.config.global_path_save + '/figure_' + self.signaltype + '/'

        self.nbcycle, self.cycles, self.sub_cycles, self.sub_cycles_NN, self.NN_sub_cycles = def_nbcycle(self.config,
                                                                                                            self.path_signal,
                                                                                                            self.fname,
                                                                                                            self.NN_data)


        ## events info

        self.f_size = np.size(self.f[0, :])

        self.df_tt, self.dt_tt, self.dext_tt, self.index_df_tt, self.number_df_tt, self.nb_df_tt, \
        self.min_indice_df_tt, self.max_indice_df_tt = self.df_tt()

    # ------------------------------------------
    def import_single(self, name, extension='npy', size=None):

        if extension == 'npy':
            to_load = self.path_signal + name + '.npy'
            single = np.load(to_load)
        else:
            recup_single = Cell(self.path_signal + name, size)
            single = recup_single.reco_cell()

        return single

    # ------------------------------------------
    def save_single(self, path_signal, data, name, extension='npy', nbfichier=None):

        if data is not None:
            if extension == 'npy':
                to_save = path_signal + name
                np.save(to_save, data)
            elif extension == 'cell':
                Cell(path_signal + name, nbfichier, data=data, extension='cell')
            else:
                Cell(path_signal + name, nbfichier, data=data, extension='csv')

    # ------------------------------------------
    def find_indice_event(self, ):
        ''' '''

        min_indice = [0 for i in range(self.nbcycle)]
        max_indice = [0 for i in range(self.nbcycle)]
        min_indice_size = np.zeros(self.nbcycle, dtype=int)
        max_indice_size = np.zeros(self.nbcycle, dtype=int)

        for i in range(self.nbcycle):

            if self.signaltype == 'm_f':
                der = Derivee(self.f[i], self.ext[i], self.t[i])
            else:
                der = Derivee(self.f[i, :], self.ext[i, :], self.t[i, :])

            # if self.display_figure:
            #     if i == 0:
            #         verfi = der.verif_der()

            findpeak = FindPeak(der.der_signe_der_f, i, brut_signal=False)

            min, max, min_size, max_size = findpeak.recup_min_max_indices()

            min_indice[i] = min
            max_indice[i] = max
            min_indice_size[i] = min_size
            max_indice_size[i] = max_size

        return min_indice, max_indice, min_indice_size, max_indice_size

    # ------------------------------------------
    def find_event(self, plot=None):

        if self.signaltype == 'm_f':
            index_event = [0 for i in range(self.nbcycle)]
            number_event = [0 for i in range(self.nbcycle)]
        else:
            index_event = np.zeros((self.nbcycle, self.f_size))
            number_event = np.ones((self.nbcycle, self.f_size)) * np.nan

        nb_event = 0

        k = 0
        for i in range(self.nbcycle):
            if self.signaltype == 'm_f':
                print(np.shape(self.f_size))

                a = np.zeros(self.f_size[i])
                b = np.ones(self.f_size[i]) * np.nan

                for j in range(self.min_indice_size[i]):
                    a[1 + self.max_indice[i][j]:2 + self.min_indice[i][j]] = 1
                    b[1 + self.max_indice[i][j]:2 + self.min_indice[i][j]] = k + j

                index_event[i] = a
                number_event[i] = b
                nb_event = nb_event + self.min_indice_size[i]
                k = k + self.min_indice_size[i] - 1

            else:
                for j in range(self.min_indice_size[i]):
                    index_event[i, 1 + self.max_indice[i][j]:2 + self.min_indice[i][j]] = 1
                    number_event[i, 1 + self.max_indice[i][j]:2 + self.min_indice[i][j]] = k + j

                nb_event = nb_event + self.min_indice_size[i]
                k = k + self.min_indice_size[i] - 1

        if self.display_figure:
            if self.signaltype != 'm_f':
                i = 0
                fig, ax = plot.belleFigure('$L_{w} (mm)$', '$F(N)$', nfigure=None)
                where_events = np.where(number_event[i, :] == 0)[0]
                ax.plot(self.ext[i, :where_events[-1] + 2], self.f[i, :where_events[-1] + 2], 'b')
                ax.plot(self.ext[i, where_events], self.f[i, where_events], 'r.')
                plot.fioritures(ax, fig, title='event0', label=None, grid=None, save=None)

                fig, ax = plot.belleFigure('$L_{w} (mm)$', '$F(N)$', nfigure=None)
                where_events = np.where(number_event[i, :] == 1)[0]
                ax.plot(self.ext[i, :where_events[-1] + 2], self.f[i, :where_events[-1] + 2], 'b')
                ax.plot(self.ext[i, where_events], self.f[i, where_events], 'r.')
                plot.fioritures(ax, fig, title='event1', label=None, grid=None, save=None)

                fig, ax = plot.belleFigure('$L_{w} (mm)$', '$F(N)$', nfigure=None)
                where_events = np.where(number_event[i, :] == 2)[0]
                ax.plot(self.ext[i, :where_events[-1] + 2], self.f[i, :where_events[-1] + 2], 'b')
                ax.plot(self.ext[i, where_events], self.f[i, where_events], 'r.')
                plot.fioritures(ax, fig, title='event2', label=None, grid=None, save=None)

        return index_event, number_event, nb_event

    # ------------------------------------------
    def ampli_events(self):

        nb_df_tt = self.nb_events
        df_tt = np.zeros(nb_df_tt)
        dt_tt = np.zeros(nb_df_tt)
        dext_tt = np.zeros(nb_df_tt)
        min_indice_df_tt = np.zeros((2, nb_df_tt))
        max_indice_df_tt = np.zeros((2, nb_df_tt))

        if self.signaltype == 'm_f':
            index_df_tt = [0 for i in range(self.nbcycle)]
            number_df_tt = [0 for i in range(self.nbcycle)]
        else:
            index_df_tt = np.zeros((self.nbcycle, self.f_size))
            number_df_tt = np.ones((self.nbcycle, self.f_size)) * np.nan

        k = 0
        for i in range(self.nbcycle):
            if self.signaltype == 'm_f':
                a = np.zeros(self.f_size[i])
                b = np.ones(self.f_size[i], dtype=int) * np.nan

                for j in range(self.min_indice_size[i]):
                    a[1 + self.max_indice[i][j]] = 1
                    b[1 + self.max_indice[i][j]] = int(k + j)
                    df_tt[k + j] = self.f[i][1 + self.max_indice[i][j]] - self.f[i][1 + self.min_indice[i][j]]
                    dt_tt[k + j] = np.abs(self.t[i][1 + self.max_indice[i][j]] - self.t[i][1 + self.min_indice[i][j]])
                    dext_tt[k + j] = self.ext[i][1 + self.max_indice[i][j]] - self.ext[i][1 + self.min_indice[i][j]]

                    min_indice_df_tt[0, k + j] = i
                    min_indice_df_tt[1, k + j] = 1 + self.min_indice[i][j]
                    max_indice_df_tt[0, k + j] = i
                    max_indice_df_tt[1, k + j] = 1 + self.max_indice[i][j]

                    # if j == 0:
                    #     m_tt[k + j] = (self.f[i][1 + self.max_indice[i][j]] - self.f[i][0]) / \
                    #                   (self.ext[i][1 + self.max_indice[i][j]] - self.ext[i][0])
                    # else:
                    #     m_tt[k + j] = (self.f[i][1 + self.max_indice[i][j]] - self.f[i][1 + self.min_indice[i][j-1]]) / \
                    #                 (self.ext[i][1 + self.max_indice[i][j]] - self.ext[i][1 + self.min_indice[i][j-1]])

                index_df_tt[i] = a
                number_df_tt[i] = b

                k = k + self.min_indice_size[i]

            else:
                for j in range(self.min_indice_size[i]):
                    index_df_tt[i, 1 + self.max_indice[i][j]] = 1
                    number_df_tt[i, 1 + self.max_indice[i][j]] = int(k + j)
                    df_tt[k + j] = self.f[i, 1 + self.max_indice[i][j]] - self.f[i, 1 + self.min_indice[i][j]]
                    dt_tt[k + j] = np.abs(self.t[i, 1 + self.max_indice[i][j]] - self.t[i, 1 + self.min_indice[i][j]])
                    dext_tt[k + j] = self.ext[i, 1 + self.max_indice[i][j]] - self.ext[i, 1 + self.min_indice[i][j]]

                    min_indice_df_tt[0, k + j] = i
                    min_indice_df_tt[1, k + j] = 1 + self.min_indice[i][j]
                    max_indice_df_tt[0, k + j] = i
                    max_indice_df_tt[1, k + j] = 1 + self.max_indice[i][j]

                    # if j == 0:
                    #     m_tt[k + j] = (self.f[i, 1 + self.max_indice[i][j]] - self.f[i, 0]) / \
                    #                   (self.t[i, 1 + self.max_indice[i][j]] - self.t[i, 0])
                    # else:
                    #     m_tt[k + j] = (self.f[i, 1 + self.max_indice[i][j]] - self.f[i, 1 + self.min_indice[i][j-1]]) / \
                    #                   (self.t[i, 1 + self.max_indice[i][j]] - self.t[i, 1 + self.min_indice[i][j-1]])

                k = k + self.min_indice_size[i]

        return df_tt, dt_tt, dext_tt, index_df_tt, number_df_tt, nb_df_tt, min_indice_df_tt, max_indice_df_tt

    # ------------------------------------------
    def df_tt(self):

        ## regarde si le fichier existe dejà :
        fileName = self.path_signal + 'df' + self.savename_df_tt + '.npy'
        fileObj = Path(fileName)

        is_fileObj = fileObj.is_file()

        if is_fileObj and self.saving_step:
            # print('df_tt déjà enregisté')

            df_tt = self.import_single('df' + self.savename_df_tt)
            dext_tt = self.import_single('dext' + self.savename_df_tt)
            dt_tt = self.import_single('dt' + self.savename_df_tt)

            index_df_tt = self.import_single('index_df' + self.savename_df_tt)
            number_df_tt = self.import_single('number_df' + self.savename_df_tt)
            nb_index_df_tt = self.import_single('nb_df' + self.savename_df_tt)

            min_indice_df_tt = self.import_single('min_indice_df' + self.savename_df_tt)
            max_indice_df_tt = self.import_single('max_indice_df' + self.savename_df_tt)

            self.min_indice = self.import_single('min_indice' + self.savename, extension='cell', size=self.nbcycle)
            self.max_indice = self.import_single('max_indice' + self.savename, extension='cell', size=self.nbcycle)

            self.min_indice_size = self.import_single('min_indice_size' + self.savename)
            self.max_indice_size = self.import_single('max_indice_size' + self.savename)

            self.index_events = self.import_single('index_events' + self.savename)
            self.number_events = self.import_single('number_events' + self.savename)
            self.nb_index_events = self.import_single('nb_events' + self.savename)

        else:
            print('df_tt non enregisté')
            self.min_indice, self.max_indice, self.min_indice_size, self.max_indice_size = self.find_indice_event()

            self.index_events, self.number_events, self.nb_events = self.find_event()

            df_tt, dt_tt, dext_tt, index_df_tt, number_df_tt, nb_index_df_tt, \
            min_indice_df_tt, max_indice_df_tt = self.ampli_events()

            if self.Sm:
                stats = Stat(self.config, df_tt)
                df_tt = df_tt / stats.m2

            if self.saving_step:

                self.save_single(self.path_signal, self.min_indice, 'min_indice' + self.savename, extension='cell',
                                 nbfichier=self.nbcycle)
                self.save_single(self.path_signal, self.max_indice, 'max_indice' + self.savename, extension='cell',
                                 nbfichier=self.nbcycle)

                self.save_single(self.path_signal, self.min_indice_size, 'min_indice_size' + self.savename)
                self.save_single(self.path_signal, self.max_indice_size, 'max_indice_size' + self.savename)

                self.save_single(self.path_signal, self.index_events, 'index_events' + self.savename)
                self.save_single(self.path_signal, self.number_events, 'number_events' + self.savename)
                self.save_single(self.path_signal, self.nb_events, 'nb_events' + self.savename)

                self.save_single(self.path_signal, df_tt, 'df' + self.savename_df_tt)
                self.save_single(self.path_signal, dext_tt, 'dext' + self.savename_df_tt)
                self.save_single(self.path_signal, dt_tt, 'dt' + self.savename_df_tt)
                self.save_single(self.path_signal, nb_index_df_tt, 'nb_df' + self.savename_df_tt)

                self.save_single(self.path_signal, min_indice_df_tt, 'min_indice_df' + self.savename_df_tt)
                self.save_single(self.path_signal, max_indice_df_tt, 'max_indice_df' + self.savename_df_tt)

                if self.signaltype == 'm_f':
                    tosave_index_df_tt = self.path_signal + 'index_df' + self.savename_df_tt
                    tosave_number_df_tt = self.path_signal + 'number_df' + self.savename_df_tt
                    Cell(tosave_index_df_tt, self.nbcycle, index_df_tt, 'cell')
                    Cell(tosave_number_df_tt, self.nbcycle, number_df_tt, 'cell')
                else:
                    self.save_single(self.path_signal, index_df_tt, 'index_df' + self.savename_df_tt)
                    self.save_single(self.path_signal, number_df_tt, 'number_df' + self.savename_df_tt)

        return df_tt, dt_tt, dext_tt, index_df_tt, number_df_tt, nb_index_df_tt, min_indice_df_tt, max_indice_df_tt

    # ------------------------------------------
    def from_array_to_np(self, array, size_i, size_j):
        "Transforme multiprocessing.Array -> np.array"
        shared_array = np.frombuffer(array.get_obj())
        return shared_array.reshape(size_i, size_j)

    # ------------------------------------------
    def _initialize_subprocess_index_number(self, shared_array_base_index, shared_array_base_number):
        "Initialise notre subprocess avec la mémoire partagée."
        memory.shared_array_base_index = shared_array_base_index
        memory.shared_array_base_number = shared_array_base_number

    # ------------------------------------------
    def _initialize_subprocess_t_bfr(self, shared_array_base_t_bfr):
        "Initialise notre subprocess avec la mémoire partagée."
        memory.shared_array_base_t_bfr = shared_array_base_t_bfr

    # ------------------------------------------
    def _initialize_subprocess_df_btw(self, shared_array_base_sum_df, shared_array_base_max_df,
                                      shared_array_base_nb_df):
        "Initialise notre subprocess avec la mémoire partagée."
        memory.shared_array_base_sum_df = shared_array_base_sum_df
        memory.shared_array_base_max_df = shared_array_base_max_df
        memory.shared_array_base_nb_df = shared_array_base_nb_df

    # ------------------------------------------
    def create_index_number_cycle(self, indice_where, cycle, where, number_signal):

        where_number_j = np.where(number_signal[cycle] == where[indice_where])[0]

        if np.size(where_number_j) != 0:
            shared_array_index = self.from_array_to_np(memory.shared_array_base_index, 1, self.f_size[cycle])
            shared_array_number = self.from_array_to_np(memory.shared_array_base_number, 1, self.f_size[cycle])

            shared_array_index[0, where_number_j] = 1
            shared_array_number[0, where_number_j] = indice_where

    # ------------------------------------------
    def create_index_number(self, indice_where, where, number_signal):

        where_number_i = np.where(number_signal == where[indice_where])[0]
        where_number_j = np.where(number_signal == where[indice_where])[1]

        shared_array_index = self.from_array_to_np(memory.shared_array_base_index, self.nbcycle, self.f_size)
        shared_array_number = self.from_array_to_np(memory.shared_array_base_number, self.nbcycle, self.f_size)

        # print(where_number_i * self.f_size + where_number_j)

        shared_array_index[where_number_i, where_number_j] = 1
        shared_array_number[where_number_i, where_number_j] = indice_where

    # ------------------------------------------
    def df_seuil(self, seuilmin, seuilmax, save_seuil=''):

        start_time = timeit.default_timer()

        if self.saving_step:
            tosave_df_seuil = self.path_signal + 'df' + self.savename_df_seuil + '_' + save_seuil
            tosave_dt_seuil = self.path_signal + 'dt' + self.savename_df_seuil + '_' + save_seuil
            tosave_dext_seuil = self.path_signal + 'dext' + self.savename_df_seuil + '_' + save_seuil
            tosave_index_df_seuil = self.path_signal + 'index_df' + self.savename_df_seuil + '_' + save_seuil
            tosave_number_df_seuil = self.path_signal + 'number_df' + self.savename_df_seuil + '_' + save_seuil
            tosave_nb_index_df_seuil = self.path_signal + 'nb_index_df' + self.savename_df_seuil + '_' + save_seuil

        ## regarde si le fichier existe dejà :
        fileName = tosave_df_seuil + '.npy'
        fileObj = Path(fileName)
        is_fileObj = fileObj.is_file()

        if is_fileObj:
            print('seuil', save_seuil, 'déjà enregisté')

            df_seuil = np.load(tosave_df_seuil + '.npy')
            dt_seuil = np.load(tosave_dt_seuil + '.npy')
            dext_seuil = np.load(tosave_dext_seuil + '.npy')
            index_df_seuil = np.load(tosave_index_df_seuil + '.npy')
            number_df_seuil = np.load(tosave_number_df_seuil + '.npy')
            nb_index_df_seuil = np.load(tosave_nb_index_df_seuil + '.npy')

        else:
            print('seuil ', save_seuil, 'non enregisté')
            if seuilmax is None:
                where = np.where(self.df_tt >= seuilmin)[0]
            else:
                where = np.where((self.df_tt >= seuilmin) & (self.df_tt < seuilmax))[0]

            df_seuil = self.df_tt[where]
            dt_seuil = self.dt_tt[where]
            dext_seuil = self.df_tt[where]

            print(__name__)
            if __name__ == "Datas.classEvent":
                print(__name__)
                shared_array_base_index = Array(ctypes.c_double, np.zeros(self.nbcycle * self.f_size))
                shared_array_base_number = Array(ctypes.c_double, np.ones(self.nbcycle * self.f_size) * np.nan)

                with Pool(processes=self.nb_process, initializer=self._initialize_subprocess_index_number,
                          initargs=(shared_array_base_index, shared_array_base_number)) as pool:
                    pool.map(partial(self.create_index_number, where=where, number_signal=self.number_df_tt),
                             range(where.size))

                index_df_seuil = self.from_array_to_np(shared_array_base_index, self.nbcycle, self.f_size)
                number_df_seuil = self.from_array_to_np(shared_array_base_number, self.nbcycle, self.f_size)
                nb_index_df_seuil = where.size

            ## save
            if self.saving_step:
                np.save(tosave_df_seuil + '.npy', df_seuil)
                np.save(tosave_dt_seuil + '.npy', dt_seuil)
                np.save(tosave_dext_seuil + '.npy', dext_seuil)
                np.save(tosave_index_df_seuil + '.npy', index_df_seuil)
                np.save(tosave_number_df_seuil + '.npy', number_df_seuil)
                np.save(tosave_nb_index_df_seuil + '.npy', nb_index_df_seuil)

        stop_time = timeit.default_timer()
        print('tps pour calculer df_seuil:', stop_time - start_time)

        return df_seuil, dt_seuil, dext_seuil, index_df_seuil, number_df_seuil, nb_index_df_seuil

    # ------------------------------------------
    def df_seuil_fast(self, df, seuilmin, seuilmax):

        if seuilmax is None:
            where = np.where(df >= seuilmin)[0]
        else:
            where = np.where((df >= seuilmin) & (df < seuilmax))[0]

        df_seuil = df[where]
        # dt_seuil = dt[where]

        return df_seuil #, dt_seuil

    # ------------------------------------------
    def time_btw_df(self, index_signal, number_signal, nb_index_signal):

        start_time = timeit.default_timer()

        t_btw = np.array([0])
        index_t_btw = index_signal
        number_t_btw = number_signal

        for i in range(self.nbcycle):
            where_df = np.where(index_signal[i, :] == 1)[0]
            t_btw_sub = self.t[i, where_df[1:]] - self.t[i, where_df[:-1]]

            index_t_btw[i, where_df[-1]] = 0
            number_t_btw[i, where_df] = number_t_btw[i, where_df] - i
            number_t_btw[i, where_df[-1]] = np.nan

            t_btw = np.hstack((t_btw, t_btw_sub))

        t_btw = t_btw[1:]

        stop_time = timeit.default_timer()
        print('tps pour calculer t_btw:', stop_time - start_time)

        return t_btw, index_t_btw, number_t_btw

    # ------------------------------------------
    def find_t_bfr_df_cell(self, j, i, where):

        shared_array_t_bfr = self.from_array_to_np(memory.shared_array_base_t_bfr, 1, self.f_size[i])

        for k in range(where.size):
            if j <= where[k]:
                shared_array_t_bfr[0, j] = self.t[i][where[k]] - self.t[i][j]
                break

    # ------------------------------------------
    def find_t_bfr_df(self, j, i, where):

        shared_array_t_bfr = self.from_array_to_np(memory.shared_array_base_t_bfr, 1, self.f_size)

        for k in range(where.size):
            if j <= where[k]:
                shared_array_t_bfr[0, j] = self.t[i, where[k]] - self.t[i, j]
                break

    # ------------------------------------------
    def time_bfr_df(self, index_signal):

        start_time = timeit.default_timer()

        if self.signaltype == 'mf':
            t_bfr = [0 for i in range(self.nbcycle)]

            for i in range(self.nbcycle):
                # prin(i)

                where = np.where(index_signal[i] == 1)[0]

                if __name__ == "classEvent":
                    shared_array_base_t_bfr = Array(ctypes.c_double, np.zeros(self.f_size[i]))

                    # prin(self.f_size[i])

                    with Pool(processes=self.nb_process, initializer=self._initialize_subprocess_t_bfr,
                              initargs=(shared_array_base_t_bfr,)) as pool:
                        pool.map(partial(self.find_t_bfr_df_cell, i=i, where=where), range(self.f_size[i]))

                    t_bfr_sub = self.from_array_to_np(shared_array_base_t_bfr, 1, self.f_size[i])

                t_bfr[i] = t_bfr_sub
        else:
            t_bfr = np.zeros((self.nbcycle, self.f_size))

            for i in range(self.nbcycle):
                # prin(i)

                where = np.where(index_signal[i, :] == 1)[0]

                if __name__ == "classEvent":
                    shared_array_base_t_bfr = Array(ctypes.c_double, np.zeros(self.f_size))

                    # prin(self.f_size)

                    with Pool(processes=self.nb_process, initializer=self._initialize_subprocess_t_bfr,
                              initargs=(shared_array_base_t_bfr,)) as pool:
                        pool.map(partial(self.find_t_bfr_df, i=i, where=where), range(self.f_size))

                    t_bfr_sub = self.from_array_to_np(shared_array_base_t_bfr, 1, self.f_size)

                t_bfr[i, :] = t_bfr_sub

        stop_time = timeit.default_timer()
        print('tps pour calculer t_bfr:', stop_time - start_time)

        return t_bfr.reshape(self.nbcycle * self.f_size)

    # ------------------------------------------
    def df_tab(self):

        df_tab = np.zeros_like(self.index_df_tt)

        where_df = np.where(self.index_df_tt == 1)
        for i in range(where_df[0].size):
            df_tab[where_df[0][i], where_df[1][i]] = self.df_tt[i]

        return df_tab

    # ------------------------------------------
    # def classes_df(self, equipro_classes, nbclasses, exposant, bfr=False):
    #
    #     histo = Histo(self.config)
    #
    #     start_time = timeit.default_timer()
    #
    #     X_df_ampli = [0 for i in range(nbclasses)]
    #     Y_df_ampli = [0 for i in range(nbclasses)]
    #
    #     X_dt_ampli = [0 for i in range(nbclasses)]
    #     Y_dt_ampli = [0 for i in range(nbclasses)]
    #
    #     X_t_btw = [0 for i in range(nbclasses)]
    #     Y_t_btw = [0 for i in range(nbclasses)]
    #
    #     if bfr:
    #         X_t_bfr = [0 for i in range(nbclasses)]
    #         Y_t_bfr = [0 for i in range(nbclasses)]
    #
    #     for i in range(nbclasses):
    #
    #         print('#### df_seuil ####')
    #
    #         df_seuil, dt_seuil, dext_seuil, index_df_seuil, number_df_seuil, nb_index_df_seuil = \
    #             self.df_seuil(equipro_classes[i], equipro_classes[i + 1])
    #
    #         df_seuil_stats = Stat(self.config, df_seuil)
    #         dt_seuil_stats = Stat(self.config, dt_seuil)
    #
    #         print('#### t_btw ####')
    #         t_btw, index_t_btw, number_t_btw, nb_index_t_btw = \
    #             self.time_btw_df(index_df_seuil, number_df_seuil, nb_index_df_seuil)
    #
    #         t_btw_stats = Stat(self.config, t_btw)
    #
    #         if bfr:
    #             print('#### t_bfr ####')
    #             t_bfr = self.time_bfr_df(index_df_seuil)
    #
    #             t_bfr_stats = Stat(self.config, t_bfr)
    #
    #         print('#### histo ####')
    #         Y_df_ampli[i], X_df_ampli[i] = histo.my_histo(df_seuil, np.round(df_seuil_stats.min, exposant),
    #                                                       np.round(df_seuil_stats.max, exposant),
    #                                                       'log', 'log', density=False, binwidth=None, nbbin=100)
    #
    #         Y_dt_ampli[i], X_dt_ampli[i] = histo.my_histo(dt_seuil, np.round(dt_seuil_stats.min, 2),
    #                                                       np.round(dt_seuil_stats.max, 2),
    #                                                       'log', 'log', density=False, binwidth=0.04, nbbin=None)
    #
    #         Y_t_btw[i], X_t_btw[i] = histo.my_histo(t_btw, np.round(t_btw_stats.min, 1),
    #                                                 np.round(t_btw_stats.max, 1),
    #                                                 'log', 'log', density=False, binwidth=0.4, nbbin=None)
    #
    #         if bfr == True:
    #             Y_t_bfr[i], X_t_bfr[i] = histo.my_histo(t_bfr, np.round(t_bfr_stats.min, 1),
    #                                                     np.round(t_bfr_stats.max, 1),
    #                                                     'log', 'log', density=True, binwidth=0.4, nbbin=None)
    #
    #     fig, ax = Fc.belleFigure('$df_c$', '$pdf(df_c)$', nfigure=None)
    #     for i in range(nbclasses):
    #         ax.plot(X_df_ampli[i], Y_df_ampli[i], '.', label='$df_{min} = %10.3f , df_{max} = %10.3f$'
    #                                                          % (np.min(X_df_ampli[i]), np.max(X_df_ampli[i])))
    #     plt.xscale('log')
    #     plt.yscale('log')
    #     plt.title('pdf df equipro_classes')
    #     plt.legend()
    #     plt.savefig(self.config.global_path_save + 'figure_' + self.signaltype + '/' + 'df_classes_pdf' + '.png')
    #     if not self.remote:
    #         plt.show()
    #
    #     fig, ax = Fc.belleFigure('$dt_c$', '$pdf(dt_c)$', nfigure=None)
    #     for i in range(nbclasses):
    #         ax.plot(X_dt_ampli[i], Y_dt_ampli[i], '.', label='$df_{min} = %10.3f , df_{max} = %10.3f$'
    #                                                          % (np.min(X_df_ampli[i]), np.max(X_df_ampli[i])))
    #     plt.xscale('log')
    #     plt.yscale('log')
    #     plt.title('pdf dt equipro_classes')
    #     plt.legend()
    #     plt.savefig(self.config.global_path_save + 'figure_' + self.signaltype + '/' + 'dt_classes_pdf' + '.png')
    #     if not self.remote:
    #         plt.show()
    #
    #     fig, ax = Fc.belleFigure('$t_{btw}$', '$pdf(t_{btw}$', nfigure=None)
    #     for i in range(nbclasses):
    #         ax.plot(X_t_btw[i], Y_t_btw[i], '.', label='$df_{min} = %10.3f , df_{max} = %10.3f$'
    #                                                    % (np.min(X_df_ampli[i]), np.max(X_df_ampli[i])))
    #     plt.xscale('log')
    #     plt.yscale('log')
    #     plt.title('pdf t btw equipro_classes')
    #     plt.legend()
    #     plt.savefig(self.config.global_path_save + 'figure_' + self.signaltype + '/' + 'dt_btw_classes_pdf' + '.png')
    #     if not self.remote:
    #         plt.show()
    #
    #     if bfr == True:
    #         fig, ax = Fc.belleFigure('$t_{bfr}$', '$pdf(t_{bfr})$', nfigure=None)
    #         for i in range(nbclasses):
    #             ax.plot(X_t_bfr[i], Y_t_bfr[i], '.', label='$df_{min} = %10.3f , df_{max} = %10.3f$'
    #                                                        % (np.min(X_df_ampli[i]), np.max(X_df_ampli[i])))
    #         # plt.xscale('log')
    #         # plt.yscale('log')
    #         plt.title('pdf t bfr equipro_classes')
    #         plt.legend()
    #         plt.savefig(
    #             self.config.global_path_save + 'figure_' + self.signaltype + '/' + 'dt_bfr_classes_pdf' + '.png')
    #         if not self.remote:
    #             plt.show()
    #
    #     stop_time = timeit.default_timer()
    #     print('tps pour traiter les equipro_classes :', stop_time - start_time)

    # ------------------------------------------
    def find_df_btw(self, indice_where, i, number_picture, where_img, df_signal, index_df, number_df,
                    nb_index_sum_df, nb_inter):

        # print('rentré dans find')
        shared_array_sum_df = self.from_array_to_np(memory.shared_array_base_sum_df, int(nb_index_sum_df), 1)
        shared_array_max_df = self.from_array_to_np(memory.shared_array_base_max_df, int(nb_index_sum_df), 1)
        shared_array_nb_df = self.from_array_to_np(memory.shared_array_base_nb_df, int(nb_index_sum_df), 1)

        # print('jai recup les shared array')

        which_img = number_picture[i, where_img[indice_where]].astype(int)


        if indice_where == 0:
            sub_index_df = index_df[i, 0:where_img[indice_where]]
            sub_number_df = number_df[i, 0:where_img[indice_where]]
        else:
            sub_index_df = index_df[i, where_img[indice_where - 1]:where_img[indice_where]]
            sub_number_df = number_df[i, where_img[indice_where - 1]:where_img[indice_where]]

        # print('j ai recup les sub array')

        if nb_inter is None:
            where_number_sub = np.where(sub_index_df == 1)[0]
        else:
            if np.where(sub_index_df == 1)[0].size >= nb_inter:
                where_number_sub = np.where(sub_index_df == 1)[0][-nb_inter:]
            else:
                where_number_sub = np.where(sub_index_df == 1)[0]

        which_number_df = sub_number_df[where_number_sub].astype(int)

        # print('j ai recup les which number df')

        shared_array_sum_df[which_img] = np.sum(df_signal[which_number_df])
        if np.size(df_signal[which_number_df]) != 0:
            shared_array_max_df[which_img] = np.max(df_signal[which_number_df])
        else:
            shared_array_max_df[which_img] = 0
        shared_array_nb_df[which_img] = np.size(which_number_df)

        # prin('numero de l indice dans where_img : {}, correspond à img {}, '.format(indice_where, which_img))

        stop_time = timeit.default_timer()
        # print('tps pour find sum df:', stop_time - start_time)

    # ------------------------------------------
    def df_btwpict(self, df, index_df, number_df, index_picture, number_picture, nb_index_picture,
                   nb_inter=None, name='tt', seuil=''):

        if self.saving_step:
            tosave_sum_df = self.path_signal + 'sum_df' + name + seuil
            tosave_max_df = self.path_signal + 'max_df' + name + seuil
            tosave_nb_df = self.path_signal + 'nb_df' + name + seuil
            tosave_index_sum_df = self.path_signal + 'index_sum_df' + name + seuil
            tosave_number_sum_df = self.path_signal + 'number_sum_df' + name + seuil
            tosave_nb_index_sum_df = self.path_signal + 'nb_index_sum_df' + name + seuil

        ## regarde si le fichier existe dejà :
        fileName = tosave_sum_df + '.npy'
        fileObj = Path(fileName)

        is_fileObj = fileObj.is_file()

        if is_fileObj:
            print('sum_df déjà enregistré')

            sum_df = np.load(tosave_sum_df + '.npy')
            max_df = np.load(tosave_max_df + '.npy')
            nb_df = np.load(tosave_nb_df + '.npy')
            index_sum_df = np.load(tosave_index_sum_df + '.npy')
            number_sum_df = np.load(tosave_number_sum_df + '.npy')
            nb_index_sum_df = np.load(tosave_nb_index_sum_df + '.npy')

        else:
            index_sum_df = index_picture
            number_sum_df = number_picture
            nb_index_sum_df = nb_index_picture

            shared_array_base_sum_df = Array(ctypes.c_double, np.zeros(int(nb_index_sum_df)))
            shared_array_base_max_df = Array(ctypes.c_double, np.zeros(int(nb_index_sum_df)))
            shared_array_base_nb_df = Array(ctypes.c_double, np.zeros(int(nb_index_sum_df)))

            for i in range(self.nbcycle):
                if __name__ == "classEvent":
                    # prin('cycle{}'.format(i))
                    where_img = np.where(index_picture[i, :] == 1)[0]

                    # print(
                    #     'première img du cycle : {}, dernière img du cycle : {}'.format(number_picture[i, int(where_img[0])],
                    #                                                                 number_picture[i, int(where_img[-1])]))

                    start_time = timeit.default_timer()
                    with Pool(processes=self.nb_process, initializer=self._initialize_subprocess_df_btw,
                              initargs=(
                                      shared_array_base_sum_df, shared_array_base_max_df, shared_array_base_nb_df)) as pool:
                        pool.map(partial(self.find_df_btw, i=i, number_picture=number_picture, where_img=where_img,
                                         df_signal=df, index_df=index_df, number_df=number_df,
                                         nb_index_sum_df=nb_index_sum_df, nb_inter=nb_inter), range(where_img.size))

                    sum_df = self.from_array_to_np(shared_array_base_sum_df, int(nb_index_sum_df), 1)
                    max_df = self.from_array_to_np(shared_array_base_max_df, int(nb_index_sum_df), 1)
                    nb_df = self.from_array_to_np(shared_array_base_nb_df, int(nb_index_sum_df), 1)

                    stop_time = timeit.default_timer()
                    print('tps pour sum df sur cycle {} : {}'.format(i, stop_time - start_time))

            if self.saving_step:
                np.save(tosave_sum_df, sum_df)
                np.save(tosave_max_df, max_df)
                np.save(tosave_nb_df, nb_df)
                np.save(tosave_index_sum_df, index_sum_df)
                np.save(tosave_number_sum_df, number_sum_df)
                np.save(tosave_nb_index_sum_df, nb_index_sum_df)

        sum_df = sum_df.reshape(np.shape(sum_df)[0])
        max_df = max_df.reshape(np.shape(max_df)[0])
        nb_df = nb_df.reshape(np.shape(nb_df)[0])

        return sum_df, max_df, nb_df, index_sum_df, number_sum_df, nb_index_sum_df

        # ------------------------------------------

    # ------------------------------------------
    def nb_df_btwpict(self, index_df, index_picture, number_picture):

        nb_df = []

        for i in range(self.nbcycle):

            where = np.where(index_picture[i, :] == 1)[0]
            # # print('size where {}'.format(where.size))
            # print('on verif pour cycle {} : premier number {} et dernier number {}'.format(i,
            #                                                                                number_picture[
            #                                                                                    i, where[0]],
            #                                                                                number_picture[
            #                                                                                    i, where[-1]]))
            for j in range(where.size):

                if j == 0:
                    sub = index_df[i, 0:where[j]]
                    # print('quand j = 0, c est img {}'.format(number_picture[i, where[j]]))
                else:
                    sub = index_df[i, where[j - 1]:where[j]]
                    # if j == where.size - 1:
                    #     print('quand j = where.size-1, c est img {}'.format(number_picture[i, where[j]]))
                nb_sub = np.sum(sub)
                nb_df.append(nb_sub)

        return np.asarray(nb_df)


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
    def __init__(self, config, field, field_seuil, seuil, fault_analyse=False, debug=False):
        """
        The constructor for InfoField.

        Parameters:
            config (class) : config associée à la l'analyse

            field (array) : une img de champ
            field_seuil (array) : babeled array, where all regions supp to seuil are assigned 1.
            seuil (int) : valeur utilisée pour seuiller les évents
            fault_analyse (bol) : est ce que l'analyse est pour étudier les fault
            debug (bol) : permet d'afficher plot des img des region pour debuguer

        """

        self.config = config

        self.nb_area, self.num_area, self.size_area, self.sum_field, \
        self.size_area_img, self.sum_field_img, self.conncomp = self.info(field, field_seuil, seuil, fault_analyse, debug)

    # ------------------------------------------
    def info(self, field, field_seuil, seuil, fault_analyse, debug):

        start_time = timeit.default_timer()

        conncomp, Nobj = measure.label(field_seuil, return_num=True)

        Reg = measure.regionprops(conncomp)

        stop_time = timeit.default_timer()
        # print('tps pour regionprop :', stop_time - start_time)

        start_time = timeit.default_timer()

        num_area = np.arange(1, Nobj + 1)
        Area = np.zeros(Nobj)
        Center = np.zeros((Nobj, 2))
        Orient = np.zeros(Nobj)

        for i in range(Nobj):
            Area[i] = Reg[i].area
            if Area[i] == 1:
                pixels = np.nonzero(conncomp == num_area[i])
                conncomp[pixels] = 0
            else:
                Center[i, :] = Reg[i].centroid
                Orient[i] = Reg[i].orientation

        num_area = num_area[np.where(Area != 1)[0]]
        Center = Center[np.where(Area != 1)[0], :]
        Orient = Orient[np.where(Area != 1)[0]]
        Area = Area[np.where(Area != 1)[0]]

        Nobj = np.size(Area)

        if debug:
            L = np.zeros_like(field_seuil)
            L_area = np.zeros_like(field_seuil)
            L_sum = np.zeros_like(field_seuil)

        sum_field = np.zeros(Nobj)

        for i in range(Nobj):
            to_sum_field = np.zeros_like(field_seuil)
            pixels = np.nonzero(conncomp == num_area[i])

            to_sum_field[pixels] = field[pixels]

            sum_field[i] = np.sum(to_sum_field)
            if sum_field[i] < seuil:
                print('Proooooooooooooooobleeeeeeeeeeeeeme')

            if debug:
                L[pixels] = i
                L_area[pixels] = Area[i]
                L_sum[pixels] = sum_field[i]

        stop_time = timeit.default_timer()
        # print('tps pour calcul sum :', stop_time - start_time)

        # if debug:
            #
            # fig, ax = Fc.belleFigure('$L_w$', '$L_c$', nfigure=None)
            # ax.imshow(L, extent=[0, 1, 0, 1])
            # plt.title('test')
            # if not self.remote:
            #     plt.show()
            #
            # fig, ax = Fc.belleFigure('$L_w$', '$L_c$', nfigure=None)
            # ax.imshow(L_area, extent=[0, 1, 0, 1])
            # plt.title('test')
            # if not self.remote:
            #     plt.show()
            #
            # fig, ax = Fc.belleFigure('$L_w$', '$L_c$', nfigure=None)
            # ax.imshow(L_sum, extent=[0, 1, 0, 1])
            # plt.title('test')
            # if not self.remote:
            #     plt.show()

        if not fault_analyse:
            conncomp = None

        return Nobj, np.arange(1, Nobj + 1), Area, sum_field, np.sum(Area), np.sum(sum_field), conncomp


# ---------------------------------------------------------------------------------------------------------------------#
class ImgEvent():
    """
    Classe qui permet e trouver les events dans les champs de déformations.

    Attributes:
        config (class) : config associée à la l'analyse

        seuil (int) : valeur utilisée pour seuiller les évents
        save_seuil (str) : matricule du seuil

        signaltype (str): 'flu', 'flu_rsc', 'rsc_flu_rsc', 'sequence' nom du dossier
        fname (str) : nom du fichier
        savename (str) : '_' + extension NN
        NN_data (str) : '', 'train', 'val', 'test'
        sep_posneg (bol) : définie la façon de traité les régions positives et négative

        saving step (bol) : pêrmet de sauver les étapes dans cas de analyse signel set, si multi set alors ne sauve pas

        path_signal (str) : chemin du dossier associé à ce signal
        to_save_fig (str) : chemin associé pour save fig

        nbcycle (int) : nombre de cycles dans le signal
        sub_cycles (list[liste]) : liste des cycles par set comptés sur nombre total de cycles dans l'analyse
        cycles (list): liste des cycles dans cette extention NN (not None seulement quand NN et single set)
        sub_cycles_NN (list[liste]) : liste des cycles par set dans ce NN comptés sur nombre total de cycles dans analyse
        NN_sub_cycles (list[liste]) : liste des cycles par set dans ce NN comptés sur nombre de cycles total dans NN

        shape (class) : class shape sur field

        nb_area_img (array) : nombre de regions par img
        S_a_img (1D array) :  tableau des nombre de maille appartenant a regions par img
        S_f_img (1D array) : tableau sum des valeurs de champs sur toutes les regions par img
        S_a (1D array) : taille en nombre de mailles de chaque region
        S_f (1D array) : sum des valeurs du champs sur les mailles d'une region, pour toutes les regions
        pict_S ( 1D array) : labeled array, chaque region d'une même img est labélisée par le numéro de l'img

    """

    # ---------------------------------------------------------#
    def __init__(self, config, f, seuil, save_seuil, signaltype, NN_data, fname, sep_posneg, saving_step=True):
        """
        The constructor for ImgEvent.

        Parameters:
            config (class) : config associée à la l'analyse

            f (1D array) : field
            seuil (int) : valeur utilisée pour seuiller les évents
            save_seuil (str) : matricule du seuil

            signaltype (str): 'flu', 'flu_rsc', 'rsc_flu_rsc', 'sequence' type du signal en force
            NN_data (str) : '', 'train', 'val', 'test' extension NN
            fname (str) : nom du champ
            sep_posneg (bol) : définie la façon de traité les régions positives et négative

            saving step (bol) : pêrmet de sauver
        """
        ## Config
        self.config = config

        self.seuil = seuil
        self.save_seuil = save_seuil

        self.NN_data = NN_data
        self.sep_posneg = sep_posneg
        self.signaltype, self.fname, self.savename, _, _ = def_names(signaltype, fname, NN_data)

        self.saving_step = saving_step

        self.path_signal = self.config.global_path_save + '/pict_event_' + self.signaltype + '/'
        self.to_save_fig = self.config.global_path_save + '/figure_pict_event_' + self.signaltype + '/'

        self.nbcycle, self.cycles, self.sub_cycles, self.sub_cycles_NN, self.NN_sub_cycles = def_nbcycle(self.config,
                                                                                                            self.path_signal,
                                                                                                            self.fname,
                                                                                                            self.NN_data)


        self.shape_f = Shape(f)

        ## regarde si le fichier existe dejà :
        if fname == 'vort' and self.sep_posneg:
            fileName = self.path_signal + 'S_f_' + self.fname + '_p_' + self.save_seuil + '.npy'
        else:
            fileName = self.path_signal + 'S_f_' + self.fname + '_' + self.save_seuil + '.npy'
        fileObj = Path(fileName)
        is_fileObj = fileObj.is_file()

        if is_fileObj:
            print('seuil déjà enregisté')
            if fname == 'vort' and self.sep_posneg:
                self.nb_area_img_p, self.nb_area_img_n, self.sum_S_a_p, self.sum_S_a_n, self.S_a_p, self.S_a_n, \
                self.sum_S_f_p, self.sum_S_f_n, self.S_f_p, self.S_f_n, \
                self.pict_S_p, self.pict_S_n = self.import_data(fname)
            else:
                self.nb_area_img, self.sum_S_a, self.S_a, self.sum_S_f, self.S_f, \
                self.pict_S = self.import_data(fname)
        else:
            print('seuil à traiter')
            info, info_p, info_n = self.reg_analyse(fname, f)

            if fname == 'vort' and self.sep_posneg:
                self.nb_area_img_p, self.sum_S_a_p, self.sum_S_f_p, self.S_a_p, self.S_f_p, \
                self.pict_S_p = self.stat_reg(info=info_p)
                self.nb_area_img_n, self.sum_S_a_n, self.sum_S_f_n, self.S_a_n, self.S_f_n, \
                self.pict_S_n = self.stat_reg(info=info_n)
            else:
                self.nb_area_img, self.sum_S_a, self.sum_S_f, self.S_a, self.S_f, \
                self.pict_S = self.stat_reg(info=info)

            ## save
            if self.saving_step:
                if fname == 'vort' and self.sep_posneg:
                    self.save_single(self.nb_area_img_p, 'nb_area_img_' + self.fname + '_p_' + self.save_seuil)
                    self.save_single(self.sum_S_a_p, 'sum_S_a_' + self.fname + '_p_' + self.save_seuil)
                    self.save_single(self.S_a_p, 'S_a_' + self.fname + '_p_' + self.save_seuil)
                    self.save_single(self.sum_S_f_p, 'sum_S_f_' + self.fname + '_p_' + self.save_seuil)
                    self.save_single(self.S_f_p, 'S_f_' + self.fname + '_p_' + self.save_seuil)
                    self.save_single(self.pict_S_p, 'pict_S_' + self.fname + '_p_' + self.save_seuil)

                    self.save_single(self.nb_area_img_n, 'nb_area_img_' + self.fname + '_n_' + self.save_seuil)
                    self.save_single(self.sum_S_a_n, 'sum_S_a_' + self.fname + '_n_' + self.save_seuil)
                    self.save_single(self.S_a_n, 'S_a_' + self.fname + '_n_' + self.save_seuil)
                    self.save_single(self.sum_S_f_n, 'sum_S_f_' + self.fname + '_n_' + self.save_seuil)
                    self.save_single(self.S_f_n, 'S_f_' + self.fname + '_n_' + self.save_seuil)
                    self.save_single(self.pict_S_n, 'pict_S_' + self.fname + '_n_' + self.save_seuil)

                else:
                    self.save_single(self.nb_area_img, 'nb_area_img_' + self.fname + '_' + self.save_seuil)
                    self.save_single(self.sum_S_a, 'sum_S_a_' + self.fname + '_' + self.save_seuil)
                    self.save_single(self.S_a, 'S_a_' + self.fname + '_' + self.save_seuil)
                    self.save_single(self.sum_S_f, 'sum_S_f_' + self.fname + '_' + self.save_seuil)
                    self.save_single(self.S_f, 'S_f_' + self.fname + '_' + self.save_seuil)
                    self.save_single(self.pict_S, 'pict_S_' + self.fname + '_' + self.save_seuil)

    # ------------------------------------------
    def import_single(self, name, extension='npy', size=None):

        if extension == 'npy':
            to_load = self.path_signal + name + '.npy'
            single = np.load(to_load)
        else:
            recup_single = Cell(self.path_signal + name, size)
            single = recup_single.reco_cell()

        return single

    # ------------------------------------------
    def save_single(self, data, name, extension='npy', nbfichier=None):

        if data is not None:
            if extension == 'npy':
                to_save = self.path_signal + name
                np.save(to_save, data)
            elif extension == 'cell':
                Cell(self.path_signal + name, nbfichier, data=data, extension='cell')
            else:
                Cell(self.path_signal + name, nbfichier, data=data, extension='csv')

    # ------------------------------------------
    def import_data(self, fname):
        if fname == 'vort' and self.sep_posneg:
            nb_area_img_p = self.import_single('nb_area_img_' + self.fname + '_p_' + self.save_seuil)
            sum_S_a_p = self.import_single('sum_S_a_' + self.fname + '_p_' + self.save_seuil)
            S_a_p = self.import_single('S_a_' + self.fname + '_p_' + self.save_seuil)
            sum_S_f_p = self.import_single('sum_S_f_' + self.fname + '_p_' + self.save_seuil)
            S_f_p = self.import_single('S_f_' + self.fname + '_p_' + self.save_seuil)
            pict_S_p = self.import_single('pict_S_' + self.fname + '_p_' + self.save_seuil)

            nb_area_img_n = self.import_single('nb_area_img_' + self.fname + '_n_' + self.save_seuil)
            sum_S_a_n = self.import_single('sum_S_a_' + self.fname + '_n_' + self.save_seuil)
            S_a_n = self.import_single('S_a_' + self.fname + '_n_' + self.save_seuil)
            sum_S_f_n = self.import_single('sum_S_f_' + self.fname + '_n_' + self.save_seuil)
            S_f_n = self.import_single('S_f_' + self.fname + '_n_' + self.save_seuil)
            pict_S_n = self.import_single('pict_S_' + self.fname + '_n_' + self.save_seuil)

            return nb_area_img_p, nb_area_img_n, sum_S_a_p, sum_S_a_n, S_a_p, S_a_n, sum_S_f_p, sum_S_f_n, S_f_p, S_f_n, \
                   pict_S_p, pict_S_n

        else:
            nb_area_img = self.import_single('nb_area_img_' + self.fname + '_' + self.save_seuil)
            sum_S_a = self.import_single('sum_S_a_' + self.fname + '_' + self.save_seuil)
            S_a = self.import_single('S_a_' + self.fname + '_' + self.save_seuil)
            sum_S_f = self.import_single('sum_S_f_' + self.fname + '_' + self.save_seuil)
            S_f = self.import_single('S_f_' + self.fname + '_' + self.save_seuil)
            pict_S = self.import_single('pict_S_' + self.fname + '_' + self.save_seuil)

            return nb_area_img, sum_S_a, S_a, sum_S_f, S_f, pict_S

    # ------------------------------------------
    def find_field_seuil(self, field, seuil):

        start_time = timeit.default_timer()

        field_seuil = np.zeros_like(field)

        for k in range(self.shape_f.nb_pict):

            for i in range(self.shape_f.size_w):
                for j in range(self.shape_f.size_c):
                    a = field[i, j, k]
                    if a > seuil:
                        field_seuil[i, j, k] = 1

        stop_time = timeit.default_timer()
        print('tps pour seuiler :', stop_time - start_time)

        return field_seuil

        # ------------------------------------------

    # ------------------------------------------
    def reg_analyse(self, fname, f):

        info = [0 for i in range(self.shape_f.nb_pict)]
        info_p = [0 for i in range(self.shape_f.nb_pict)]
        info_n = [0 for i in range(self.shape_f.nb_pict)]

        if fname == 'vort' and self.sep_posneg:
            field_seuil_p = self.find_field_seuil(f, self.seuil)
            field_seuil_n = self.find_field_seuil(-f, self.seuil)
        elif fname == 'vort' and not self.sep_posneg:
            field_seuil = self.find_field_seuil(np.abs(f), self.seuil)
        else:
            field_seuil = self.find_field_seuil(f, self.seuil)

        start_time = timeit.default_timer()

        for k in range(self.shape_f.nb_pict):
            v = f[:, :, k]

            if fname == 'vort' and self.sep_posneg:
                vs_p = field_seuil_p[:, :, k]
                vs_n = field_seuil_n[:, :, k]

                info_p[k] = InfoField(self.config, v, vs_p, self.seuil)
                info_n[k] = InfoField(self.config, -v, vs_n, self.seuil)
                info[k] = None
            elif fname == 'vort' and not self.sep_posneg:
                vs = field_seuil[:, :, k]

                info[k] = InfoField(self.config, np.abs(v), vs, self.seuil)
                info_p[k] = None
                info_n[k] = None
            else:
                vs = field_seuil[:, :, k]

                info[k] = InfoField(self.config, v, vs, self.seuil)
                info_p[k] = None
                info_n[k] = None

        stop_time = timeit.default_timer()
        print('tps pour seuilreg :', stop_time - start_time)

        return info, info_p, info_n

        # ------------------------------------------

    # ------------------------------------------
    def stat_reg(self, info):

        start_time = timeit.default_timer()

        nb_area_tot = 0

        for k in range(self.shape_f.nb_pict):
            nb_area_tot = nb_area_tot + info[k].nb_area

        nb_area_img = np.zeros(self.shape_f.nb_pict)
        sum_S_a = np.zeros(self.shape_f.nb_pict)
        sum_S_f = np.zeros(self.shape_f.nb_pict)
        S_a = np.zeros(nb_area_tot)
        S_f = np.zeros(nb_area_tot)
        pict_S = np.zeros(nb_area_tot)

        ## recup info par reg
        j = 0
        for k in range(self.shape_f.nb_pict):
            subinfo = info[k]

            nb_area_img[k] = subinfo.nb_area

            for i in range(subinfo.nb_area):
                S_a[j + i] = subinfo.size_area[i]
                S_f[j + i] = subinfo.sum_field[i]
                pict_S[j + i] = k

                if subinfo.sum_field[i] < self.seuil:
                    print('Proooooooooooooooobleeeeeeeeeeeeeme', i, k)

            j = j + subinfo.nb_area

            sum_S_a[k] = subinfo.size_area_img
            sum_S_f[k] = subinfo.sum_field_img

        stop_time = timeit.default_timer()
        print('tps pour stat_reg :', stop_time - start_time)

        return nb_area_img, sum_S_a, sum_S_f, S_a, S_f, pict_S
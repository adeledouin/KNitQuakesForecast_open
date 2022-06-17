import timeit
from pathlib import Path
import numpy as np

from Datas.classCell import Cell
from Datas.classStat import Stat, Histo, Shape

def def_names(signaltype, fname, NN_data):
    """
    Function to define folders and files names

    Parameters:
        signaltype (str) : 'flu', 'flu_rsc', 'rsc_flu_rsc', 'sequence'
        fname (str) : None if pas différent de signaltype
        NN_data (str) : '', 'train', 'val', 'test'

    Returns:
        output (str) : nom du dossier, nomd du fichier avec extension NN, extention NN
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

    return signaltype, fname, savename

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


# ------------------------------------------
class SignalForce():
    """
    Classe qui permet de charger signal en force et ses dépendances.

    Attributes:
        config (class) : config associée à la l'analyse

        signaltype (str): 'flu', 'flu_rsc', 'rsc_flu_rsc', 'sequence' nom du dossier
        fname (str) : nom du fichier - None if pas différent de signaltype
        NN_data (str) : '', 'train', 'val', 'test' extention NN
        savename (str) : '_' + extension NN
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
        index_picture (array) : index des img associées au signal
        number_picture (array) : numero des img associées au signal - si mix comptée par set
        number_picture (array ou None) : numero des img associées au signal comptée sur le total - not None si mix
        nb_index_picture (int ou array) : nombre total d'img associées au signal - si mix array des nombre d'img par set
        """

    # ---------------------------------------------------------#
    def __init__(self, config, signaltype, NN_data, fname=None):
        """
        The constructor for SignalForce.

        Parameters:
            config (class) : config associée à la l'analyse

            signaltype (str): 'flu', 'flu_rsc', 'rsc_flu_rsc', 'sequence' type du signal en force
            NN_data (str) : '', 'train', 'val', 'test' extension NN
            fname (str) : nom du signal - None if pas différent de signaltype

        """

        ## Config
        self.config = config

        self.NN_data = NN_data
        self.signaltype, self.fname, self.savename = def_names(signaltype, fname, NN_data)

        self.path_signal = self.config.global_path_load + self.signaltype + '/'

        self.nbcycle, self.cycles, self.sub_cycles, self.sub_cycles_NN, self.NN_sub_cycles = def_nbcycle(self.config,
                                                                                                            self.path_signal,
                                                                                                            self.fname,
                                                                                                            self.NN_data)

        ## events info
        self.f, self.t, self.ext, self.f_size, self.index_picture, \
        self.number_picture, self.numbertot_picture, self.nb_index_picture = self.import_data()

    # ------------------------------------------
    def import_single(self, name, extension='npy', size=None):

        if extension == 'npy':
            to_load = self.path_signal + name + '.npy'
            # print(to_load)
            single = np.load(to_load, allow_pickle=True)
        else:
            recup_single = Cell(self.path_signal + name, size)
            single = recup_single.reco_cell()

        return single

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
    def save_data(self, signaltype, NN_data, f, t, ext, f_size, index, number, numbertot, nb_index,
                  size, cycles, sub_cycles, sub_cycles_NN, NN_sub_cycles, fname=None):

        signaltype, fname, _ = def_names(signaltype, fname, NN_data)

        path_signal = self.config.global_path_save + signaltype + '/'

        self.save_single(path_signal, f, fname)
        self.save_single(path_signal, t, 't_' + fname)
        self.save_single(path_signal, ext, 'ext_' + fname)
        self.save_single(path_signal, f_size, + fname + '_size')

        if self.config.img:
            # print(np.shape(index))
            self.save_single(path_signal, index, 'index_picture_' + self.fname)
            self.save_single(path_signal, number, 'number_picture_' + self.fname)
            if self.config.mix_set:
                self.save_single(path_signal, numbertot, 'numbertot_picture_' + self.fname)
            self.save_single(path_signal, nb_index, 'nb_index_picture_' + self.fname)

        self.save_single(path_signal, size, fname + '_size')
        self.save_single(path_signal, cycles, fname + '_cycles')
        self.save_single(path_signal, sub_cycles_NN, fname + '_sub_cycles_NN', extension='cell',
                         nbfichier=self.config.nb_set)
        self.save_single(path_signal, NN_sub_cycles, fname + '_NN_sub_cycles', extension='cell',
                         nbfichier=self.config.nb_set)

    # ------------------------------------------
    def import_data(self):

        f = self.import_single(self.fname)
        t = self.import_single('t_' + self.fname)
        ext = self.import_single('ext_' + self.fname)
        # f_size = self.import_single(self.fname + '_size')
        f_size = None

        toload_numbertot_picture = self.path_signal + 'numbertot_picture_' + self.fname + '.npy'

        if self.config.img:
            index_picture = self.import_single('index_picture_' + self.fname)
            number_picture = self.import_single('number_picture_' + self.fname)
            nb_index_picture = self.import_single('nb_index_picture_' + self.fname)
            fileObj = Path(toload_numbertot_picture)
            is_fileObj = fileObj.is_file()
            if is_fileObj:
                numbertot_picture = np.load(toload_numbertot_picture)
            else:
                numbertot_picture = None

        if self.config.img:
            return f, t, ext, f_size, index_picture, number_picture, numbertot_picture, nb_index_picture
        else:
            return f, t, ext, f_size, None, None, None, None


# ------------------------------------------
class SignalImg():
    """
    Classe qui permet de charger les imgs et leurs dépendances.

    Attributes:
        config (class) : config associée à la l'analyse

        signaltype (str): 'flu', 'flu_rsc', 'rsc_flu_rsc', 'sequence' nom du dossier
        fname (str) : nom du fichier
        savename (str) : '_' + extension NN
        NN_data (str) : '', 'train', 'val', 'test'

        saving step (bol) : pêrmet de sauver les étapes dans cas de analyse signel set, si multi set alors ne sauve pas

        path_signal (str) : chemin du dossier associé à ce signal
        to_save_fig (str) : chemin associé pour save fig
        to_save_film (str) : chemin associé pour save film de field

        nbcycle (int) : nombre de cycles dans le signal
        sub_cycles (list[liste]) : liste des cycles par set comptés sur nombre total de cycles dans l'analyse
        cycles (list): liste des cycles dans cette extention NN (not None seulement quand NN et single set)
        sub_cycles_NN (list[liste]) : liste des cycles par set dans ce NN comptés sur nombre total de cycles dans analyse
        NN_sub_cycles (list[liste]) : liste des cycles par set dans ce NN comptés sur nombre de cycles total dans NN

        index_picture (array) : index des img associées au signal
        number_picture (array) : numero des img associées au signal - si mix comptée par set
        number_picture (array ou None) : numero des img associées au signal comptée sur le total - not None si mix
        nb_index_picture (int ou array) : nombre total d'img associées au signal - si mix array des nombre d'img par set

    """

    # ---------------------------------------------------------#
    def __init__(self, config, signaltype, NN_data, fname=None, fields=True, saving_step=True):
        """
        The constructor for SignalImg.

        Parameters:
            config (class) : config associée à la l'analyse

            signaltype (str): 'flu', 'flu_rsc', 'rsc_flu_rsc', 'sequence' type du signal en force
            NN_data (str) : '', 'train', 'val', 'test' extension NN
            fname (str) : nom du signal - None if pas différent de signaltype

            fields (bol) : est ce qu'on load en memoire tous les fields
            saving step (bol) : pêrmet de sauver les étapes dans cas de analyse signel set, si multi set alors ne sauve pas
        """

        ## Config
        self.config = config

        self.NN_data = NN_data
        self.signaltype, self.fname, self.savename = def_names(signaltype, fname, NN_data)

        self.saving_step = saving_step

        self.path_signal = self.config.global_path_load + '/pict_event_' + self.signaltype + '/'

        self.nbcycle, self.cycles, self.sub_cycles, self.sub_cycles_NN, self.NN_sub_cycles = def_nbcycle(self.config,
                                                                                                            self.path_signal,
                                                                                                            self.fname,
                                                                                                            self.NN_data)

        ## events info
        self.index_picture, self.number_picture, self.numbertot_picture, self.nb_index_picture = self.import_index()

        if fields:
            if not self.config.mix_set:
                self.vit_x, self.vit_y, self.vit_X, self.vit_Y, self.slip_x, self.slip_y, self.slip_X, self.slip_Y, \
                self.vort, self.dev, self.shear, self.div, self.posx, self.posy, self.posX, self.posY = self.import_fields()

                self.vit, self.slip = self.import_norm()

                self.abs_vort = np.abs(self.vort)

                self.sum_vit = self.sum_field(self.vit)
                self.sum_slip = self.sum_field(self.slip)
                self.sum_vort = self.sum_field(self.vort)
                self.sum_abs_vort = self.sum_field(self.abs_vort)
                self.sum_dev = self.sum_field(self.dev)

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
    def save_data(self, signaltype, NN_data, f, t, ext, f_size, index, number, numbertot, nb_index,
                  size, cycles, sub_cycles, sub_cycles_NN, NN_sub_cycles, fname=None):

        signaltype, fname, _ = def_names(signaltype, fname, NN_data)

        path_signal = self.config.global_path_save + '/pict_event_' + signaltype + '/'

        self.save_single(path_signal, f, fname)
        self.save_single(path_signal, t, 't_' + fname)
        self.save_single(path_signal, ext, 'ext_' + fname)
        self.save_single(path_signal, f_size, 'size_' + fname)

        if self.config.img:
            self.save_single(path_signal, index, 'index_picture_' + self.fname)
            self.save_single(path_signal, number, 'number_picture_' + self.fname)
            if self.config.mix_set:
                self.save_single(path_signal, numbertot, 'numbertot_picture_' + self.fname)
            self.save_single(path_signal, nb_index, 'nb_index_picture_' + self.fname)

        self.save_single(path_signal, size, fname + '_size')
        self.save_single(path_signal, cycles, fname + '_cycles')
        self.save_single(path_signal, sub_cycles_NN, fname + '_sub_cycles_NN', extension='cell',
                         nbfichier=self.config.nb_set)
        self.save_single(path_signal, NN_sub_cycles, fname + '_NN_sub_cycles', extension='cell',
                         nbfichier=self.config.nb_set)

    # ------------------------------------------
    def import_index(self):

        index_picture = self.import_single('index_picture_' + self.fname)
        number_picture = self.import_single('number_picture_' + self.fname)
        nb_index_picture = self.import_single('nb_index_picture_' + self.fname)
        if self.config.mix_set:
            numbertot_picture = self.import_single('numbertot_picture_' + self.fname)
        else:
            numbertot_picture = None

        return index_picture, number_picture, numbertot_picture, nb_index_picture

    # ------------------------------------------
    def import_fields(self):

        start_time = timeit.default_timer()

        vit_x = self.import_field('vit_x')
        vit_y = self.import_field('vit_y')
        vit_X = self.import_field('vit_x_XY')
        vit_Y = self.import_field('vit_y_XY')
        slip_x = self.import_field('slip_x')
        slip_y = self.import_field('slip_y')
        slip_X = self.import_field('slip_x_XY')
        slip_Y = self.import_field('slip_y_XY')
        vort = self.import_field('vort')
        dev = self.import_field('dev')
        shear = self.import_field('shear')
        div = self.import_field('div')
        posx = self.import_field('posx')
        posy = self.import_field('posy')
        posX = self.import_field('posx_XY')
        posY = self.import_field('posy_XY')

        stop_time = timeit.default_timer()
        print('tps pour importer :', stop_time - start_time)

        return vit_x, vit_y, vit_X, vit_Y, slip_x, slip_y, slip_X, slip_Y, vort, dev, shear, div, posx, posy, posX, posY

    # ------------------------------------------
    def import_field(self, name, num_set=None):
        if num_set is None:
            toload_field = self.path_signal + name + self.savename + '.npy'
        else:
            toload_field = self.path_signal + name + self.savename + '_{}'.format(num_set) + '.npy'

        field = np.load(toload_field)

        return field

    # ------------------------------------------
    def create_norm(self, name):

        fileName = self.path_signal + name + self.savename + '.npy'
        fileObj = Path(fileName)

        is_fileObj = fileObj.is_file()

        if not is_fileObj:
            field_x = self.import_field(name + '_x')
            field_y = self.import_field(name + '_y')
            field = self.norm(field_x, field_y)

            if self.saving_step:
                # print('start saving')
                tosave_field = self.path_signal + name + self.savename

                np.save(tosave_field, field)

    # ------------------------------------------
    def import_norm(self):
        fileName = self.path_signal + 'slip' + '.npy'
        fileObj = Path(fileName)

        is_fileObj = fileObj.is_file()

        if not is_fileObj:
            vit = self.norm(self.vit_x, self.vit_y)
            slip = self.norm(self.slip_x, self.slip_y)

            if self.saving_step:
                # print('start saving')
                tosave_vit = self.path_signal + 'vit'
                tosave_slip = self.path_signal + 'slip'

                np.save(tosave_vit, vit)
                np.save(tosave_slip, slip)
        else:
            toload_vit = self.path_signal + 'vit' + '.npy'
            toload_slip = self.path_signal + 'slip' + '.npy'

            vit = np.load(toload_vit)
            slip = np.load(toload_slip)

        return vit, slip

    # ------------------------------------------
    def norm(self, field_x, field_y):
        field_shape = Shape(field_x)

        norm = np.sqrt(field_x ** 2 + field_y ** 2)

        for i in range(field_shape.nb_pict):
            norm[:, :, i] = np.sqrt(field_x[:, :, i] ** 2 + field_y[:, :, i] ** 2)

        return norm

    # ------------------------------------------
    def sum_field(self, field, abs=False):

        shape_field = Shape(field)
        sum_field = np.zeros(shape_field.nb_pict)

        for i in range(shape_field.nb_pict):
            if abs:
                sum_field[i] = np.sum(np.abs(field[:, :, i]))
            else:
                sum_field[i] = np.sum(field[:, :, i])

        return sum_field

    # ------------------------------------------
    def sum_field_mix(self, name, abs=False, num_set=None):

        if abs and num_set is None:
            fileName = self.path_signal + 'sum_abs_' + name + self.savename + '.npy'
        elif abs and num_set is not None:
            fileName = self.path_signal + 'sum_abs_' + name + self.savename + '_{}'.format(num_set) + '.npy'
        elif not abs and num_set is None:
            fileName = self.path_signal + 'sum_' + name + self.savename + '.npy'
        else:
            fileName = self.path_signal + 'sum_' + name + self.savename + '_{}'.format(num_set) + '.npy'

        fileObj = Path(fileName)

        is_fileObj = fileObj.is_file()

        if not is_fileObj:
            field = self.import_field(name, num_set)

            shape_field = Shape(field)
            sum_field = np.zeros(shape_field.nb_pict)

            for i in range(shape_field.nb_pict):
                sum_field[i] = np.sum(field[:, :, i].reshape((shape_field.size_w * shape_field.size_c)))
        else:
            if abs:
                sum_field = self.import_field('sum_abs_' + name, num_set)
            else:
                sum_field = self.import_field('sum_' + name, num_set)

        return sum_field

    # ------------------------------------------
    def film_field(self, config_plot, plot, k_ini, k_fin):

        shape_vort = Shape(self.vort)

        for k in range(k_ini, k_fin + 1):
            where_new = np.where(self.number_picture == k)
            cycle_new = where_new[0][0]
            insidecycle_new = where_new[1][0]

            ### regardes les champs de vit et def

            vit_x = self.vit_x[1:-1, 2:-2, k]
            vit_y = self.vit_y[1:-1, 2:-2, k]
            vort = self.vort[:, :, k]
            lc = np.arange(shape_vort.size_c)
            lw = np.arange(shape_vort.size_w)
            X, Y = np.meshgrid(lc, lw)

            plot.plot_field(config_plot, vort, X, Y, vit_x, vit_y, k, 'vort_vit',
                            save=self.to_save_film + 'vit/', title='vort vit field')

            ### regardes les champs de slip et def
            slip_x = self.slip_x[:, :, k]
            slip_y = self.slip_y[:, :, k]
            vort = self.vort[:, :, k]
            lc = np.arange(shape_vort.size_c)
            lw = np.arange(shape_vort.size_w)
            X, Y = np.meshgrid(lc, lw)

            plot.plot_field(config_plot, vort, X, Y, slip_x, slip_y, k, 'vort_slip',
                            save=self.to_save_film + 'slip/', title='vort slip field')

    # ------------------------------------------
    def reshape_all_fields(self, names_tosave, new_signal, prev_fields, sub_NN=False):

        for i in range(np.size(names_tosave)):

            save = self.config.global_path_save + '/pict_event_' + new_signal.signaltype + '/'

            if not self.config.mix_set:
                fileName = save + names_tosave[i] + new_signal.savename + '.npy'
            else:
                fileName = save + names_tosave[i] + new_signal.savename + '_{}'.format(
                    self.config.nb_set - 1) + '.npy'
            fileObj = Path(fileName)
            is_fileObj = fileObj.is_file()


            if not is_fileObj:
                if not self.config.mix_set:
                    prev_field = prev_fields[i]
                    print(prev_field.shape)
                else:
                    prev_field = prev_fields
                print('reshape de {} has to be done'.format((names_tosave[i])))
                self.reshape_field(save, names_tosave[i], new_signal, prev_field, sub_NN)
            else:
                print('reshape de {} is alredy done'.format((names_tosave[i])))

    # ------------------------------------------
    def reshape_field(self, save, name_tosave, new_signal, prev_field, sub_NN):

        if not self.config.mix_set:

            field, number_picture, _, nb_pict_set = self.reshape(np.arange(new_signal.nbcycle),
                                                                 np.ones_like(new_signal.number_picture) * np.nan,
                                                                 prev_field,
                                                                 new_signal.index_picture,
                                                                 new_signal.number_picture)
            index_picture = new_signal.index_picture
            nb_index_picture = new_signal.nb_index_picture

            shape = Shape(field)

            print('verif reshape img dans set : nb_index_picture - nb_picture_set = {}'.format(
                nb_index_picture - nb_pict_set))

            if self.saving_step:
                self.save_single(save, field, name_tosave + new_signal.savename)
                self.save_single(save, np.array([shape.size_w, shape.size_c, shape.nb_pict]),
                                 name_tosave + new_signal.savename + '_shape')

        else:
            if sub_NN:
                print('on cherche {} pict dans 0 et {} dans 1'.format(new_signal.nb_index_picture,
                                                                      new_signal.nb_index_picture))
                number_picture, numbertot_picture = self.reshape_mix(save, new_signal.savename, name_tosave,
                                                                     new_signal.NN_sub_cycles,
                                                                     new_signal.index_picture,
                                                                     new_signal.number_picture,
                                                                     new_signal.nb_index_picture)
            else:
                number_picture, numbertot_picture = self.reshape_mix(save, new_signal.savename, name_tosave,
                                                                     self.config.NN_sub_cycles,
                                                                     new_signal.index_picture,
                                                                     new_signal.number_picture,
                                                                     new_signal.nb_index_picture)
            index_picture = new_signal.index_picture
            nb_index_picture = new_signal.nb_index_picture

            if self.saving_step:
                self.save_single(save, numbertot_picture, 'numbertot_picture_' + new_signal.fname)

        if self.saving_step:
            self.save_single(save, index_picture, 'index_picture_' + new_signal.fname)
            self.save_single(save, number_picture, 'number_picture_' + new_signal.fname)
            self.save_single(new_signal.path_signal, number_picture, 'new_number_picture_' + new_signal.fname)
            self.save_single(save, nb_index_picture, 'nb_index_picture_' + new_signal.fname)

    # ------------------------------------------
    def reshape_mix(self, save, savename, name_tosave, sub_cycles, index_picture, number_picture, nb_index):

        nb_pict_so_far = 0
        new_number_picture = np.ones_like(number_picture) * np.nan
        numbertot_picture = np.ones_like(number_picture) * np.nan

        # print_number(None, number_picture)
        for i in range(self.config.nb_set):

            field = self.import_field(name_tosave, num_set=i)
            # print('nb_pict_dans set', np.shape(field)[2])
            # print('nb index dans signalNN', nb_index[i])

            # print('sub_c size', np.size(sub_c))
            field, new_number_picture, numbertot_picture, nb_pict_set = self.reshape(sub_cycles[i], new_number_picture,
                                                                                     field,
                                                                                     index_picture, number_picture,
                                                                                     numbertot_picture,
                                                                                     nb_pict_so_far)

            nb_pict_so_far = nb_pict_so_far + nb_pict_set

            print('verif reshape img dans set : nb_index_picture - nb_picture_set = {}'.format(
                nb_index[i] - nb_pict_set))

            shape = Shape(field)

            ## save
            if self.saving_step:
                tosave_field = save + name_tosave + savename + '_{}'.format(i)
                tosave_shape_field = save + name_tosave + savename + '_shape' + '_{}'.format(i)

                np.save(tosave_field, field)
                np.save(tosave_shape_field, np.array([shape.size_w, shape.size_c, shape.nb_pict]))

        print('verif reshape img tot : nbtot_index_picture - nb_picture_son_far = {}'.format(
            np.sum(nb_index) - nb_pict_so_far))

        return new_number_picture, numbertot_picture

    # ------------------------------------------
    def reshape(self, sub_c, new_number_picture, field, index_picture, number_picture,
                numbertot_picture=None, nb_pict_so_far=None):

        start_time = timeit.default_timer()

        # print('subcycles', np.size(sub_c), sub_c)

        # print('shape field', np.shape(field))

        sub_index = index_picture[sub_c, :]
        sub_number = number_picture[sub_c, :]
        sub_nb_index = np.where(sub_index.reshape((sub_index.shape[0] * sub_index.shape[1])) == 1)[0].size

        # print_number(None, sub_number)
        # print('index size', np.shape(sub_index))
        # print('nb_index', sub_nb_index)

        which_pict = np.zeros(sub_nb_index, dtype=int)
        # print('size which pict', np.size(which_pict))

        k = 0
        for l in range(np.size(sub_c)):
            # print('cycle dans set ', l, 'nb_pict_ds_set_so_far', k)

            where_pict = np.where(sub_index[l, :] == 1)[0]
            # prin('taille where pict dans cycle ', which_pict[k:k + where_pict.size].size)
            # prin('taille number to keep dans cycle ', sub_number[l, where_pict].size)

            # print('cest la')
            which_pict[k:k + where_pict.size] = sub_number[l, where_pict]

            # prin('sur new_number : cycle = ', l, 'ini = ', k,
            #       'fin = ', k + where_pict.size)

            # print('c est ici')
            new_number = np.arange(k, k + where_pict.size)
            new_number_picture[sub_c[l], where_pict] = new_number

            if self.config.mix_set:
                # print('ah non la')
                new_numbertot = np.arange(nb_pict_so_far + k, nb_pict_so_far + k + where_pict.size)
                numbertot_picture[sub_c[l], where_pict] = new_numbertot
            else:
                numbertot_picture = None
            k = k + where_pict.size

        # print_number(None, new_number_picture)
        # print_number(None, numbertot_picture)
        # print('et ba non')
        # print(which_pict[0], which_pict[-1])
        # print(np.shape(field)[2])
        # print(np.max(which_pict))
        field = field[:, :, which_pict]
        nb_pict_set = k
        # print('j ai reussis?')
        return field, new_number_picture, numbertot_picture, nb_pict_set


# ------------------------------------------
class SignalSeq():
    '''  '''

    # ---------------------------------------------------------#
    def __init__(self, config, remote, signaltype, NN_data, display_figure=False,
                 saving_step=True):
        ''' signaltype : 'flu' ou 'flu_rsc'
            NN_data : '' ou 'train' ou 'val' ou 'test'
            nb_processes : nombre de coeurs à utiilisé pour le mutiprocessing
            display_figure : affiche les figures pour vérifier pas de pb dans l'analyse
            saving step : pêrmet de sauver les étapes dans cas de analyse signel set, si multi set alors ne sauve pas'''

        ## Config
        self.config = config
        self.remote = remote

        self.NN_data = NN_data
        self.signaltype = signaltype + '_NN'
        self.fname = signaltype + '_' + self.NN_data
        self.savename = '_' + self.NN_data

        self.nb_process = config.nb_process
        self.display_figure = display_figure
        self.saving_step = saving_step

        self.to_load = self.config.global_path_load + '/pict_event_' + self.signaltype + '/'
        self.to_save = self.config.global_path_load + '/pict_event_' + self.signaltype + '/'


# ------------------------------------------
class VariationsScalar():
    """
    Classe qui permet d'etudier les variations d'un signal scalair en fonction des cycles, tu tps, ou des deux.

    Attributes:
        config (class) : config associée à la l'analyse

        pourcentage (int) : pourcentage de la taille du signal en tps utilisé pour créer fenetre temporelle

        directsignal (bol) : vrai si signal est un signal de force, faux sinon - img, event...
        signaltype (str): 'flu', 'flu_rsc', 'rsc_flu_rsc', 'sequence' nom du dossier
        fname (str) : nom du signal
        savename (str) : nom du fichier
        NN_data (str) : '', 'train', 'val', 'test'
        ftype (str) : 'force' ou 'img'
        sep_pos_neg (bol) : est ce qu'on a separé les events positif et négatifs

        path_signal (str) : chemin du dossier associé à ce signal

        nbcycle (int) : nombre de cycles dans le signal
        sub_cycles (list[liste]) : liste des cycles par set comptés sur nombre total de cycles dans l'analyse
        cycles (list): liste des cycles dans cette extention NN (not None seulement quand NN et single set)
        sub_cycles_NN (list[liste]) : liste des cycles par set dans ce NN comptés sur nombre total de cycles dans analyse
        NN_sub_cycles (list[liste]) : liste des cycles par set dans ce NN comptés sur nombre de cycles total dans NN

        histo (class) : classe pour créer les histo

        f (array) : signal

        f_c (array) : signal reshape par cycle

        shape_f (class) : class shape sur signal
        ndim_to_1_dim (array) : signal reshape en 1D

        stats_f (class) : class stat sur signal

        stat_f_cycle (list(class)) : liste des class stat par cycle
        stats_f_cycle_tps (list(list(class))) : liste par cycle de liste des class stats sur fenetre de tps
        pente_stats_f_cycle_tps (array) : pentes par cycle de l'évolution de la moyenne et de la variance au cour du temps
        odrdonnée_stats_f_cycle_tps (array) : ordonnée à l'origine par cycle de l'évolution de la moyenne et de la variance au cour du temps
        stats_f_tps (list) : liste des class stat calculée sur fenetre de temps et tous les cycles

        """

    # ---------------------------------------------------------#
    def __init__(self, config, pourcentage, f, ext, t, index, number, directsignal, signaltype, NN_data, ftype,
                 fname, sep_posneg=False, pict_S=None, rsc=False, stats=False, multi_seuils=False):
        """
        The constructor for VariationsScalar.

        Parameters:
            config (class) : config associée à la l'analyse

            pourcentage (int) : pourcentage de la taille du signal en tps utilisé pour créer fenetre temporelle

            f (array) : signal
            ext (array) : extension associé au signal
            t (array) : tps associé au signal
            index (array) : index associé au signal
            number (array) : number associés au signal
            nb_index : nombre d'index dans le signal

            directsignal (bol) : vrai si signal est un signal de force, faux sinon - img, event...
            signaltype (str): 'flu', 'flu_rsc', 'rsc_flu_rsc', 'sequence' type du signal en force
            fname (str) : nom du signal
            NN_data (str) : '', 'train', 'val', 'test' extension NN
            ftype (str) : 'force' ou 'img'

            sep_pos_neg (bol) : est ce qu'on a separé les events positif et négatifs
            pict_S (array) :
            rsc (bol) : est ce qu'on va rescaler des données
            stats (bol) : est ce qu'o va regarder les stats des données
            multi_seuils (bol) : garde signal en memoir si analyse sur multi seuil
        """

        ## Config
        self.config = config

        self.pourcentage = pourcentage

        self.directsignal = directsignal
        self.ftype = ftype
        self.fname = fname
        self.NN_data = NN_data
        self.sep_posneg = sep_posneg

        self.signaltype, self.savename, _ = def_names(signaltype, None, NN_data)

        self.path_signal = self.config.global_path_save + self.signaltype + '/'

        self.nbcycle, self.cycles, self.sub_cycles, self.sub_cycles_NN, self.NN_sub_cycles = def_nbcycle(self.config,
                                                                                                         self.path_signal,
                                                                                                         self.savename,
                                                                                                         self.NN_data)

        self.histo = Histo(self.config)

        if multi_seuils:
            self.f = f

        if not directsignal:
            self.f_c = self.reshape_c(f, index, number, pict_S)

        self.shape_f = Shape(f)
        if self.shape_f.dim != 1:
            self.ndim_to_1dim = self.shape_f.ndim_to_1dim(f)

        print('stats f')
        if self.shape_f.dim == 1:
            self.stats_f = Stat(self.config, f)
        else:
            self.stats_f = Stat(self.config, self.ndim_to_1dim)

        if stats:
            print('stats f cycle')
            self.stats_f_cycle = self.stats_par_cycle(f)
            print('stats f cycle tps')
            self.stats_f_cycle_tps = self.stats_par_cycle_par_fenetre_temps(f, t)

            print('pente stats f cycle tps')
            self.pente_stats_f_cycle_tps, self.ordonnee_stats_f_cycle_tps = self.coeff_stats_par_cycle_par_fenetre_temps()

        if rsc or stats:
            print('stats f tps')
            self.stats_f_tps = self.stats_par_fenetre_temps(f, t)

    # ------------------------------------------
    def reshape_c(self, f, index, number, pict_S):

        f_c = [0 for i in range(self.nbcycle)]

        for i in range(self.nbcycle):
            if self.fname == 'S_a' or self.fname == 'S_f':
                where_pict = np.where(index[i, :] == 1)[0]
                which_pict = number[i, where_pict].astype(int)
                sub_f_c = []
                for j in range(np.size(which_pict)):
                    where = np.where(pict_S == which_pict[j])
                    sub_f_c = np.concatenate((sub_f_c, f[where]))
                f_c[i] = sub_f_c
            else:
                where = np.where(index[i, :] == 1)[0]
                which = number[i, where].astype(int)
                f_c[i] = f[which]

        return f_c

    # ------------------------------------------
    def coeff_stats_par_cycle_par_fenetre_temps(self):

        a = np.zeros((2, self.nbcycle))
        b = np.zeros((2, self.nbcycle))

        for i in range(self.nbcycle):
            stat = self.stats_f_cycle_tps[i]
            liny_mean = [stat[j].mean for j in range(np.size(stat))]
            liny_var = [stat[j].var for j in range(np.size(stat))]
            linx = np.arange(np.size(stat))

            coef_distri_mean, x_mean, y_mean = self.histo.regression(linx, liny_mean, np.min(linx), np.max(linx),
                                                                     x_axis='lin', y_axis='lin')
            coef_distri_var, x_var, y_var = self.histo.regression(linx, liny_var, np.min(linx), np.max(linx),
                                                                  x_axis='lin', y_axis='lin')

            a[0, i] = coef_distri_mean[0]
            b[0, i] = coef_distri_mean[1]

            a[1, i] = coef_distri_var[0]
            b[1, i] = coef_distri_var[1]

        return a, b

    # ------------------------------------------
    def stats_par_cycle_par_fenetre_temps(self, f, t):

        stats_f_cycle_tps = [0 for i in range(self.nbcycle)]

        t_size = np.size(t[0, :])
        window = int(np.round(self.pourcentage / 1000 * t_size) * 10)

        if self.directsignal:

            new_t_size = int(t_size - window)
            # print('{} % du signal représente {} points'.format(self.pourcentage, new_t_size))

            for i in range(self.nbcycle):
                stats_f_tps = [0 for k in range(new_t_size)]

                for j in range(new_t_size):
                    sub_f = f[i, j:j + window]
                    stats_f_tps[j] = Stat(self.config, sub_f)

                stats_f_cycle_tps[i] = stats_f_tps
        elif not self.directsignal and self.ftype == 'force':
            for i in range(self.nbcycle):
                window = int(np.round(self.pourcentage / 1000 * np.size(self.f_c[i])) * 10)
                new_t_size = int(np.size(self.f_c[i]) - window)
                # print('{} % des events représente {} evenements sur cycle {}'.format(self.pourcentage, new_t_size, i))
                stats_f_tps = [0 for i in range(new_t_size)]

                for j in range(new_t_size):
                    stats_f_tps[j] = Stat(self.config, self.f_c[i][j: j + window])

                stats_f_cycle_tps[i] = stats_f_tps
        else:
            for i in range(self.nbcycle):
                delta_t_pict = self.config.dict_exp['delta_t_pict']
                tps_window = window / self.config.fr
                window_img = int(np.round(tps_window / delta_t_pict))
                # print('{} % du signal représente en moyenne {} images'.format(self.pourcentage, window_img))
                new_t_size = int(np.size(self.f_c[i]) - window_img)
                stats_f_tps = [0 for i in range(new_t_size)]

                for j in range(new_t_size):
                    stats_f_tps[j] = Stat(self.config, self.f_c[i][j: j + window_img])

                stats_f_cycle_tps[i] = stats_f_tps

        return stats_f_cycle_tps

    # ------------------------------------------
    def stats_par_cycle(self, f):

        stats_f_cycle = [0 for i in range(self.nbcycle)]

        if self.directsignal:
            for i in range(self.nbcycle):
                stats_f_cycle[i] = Stat(self.config, f[i, :])

        else:
            for i in range(self.nbcycle):
                stats_f_cycle[i] = Stat(self.config, self.f_c[i])

        return stats_f_cycle

    # ------------------------------------------
    def stats_par_fenetre_temps(self, f, t):

        t_size = np.size(t[0, :])
        window = int(np.round(self.pourcentage / 1000 * t_size) * 10)

        if self.directsignal:
            new_t_size = int(t_size - window)
            stats_f_tps = [0 for i in range(new_t_size)]
            # print('{} % du signal représente {} points'.format(self.pourcentage, new_t_size))

            for i in range(new_t_size):
                sub_f = f[:, i:i + window]
                shape_sub_f = Shape(sub_f)
                stats_f_tps[i] = Stat(self.config, shape_sub_f.ndim_to_1dim(sub_f))
        elif not self.directsignal and self.ftype == 'force':
            stats_f_tps = None
        else:
            delta_t_pict = self.config.dict_exp['delta_t_pict']
            tps_window = window / self.config.fr
            window_img = int(np.round(tps_window / delta_t_pict))
            # print('{} % du signal représente en moyenne {} images'.format(self.pourcentage, window_img))

            new_t_size = int(t_size - window)
            for i in range(self.nbcycle):
                if int(np.size(self.f_c[i]) - window_img) < new_t_size:
                    new_t_size = int(np.size(self.f_c[i]) - window_img)

            stats_f_tps = [0 for i in range(new_t_size)]

            for i in range(new_t_size):
                sub_f = np.zeros((self.nbcycle, window_img))
                for j in range(self.nbcycle):
                    sub_f[j, :] = self.f_c[j][i:i + window_img]

                stats_f_tps[i] = Stat(self.config, sub_f.reshape(self.nbcycle * window_img))

        return stats_f_tps

    # ------------------------------------------
    def size_rsc_array(self, t):
        t_size = np.size(t)
        window = int(np.round(self.pourcentage / 1000 * t_size) * 10)
        new_t_size = int(t_size - window)

        return new_t_size

    # ------------------------------------------
    def rsc_par_fenetre_tps(self, f, ext, t, index_picture, number_picture, numbertot_picture=None, mean=True,
                            var=True):

        # print('number', number_picture.shape)
        # print_number(None, number=number_picture)
        # print('numbertot', numbertot_picture.shape)
        # print_number(None, number=numbertot_picture)

        nb_c = np.shape(f)[0]
        t_size = np.size(t[0, :])
        window = int(np.round(self.pourcentage / 1000 * t_size) * 10)

        new_t_size = int(t_size - window)
        # print('{} % du signal représente {} points'.format(self.pourcentage, new_t_size))

        newstart = int(np.round(window / 2, 0))

        f_rsc = np.zeros((nb_c, new_t_size))
        ext_rsc = np.zeros((nb_c, new_t_size))
        t_rsc = np.zeros((nb_c, new_t_size))
        index_picture_rsc = np.zeros((nb_c, new_t_size))
        number_picture_rsc = np.zeros((nb_c, new_t_size))
        numbertot_picture_rsc = np.zeros((nb_c, new_t_size))

        for i in range(new_t_size):
            if mean and var:
                f_rsc[:, i] = (f[:, i + newstart] - self.stats_f_tps[i].mean) / np.sqrt(self.stats_f_tps[i].var)
            elif not mean and var:
                f_rsc[:, i] = f[:, i + newstart] / np.sqrt(self.stats_f_tps[i].var)
            ext_rsc[:, i] = ext[:, i + newstart]
            t_rsc[:, i] = t[:, i + newstart]
            if self.config.img:
                index_picture_rsc[:, i] = index_picture[:, i + newstart]
                number_picture_rsc[:, i] = number_picture[:, i + newstart]
                if self.config.mix_set:
                    numbertot_picture_rsc[:, i] = numbertot_picture[:, i + newstart]
            else:
                index_picture_rsc = None
                number_picture_rsc = None
                numbertot_picture_rsc = None

        if self.config.img:
            nb_index_picture_rsc = 0
            for i in range(nb_c):
                nb_index_picture_rsc = nb_index_picture_rsc + np.sum(index_picture_rsc[i, :])
                # print('nombre de picture dans rsc :', nb_index_picture_rsc)
        else:
            nb_index_picture_rsc = None

        # print(f_rsc.shape)
        # print('number', number_picture_rsc.shape)
        # print_number(None, number=number_picture_rsc)
        # print('numbertot', numbertot_picture_rsc.shape)
        # print_number(None, number=numbertot_picture_rsc)
        return f_rsc, ext_rsc, t_rsc, new_t_size, index_picture_rsc, number_picture_rsc, numbertot_picture_rsc, nb_index_picture_rsc


# ------------------------------------------
class VariationsField():
    """
    Classe qui permet d'etudier les variations d'un signal champ en fonction des cycles.

    Attributes:
        config (class) : config associée à la l'analyse

        signaltype (str): 'flu', 'flu_rsc', 'rsc_flu_rsc', 'sequence' nom du dossier
        fname (str) : nom du champs
        savename (str) : nom du fichier
        NN_data (str) : '', 'train', 'val', 'test'

        path_signal (str) : chemin du dossier associé à ce signal

        nbcycle (int) : nombre de cycles dans le signal
        sub_cycles (list[liste]) : liste des cycles par set comptés sur nombre total de cycles dans l'analyse
        cycles (list): liste des cycles dans cette extention NN (not None seulement quand NN et single set)
        sub_cycles_NN (list[liste]) : liste des cycles par set dans ce NN comptés sur nombre total de cycles dans analyse
        NN_sub_cycles (list[liste]) : liste des cycles par set dans ce NN comptés sur nombre de cycles total dans NN

        histo (class) : classe pour créer les histo

        f_c (array) : signal reshape par cycle

        shape_f (class) : class shape sur signal

        stats_f (class) : class stat sur signal

        stat_f_cycle (list(class)) : liste des class stat par cycle

    """

    # ---------------------------------------------------------#
    def __init__(self, config, f, ext, t, index, number, signaltype, NN_data, fname, set,
                 rsc=False, stats=False):
        """
        The constructor for VariationsField.

        Parameters:
            config (class) : config associée à la l'analyse

            f (array) : signal
            ext (array) : extension associé au signal
            t (array) : tps associé au signal
            index (array) : index associé au signal
            number (array) : number associés au signal
            nb_index : nombre d'index dans le signal

            signaltype (str): 'flu', 'flu_rsc', 'rsc_flu_rsc', 'sequence' type du signal en force
            fname (str) : nom du champs
            NN_data (str) : '', 'train', 'val', 'test' extension NN

            rsc (bol) : est ce qu'on va rescaler des données
            stats (bol) : est ce qu'o va regarder les stats des données
            """

        ## Config
        self.config = config

        self.fname = fname
        self.NN_data = NN_data

        self.signaltype, self.savename, _ = def_names(signaltype, None, NN_data)

        self.path_signal = self.config.global_path_load + self.signaltype + '/'

        nbcycle, self.cycles, sub_cycles, self.sub_cycles_NN, NN_sub_cycles = def_nbcycle(self.config,
                                                                                                         self.path_signal,
                                                                                                         self.savename,
                                                                                                         self.NN_data)
        self.NN_sub_cycles = NN_sub_cycles[set]
        self.sub_cycles = sub_cycles[set]
        self.nbcycle = np.size(self.NN_sub_cycles)

        print(self.nbcycle, self.NN_sub_cycles)
        print(index.shape, number.shape)
        self.histo = Histo(self.config)

        self.f_c = self.reshape_c(f, index, number)

        self.shape_f = Shape(f)

        print('stats f')
        if self.shape_f.dim == 1:
            self.stats_f = Stat(self.config, f)
        else:
            self.stats_f = Stat(self.config, self.shape_f.ndim_to_1dim(f))

        if stats or rsc:
            self.stats_f_cycle = self.stats_par_cycle()

        # ------------------------------------------

    # ------------------------------------------
    def reshape_c(self, f, index, number):

        f_c = [0 for i in range(self.nbcycle)]

        print(index.shape, number.shape)
        for i in range(self.nbcycle):
            where = np.where(index[i, :] == 1)[0]
            which = number[i, where].astype(int)
            f_c[i] = f[:, :, which]

        return f_c

    # ------------------------------------------
    def stats_par_cycle(self):

        stats_f_cycle = [0 for i in range(self.nbcycle)]

        for i in range(self.nbcycle):
            shape = Shape(self.f_c[i])
            stats_f_cycle[i] = Stat(self.config, shape.ndim_to_1dim(self.f_c[i]))

        return stats_f_cycle

    # ------------------------------------------
    def rsc(self, field, mean=True, var=True):

        if mean and var:
            field_rsc = (field - self.stats_f.mean) / np.sqrt(self.stats_f.var)
        elif not mean and var:
            field_rsc = field / np.sqrt(self.stats_f.var)

        return field_rsc
import numpy as np
import math


def print_spec(config_pred, pred, num_test):
    save_dict_version = config_pred.global_path_save + 'net/{}/config_pred_{}.npy'.format(pred, num_test)
    save_dict_model = config_pred.global_path_save + 'net/{}/model_{}.npy'.format(pred, num_test)

    dict_version = np.load(save_dict_version, allow_pickle=True).flat[0]
    dict_model = np.load(save_dict_model, allow_pickle=True).flat[0]

    print('--- dict version ---')
    print('ref: {} | version work: {}'.format(dict_version['ref'], dict_version['version_work']))
    print('false data: {} | input data: {}'.format(dict_version['false_data'], dict_version['input_data']))
    print('NN data: {} | fields: {}'.format(dict_version['NN_data'], dict_version['fields']))
    print('output type: {} | label : {}'.format(dict_version['output_type'], dict_version['label_save']))
    print('shuffle: {}'.format(dict_version['shuffle']))
    if dict_version['output_type'] == 'class':
        print('channel: {} | nb equipro_classes: {} | futur: {}'.format(dict_version['channel'],
                                                                        dict_version['nb_classes'],
                                                                        dict_version['futur']))
    else:
        print('channel: {} | futur pred: {}'.format(dict_version['channel'], dict_version['futur']))
    print('seq size: {} | overlap step : {}'.format(dict_version['seq_size'], dict_version['overlap_step']))
    print('batch size: {} '.format(dict_version['batch_size']))

    print('--- model spec ---')
    print('model name: {} | magic number {}'.format(dict_model['model_name'], dict_model['magic_number']))
    print('balance dataset: {}'.format(dict_model['balance_dataset']))
    print('layer param: {}'.format(dict_model['layers_param']))
    print('optimizer param: {}'.format(dict_model['opti_param']))
    print('criterion param: {}'.format(dict_model['criterion_param']))


class ConfigData():
    ''' def paramètres de la manip
    et paramètres de l'analyse '''

    # ---------------------------------------------------------#
    def __init__(self, path_from_root, dictionnaire_param_exp):
        '''param ref : définit le set auquel appartient la manip
        param mix : est ce que mix set ?
        param num_set : nombre de set
         param date : date de la manip
         param nexp : numero de test de prestat
         param version : version du code
         param globa_path : chemin global pour atteindre le fichier d'analyse
         param v : vitesse de traction en mm/s
         param Lw_0 : extension initiale
         param Lw_max : extension maximale
         param fr : frequence d'acquisition de l'instron
         param prescycle : nombre de pres cycles
         param mincycle : numero du premier cycle à garder
         param maxcycle : numero du dernier cycle à garder
         param nbcycle : nombre de cycles à garder
         '''

        self.path_from_root = path_from_root
        self.dict_exp = dictionnaire_param_exp

        self.mix_set = self.dict_exp['mix_set']
        self.nb_set = self.dict_exp['nb_set']
        self.img = self.dict_exp['img']

        self.ref = self.dict_exp['ref']
        self.version_raw = self.dict_exp['version_raw']
        self.version_work = self.dict_exp['version_work']

        self.v = self.dict_exp['vitesse']
        self.Lw_0 = self.dict_exp['Lw_0']
        self.Lw_i = self.dict_exp['Lw_i']
        self.Lw_max = self.dict_exp['Lw_max']
        self.fr = self.dict_exp['fr_instron']  # Hz => dt = 0.04s
        self.sursample = self.dict_exp['sursample']
        self.reso_instron = 0.00015  # mm

        self.prescycle = self.dict_exp['prescycle']
        self.mincycle = self.dict_exp['mincycle']
        self.maxcycle = self.dict_exp['maxcycle']
        self.nbcycle = np.asarray(self.maxcycle) - np.asarray(self.mincycle) + 1

        if self.mix_set:
            self.sub_cycles = [0 for i in range(self.nb_set)]
            k = 0
            cycles_tot = np.arange(np.sum(self.nbcycle))
            for i in range(self.nb_set):
                self.sub_cycles[i] = cycles_tot[k: k + self.nbcycle[i]]
                k = k + self.nbcycle[i]

        self.nb_process = 15

        self.global_path_load = '/{}/KnitAnalyse/'.format(path_from_root) + self.ref + \
                                '/mix' + '/version%d/' % (self.version_raw)
        self.global_path_save = '/{}/KnitQuakesForecast/'.format(path_from_root) + self.ref + \
                                '/input/version%d/' % (self.version_work)

        if self.img:
            self.fields = self.dict_exp['fields']
        else:
            self.fields = None
    # ------------------------------------------
    def config_scalarevent(self, signaltype):
        '''config pour class scalarevents_brut
        param exposants : exposants ddes seuils utilisés dans le plot pdf des events
        param seuils : seuil utilisés dans le plot pdf events
        param nb_seuils : nb de seuls testé dans le plot pdf events
        param wich_seuil : index du seuil à utilisé pour la stat des evenements
        param : nbclasses : nombre de equipro_classes quand plot les stats des events'''

        exposants = self.dict_exp['config_scalarevent_flu_exposants']
        seuils = self.dict_exp['config_scalarevent_flu_seuils']
        save_seuils = self.dict_exp['config_scalarevent_flu_save_seuils']
        nb_seuils = self.dict_exp['config_scalarevent_flu_nb_seuils']
        which_seuil = self.dict_exp['config_scalarevent_flu_which_seuil']
        nbclasses = self.dict_exp['config_scalarevent_flu_nbclasses']

        return exposants, seuils, save_seuils, nb_seuils, which_seuil, nbclasses


class ConfigPred():
    ''' def paramètres de la manip
    et paramètres de l'analyse '''

    # ---------------------------------------------------------#
    def __init__(self, dictionnaire_param_exp, config_data):
        '''param ref : définit le set auquel appartient la manip
         '''

        self.dict_exp = dictionnaire_param_exp

        self.ref = self.dict_exp['ref']
        self.version_work = config_data.version_work

        self.mix_set = config_data.mix_set
        self.nb_set = config_data.nb_set
        self.mincycle = config_data.mincycle
        self.maxcycle = config_data.maxcycle
        self.nbcycle = np.asarray(self.maxcycle) - np.asarray(self.mincycle) + 1

        if self.mix_set:
            self.sub_cycles = [0 for i in range(self.nb_set)]
            k = 0
            cycles_tot = np.arange(np.sum(self.nbcycle))
            for i in range(self.nb_set):
                self.sub_cycles[i] = cycles_tot[k: k + self.nbcycle[i]]
                k = k + self.nbcycle[i]

        self.nb_process = 20

        self.global_path_load = '/{}/KnitQuakesForecast/'.format(config_data.path_from_root) + self.ref + \
                                '/input/version%d/' % (self.version_work)
        self.global_path_save = '/{}/KnitQuakesForecast/'.format(config_data.path_from_root) + self.ref + \
                                '/output/version%d/' % (self.version_work)

        # info on pred
        self.input_data = self.dict_exp['input_data']
        self.output_type = self.dict_exp['output_type']

        # info on data
        self.false_data = self.dict_exp['false_data']
        self.NN_data = self.dict_exp['NN_data']
        self.fields = self.dict_exp['fields']
        self.img = config_data.img
        self.label_name = self.dict_exp['label_name']
        self.label_save = self.dict_exp['label_save']

        self.channel = int(self.dict_exp['channel'])
        self.seq_size = int(self.dict_exp['seq_size'])
        self.overlap_step = int(self.dict_exp['overlap_step'])
        self.futur = int(self.dict_exp['futur'])
        if self.output_type == 'img_vort':
            self.output_shape = int(self.dict_exp['out_channel'])
            self.tanh_std = (self.dict_exp['tanh_std'])
            self.tanh_seuil = (self.dict_exp['tanh_seuil'])
        elif self.output_type == 'digit_img_vort':
            self.output_shape = int(self.dict_exp['out_channel'])
            self.tanh_std = (self.dict_exp['tanh_std'])
            self.tanh_seuil = (self.dict_exp['tanh_seuil'])
        elif self.output_type == 'class':
            self.equipro = (self.dict_exp['equipro'])
            self.output_shape = int(self.dict_exp['nb_classes'])
            if self.label_save == 'sum_expdf':
                self.tau = int(self.dict_exp['tau'])
            elif self.label_save == 'sum_gammadf':
                self.k = int(self.dict_exp['k'])
                self.theta = self.dict_exp['theta']
            elif self.label_save == 'sum_gaussdf':
                self.mu = int(self.dict_exp['mu'])
                self.sigma = int(self.dict_exp['sigma'])
                self.label_save = '{}_sigma{}_mu{}'.format(self.label_save, self.sigma, self.mu)
        else:
            self.tau = int(self.dict_exp['tau'])
            self.label_rsc = self.dict_exp['label_rsc']
            self.output_shape = int(self.dict_exp['pred_size'])

        # info on training
        self.batch_size = int(self.dict_exp['batch_size'])
        self.shuffle = self.dict_exp['shuffle']
        self.criterion = self.dict_exp['criterion']
        self.model = None

    def set_model_attribute(self, model):
        self.model = model

    def set_weight_classes(self, weight):
        self.criterion['weight'] = weight

    def set_output_shape(self, new_output_shape):
        self.output_shape = new_output_shape

    def spec_model_scalar(self, ):
        '''une liste de modèles prédéfinis et leur identifiant unique (magic_number)'''

        # ------------- dense
        model0 = {'model_name': 'dense1d',
                  'balance_dataset': True, 'magic_number': 1000,
                  'layers_param': {'dropout': 0.25,
                                   'batch_norm': True,
                                   'dense1': 1024,
                                   'dense2': 512},
                  'opti_param': {'name_opti': 'adam', 'lr': 0.005, 'weight_decay': 0.001,
                                 'scheduler': 'stepLR', 'stepsize': 50, 'gamma': 0.5},
                  'criterion_param': self.criterion}

        model1 = {'model_name': 'dense1d',
                  'balance_dataset': True, 'magic_number': 1001,
                  'layers_param': {'dropout': 0.25,
                                   'batch_norm': True,
                                   'dense1': 1024,
                                   'dense2': 512},
                  'opti_param': {'name_opti': 'adam', 'lr': 0.01, 'weight_decay': 0.001,
                                 'scheduler': None},
                  'criterion_param': self.criterion}

        model2 = {'model_name': 'dense1d',
                  'balance_dataset': True, 'magic_number': 1002,
                  'layers_param': {'dropout': 0.25,
                                   'batch_norm': True,
                                   'dense1': 16,
                                   'dense2': 32},
                  'opti_param': {'name_opti': 'adam', 'lr': 0.01, 'weight_decay': 0.0001,
                                 'scheduler': 'stepLR', 'stepsize': 50, 'gamma': 0.5},
                  'criterion_param': self.criterion}

        model3 = {'model_name': 'dense1d',
                  'balance_dataset': True, 'magic_number': 1003,
                  'layers_param': {'dropout': 0.25,
                                   'batch_norm': True,
                                   'dense1': 4,
                                   'dense2': 8},
                  'opti_param': {'name_opti': 'adam', 'lr': 0.01, 'weight_decay': 0.0001,
                                 'scheduler': 'stepLR', 'stepsize': 50, 'gamma': 0.5},
                  'criterion_param': self.criterion}

        # ------------- conv into dense
        model4 = {'model_name': 'conv1d_into_dense',
                  'balance_dataset': True, 'magic_number': 2004,
                  'layers_param': {'dropout': 0.25,
                                   'batch_norm': True,
                                   'cnn1': 16, 'cnn2': 32, 'cnn3': 64,
                                   'kernel_1': 7, 'kernel_2': 3, 'kernel_3': 3,
                                   'maxpool_1': 2, 'maxpool_2': 2, 'maxpool_3': 2,
                                   'dense1': 512,
                                   'dense2': 512},
                  'opti_param': {'name_opti': 'adam', 'lr': 0.05, 'weight_decay': 0.0001,
                                 'scheduler': None},
                  'criterion_param': self.criterion}

        model5 = {'model_name': 'conv1d_into_dense',
                  'balance_dataset': True, 'magic_number': 2005,
                  'layers_param': {'dropout': 0.25,
                                   'batch_norm': True,
                                   'cnn1': 16, 'cnn2': 32, 'cnn3': 64,
                                   'kernel_1': 7, 'kernel_2': 3, 'kernel_3': 3,
                                   'maxpool_1': 2, 'maxpool_2': 2, 'maxpool_3': 2,
                                   'dense1': 512,
                                   'dense2': 512},
                  'opti_param': {'name_opti': 'adadelta', 'lr': 0.005,  'weight_decay': 0.001,
                                 'scheduler': None},
                  'criterion_param': self.criterion}

        model6 = {'model_name': 'conv1d_into_dense',
                  'balance_dataset': True, 'magic_number': 2006,
                  'layers_param': {'dropout': 0.25,
                                   'batch_norm': True,
                                   'cnn1': 16, 'cnn2': 32, 'cnn3': 64,
                                   'kernel_1': 7, 'kernel_2': 3, 'kernel_3': 3,
                                   'maxpool_1': 2, 'maxpool_2': 2, 'maxpool_3': 2,
                                   'dense1': 512,
                                   'dense2': 512},
                  'opti_param': {'name_opti': 'adadelta', 'lr': 0.01, 'weight_decay': 0.1,
                                 'scheduler': None},
                  'criterion_param': self.criterion}

        # lr scheduler
        model7 = {'model_name': 'conv1d_into_dense',
                  'balance_dataset': True, 'magic_number': 2007,
                  'layers_param': {'dropout': 0.25,
                                   'batch_norm': True,
                                   'cnn1': 16, 'cnn2': 32, 'cnn3': 64,
                                   'kernel_1': 7, 'kernel_2': 3, 'kernel_3': 3,
                                   'maxpool_1': 2, 'maxpool_2': 2, 'maxpool_3': 2,
                                   'dense1': 512,
                                   'dense2': 512},
                  'opti_param': {'name_opti': 'adadelta', 'lr': 0.01, 'weight_decay': 0.1,
                                 'scheduler': 'multistepLR', 'gamma': 0.5, 'milestones': [30, 60, 90, 120, 150]},
                  'criterion_param': self.criterion}

        model8 = {'model_name': 'conv1d_into_dense',
                  'balance_dataset': True, 'magic_number': 2008,
                  'layers_param': {'dropout': 0.25,
                                   'batch_norm': True,
                                   'cnn1': 16, 'cnn2': 32, 'cnn3': 64,
                                   'kernel_1': 7, 'kernel_2': 3, 'kernel_3': 3,
                                   'maxpool_1': 2, 'maxpool_2': 2, 'maxpool_3': 2,
                                   'dense1': 512,
                                   'dense2': 512},
                  'opti_param': {'name_opti': 'adadelta', 'lr': 0.01, 'weight_decay': 0.01,
                                 'scheduler': 'multistepLR', 'gamma': 0.5, 'milestones': [30, 80, 120, 140]},
                  'criterion_param': self.criterion}

        # ------------- conv into lstm
        model9 = {'model_name': 'conv1d_into_lstm',
                  'balance_dataset': True, 'magic_number': 3009,
                  'layers_param': {'dropout': 0.25,
                                   'batch_norm': True,
                                   'cnn1': 512, 'cnn2': None, 'cnn3': None,
                                   'kernel_1': 7, 'kernel_2': 3, 'kernel_3': 3,
                                   'maxpool_1': 2, 'maxpool_2': 2, 'maxpool_3': 2,
                                   'nb_lstm': 1,
                                   'lstm_hidden_size': 64},
                  'opti_param': {'name_opti': 'adam', 'lr': 0.005, 'weight_decay': 0.0001,
                                 'scheduler': None},
                  'criterion_param': self.criterion}

        model10 = {'model_name': 'conv1d_into_lstm',
                   'balance_dataset': True, 'magic_number': 3010,
                   'layers_param': {'dropout': 0.25,
                                    'batch_norm': True,
                                    'cnn1': 16, 'cnn2': 32, 'cnn3': 64,
                                    'kernel_1': 7, 'kernel_2': 3, 'kernel_3': 3,
                                    'maxpool_1': 2, 'maxpool_2': None, 'maxpool_3': None,
                                    'nb_lstm': 2,
                                    'lstm_hidden_size': 64},
                   'opti_param': {'name_opti': 'adadelta', 'lr': 0.01, 'weight_decay': 0.01,
                                  'scheduler': None},
                   'criterion_param': self.criterion}

        model11 = {'model_name': 'conv1d_into_lstm',
                   'balance_dataset': True, 'magic_number': 3011,
                   'layers_param': {'dropout': 0.25,
                                    'batch_norm': True,
                                    'cnn1': 512, 'cnn2': None, 'cnn3': None,
                                    'kernel_1': 7, 'kernel_2': 3, 'kernel_3': 3,
                                    'maxpool_1': 2, 'maxpool_2': 2, 'maxpool_3': 2,
                                    'nb_lstm': 2,
                                    'lstm_hidden_size': 64},
                   'opti_param': {'name_opti': 'adam', 'lr': 0.001, 'weight_decay': 0.001,
                                  'scheduler': None},
                   'criterion_param': self.criterion}

        model12 = {'model_name': 'conv1d_into_lstm',
                   'balance_dataset': True, 'magic_number': 3012,
                   'layers_param': {'dropout': 0.25,
                                    'batch_norm': True,
                                    'cnn1': 16, 'cnn2': 32, 'cnn3': 64,
                                    'kernel_1': 7, 'kernel_2': 3, 'kernel_3': 3,
                                    'maxpool_1': 2, 'maxpool_2': None, 'maxpool_3': None,
                                    'nb_lstm': 2,
                                    'lstm_hidden_size': 64},
                   'opti_param': {'name_opti': 'adam', 'lr': 0.001, 'weight_decay': 0.001,
                                  'scheduler': None},
                   'criterion_param': self.criterion}

        # lr scheduler
        model13 = {'model_name': 'conv1d_into_lstm',
                   'balance_dataset': True, 'magic_number': 3013,
                   'layers_param': {'dropout': 0.25,
                                    'batch_norm': True,
                                    'cnn1': 512, 'cnn2': None, 'cnn3': None,
                                    'kernel_1': 7, 'kernel_2': 3, 'kernel_3': 3,
                                    'maxpool_1': 2, 'maxpool_2': 2, 'maxpool_3': 2,
                                    'nb_lstm': 2,
                                    'lstm_hidden_size': 64},
                   'opti_param': {'name_opti': 'adadelta', 'lr': 0.05, 'weight_decay': 0.01,
                                  'scheduler': 'multistepLR', 'gamma': 0.5,
                                  'milestones': [50, 100, 150, 200, 250, 300]},
                   'criterion_param': self.criterion}

        model14 = {'model_name': 'conv1d_into_lstm',
                   'balance_dataset': True, 'magic_number': 3014,
                   'layers_param': {'dropout': 0.25,
                                    'batch_norm': True,
                                    'cnn1': 16, 'cnn2': 32, 'cnn3': 64,
                                    'kernel_1': 7, 'kernel_2': 3, 'kernel_3': 3,
                                    'maxpool_1': 2, 'maxpool_2': None, 'maxpool_3': None,
                                    'nb_lstm': 2,
                                    'lstm_hidden_size': 64},
                   'opti_param': {'name_opti': 'adadelta', 'lr': 0.05, 'weight_decay': 0.01,
                                  'scheduler': 'multistepLR', 'gamma': 0.5,
                                  'milestones': [50, 100, 150, 200, 250, 300]},
                   'criterion_param': self.criterion}

        # ------------- transformer
        model15 = {'model_name': 'transformer',
                   'balance_dataset': True, 'magic_number': 4015,
                   'layers_param': {'dropout': 0.25,
                                    'embedding_type': 'lin', 'd_model': 256, 'dim_feedforward': 1024,
                                    'nhead': 4, 'nlayers': 3},
                   'opti_param': {'name_opti': 'noam', 'factor': 1 / np.sqrt(2), 'warmup': 9000,
                                  'scheduler': None},
                   'criterion_param': self.criterion}

        model16 = {'model_name': 'transformer',
                   'balance_dataset': True, 'magic_number': 4016,
                   'layers_param': {'dropout': 0.25,
                                    'embedding_type': 'lin', 'd_model': 256, 'dim_feedforward': 1024,
                                    'nhead': 4, 'nlayers': 3},
                   'opti_param': {'name_opti': 'noam', 'factor': 1.5 / np.sqrt(2), 'warmup': 9000,
                                  'scheduler': None},
                   'criterion_param': self.criterion}

        model17 = {'model_name': 'transformer',
                   'balance_dataset': True, 'magic_number': 4017,
                   'layers_param': {'dropout': 0.25,
                                    'embedding_type': 'lin', 'd_model': 128, 'dim_feedforward': 512,
                                    'nhead': 4, 'nlayers': 3},
                   'opti_param': {'name_opti': 'noam', 'factor': 1.5 / np.sqrt(2), 'warmup': 18000,
                                  'scheduler': None},
                   'criterion_param': self.criterion}

        # ------------- ResNet
        model18 = {'model_name': 'MSResNet',
                   'balance_dataset': True, 'magic_number': 5018,
                   'layers_param': {'layers': [1, 1, 1, 1]},
                   'opti_param': {'name_opti': 'adadelta', 'lr': 0.05, 'weight_decay': 0.01,
                                  'scheduler': 'multistepLR', 'gamma': 0.5,
                                  'milestones': [50, 100, 150, 200, 250, 300]},
                   'criterion_param': self.criterion}

        model19 = {'model_name': 'ResNet',
                   'balance_dataset': True, 'magic_number': 5019,
                   'layers_param': {'num_res_net': '18'},
                   'opti_param': {'name_opti': 'adadelta', 'lr': 0.05, 'weight_decay': 0.01,
                                  'scheduler': 'multistepLR', 'gamma': 0.5,
                                  'milestones': [50, 100, 150, 200, 250, 300]},
                   'criterion_param': self.criterion}

        model20 = {'model_name': 'ResNetFred',
                   'balance_dataset': True, 'magic_number': 5020,
                   'layers_param': {},
                   'opti_param': {'name_opti': 'adadelta', 'lr': 0.005, 'weight_decay': 0.01,
                                  'scheduler': 'multistepLR', 'gamma': 0.5,
                                  'milestones': [50, 100, 150, 200, 250, 300]},
                   'criterion_param': self.criterion}

        model21 = {'model_name': 'ResNet',
                   'balance_dataset': True, 'magic_number': 5021,
                   'layers_param': {'num_res_net': '50'},
                   'opti_param': {'name_opti': 'adadelta', 'lr': 0.05, 'weight_decay': 0.01,
                                  'scheduler': 'multistepLR', 'gamma': 0.5,
                                  'milestones': [50, 100, 150, 200, 250, 300]},
                   'criterion_param': self.criterion}

        model22 = {'model_name': 'ResNeXt',
                   'balance_dataset': True, 'magic_number': 5022,
                   'layers_param': {},
                   'opti_param': {'name_opti': 'adadelta', 'lr': 0.05, 'weight_decay': 0.01,
                                  'scheduler': 'multistepLR', 'gamma': 0.5,
                                  'milestones': [50, 100, 150, 200, 250, 300]},
                   'criterion_param': self.criterion}

        # ------------- tests optimizer on resnet
        model23 = {'model_name': 'ResNet',
                   'balance_dataset': True, 'magic_number': 5023,
                   'layers_param': {'num_res_net': '18'},
                   'opti_param': {'name_opti': 'adam', 'lr': 0.01, 'weight_decay': 0.0001,
                                  'scheduler': 'multistepLR', 'gamma': 0.5,
                                  'milestones': [50, 100, 150, 200, 250, 300]},
                   'criterion_param': self.criterion}

        model24 = {'model_name': 'ResNet',
                   'balance_dataset': True, 'magic_number': 5024,
                   'layers_param': {'num_res_net': '18'},
                   'opti_param': {'name_opti': 'adam', 'lr': 0.01, 'weight_decay': 0.001,
                                  'scheduler': 'multistepLR', 'gamma': 0.5,
                                  'milestones': [50, 100, 150, 200, 250, 300]},
                   'criterion_param': self.criterion}

        model25 = {'model_name': 'ResNet',
                   'balance_dataset': True, 'magic_number': 5025,
                   'layers_param': {'num_res_net': '50'},
                   'opti_param': {'name_opti': 'adam', 'lr': 0.01, 'weight_decay': 0.0001,
                                  'scheduler': 'multistepLR', 'gamma': 0.5,
                                  'milestones': [50, 100, 150, 200, 250, 300]},
                   'criterion_param': self.criterion}

        model26 = {'model_name': 'ResNet',
                   'balance_dataset': True, 'magic_number': 5026,
                   'layers_param': {'num_res_net': '18_little'},
                   'opti_param': {'name_opti': 'adam', 'lr': 0.01, 'weight_decay': 0.001,
                                  'scheduler': 'multistepLR', 'gamma': 0.5,
                                  'milestones': [50, 100, 150, 200, 250, 300]},
                   'criterion_param': self.criterion}

        models = [model0, model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11,
                  model12, model13, model14, model15, model16, model17, model18, model19, model20, model21, model22,
                  model23, model24, model25, model26]

        return models[self.model]

import numpy as np
from torch.utils.data import DataLoader
import torchvision
import warnings
from pathlib import Path
import time

warnings.filterwarnings("ignore")

from Datas.classSignal import SignalForce, SignalImg,  Shape
from Datas.classEvent import ForceEvent
from Datas.classData import CreateDataField
from Datas.classLoader import DataSetTIMEFIELD, DataSetFalseData


def create_sequences_field(plot, config_data, config_pred, remote):
    signal_img = [SignalImg(config_pred, 'flu_rsc', NN) for NN in config_pred.NN]

    sw, sc = signal_img[0].import_single('sw_sc')

    create_data = CreateDataField(config_data, config_pred, remote, sw, sc)

    fileName = config_pred.global_path_load + \
               'pict_event_sequence_NN/{}seqsize_{}step_{}futur/'.format(config_pred.seq_size,
                                                                         config_pred.overlap_step,
                                                                         config_pred.futur) + 'sequence_train_0.npy'
    fileObj = Path(fileName)
    is_fileObj = fileObj.is_file()
    print('path : {}'.format(fileName))
    print('sequences existent : {}'.format(is_fileObj))
    if not is_fileObj:
        dict_seq, nb_seq = create_data.NN_sequences(signal_img)
        classes_edges = create_data.classes(plot, signal_img[0], display_fig=True)
    else:
        # classes_edges = create_data.load_classes_edges()
        classes_edges = create_data.classes(plot, signal_img[0], display_fig=True)
        dict_seq, nb_seq = create_data.load_dict_sequences()

    return nb_seq, dict_seq, classes_edges, sw, sc


def create_generator_field(args, config_pred, dict_seq, classes_edges):

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    path_seqence = config_pred.global_path_load \
                   + '/pict_event_sequence_NN/{}seqsize_{}step_{}futur/'.format(config_pred.seq_size,
                                                                                config_pred.overlap_step,
                                                                                config_pred.futur)

    train_dataset = DataSetTIMEFIELD(path_seqence, dict_seq['train'], 'train', classes_edges,
                                     transform=torchvision.transforms.ToTensor())

    val_dataset = DataSetTIMEFIELD(path_seqence, dict_seq['val'], 'val', classes_edges,
                                   transform=torchvision.transforms.ToTensor())

    test_dataset = DataSetTIMEFIELD(path_seqence, dict_seq['test'], 'test', classes_edges,
                                    transform=torchvision.transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=config_pred.batch_size, shuffle=True, **kwargs)

    val_loader = DataLoader(val_dataset, batch_size=config_pred.batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(test_dataset, batch_size=config_pred.batch_size, shuffle=True, **kwargs)

    return train_loader, val_loader, test_loader


def create_sequences_false_data():

    return 'bmla'


def create_generator_false_data(args, config_pred, X, Y):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    train_dataset = DataSetFalseData(X, Y)

    train_loader = DataLoader(train_dataset, batch_size=config_pred.batch_size, shuffle=True, **kwargs)

    return train_loader, train_loader, train_loader
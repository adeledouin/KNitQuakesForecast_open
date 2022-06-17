from classConfig import ConfigData, ConfigPred
import Config_data
import Config_pred
from dictdata import dictdata
import numpy as np

import logging

logging.basicConfig(format='| %(levelname)s | %(asctime)s | %(message)s', level=logging.DEBUG)

# %% ################### Args ##################################
remote = False

path_from_root = '/path_from_root/'

date = '220119'
ref_tricot = 'knit005_'
n_exp = 'mix_'
version_data = 'v2'
version_pred = 'v3_5_4'
trainsize = 1000000
tau = 4

config_data = ConfigData(path_from_root, Config_data.exp[ref_tricot+n_exp+version_data])
NAME_EXP = ref_tricot + version_pred
config_pred = ConfigPred(Config_pred.exp_scalar[NAME_EXP], config_data)

fileName = config_pred.global_path_load + 'sequence_NN/dict_sequences_{}_{}seqsize_{}step_{}futur.npy'.format(
    config_pred.input_data, config_pred.seq_size, config_pred.overlap_step, config_pred.futur)

dict_sequence = np.load(fileName, allow_pickle=True).flat[0]

fileName = config_pred.global_path_load + 'sequence_NN/{}_classes_edges_{}_futur_{}.npy'.format(
    config_pred.output_shape, config_pred.label_save,
    config_pred.futur)

classes_edges = np.load(fileName)

print('train', dict_sequence['train'].shape)
print('test', dict_sequence['test'].shape)

dates = np.array(['210726', '210822', '210822', '210822', '210726', '210825', '210825', '210825', '210726', '210825', '210830', '210830',
        '210726', '210727', '210727', '210727', '210703', '210803', '210803', '210803', '210803', '210803', '210803', '210811',
        '210727', '210728', '210728', '210729', '210811', '210811', '210811', '210812', '210812', '210812', '210812', '210815',
        '210729', '210729', '210729', '210130', '210815', '210815', '210815', '210818', '210818', '210818', '210818', '210822',
                  '220119'])


version_preds = np.array(['v1_2_1', 'v1_3_1', 'v1_4_1', 'v1_5_1', 'v1_2_2', 'v1_3_2', 'v1_4_2', 'v1_5_2', 'v1_2_3', 'v1_3_3', 'v1_4_3', 'v1_5_3',
                'v2_2_1', 'v2_3_1', 'v2_4_1', 'v2_5_1', 'v2_2_2', 'v2_3_2', 'v2_4_2', 'v2_5_2', 'v2_2_3', 'v2_3_3', 'v2_4_3', 'v2_5_3',
                'v3_2_1', 'v3_3_1', 'v3_4_1', 'v3_5_1', 'v3_2_2', 'v3_3_2', 'v3_4_2', 'v3_5_2', 'v3_2_3', 'v3_3_3', 'v3_4_3', 'v3_5_3',
                'v4_2_1', 'v4_3_1', 'v4_4_1', 'v4_5_1', 'v4_2_2', 'v4_3_2', 'v4_4_2', 'v4_5_2', 'v4_2_3', 'v4_3_3', 'v4_4_3', 'v4_5_3',
                          'v3_5_4'])
model = 5023


def create_ref_sets(date_2, version_pred_2, sub_version_2=''):

    print(date_2, version_pred_2)
    NAME_EXP_2 = ref_tricot + version_pred_2
    config_pred_2 = ConfigPred(Config_pred.exp_scalar[NAME_EXP_2], config_data)

    fileName_2 = config_pred_2.global_path_load + 'sequence_NN/dict_sequences_{}_{}seqsize_{}step_{}futur.npy'.format(
        config_pred_2.input_data, config_pred_2.seq_size, config_pred_2.overlap_step, config_pred_2.futur)

    dict_sequence_2 = np.load(fileName_2, allow_pickle=True).flat[0]

    if sub_version_2 == 'reverse':
        fileName_2 = config_pred_2.global_path_load + 'sequence_NN/{}_reversed2_classes_edges_{}_futur_{}.npy'.format(
            config_pred_2.output_shape, config_pred_2.label_save,
            config_pred_2.futur)
    else:
        fileName_2 = config_pred_2.global_path_load + 'sequence_NN/{}_classes_edges_{}_futur_{}.npy'.format(
            config_pred_2.output_shape, config_pred_2.label_save,
            config_pred_2.futur)

    classes_edges_2 = np.load(fileName_2)

    print('train', dict_sequence_2['train'].shape)
    print('test', dict_sequence['test'].shape)
    print('test', dict_sequence_2['test'].shape)

    print('for train')
    where_train = np.array([])
    for i in range(np.max(dict_sequence['train'][:, 0])+1):
        where_sub_train = np.where(dict_sequence['train'][:, 0] == i)[0]
        where_sub_train_2 = np.where(dict_sequence_2['train'][:, 0] == i)[0]
        # print(i, where_sub_train.size)
        # print(i, where_sub_train_2.size)

        selections = set(dict_sequence['train'][where_sub_train, 1])
        indexes = [ind for ind, el in enumerate(dict_sequence_2['train'][where_sub_train_2, 1]) if el in selections]
        # print(i, np.size(indexes))

        if np.size(indexes) != where_sub_train.size:
            print('issue cycle {}'.format(i))
        where_train = np.concatenate((where_train, where_sub_train_2[indexes])).astype(int)

    print('for val')
    where_val = np.array([])
    for i in range(np.max(dict_sequence['val'][:, 0]) + 1):
        where_sub_val = np.where(dict_sequence['val'][:, 0] == i)[0]
        where_sub_val_2 = np.where(dict_sequence_2['val'][:, 0] == i)[0]
        # print(i, where_sub_val.size)
        # print(i, where_sub_val_2.size)

        selections = set(dict_sequence['val'][where_sub_val, 1])
        indexes = [ind for ind, el in enumerate(dict_sequence_2['val'][where_sub_val_2, 1]) if el in selections]
        # print(i, np.size(indexes))

        if np.size(indexes) != where_sub_val.size:
            print('issue cycle {}'.format(i))
        where_val = np.concatenate((where_val, where_sub_val_2[indexes])).astype(int)

    print('for test')
    where_test = np.array([])
    for i in range(np.max(dict_sequence['test'][:, 0])+1):
        where_sub_test = np.where(dict_sequence['test'][:, 0] == i)[0]
        where_sub_test_2 = np.where(dict_sequence_2['test'][:, 0] == i)[0]
        # print(i, where_sub_test.size)
        # print(i, where_sub_test_2.size)

        selections = set(dict_sequence['test'][where_sub_test, 1])
        indexes = [ind for ind, el in enumerate(dict_sequence_2['test'][where_sub_test_2, 1]) if el in selections]
        # print(i, np.size(indexes))

        if np.size(indexes) != where_sub_test.size:
            print('issue cycle {}'.format(i))
        where_test = np.concatenate((where_test, where_sub_test_2[indexes])).astype(int)

    ## callbacks
    load_test_callback = config_pred.global_path_save + 'callback/scalar/test_callback_{}_{}_m{}_{}seq{}.npy'.format(
        date_2, version_pred_2, model, trainsize, sub_version_2)

    callback = np.load(load_test_callback, allow_pickle=True).flat[0]
    y_value = callback['y_value'].cpu().data.numpy()
    y_target = callback['y_target'].cpu().data.numpy().astype(int)
    y_pred = callback['y_pred'].cpu().data.numpy().astype(int)

    test_callback_ref = dictdata()
    test_callback_ref.add('y_value', y_value[where_test])
    test_callback_ref.add('y_target', y_target[where_test])
    test_callback_ref.add('y_pred', y_pred[where_test])

    np.save(
        config_pred.global_path_save + 'callback/scalar/test_callback_reftau{}_{}_{}_m{}_{}seq{}.npy'.format(
            tau, date_2, version_pred_2, model, trainsize, sub_version_2), test_callback_ref)

    np.save(
        config_pred.global_path_save + 'sequences_RL/test_callback_reftau{}_{}_{}_m{}_{}seq{}.npy'.format(
            tau, date_2, version_pred_2, model, trainsize, sub_version_2), test_callback_ref)

    ## values
    fileName = config_pred_2.global_path_load + 'sequence_NN/dict_all_value_for_class_{}_seqsize_{}_futur_{}.npy'.format(
        config_pred_2.label_save, config_pred_2.seq_size, config_pred_2.futur)

    dict_all_value = np.load(fileName, allow_pickle=True).flat[0]

    dict_all_value_ref = dictdata()
    dict_all_value_ref.add('train', dict_all_value['train'][where_train])
    dict_all_value_ref.add('val', dict_all_value['val'][where_val])
    dict_all_value_ref.add('test', dict_all_value['test'][where_test])

    np.save(
        config_pred.global_path_load + 'sequence_NN/dict_all_value_reftau{}_for_class_{}_seqsize_{}_futur_{}.npy'.format(
            version_data, tau, config_pred_2.label_save, config_pred.seq_size, config_pred_2.futur), dict_all_value_ref)

    np.save(
        config_pred.global_path_load + 'sequences_RL/dict_all_value_reftau{}_for_class_{}_futur_{}.npy'.format(
            version_data, tau, config_pred_2.label_save, config_pred_2.futur), dict_all_value_ref)

    # sequences
    dict_sequence_ref = dictdata()
    dict_sequence_ref.add('train', dict_sequence_2['train'][where_train, :])
    dict_sequence_ref.add('val', dict_sequence_2['val'][where_val, :])
    dict_sequence_ref.add('test', dict_sequence_2['test'][where_test, :])

    np.save(
        config_pred.global_path_load + 'sequences_RL/dict_sequence_reftau{}_{}_{}seqsize_{}step_{}futur.npy'.format(
            version_data, tau, config_pred_2.input_data, config_pred_2.seq_size, config_pred_2.overlap_step, config_pred_2.futur),
        dict_sequence_ref)

    ## class edges

    if sub_version_2 == 'reverse':
        np.save(
            config_pred.global_path_load + 'sequences_RL/{}_reversed2_classes_edges_{}_futur_{}.npy'.format(
                version_data, config_pred_2.output_shape, config_pred_2.label_save,
                config_pred_2.futur), classes_edges_2)
    else:
        np.save(
            config_pred.global_path_load + 'sequences_RL/{}_classes_edges_{}_futur_{}.npy'.format(
                version_data, config_pred_2.output_shape, config_pred_2.label_save,
                config_pred_2.futur), classes_edges_2)


for i in range(np.size(dates)):
    create_ref_sets(dates[i], version_preds[i]) #, sub_versions_2[i])

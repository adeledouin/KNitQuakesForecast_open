import numpy as np

import logging

logging.basicConfig(level=logging.INFO)

from classConfig import ConfigData, ConfigPred
import Config_data
import Config_pred


# %% ################### Args ##################################
remote = False

path_from_root = '/path_from_root/'

ref_tricot = 'knit005_'
n_exp = 'mix_'
version_data = 'v2'
sub_version = ''
model = 5023
trainsize = 1000000
tau = 4
info_size_max = 256
nb_steps = 5000
nb_episodes = 300
nb_eval = 50

config_data = ConfigData(path_from_root, Config_data.exp[ref_tricot + n_exp + version_data])

date = 220524
version_pred = 'v1_2_1'
print(date, version_pred)
NAME_EXP = ref_tricot + version_pred
config_pred = ConfigPred(Config_pred.exp_scalar[NAME_EXP], config_data)

fileName = config_pred.global_path_load + 'sequences_RL/dict_sequence_reftau{}_{}_{}seqsize_{}step_{}futur.npy'.format(
    tau, config_pred.input_data, config_pred.seq_size, config_pred.overlap_step, config_pred.futur)

dict_sequence = np.load(fileName, allow_pickle=True).flat[0]

print('train', dict_sequence['train'].shape)
print('test', dict_sequence['test'].shape)

fileName = config_pred.global_path_load + 'sequences_RL/dict_all_value_reftau{}_for_class_{}_futur_{}.npy'.format(
    tau, config_pred.label_save, config_pred.futur)

dict_all_value = np.load(fileName, allow_pickle=True).flat[0]

load_test_callback = config_pred.global_path_load + 'sequences_RL/test_callback_reftau{}_{}_{}_m{}_{}seq{}.npy'.format(
    tau, date, version_pred, model, trainsize, sub_version)
callback = np.load(load_test_callback, allow_pickle=True).flat[0]
y_value = callback['y_value']
y_target = callback['y_target']
y_pred = callback['y_pred']

cycles_in_train = np.max(dict_sequence['train'][:, 0])
cycles_in_test = np.max(dict_sequence['test'][:, 0])
t0_max = np.max(dict_sequence['train'][:, 1])

sequence_train = dict_sequence['train']
sequence_test = dict_sequence['test'][dict_sequence['test'][:, 0] < cycles_in_test - 9]
sequence_eval = dict_sequence['test'][(dict_sequence['test'][:, 0] >= cycles_in_test - 9) & (dict_sequence['test'][:, 0] < cycles_in_test - 4)]
sequence_comp = dict_sequence['test'][dict_sequence['test'][:, 0] >= cycles_in_test - 4]

cycle_in_trainRL = np.arange(np.min(sequence_train[:, 0]), np.max(sequence_train[:, 0]) + 1)
cycle_in_testRL = np.arange(np.min(sequence_test[:, 0]), np.max(sequence_test[:, 0]) + 1)
cycle_in_eval = np.arange(np.min(sequence_eval[:, 0]), np.max(sequence_eval[:, 0]) + 1)
cycle_in_comp = np.arange(np.min(sequence_comp[:, 0]), np.max(sequence_comp[:, 0]) + 1)

possible_index_in_time_learing = np.arange(info_size_max - 1, t0_max + 1, nb_steps)
possible_index_in_time_eval = np.arange(info_size_max - 1, t0_max + 1 - nb_steps, 1000)
possible_index_in_time_comp = np.arange(info_size_max - 1, t0_max + 1 - nb_steps, 1000)

print('ce qui fait :')
print('pour testRL  {} seq'.format(cycle_in_testRL.size*possible_index_in_time_learing[:-1].size))
print('pour eval {} seq'.format(cycle_in_eval.size*possible_index_in_time_eval[:-1].size))
print('pour comp {} seq'.format(cycle_in_comp.size*possible_index_in_time_comp[:-1].size))

nb_comp = cycle_in_comp.size*possible_index_in_time_comp[:-1].size

# ------------------------------------------
def random_indices(cycle, tps, nb_episodes):
    pickrandom_lin = np.random.choice(cycle * tps, nb_episodes, replace=False)
    pickrandom_mat = np.unravel_index(pickrandom_lin, (cycle, tps))

    return pickrandom_mat[0], pickrandom_mat[1]


# ------------------------------------------
def all_indices(cycle, tps, ):
    lin = np.arange(cycle * tps)
    mat = np.unravel_index(lin, (cycle, tps))

    return mat[0], mat[1]


# ------------------------------------------
def indexes_in_sequence():
    cycles, tps = random_indices(cycle_in_eval.size, possible_index_in_time_eval[:-1].size, nb_eval)
    indexes = np.vstack((cycle_in_eval[cycles], possible_index_in_time_eval[:-1][tps])).transpose()

    np.save(config_pred.global_path_load + 'sequences_RL/indexes_evalsetreftau{}_{}eval_{}steps.npy'.format(tau, nb_eval, nb_steps),
            indexes)

    cycles, tps = all_indices(cycle_in_comp.size, possible_index_in_time_comp[:-1].size)
    indexes = np.vstack((cycle_in_comp[cycles], possible_index_in_time_comp[:-1][tps])).transpose()

    np.save(
        config_pred.global_path_load + 'sequences_RL/indexes_compsetreftau{}_{}eval_{}steps.npy'.format(tau, nb_comp,
                                                                                                        nb_steps),
        indexes)

    cycles, tps = random_indices(cycle_in_trainRL.size, possible_index_in_time_learing.size, nb_episodes)
    indexes = np.vstack((cycle_in_trainRL[cycles], possible_index_in_time_learing[tps])).transpose()
    np.save(
        config_pred.global_path_load + 'sequences_RL/indexes_trainsetreftau{}_{}ep_{}steps.npy'.format(tau, nb_episodes, nb_steps),
        indexes)

    cycles, tps = random_indices(cycle_in_testRL.size, possible_index_in_time_learing.size, nb_episodes)
    indexes = np.vstack((cycle_in_testRL[cycles], possible_index_in_time_learing[tps])).transpose()
    np.save(
        config_pred.global_path_load + 'sequences_RL/indexes_testsetreftau{}_{}ep_{}steps.npy'.format(tau, nb_episodes, nb_steps),
        indexes)


indexes_in_sequence()

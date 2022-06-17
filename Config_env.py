import numpy as np

def_damage = {'d_5_50': np.array([0, 0, 0, 5, 50]),
              'd_7_70': np.array([0, 0, 0, 7, 70]),
              'd_10_100': np.array([0, 0, 0, 10, 100]),
              'd_20_200': np.array([0, 0, 0, 20, 200]),
              'd_0_200': np.array([0, 0, 0, 0, 200]),
              'd_0_300': np.array([0, 0, 0, 0, 300]),
              'd_15_150': np.array([0, 0, 0, 15, 150]),
              'd_25_250': np.array([0, 0, 0, 25, 250]),
              'd_30_300': np.array([0, 0, 0, 30, 300]),
              'd_35_350': np.array([0, 0, 0, 35, 350]),
              'd_40_400': np.array([0, 0, 0, 40, 400]),
              'd_50_500': np.array([0, 0, 0, 50, 500])}

exp = dict()

##################
exp['01'] = {'lab_args': {'nb_eval': 50, 'info_size': 1, 'model_set': 'test', 'RL_set': 'train'},
             'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
             'knowledge_type': 'NN'}

exp['01-eval'] = {'lab_args': {'nb_eval': 50, 'info_size': 1, 'model_set': 'test', 'RL_set': 'test'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['01-comp'] = {'lab_args': {'nb_eval': 185, 'info_size': 1, 'model_set': 'test', 'RL_set': 'comp'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['11'] = {'lab_args': {'nb_eval': 50, 'info_size': 1, 'model_set': 'train', 'RL_set': 'train'},
             'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
             'knowledge_type': 'god'}

exp['11-eval'] = {'lab_args': {'nb_eval': 50, 'info_size': 1, 'model_set': 'test', 'RL_set': 'test'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'god'}

exp['11-comp'] = {'lab_args': {'nb_eval': 185, 'info_size': 1, 'model_set': 'test', 'RL_set': 'comp'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'god'}

# info size 2
exp['02'] = {'lab_args': {'nb_eval': 50, 'info_size': 2, 'model_set': 'test', 'RL_set': 'train'},
             'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
             'knowledge_type': 'NN'}

exp['02-eval'] = {'lab_args': {'nb_eval': 50, 'info_size': 2, 'model_set': 'test', 'RL_set': 'test'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['02-comp'] = {'lab_args': {'nb_eval': 185, 'info_size': 2, 'model_set': 'test', 'RL_set': 'comp'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['12'] = {'lab_args': {'nb_eval': 50, 'info_size': 2, 'model_set': 'train', 'RL_set': 'train'},
             'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
             'knowledge_type': 'god'}

exp['12-eval'] = {'lab_args': {'nb_eval': 50, 'info_size': 2, 'model_set': 'test', 'RL_set': 'test'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'god'}

exp['12-comp'] = {'lab_args': {'nb_eval': 185, 'info_size': 2, 'model_set': 'test', 'RL_set': 'comp'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'god'}

# info size 4
exp['03'] = {'lab_args': {'nb_eval': 50, 'info_size': 4, 'model_set': 'test', 'RL_set': 'train'},
             'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
             'knowledge_type': 'NN'}

exp['03-eval'] = {'lab_args': {'nb_eval': 50, 'info_size': 4, 'model_set': 'test', 'RL_set': 'test'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['03-comp'] = {'lab_args': {'nb_eval': 185, 'info_size': 4, 'model_set': 'test', 'RL_set': 'comp'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['13'] = {'lab_args': {'nb_eval': 50, 'info_size': 4, 'model_set': 'train', 'RL_set': 'train'},
             'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
             'knowledge_type': 'god'}

exp['13-eval'] = {'lab_args': {'nb_eval': 50, 'info_size': 4, 'model_set': 'test', 'RL_set': 'test'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'god'}

exp['13-comp'] = {'lab_args': {'nb_eval': 185, 'info_size': 4, 'model_set': 'test', 'RL_set': 'comp'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'god'}

# info size 8

exp['04'] = {'lab_args': {'nb_eval': 50, 'info_size': 8, 'model_set': 'test', 'RL_set': 'train'},
             'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
             'knowledge_type': 'NN'}

exp['04-eval'] = {'lab_args': {'nb_eval': 50, 'info_size': 8, 'model_set': 'test', 'RL_set': 'test'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['04-comp'] = {'lab_args': {'nb_eval': 185, 'info_size': 8, 'model_set': 'test', 'RL_set': 'comp'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['14'] = {'lab_args': {'nb_eval': 50, 'info_size': 8, 'model_set': 'train', 'RL_set': 'train'},
             'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
             'knowledge_type': 'god'}

exp['14-eval'] = {'lab_args': {'nb_eval': 50, 'info_size': 8, 'model_set': 'test', 'RL_set': 'test'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'god'}

exp['14-comp'] = {'lab_args': {'nb_eval': 185, 'info_size': 8, 'model_set': 'test', 'RL_set': 'comp'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'god'}

# info size 16
exp['05'] = {'lab_args': {'nb_eval': 50, 'info_size': 16, 'model_set': 'test', 'RL_set': 'train'},
             'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
             'knowledge_type': 'NN'}

exp['05-eval'] = {'lab_args': {'nb_eval': 50, 'info_size': 16, 'model_set': 'test', 'RL_set': 'test'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['05-comp'] = {'lab_args': {'nb_eval': 185, 'info_size': 16, 'model_set': 'test', 'RL_set': 'comp'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['15'] = {'lab_args': {'nb_eval': 50, 'info_size': 16, 'model_set': 'train', 'RL_set': 'train'},
             'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
             'knowledge_type': 'god'}

exp['15-eval'] = {'lab_args': {'nb_eval': 50, 'info_size': 16, 'model_set': 'test', 'RL_set': 'test'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'god'}

exp['15-comp'] = {'lab_args': {'nb_eval': 185, 'info_size': 16, 'model_set': 'test', 'RL_set': 'comp'},
                  'knowledge_type': 'god'}

# info size 32
exp['06'] = {'lab_args': {'nb_eval': 50, 'info_size': 32, 'model_set': 'test', 'RL_set': 'train'},
             'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
             'knowledge_type': 'NN'}

exp['06-eval'] = {'lab_args': {'nb_eval': 50, 'info_size': 32, 'model_set': 'test', 'RL_set': 'test'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['06-comp'] = {'lab_args': {'nb_eval': 185, 'info_size': 32, 'model_set': 'test', 'RL_set': 'comp'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['16'] = {'lab_args': {'nb_eval': 50, 'info_size': 32, 'model_set': 'train', 'RL_set': 'train'},
             'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
             'knowledge_type': 'god'}

exp['16-eval'] = {'lab_args': {'nb_eval': 50, 'info_size': 32, 'model_set': 'test', 'RL_set': 'test'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'god'}

exp['16-comp'] = {'lab_args': {'nb_eval': 185, 'info_size': 32, 'model_set': 'test', 'RL_set': 'comp'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'god'}

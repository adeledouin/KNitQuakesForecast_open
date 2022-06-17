import numpy as np

exp = dict()

###################

exp['01'] = {'lab_args': {'nb_eval': 50, 'info_size': 0, 'model_set': 'test', 'RL_set': 'train'},
             'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
             'knowledge_type': 'NN'}

exp['01-eval'] = {'lab_args': {'nb_eval': 50, 'info_size': 0, 'model_set': 'test', 'RL_set': 'test'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['01-comp'] = {'lab_args': {'nb_eval': 185, 'info_size': 0, 'model_set': 'test', 'RL_set': 'comp'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['02'] = {'lab_args': {'nb_eval': 50, 'info_size': 1, 'model_set': 'test', 'RL_set': 'train'},
             'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
             'knowledge_type': 'NN'}

exp['02-eval'] = {'lab_args': {'nb_eval': 50, 'info_size': 1, 'model_set': 'test', 'RL_set': 'test'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['02-comp'] = {'lab_args': {'nb_eval': 185, 'info_size': 1, 'model_set': 'test', 'RL_set': 'comp'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['03'] = {'lab_args': {'nb_eval': 50, 'info_size': 2, 'model_set': 'test', 'RL_set': 'train'},
             'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
             'knowledge_type': 'NN'}

exp['03-eval'] = {'lab_args': {'nb_eval': 50, 'info_size': 2, 'model_set': 'test', 'RL_set': 'test'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['03-comp'] = {'lab_args': {'nb_eval': 185, 'info_size': 2, 'model_set': 'test', 'RL_set': 'comp'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['04'] = {'lab_args': {'nb_eval': 50, 'info_size': 4, 'model_set': 'test', 'RL_set': 'train'},
             'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
             'knowledge_type': 'NN'}

exp['04-eval'] = {'lab_args': {'nb_eval': 50, 'info_size': 4, 'model_set': 'test', 'RL_set': 'test'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['04-comp'] = {'lab_args': {'nb_eval': 185, 'info_size': 4, 'model_set': 'test', 'RL_set': 'comp'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['05'] = {'lab_args': {'nb_eval': 50, 'info_size': 8, 'model_set': 'test', 'RL_set': 'train'},
             'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
             'knowledge_type': 'NN'}

exp['05-eval'] = {'lab_args': {'nb_eval': 50, 'info_size': 8, 'model_set': 'test', 'RL_set': 'test'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['05-comp'] = {'lab_args': {'nb_eval': 185, 'info_size': 8, 'model_set': 'test', 'RL_set': 'comp'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['06'] = {'lab_args': {'nb_eval': 50, 'info_size': 16, 'model_set': 'test', 'RL_set': 'train'},
             'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
             'knowledge_type': 'NN'}

exp['06-eval'] = {'lab_args': {'nb_eval': 50, 'info_size': 16, 'model_set': 'test', 'RL_set': 'test'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['06-comp'] = {'lab_args': {'nb_eval': 185, 'info_size': 16, 'model_set': 'test', 'RL_set': 'comp'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['07'] = {'lab_args': {'nb_eval': 50, 'info_size': 32, 'model_set': 'test', 'RL_set': 'train'},
             'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
             'knowledge_type': 'NN'}

exp['07-eval'] = {'lab_args': {'nb_eval': 50, 'info_size': 32, 'model_set': 'test', 'RL_set': 'test'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}

exp['07-comp'] = {'lab_args': {'nb_eval': 185, 'info_size': 32, 'model_set': 'test', 'RL_set': 'comp'},
                  'decade': np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1]), 'dt': 1,
                  'knowledge_type': 'NN'}


exp = dict()

# %% knit_005
exp['knit005_mix_v1'] = {'ref': 'knit_005', 'date': [201029, 201029, 201029, 201029, 201029, 201029, 201029, 201029,
                                                     201029, 201116, 201119, 201119, 201119, 201119, 201119, 201119,
                                                     201119, 201119, 201119, 201119],
                         'nexp': [1, 2, 3, 4, 5, 6, 7, 8, 9, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         'version_raw': 1, 'version_work': 1,
                         'vitesse': 0.005, 'Lw_0': 170, 'Lw_i': 210, 'Lw_max': 219.99,
                         'fr_instron': 25, 'sursample': False,
                         'delta_z': 1, 'delta_t_pict': 5,
                         'nb_point_eff': 1,
                         'prescycle': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                         'mincycle': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                         'maxcycle': [31, 39, 36, 36, 31, 31, 23, 27, 17, 39, 36, 39, 39, 39, 31, 39, 33, 31, 33, 33],
                         'mix_set': True, 'nb_set': 20, 'img': False,
                         'nb_process': 15,
                         'config_corr_fraction': 10,

                         'methode_rsc_img': 1,
                         'fields': ['vit_x', 'vit_y', 'vort'],

                         'config_scalarevent_flu_exposants': [5, 4, 4, 3, 3],
                         'config_scalarevent_flu_seuils': [5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
                         'config_scalarevent_flu_save_seuils': ['5_5', '1_4', '5_4', '1_3', '5_3'],
                         'config_scalarevent_flu_nb_seuils': 5,
                         'config_scalarevent_flu_which_seuil': 4,
                         'config_scalarevent_flu_nbclasses': 3,
                         }

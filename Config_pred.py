exp_scalar = dict()

# %% ##### scalar

exp_scalar['knit005_v03_5_1'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum (exp *\delta f)', 'label_save': 'sum_expdf',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 5, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 20, 'tau': 7,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss',
                                           'weight': None}}

#max df
exp_scalar['knit005_v1_2_0'] = {'ref': 'knit_005',
                            'false_data': False, 'input_data': 'f_rsc',
                            'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                            'output_type': 'class', 'label_name': 'Max(\delta f)', 'label_save': 'max_df',
                            'shuffle': True,
                            'channel': 3, 'nb_classes': 2, 'equipro': True,
                            'seq_size': 256, 'overlap_step': 1, 'futur': 1,
                            'batch_size': 512,
                            'criterion': {'name_criterion': 'cross_entropy_loss',
                                          'weight': None}}

exp_scalar['knit005_v1_3_0'] = {'ref': 'knit_005',
                            'false_data': False, 'input_data': 'f_rsc',
                            'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                            'output_type': 'class', 'label_name': 'Max(\delta f)', 'label_save': 'max_df',
                            'shuffle': True,
                            'channel': 3, 'nb_classes': 3, 'equipro': True,
                            'seq_size': 256, 'overlap_step': 1, 'futur': 1,
                            'batch_size': 512,
                            'criterion': {'name_criterion': 'cross_entropy_loss',
                                          'weight': None}}

exp_scalar['knit005_v1_4_0'] = {'ref': 'knit_005',
                            'false_data': False, 'input_data': 'f_rsc',
                            'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                            'output_type': 'class', 'label_name': 'Max(\delta f)', 'label_save': 'max_df',
                            'shuffle': True,
                            'channel': 3, 'nb_classes': 4, 'equipro': True,
                            'seq_size': 256, 'overlap_step': 1, 'futur': 1,
                            'batch_size': 512,
                            'criterion': {'name_criterion': 'cross_entropy_loss',
                                          'weight': None}}

exp_scalar['knit005_v1_5_0'] = {'ref': 'knit_005',
                            'false_data': False, 'input_data': 'f_rsc',
                            'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                            'output_type': 'class', 'label_name': 'Max(\delta f)', 'label_save': 'max_df',
                            'shuffle': True,
                            'channel': 3, 'nb_classes': 5, 'equipro': True,
                            'seq_size': 256, 'overlap_step': 1, 'futur': 1,
                            'batch_size': 512,
                            'criterion': {'name_criterion': 'cross_entropy_loss',
                                          'weight': None}}


exp_scalar['knit005_v1_2_1'] = {'ref': 'knit_005',
                            'false_data': False, 'input_data': 'f_rsc',
                            'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                            'output_type': 'class', 'label_name': 'Max(\delta f)', 'label_save': 'max_df',
                            'shuffle': True,
                            'channel': 3, 'nb_classes': 2, 'equipro': True,
                            'seq_size': 256, 'overlap_step': 1, 'futur': 20,
                            'batch_size': 512,
                            'criterion': {'name_criterion': 'cross_entropy_loss',
                                          'weight': None}}

exp_scalar['knit005_v1_3_1'] = {'ref': 'knit_005',
                            'false_data': False, 'input_data': 'f_rsc',
                            'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                            'output_type': 'class', 'label_name': 'Max(\delta f)', 'label_save': 'max_df',
                            'shuffle': True,
                            'channel': 3, 'nb_classes': 3, 'equipro': True,
                            'seq_size': 256, 'overlap_step': 1, 'futur': 20,
                            'batch_size': 512,
                            'criterion': {'name_criterion': 'cross_entropy_loss',
                                          'weight': None}}

exp_scalar['knit005_v1_4_1'] = {'ref': 'knit_005',
                            'false_data': False, 'input_data': 'f_rsc',
                            'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                            'output_type': 'class', 'label_name': 'Max(\delta f)', 'label_save': 'max_df',
                            'shuffle': True,
                            'channel': 3, 'nb_classes': 4, 'equipro': True,
                            'seq_size': 256, 'overlap_step': 1, 'futur': 20,
                            'batch_size': 512,
                            'criterion': {'name_criterion': 'cross_entropy_loss',
                                          'weight': None}}

exp_scalar['knit005_v1_5_1'] = {'ref': 'knit_005',
                            'false_data': False, 'input_data': 'f_rsc',
                            'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                            'output_type': 'class', 'label_name': 'Max(\delta f)', 'label_save': 'max_df',
                            'shuffle': True,
                            'channel': 3, 'nb_classes': 5, 'equipro': True,
                            'seq_size': 256, 'overlap_step': 1, 'futur': 20,
                            'batch_size': 512,
                            'criterion': {'name_criterion': 'cross_entropy_loss',
                                          'weight': None}}


exp_scalar['knit005_v1_2_2'] = {'ref': 'knit_005',
                            'false_data': False, 'input_data': 'f_rsc',
                            'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                            'output_type': 'class', 'label_name': 'Max(\delta f)', 'label_save': 'max_df',
                            'shuffle': True,
                            'channel': 3, 'nb_classes': 2, 'equipro': True,
                            'seq_size': 256, 'overlap_step': 1, 'futur': 40,
                            'batch_size': 512,
                            'criterion': {'name_criterion': 'cross_entropy_loss',
                                          'weight': None}}

exp_scalar['knit005_v1_3_2'] = {'ref': 'knit_005',
                            'false_data': False, 'input_data': 'f_rsc',
                            'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                            'output_type': 'class', 'label_name': 'Max(\delta f)', 'label_save': 'max_df',
                            'shuffle': True,
                            'channel': 3, 'nb_classes': 3, 'equipro': True,
                            'seq_size': 256, 'overlap_step': 1, 'futur': 40,
                            'batch_size': 512,
                            'criterion': {'name_criterion': 'cross_entropy_loss',
                                          'weight': None}}

exp_scalar['knit005_v1_4_2'] = {'ref': 'knit_005',
                            'false_data': False, 'input_data': 'f_rsc',
                            'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                            'output_type': 'class', 'label_name': 'Max(\delta f)', 'label_save': 'max_df',
                            'shuffle': True,
                            'channel': 3, 'nb_classes': 4, 'equipro': True,
                            'seq_size': 256, 'overlap_step': 1, 'futur': 40,
                            'batch_size': 512,
                            'criterion': {'name_criterion': 'cross_entropy_loss',
                                          'weight': None}}

exp_scalar['knit005_v1_5_2'] = {'ref': 'knit_005',
                            'false_data': False, 'input_data': 'f_rsc',
                            'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                            'output_type': 'class', 'label_name': 'Max(\delta f)', 'label_save': 'max_df',
                            'shuffle': True,
                            'channel': 3, 'nb_classes': 5, 'equipro': True,
                            'seq_size': 256, 'overlap_step': 1, 'futur': 40,
                            'batch_size': 512,
                            'criterion': {'name_criterion': 'cross_entropy_loss',
                                          'weight': None}}


exp_scalar['knit005_v1_2_3'] = {'ref': 'knit_005',
                            'false_data': False, 'input_data': 'f_rsc',
                            'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                            'output_type': 'class', 'label_name': 'Max(\delta f)', 'label_save': 'max_df',
                            'shuffle': True,
                            'channel': 3, 'nb_classes': 2, 'equipro': True,
                            'seq_size': 256, 'overlap_step': 1, 'futur': 60,
                            'batch_size': 512,
                            'criterion': {'name_criterion': 'cross_entropy_loss',
                                          'weight': None}}

exp_scalar['knit005_v1_3_3'] = {'ref': 'knit_005',
                            'false_data': False, 'input_data': 'f_rsc',
                            'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                            'output_type': 'class', 'label_name': 'Max(\delta f)', 'label_save': 'max_df',
                            'shuffle': True,
                            'channel': 3, 'nb_classes': 3, 'equipro': True,
                            'seq_size': 256, 'overlap_step': 1, 'futur': 60,
                            'batch_size': 512,
                            'criterion': {'name_criterion': 'cross_entropy_loss',
                                          'weight': None}}

exp_scalar['knit005_v1_4_3'] = {'ref': 'knit_005',
                            'false_data': False, 'input_data': 'f_rsc',
                            'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                            'output_type': 'class', 'label_name': 'Max(\delta f)', 'label_save': 'max_df',
                            'shuffle': True,
                            'channel': 3, 'nb_classes': 4, 'equipro': True,
                            'seq_size': 256, 'overlap_step': 1, 'futur': 60,
                            'batch_size': 512,
                            'criterion': {'name_criterion': 'cross_entropy_loss',
                                          'weight': None}}

exp_scalar['knit005_v1_5_3'] = {'ref': 'knit_005',
                            'false_data': False, 'input_data': 'f_rsc',
                            'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                            'output_type': 'class', 'label_name': 'Max(\delta f)', 'label_save': 'max_df',
                            'shuffle': True,
                            'channel': 3, 'nb_classes': 5, 'equipro': True,
                            'seq_size': 256, 'overlap_step': 1, 'futur': 60,
                            'batch_size': 512,
                            'criterion': {'name_criterion': 'cross_entropy_loss',
                                          'weight': None}}

exp_scalar['knit005_v1_5_4_512'] = {'ref': 'knit_005',
                            'false_data': False, 'input_data': 'f_rsc',
                            'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                            'output_type': 'class', 'label_name': 'Max(\delta f)', 'label_save': 'max_df',
                            'shuffle': True,
                            'channel': 3, 'nb_classes': 5, 'equipro': True,
                            'seq_size': 512, 'overlap_step': 1, 'futur': 125,
                            'batch_size': 512,
                            'criterion': {'name_criterion': 'cross_entropy_loss',
                                          'weight': None}}

#sum_df
exp_scalar['knit005_v2_2_1_0'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum(\delta f)', 'label_save': 'sum_df',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 1, 'nb_classes': 2, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 20,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss','weight': None}}

exp_scalar['knit005_v2_2_1_1'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum(\delta f)', 'label_save': 'sum_df',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 2, 'nb_classes': 2, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 20,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss','weight': None}}

exp_scalar['knit005_v2_2_1'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum(\delta f)', 'label_save': 'sum_df',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 2, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 20,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss','weight': None}}

exp_scalar['knit005_v2_2_2'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum(\delta f)', 'label_save': 'sum_df',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 2, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 40,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss','weight': None}}

exp_scalar['knit005_v2_2_3'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum(\delta f)', 'label_save': 'sum_df',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 2, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 60,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss','weight': None}}


exp_scalar['knit005_v2_3_1'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum(\delta f)', 'label_save': 'sum_df',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 3, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 20,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss','weight': None}}

exp_scalar['knit005_v2_3_2'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum(\delta f)', 'label_save': 'sum_df',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 3, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 40,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss','weight': None}}

exp_scalar['knit005_v2_3_3'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum(\delta f)', 'label_save': 'sum_df',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 3, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 60,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss','weight': None}}


exp_scalar['knit005_v2_4_1'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum(\delta f)', 'label_save': 'sum_df',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 4, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 20,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss','weight': None}}

exp_scalar['knit005_v2_4_2'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum(\delta f)', 'label_save': 'sum_df',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 4, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 40,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss','weight': None}}

exp_scalar['knit005_v2_4_3'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum(\delta f)', 'label_save': 'sum_df',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 4, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 60,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss','weight': None}}


exp_scalar['knit005_v2_5_1'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum(\delta f)', 'label_save': 'sum_df',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 5, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 20,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss','weight': None}}

exp_scalar['knit005_v2_5_2'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum(\delta f)', 'label_save': 'sum_df',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 5, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 40,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss','weight': None}}

exp_scalar['knit005_v2_5_3'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum(\delta f)', 'label_save': 'sum_df',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 5, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 60,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss','weight': None}}

exp_scalar['knit005_v2_5_4'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum(\delta f)', 'label_save': 'sum_df',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 5, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 125,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss','weight': None}}

exp_scalar['knit005_v2_5_4_512'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum(\delta f)', 'label_save': 'sum_df',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 5, 'equipro': True,
                             'seq_size': 512, 'overlap_step': 1, 'futur': 125,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss','weight': None}}

# sum expdf
exp_scalar['knit005_v3_2_1'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum (exp *\delta f)', 'label_save': 'sum_expdf',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 2, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 20, 'tau': 7,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss',
                                           'weight': None}}

exp_scalar['knit005_v3_2_2'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum (exp *\delta f)', 'label_save': 'sum_expdf',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 2, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 40, 'tau': 13,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss',
                                           'weight': None}}

exp_scalar['knit005_v3_2_3'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum (exp *\delta f)', 'label_save': 'sum_expdf',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 2, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 60, 'tau': 20,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss',
                                           'weight': None}}

exp_scalar['knit005_v3_2_1_64'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum (exp *\delta f)', 'label_save': 'sum_expdf',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 2, 'equipro': True,
                             'seq_size': 64, 'overlap_step': 1, 'futur': 20, 'tau': 7,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss',
                                           'weight': None}}


exp_scalar['knit005_v3_3_1'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum (exp *\delta f)', 'label_save': 'sum_expdf',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 3, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 20, 'tau': 7,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss',
                                           'weight': None}}

exp_scalar['knit005_v3_3_2'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum (exp *\delta f)', 'label_save': 'sum_expdf',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 3, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 40, 'tau': 13,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss',
                                           'weight': None}}

exp_scalar['knit005_v3_3_3'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum (exp *\delta f)', 'label_save': 'sum_expdf',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 3, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 60, 'tau': 20,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss',
                                           'weight': None}}

exp_scalar['knit005_v3_3_1_64'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum (exp *\delta f)', 'label_save': 'sum_expdf',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 3, 'equipro': True,
                             'seq_size': 64, 'overlap_step': 1, 'futur': 20, 'tau': 7,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss',
                                           'weight': None}}


exp_scalar['knit005_v3_4_1'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum (exp *\delta f)', 'label_save': 'sum_expdf',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 4, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 20, 'tau': 7,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss',
                                           'weight': None}}

exp_scalar['knit005_v3_4_2'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum (exp *\delta f)', 'label_save': 'sum_expdf',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 4, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 40, 'tau': 13,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss',
                                           'weight': None}}

exp_scalar['knit005_v3_4_3'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum (exp *\delta f)', 'label_save': 'sum_expdf',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 4, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 60, 'tau': 20,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss',
                                           'weight': None}}

exp_scalar['knit005_v3_4_1_64'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum (exp *\delta f)', 'label_save': 'sum_expdf',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 4, 'equipro': True,
                             'seq_size': 64, 'overlap_step': 1, 'futur': 20, 'tau': 7,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss',
                                           'weight': None}}


exp_scalar['knit005_v3_5_1'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum (exp *\delta f)', 'label_save': 'sum_expdf',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 5, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 20, 'tau': 7,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss',
                                           'weight': None}}

exp_scalar['knit005_v3_5_2'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum (exp *\delta f)', 'label_save': 'sum_expdf',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 5, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 40, 'tau': 13,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss',
                                           'weight': None}}

exp_scalar['knit005_v3_5_3'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum (exp *\delta f)', 'label_save': 'sum_expdf',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 5, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 60, 'tau': 20,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss',
                                           'weight': None}}


exp_scalar['knit005_v3_10_1'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum (exp *\delta f)', 'label_save': 'sum_expdf',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 10, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 20, 'tau': 7,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss',
                                           'weight': None}}


exp_scalar['knit005_v3_5_4'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum (exp *\delta f)', 'label_save': 'sum_expdf',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 5, 'equipro': True,
                             'seq_size': 256, 'overlap_step': 1, 'futur': 125, 'tau': 42,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss',
                                           'weight': None}}

exp_scalar['knit005_v3_5_4_512'] = {'ref': 'knit_005',
                             'false_data': False, 'input_data': 'f_rsc',
                             'NN_data': ['train', 'val', 'test'], 'fields': ['vit_x', 'vit_y', 'vort'],
                             'output_type': 'class', 'label_name': '\sum (exp *\delta f)', 'label_save': 'sum_expdf',
                             'label_rsc': None,
                             'shuffle': True,
                             'channel': 3, 'nb_classes': 5, 'equipro': True,
                             'seq_size': 512, 'overlap_step': 1, 'futur': 125, 'tau': 42,
                             'batch_size': 512,
                             'criterion': {'name_criterion': 'cross_entropy_loss',
                                           'weight': None}}

import warnings
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.metrics import classification_report
from matplotlib.text import TextPath
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

warnings.filterwarnings("ignore")
legend_properties = {'weight':'normal'}

from classConfig import ConfigData, ConfigPred
import Config_data
import Config_pred
from classPlot import ClassPlot
from Datas.classStat import Histo, Stat
from Datas.classSignal import SignalForce
from Datas.classEvent import ForceEvent
from Datas.classLoader import DataSetTIMESCALAR
from dictdata import dictdata

# %% ################### Args ##################################
remote = True

tau = 4

date = np.array(['210726', '210822', '210822', '210822', '210726', '210825', '210825', '210825', '210726', '210825', '210830', '210830',
        '210726', '210727', '210727', '210727', '210703', '210803', '210803', '210803', '210803', '210803', '210803', '210811',
        '210727', '210728', '210728', '210729', '210811', '210811', '210811', '210812', '210812', '210812', '210812', '210815'])


version_pred = np.array(['v1_2_1', 'v1_3_1', 'v1_4_1', 'v1_5_1', 'v1_2_2', 'v1_3_2', 'v1_4_2', 'v1_5_2', 'v1_2_3', 'v1_3_3', 'v1_4_3', 'v1_5_3',
                'v2_2_1', 'v2_3_1', 'v2_4_1', 'v2_5_1', 'v2_2_2', 'v2_3_2', 'v2_4_2', 'v2_5_2', 'v2_2_3', 'v2_3_3', 'v2_4_3', 'v2_5_3',
                'v3_2_1', 'v3_3_1', 'v3_4_1', 'v3_5_1', 'v3_2_2', 'v3_3_2', 'v3_4_2', 'v3_5_2', 'v3_2_3', 'v3_3_3', 'v3_4_3', 'v3_5_3'])
model = 5023

def rpz_sub(date, version_pred, sub_version=''):
    ref_tricot = 'knit005_'
    n_exp = 'mix_'
    version_data = 'v22'
    model = 23
    num_target = version_pred[1]
    num_tau = version_pred[-1]

    multi_features = True
    trainsize = 1000000


    class Args:
        def __init__(self):
            self.d = date
            self.remote = remote
            self.cuda = False
            self.vpred = version_pred
            self.subv = sub_version
            self.pred = 'scalar'
            self.verif_stats_output = False
            self.trainsize = trainsize


    args = Args()

    NAME_EXP = ref_tricot + n_exp + version_data
    config_data = ConfigData(Config_data.exp[NAME_EXP])

    NAME_EXP = ref_tricot + version_pred
    if args.pred == 'scalar':
        version_exp = Config_pred.exp_scalar[NAME_EXP]
        config_pred = ConfigPred(Config_pred.exp_scalar[NAME_EXP], config_data)
    else:
        version_exp = Config_pred.exp_img[NAME_EXP]
        config_pred = ConfigPred(Config_pred.exp_img[NAME_EXP], config_data)

    config_pred.set_model_attribute(model)
    if args.pred == 'scalar':
        model_spec = config_pred.spec_model_scalar()
    else:
        model_spec = config_pred.spec_model_img()

    histo = Histo(config_data)
    plot = ClassPlot(args.remote, histo)

    # %% ################### Create sequences ##################################

    dict_all_value = np.load(
        '/data/Douin/These/knit_quakes_forecast/knit_005/input/version22/sequence_NN/dict_all_value_reftau{}_for_class_{}_seqsize_{}_futur_{}.npy'.format(
            tau, config_pred.label_save, config_pred.seq_size, config_pred.futur), allow_pickle=True).flat[0]

    ## class edges
    classes_edges = np.load(
        '/data/Douin/These/knit_quakes_forecast/knit_005/input/version22/sequences_RL/{}_classes_edges_{}_futur_{}.npy'.format(
            config_pred.output_shape, config_pred.label_save,
            config_pred.futur))

    signal_flu = SignalForce(config_data, 'flu_rsc', 'train')
    signalevent = ForceEvent(config_data, signal_flu.f, signal_flu.ext, signal_flu.t,
                             'flu_rsc', 'train', Sm=False)

    # %% ################### save ##################################

    path_Tex = './representation_apprentissage_classification_on_reftau{}_set/'.format(tau)
    num_TeX = '{}_{}_{}'.format(num_target, config_pred.output_shape, num_tau)


    # %% ################### plot callback learning ##################################
    from pathlib import Path

    save_callback = '/data/Douin/These/knit_quakes_forecast/knit_005/output/version22/callback/scalar/callback_reftau{}_{}_{}_m{}_{}seq{}.npy'.format(
        tau, args.d, args.vpred, model_spec['magic_number'], args.trainsize, args.subv)

    fileName = save_callback
    fileObj = Path(fileName)
    is_fileObj = fileObj.is_file()
    if is_fileObj:
        callback = np.load(save_callback, allow_pickle=True).flat[0]

        # ------- loss and acc
        val_loss = callback['val_loss']
        val_acc = callback['val_acc']
        loss = callback['loss']
        acc = callback['acc']

        fig, ax = plot.belleFigure('${}$'.format('epoch'), '${}$'.format('loss'), nfigure=None)
        ax.plot(np.arange(np.size(loss)), loss, color='#1f77b4', marker='*', linestyle="None", label='train')
        ax.plot(np.arange(np.size(val_loss)), val_loss, color='#ff7f0e', marker='*', linestyle="None", label='val')
        save = path_Tex + '{}'.format('loss_') + num_TeX
        plot.fioritures(ax, fig, title=None, label=True,
                        grid=None, save=save)

        fig, ax = plot.belleFigure('${}$'.format('epochs'), '${}$'.format('acc'), nfigure=None)
        ax.plot(np.arange(np.size(acc)), acc, color='#1f77b4', marker='*', linestyle="None", label='train')
        ax.plot(np.arange(np.size(val_acc)), val_acc, color='#ff7f0e', marker='*', linestyle="None", label='val')
        save = path_Tex + '{}'.format('acc_') + num_TeX
        plot.fioritures(ax, fig, title=None, label=True, grid=None,
                        save=save)

    # %% ################### plot callback rpz classification ##################################

    def make_grid_classes(classes_edges):
        grid = np.zeros(np.size(classes_edges))
        for i in range(np.size(classes_edges)):
            grid[i] = classes_edges[i]
        return grid


    def accuracy_value_dispach(y_value, y_target, y_pred):
        g_value = []
        w_value = []

        for i in range(np.size(y_value)):
            if y_pred[i] == y_target[i]:
                g_value.append(y_value[i])
            else:
                w_value.append(y_value[i])

        return np.asarray(g_value), np.asarray(w_value)

    # ------- stat on pred
    save_test_callback = '/data/Douin/These/knit_quakes_forecast/knit_005/output/version22/callback/scalar/test_callback_reftau{}_{}_{}_m{}_{}seq{}.npy'.format(
        tau, args.d, args.vpred, model_spec['magic_number'], args.trainsize, args.subv)

    callback = np.load(save_test_callback, allow_pickle=True).flat[0]
    y_value = callback['y_value'] #.cpu().data.numpy()
    y_target = callback['y_target'] #.cpu().data.numpy()
    y_pred = callback['y_pred'] #.cpu().data.numpy()

    save_metric_supp = '/data/Douin/These/knit_quakes_forecast/knit_005/output/version22/callback/scalar/metric_supp_reftau{}_{}_{}_m{}_{}seq{}.npy'.format(
        tau, args.d, args.vpred, model_spec['magic_number'], args.trainsize, args.subv)

    # ----------- Class pour training
    nbclasses = config_pred.output_shape
    log_classes = classes_edges
    print('#---- training class')
    print(log_classes)
    print('---- le seuil interclasse est {}'.format(log_classes[1:-1]))

    conf = np.asarray(confusion_matrix(y_target, y_pred))
    print('--- count')
    print(conf)

    # ----------- Class learning
    if nbclasses == 2:
        x_conf = np.array([5e-4, 3e-1])
    elif nbclasses == 3:
        x_conf = np.array([5e-4, 3e-2, 2e0])
    elif nbclasses == 4:
        x_conf = np.array([5e-4, 1e-2, 1e-1, 2e0])
    else:
        x_conf = [3e-3, 1e-2, 1e-1, 1e0, 5e0]
    class_grid = np.concatenate((np.array([1e-4]), classes_edges[1::]))

    # ----------- Class physiques
    decade = np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 3e1])
    name_decade = ['decade_{}'.format(i) for i in range(decade.size - 1)]
    decade_nzero = np.array([5e-3, 3e-2, 3e-1, 3e0, 3e1])
    x_count = [9e-4, 1e-2, 1e-1, 1e0, 5e0]
    decade_grid = np.array([1e-4, 5e-3, 3e-2, 3e-1, 3e0,  1e1])
    major = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]

    decade_post_class = dictdata()
    for i in range(nbclasses):
        decade_c = np.hstack((np.array([log_classes[i + 1]]), decade[decade > log_classes[i + 1]]))
        decade_c[-1] = np.max(y_value[y_pred == i])
        decade_post_class.add('decade_post_{}'.format(i), decade_c)

    print('#---- class pysiques')
    print('- decades tot : {}'.format(np.round(decade, 3)))
    print('- decades supp physiqcal noise : {}'.format(np.round(decade_nzero, 3)))

    # %% # ----------- Targets
    title = True
    label = True
    grid = decade_grid
    fig, ax = plot.belleFigure('$log10$', '${}$'.format('Pdf_{log10}'), nfigure=None)
    y_Pdf, x_Pdf = plot.histo.my_histo(signalevent.df_tt, 1e-4,
                                       np.max(signalevent.df_tt),
                                       'log', 'log', density=2, binwidth=None, nbbin=70)

    ax.plot(x_Pdf, y_Pdf, '.', label='$\delta f$')
    y_Pdf, x_Pdf = plot.histo.my_histo(dict_all_value['test'], 1e-4,
                                       np.max(dict_all_value['test']),
                                       'log', 'log', density=2, binwidth=None, nbbin=70)

    ax.plot(x_Pdf, y_Pdf, '.', label='${}$'.format(config_pred.label_name))
    plot.plt.xscale('log')
    plot.plt.yscale('log')
    for i in range(nbclasses + 1):
        plot.plt.axvline(x=class_grid[i], color='k')
    save = None
    plot.fioritures(ax, fig, title=None, label=label, grid=grid, save=save, major=major)

    yname = config_pred.label_name
    ysave = config_pred.label_save
    title = True
    label = True
    grid = decade_grid
    fig, ax = plot.belleFigure('$log10({})$'.format(yname), '${}({})$'.format('Count_{log10}', yname), nfigure=None)
    for i in range(nbclasses):
        where = np.where((y_value >= log_classes[i]) & (y_value <= log_classes[i + 1]))[0]
        y_Pdf, x_Pdf = plot.histo.my_histo(y_value[where], log_classes[i] if log_classes[i] != 0 else 1e-4,
                                           np.max(y_value[where]),
                                           'log', 'log', density=1, binwidth=None, nbbin=70)

        ax.plot(x_Pdf, y_Pdf, '.', label='class {}'.format(i))
    plot.plt.xscale('log')
    plot.plt.yscale('log')
    for i in range(nbclasses + 1):
        plot.plt.axvline(x=class_grid[i], color='k')
    save = None
    plot.fioritures(ax, fig, title=None, label=label, grid=grid, save=save, major=major)

    yname = '\delta f'
    ysave = 'df'
    title = True
    label = True
    grid = decade_grid
    fig, ax = plot.belleFigure('$log10({})$'.format(yname), '${}({})$'.format('Pdf_{log10}', yname), nfigure=None)
    for i in range(nbclasses):
        where = np.where((signalevent.df_tt >= log_classes[i]) & (signalevent.df_tt <= log_classes[i + 1]))[0]
        y_Pdf, x_Pdf = plot.histo.my_histo(signalevent.df_tt[where], log_classes[i] if log_classes[i] != 0 else 1e-4,
                                           np.max(signalevent.df_tt[where]),
                                           'log', 'log', density=1, binwidth=None, nbbin=70)

        ax.plot(x_Pdf, y_Pdf, '.', label='class {}'.format(i))
    plot.plt.xscale('log')
    plot.plt.yscale('log')
    for i in range(nbclasses + 1):
        plot.plt.axvline(x=class_grid[i], color='k')
    save = None
    plot.fioritures(ax, fig, title=None, label=label, grid=grid, save=save, major=major)

    # %% # ----------- Normalisations
    norm_all = dict_all_value['test'].size
    norm_pred = np.asarray([np.sum(conf[:, i]) for i in range(nbclasses)])
    norm_target = np.asarray([np.sum(conf[i, :]) for i in range(nbclasses)])
    norm_decade, _ = np.histogram(dict_all_value['test'], bins=decade, density=False)

    print('#---- normalisation')
    print('il y a {} elmnts dans testset'.format(norm_all))
    print('il y a {} target elmnts par classe soit {}'.format(norm_target, np.round(norm_target / norm_all * 100, 0).astype(int)))
    print('il y a {} pred elmnts par classe soit {}%'.format(norm_pred, np.round(norm_pred / norm_all * 100, 0).astype(int)))
    print('il y a {} elmnts par decade soit {}%'.format(norm_decade, np.round(norm_decade / norm_all * 100, 0).astype(int)))

    # %% # ----------- standart stats

    pred_target_metric = dictdata()

    pred_target_metric.add('conf', conf)

    pred_target_metric.add('acc', np.round(np.sum(conf[i, i] / norm_all for i in range(nbclasses)), 2))

    pred_target_metric.add('precision', np.round([conf[i, i] / norm_pred[i] for i in range(nbclasses)], 2))
    pred_target_metric.add('micro_precision', np.round(
        np.sum(conf[i, i] for i in range(nbclasses)) / np.sum(norm_pred[i] for i in range(nbclasses)), 2))
    pred_target_metric.add('macro_precision',
                           np.round(np.sum(conf[i, i] / norm_pred[i] for i in range(nbclasses)) / nbclasses, 2))

    pred_target_metric.add('recall', np.round([conf[i, i] / norm_target[i] for i in range(nbclasses)], 2))
    pred_target_metric.add('micro_recall', np.round(
        np.sum(conf[i, i] for i in range(nbclasses)) / np.sum(norm_target[i] for i in range(nbclasses)), 2))
    pred_target_metric.add('macro_recall',
                           np.round(np.sum(conf[i, i] / norm_target[i] for i in range(nbclasses)) / nbclasses, 2))

    pred_target_metric.add('f1-score', 2 * pred_target_metric['precision'] * pred_target_metric['recall'] / (
                pred_target_metric['precision'] + pred_target_metric['recall']))
    pred_target_metric.add('micro_f1-score',
                           2 * pred_target_metric['micro_precision'] * pred_target_metric['micro_recall'] / (
                                       pred_target_metric['micro_precision'] + pred_target_metric['micro_recall']))
    pred_target_metric.add('macro_f1-score',
                           2 * pred_target_metric['macro_precision'] * pred_target_metric['macro_recall'] / (
                                       pred_target_metric['macro_precision'] + pred_target_metric['macro_recall']))

    # %% # ----------- standart metrics
    print('#---- stadart metric')
    print('acc = {} \n \n'.format(pred_target_metric['acc']))

    print('precision = {} par class'.format(pred_target_metric['precision']))
    print('macro precision = {}\n \n'.format((pred_target_metric['macro_precision'])))

    print('recall = {} par class'.format(pred_target_metric['recall']))
    print('macro recall = {} \n \n'.format((pred_target_metric['macro_recall'])))

    print('f1-score = {} par class'.format(pred_target_metric['f1-score']))
    print('macro f1-score = {} \n \n'.format((pred_target_metric['macro_f1-score'])))

    # print('micro precision = {}'.format((pred_target_metric['micro_precision'])))
    # print('micro recall = {}'.format((pred_target_metric['micro_recall'])))
    # print('micro f1-score = {} \n \n'.format((pred_target_metric['micro_f1-score'])))

    # %% # ----------- Preds
    value_good_classe, value_wrong_classe = accuracy_value_dispach(y_value, y_target, y_pred)

    grid = make_grid_classes(log_classes)
    fig, ax = plot.belleFigure('${}$'.format('bla'), '${}$'.format('values'), nfigure=None)
    ax.plot(np.arange(np.size(value_good_classe)), value_good_classe, '.', label='good class')
    ax.plot(np.arange(np.size(value_wrong_classe)), value_wrong_classe, 'r.', label='wrong class')
    ax.set_yticks(grid, minor=False)
    ax.yaxis.grid(True)
    save = None
    plot.fioritures(ax, fig, title=None, label=None, grid=None, save=None)

    value_wrong_classe = value_wrong_classe[value_wrong_classe != 0]
    value_good_classe = value_good_classe[value_good_classe != 0]

    stats_good_v = Stat(config_data, value_good_classe)
    stats_wrong_v = Stat(config_data, value_wrong_classe)

    good_y_Pdf, good_x_Pdf = plot.histo.my_histo(value_good_classe, 1e-4, stats_good_v.max,
                                                 'log', 'log', density=2, binwidth=None, nbbin=70)
    wrong_y_Pdf, wrong_x_Pdf = plot.histo.my_histo(value_wrong_classe, 1e-4, stats_good_v.max,
                                                   'log', 'log', density=2, binwidth=None, nbbin=70)

    yname = config_pred.label_name
    ysave = config_pred.label_save
    title = True
    label = True
    grid = decade_grid
    fig, ax = plot.belleFigure('$log10({})$'.format(yname), '${}({})$'.format('Pdf_{log10}', yname), nfigure=None)
    ax.plot(good_x_Pdf, good_y_Pdf, '.', label='good pred')
    ax.plot(wrong_x_Pdf, wrong_y_Pdf, 'r.', label='wrong pred')
    plot.plt.xscale('log')
    plot.plt.yscale('log')
    for i in range(nbclasses + 1):
        plot.plt.axvline(x=class_grid[i], color='k')
    save = path_Tex + '{}'.format('Pdf_output_values_density2_') + num_TeX
    plot.fioritures(ax, fig, title=None, label=label, grid=decade_grid, save=save)

    X_good = [0 for i in range(nbclasses)]
    Y_good = [0 for i in range(nbclasses)]

    X_wrong = [0 for i in range(nbclasses)]
    Y_wrong = [0 for i in range(nbclasses)]

    for i in range(nbclasses):
        df_seuil_good = signalevent.df_seuil_fast(value_good_classe, log_classes[i], log_classes[i + 1])
        df_seuil_wrong = signalevent.df_seuil_fast(value_wrong_classe, log_classes[i],
                                                      log_classes[i + 1])

        if np.size(df_seuil_good) != 0:
            stats_good_v = Stat(config_data, df_seuil_good)
            Y_good[i], X_good[i] = histo.my_histo(df_seuil_good, 1e-4, stats_good_v.max, 'log',
                                                  'log', density=1, binwidth=None, nbbin=70)
        else:
            Y_good[i] = None
            X_good[i] = None

        if np.size(df_seuil_wrong) != 0:
            stats_wrong_v = Stat(config_data, df_seuil_wrong)

            Y_wrong[i], X_wrong[i] = histo.my_histo(df_seuil_wrong, 1e-4, stats_good_v.max, 'log',
                                                    'log', density=1, binwidth=None, nbbin=70)
        else:
            Y_wrong[i] = None
            X_wrong[i] = None

    fig, ax = plot.belleFigure('$log10({})$'.format(yname), '${}({})$'.format('Count_{log10}', yname), nfigure=None)
    for i in range(nbclasses):
        if X_good[i] is not None:
            ax.plot(X_good[i], Y_good[i], color='#1f77b4', marker='.', linestyle="None",
                    label='good ; seuil =' + str(log_classes[i]) + 'N')
        if X_wrong[i] is not None:
            ax.plot(X_wrong[i], Y_wrong[i], 'r.',
                    label='wrong; seuil =' + str(log_classes[i]) + 'N')
    plot.plt.xscale('log')
    plot.plt.yscale('log')
    patchs = [mlines.Line2D([], [], color='#1f77b4', marker='.', linestyle='None', label='good pred'),
              mlines.Line2D([], [],color='r', marker='.', linestyle='None', label='wrong pred')]
    for i in range(nbclasses + 1):
        plot.plt.axvline(x=class_grid[i], color='k')
    save = path_Tex + '{}'.format('Pdf_output_values_density1_') + num_TeX
    plot.plt.legend(prop=legend_properties,handles=patchs)
    plot.plt.show()
    plot.fioritures(ax, fig, title=None, label=None, grid=decade_grid, save=save)

    colorpred = ['C0', 'C1', 'C2', 'C3', 'C4']
    symboletarget = ['0', '1', '2', '3', '4']
    stats_name = 'P(pred|target)'
    fig, ax = plot.belleFigure('$log10({})$ - classes targets'.format(yname), '${}$'.format(stats_name), nfigure=None)
    for i in range(nbclasses):
        for k in range(nbclasses):
            y = (conf[i, k] / norm_target[i])
            marker = TextPath((0, 0), symboletarget[i])
            # marker = mpath.Path(marker)
            ax.plot(x_conf[i], y, '.', marker=marker, markersize=20, color=colorpred[k])
            plot.plt.plot([class_grid[i], class_grid[i+1]], [y, y], color=colorpred[k])
        plot.plt.xscale('log')
    patchs = [0 for _ in range(nbclasses)]
    for i in range(nbclasses+1):
        plot.plt.axvline(x=class_grid[i], color='k')
    for i in range(nbclasses):
        patchs[i] = mpatches.Patch(color=colorpred[i], label='pred = {}'.format(i))
    plot.plt.legend(prop=legend_properties,handles=patchs)
    plot.plt.show()
    save = path_Tex + '{}'.format('p_pred_target_') + num_TeX
    plot.fioritures(ax, fig, title=None, label=None, grid=class_grid, save=save)

    stats_name = 'P(target|pred)'
    fig, ax = plot.belleFigure('$log10({})$ - classes prediction'.format(yname), '${}$'.format(stats_name), nfigure=None)
    for i in range(nbclasses):
        for k in range(nbclasses):
            y = (conf[i, k] / norm_pred[k])
            marker = TextPath((0, 0), symboletarget[i])
            # marker = mpath.Path(marker)
            ax.plot(x_conf[k], y, '.', marker=marker, markersize=20, color=colorpred[k])
            plot.plt.plot([class_grid[k], class_grid[k+1]], [y, y], color=colorpred[k])
        plot.plt.xscale('log')
    patchs = [0 for _ in range(nbclasses)]
    for i in range(nbclasses+1):
        plot.plt.axvline(x=class_grid[i], color='k')
    for i in range(nbclasses):
        patchs[i] = mpatches.Patch(color=colorpred[i], label='pred = {}'.format(i))
    plot.plt.legend(prop=legend_properties,handles=patchs)
    plot.plt.show()
    save = path_Tex + '{}'.format('p_target_pred_') + num_TeX
    plot.fioritures(ax, fig, title=None, label=None, grid=class_grid, save=save)

    # %%# ----------- value on decades
    # -- pred_decades
    conf_decade_on_pred = np.zeros((decade.size - 1, nbclasses))
    for i in range(nbclasses):
        sub_value = y_value[y_pred == i]
        count, _ = np.histogram(sub_value, bins=decade, density=False)
        conf_decade_on_pred[:, i] = count

    pred_decades_metric = dictdata()

    p_decade_sachant_pred = dictdata()
    for i in range(nbclasses):
        p_decade_sachant_pred.add('pred_class_{}'.format(i), np.round([conf_decade_on_pred[:, i] / norm_pred[i]], 2)[0])

    p_pred_sachant_decade = dictdata()
    for i in range(nbclasses):
        stats = np.array([np.round([conf_decade_on_pred[j, i] / norm_decade[j]], 2)[0] for j in range(norm_decade.size)])
        p_pred_sachant_decade.add('pred_class_{}'.format(i), stats)

    pred_decades_metric.add('conf_decade', conf_decade_on_pred)
    pred_decades_metric.add('p_decade_sachant_pred', p_decade_sachant_pred)
    pred_decades_metric.add('p_pred_sachant_decade', p_pred_sachant_decade)

    # print('--- count en term de decades par pred')
    # print(conf_decade_on_pred)
    # print('#---- pred/decades metrics')
    # print('P(decade=i|pred=j) = \n {}'.format(p_decade_sachant_pred))
    #
    print('P(pred=i|decade=j) = \n{}'.format(p_pred_sachant_decade))
    for j in range(nbclasses):
        print('P((pred={}|decade=i) = {} \n'.format(j, p_pred_sachant_decade['pred_class_{}'.format(j)]))

    stats_name = 'P(decade|pred)'
    fig, ax = plot.belleFigure('$log10({})$'.format(yname), '${}$'.format(stats_name), nfigure=None)
    for i in range(nbclasses):
        ax.plot(x_count, p_decade_sachant_pred['pred_class_{}'.format(i)], '.', marker="None")
        for k in range(decade.size - 1):
            if p_decade_sachant_pred['pred_class_{}'.format(i)][k] != 0:
                plot.plt.plot([decade[k], decade[k + 1]], [p_decade_sachant_pred['pred_class_{}'.format(i)][k], p_decade_sachant_pred['pred_class_{}'.format(i)][k]], color=colorpred[i])
    plot.plt.xscale('log')
    save = None
    patchs = [0 for _ in range(nbclasses)]
    plot.fioritures(ax, fig, title=None, label=False, grid=decade_grid, save=save)
    for i in range(nbclasses):
        patchs[i] = mpatches.Patch(color=colorpred[i], label='pred = {}'.format(i))
    plot.plt.legend(prop=legend_properties,handles=patchs)
    plot.plt.show()


    stats_name = 'P(pred|decade)'
    fig, ax = plot.belleFigure('$log10({})$'.format(yname), '${}$'.format(stats_name), nfigure=None)
    for i in range(nbclasses):
        ax.plot(x_count, p_pred_sachant_decade['pred_class_{}'.format(i)], '.', marker="None")
        for k in range(decade.size - 1):
            if p_decade_sachant_pred['pred_class_{}'.format(i)][k] != 0:
                plot.plt.plot([decade[k], decade[k + 1]], [p_pred_sachant_decade['pred_class_{}'.format(i)][k], p_pred_sachant_decade['pred_class_{}'.format(i)][k]], color=colorpred[i])
    plot.plt.xscale('log')
    save = None
    patchs = [0 for _ in range(nbclasses)]
    plot.fioritures(ax, fig, title=None, label=False, grid=decade_grid, save=save)
    for i in range(nbclasses):
        patchs[i] = mpatches.Patch(color=colorpred[i], label='pred = {}'.format(i))
    plot.plt.legend(prop=legend_properties,handles=patchs)
    plot.plt.show()

    # -- pred_target_decades
    conf_decade_on_pred_target = np.zeros((decade.size - 1, nbclasses, nbclasses))
    for i in range(nbclasses):
        for j in range(nbclasses):
            where = np.where((y_pred == j) & (y_target == i))
            sub_value = y_value[where]
            count, _ = np.histogram(sub_value, bins=decade, density=False)
            conf_decade_on_pred_target[:, i, j] = count

    pred_target_decades_metric = dictdata()

    p_decade_target_sachant_pred = dictdata()
    for i in range(nbclasses):
        for j in range(nbclasses):
            p_decade_target_sachant_pred.add('target_class_{}_pred_class_{}'.format(i, j), np.round(conf_decade_on_pred_target[:, i, j] / norm_pred[j], 2))

    p_decade_pred_sachant_target = dictdata()
    for i in range(nbclasses):
        for j in range(nbclasses):
            p_decade_pred_sachant_target.add('target_class_{}_pred_class_{}'.format(i, j), np.round(conf_decade_on_pred_target[:, i, j] / norm_target[i], 2))

    p_pred_target_sachant_decade = dictdata()
    for i in range(nbclasses):
        for j in range(nbclasses):
            stats = np.array(
                [np.round([conf_decade_on_pred_target[k, i, j] / norm_decade[k]], 2)[0] for k in range(norm_decade.size)])
            p_pred_target_sachant_decade.add('target_class_{}_pred_class_{}'.format(i, j), stats)

    pred_target_decades_metric.add('conf_decade_on_pred_target', conf_decade_on_pred_target)
    pred_target_decades_metric.add('p_decade_target_sachant_pred', p_decade_target_sachant_pred)
    pred_target_decades_metric.add('p_decade_pred_sachant_target', p_decade_pred_sachant_target)
    pred_target_decades_metric.add('p_pred_sachant_decade', p_pred_target_sachant_decade)

    precision_decade = np.round(np.sum(conf_decade_on_pred_target[:, j, j] / conf[j, j] for j in range(nbclasses)), 2)
    recall_decade = np.round(np.sum(conf_decade_on_pred_target[:, j , j] for j in range(nbclasses)) / norm_decade, 2)

    micro_recall_decade = np.round(np.sum(np.sum(conf_decade_on_pred_target[:, i, i] for i in range(nbclasses)))/norm_all, 2)
    micro_precision_decade = np.round(np.sum(np.sum(conf_decade_on_pred_target[:, i, i] for i in range(nbclasses)))/norm_all, 2)

    macro_precision_decade = np.round(np.sum(np.sum(conf_decade_on_pred_target[:, j, j] / conf[j, j] for j in range(nbclasses))) / (decade.size-1), 2)
    macro_recall_decade = np.round(np.sum(np.sum(conf_decade_on_pred_target[:, j , j] for j in range(nbclasses)) / norm_decade) / (decade.size-1), 2)
    macro_f1_score_decade = np.round(2 * macro_precision_decade * macro_recall_decade / (
                macro_precision_decade + macro_recall_decade), 2)

    pred_target_decades_metric.add('precision_decade', precision_decade)
    pred_target_decades_metric.add('recall_decade', recall_decade)
    pred_target_decades_metric.add('micro_precision_decade', micro_precision_decade)
    pred_target_decades_metric.add('micro_recall_decade', micro_recall_decade)
    pred_target_decades_metric.add('macro_precision_decade', macro_precision_decade)
    pred_target_decades_metric.add('macro_recall_decade', macro_recall_decade)
    pred_target_decades_metric.add('macro_f1_score_decade', macro_f1_score_decade)

    # print('--- count en term de decades par pred_target')
    # for i in range(decade.size-1):
    #     print('pour decade = {}'.format(i))
    #     print(conf_decade_on_pred_target[i, :, :])
    #
    #
    # print('#---- pred target decade count')
    # for i in range(nbclasses):
    #     for j in range(nbclasses):
    #         print('P(decade=i & pred={} & target={}) = {} \n'.format(j, i, conf_decade_on_pred_target[:, i, j]))
    #
    # print('#---- decades_target/pred metrics')
    # for i in range(nbclasses):
    #     for j in range(nbclasses):
    #         print('P((decade=i & target={})|pred={}) = {} \n'.format(j, i, p_decade_target_sachant_pred['target_class_{}_pred_class_{}'.format(j, i)]))
    #
    #
    # print('#---- decades_pred/target metrics')
    # for i in range(nbclasses):
    #     for j in range(nbclasses):
    #         print('P((decade=i & pred={})|target={}) = {} \n'.format(i, j, p_decade_pred_sachant_target['target_class_{}_pred_class_{}'.format(j, i)]))
    #
    print('#---- pred_target/decades metrics')
    for i in range(nbclasses):
        for j in range(nbclasses):
            print('P((pred={} & target={}|decade=i) = {} \n'.format(j, i, p_pred_target_sachant_decade['target_class_{}_pred_class_{}'.format(i, j)]))

    colorpred = ['C0', 'C1', 'C2', 'C3', 'C4']
    symboletarget = ['0', '1', '2', '3', '4']
    x_count = np.asarray(x_count)
    fig, ax = plot.belleFigure('$log10({})$ - classes target'.format('\Sigma(\delta f)'), '${}$'.format('P((decade \cup target)|pred)'), nfigure=None)
    for i in range(nbclasses):
        for j in range(nbclasses):
            y = p_decade_target_sachant_pred['target_class_{}_pred_class_{}'.format(i, j)]
            # print(np.sum(y))
            marker = TextPath((0, 0), symboletarget[i])
            # marker = mpath.Path(marker)
            ax.plot(x_count[y != 0], y[y != 0], '.', marker=marker, markersize=20, color=colorpred[j])
            for k in range(decade.size-1):
                if y[k] != 0:
                    plot.plt.plot([decade_grid[k], decade_grid[k+1]], [y[k], y[k]], color=colorpred[j])
        plot.plt.xscale('log')
    patchs = [0 for _ in range(nbclasses)]
    for i in range(nbclasses+1):
        plot.plt.axvline(x=class_grid[i], color='k')
    save = path_Tex + '{}'.format('p_target_decade_pred_') + num_TeX
    for i in range(nbclasses):
        patchs[i] = mpatches.Patch(color=colorpred[i], label='pred = {}'.format(i))
    plot.plt.legend(prop=legend_properties,handles=patchs)
    plot.plt.show()
    plot.fioritures(ax, fig, title=None, label=None, grid=decade_grid, save=save)


    fig, ax = plot.belleFigure('$log10({})$ - classes target'.format('\Sigma(\delta f)'), '${}$'.format('P((pred \cup target)|decade)'), nfigure=None)
    for i in range(nbclasses):
        for j in range(nbclasses):
            y = p_pred_target_sachant_decade['target_class_{}_pred_class_{}'.format(i, j)]
            marker = TextPath((0, 0), symboletarget[i])
            # marker = mpath.Path(marker)
            ax.plot(x_count[y != 0], y[y != 0], '.', marker=marker, markersize=20, color=colorpred[j])
            for k in range(decade.size-1):
                if y[k] != 0:
                    plot.plt.plot([decade_grid[k], decade_grid[k+1]], [y[k], y[k]], color=colorpred[j])
        plot.plt.xscale('log')
    patchs = [0 for _ in range(nbclasses)]
    for i in range(nbclasses+1):
        plot.plt.axvline(x=class_grid[i], color='k')
    save = path_Tex + '{}'.format('p_target_pred_decade_') + num_TeX
    for i in range(nbclasses):
        patchs[i] = mpatches.Patch(color=colorpred[i], label='pred = {}'.format(i))
    plot.plt.legend(prop=legend_properties,handles=patchs)
    plot.plt.show()
    plot.fioritures(ax, fig, title=None, label=None, grid=decade_grid, save=save)
    #
    # fig, ax = plot.belleFigure('$log10({})$'.format('\Sigma(\delta f)'), '${}$'.format('P((decade \cup pred)|target)'), nfigure=None)
    # for i in range(nbclasses):
    #     for j in range(nbclasses):
    #         y = p_decade_pred_sachant_target['target_class_{}_pred_class_{}'.format(i, j)]
    #         # print(np.sum(y))
    #         marker = TextPath((0, 0), symboletarget[i])
    #         # marker = mpath.Path(marker)
    #         ax.plot(x_count[y != 0], y[y != 0], '.', marker=marker, markersize=20, color=colorpred[j])
    #         for k in range(decade.size-1):
    #             if y[k] != 0:
    #                 plot.plt.plot([decade[k], decade[k+1]], [y[k], y[k]], color=colorpred[j])
    #     plot.plt.xscale('log')
    # patchs = [0 for _ in range(nbclasses)]
    # for i in range(nbclasses+1):
    #     plot.plt.axvline(x=class_grid[i], color='k')
    # save = path_Tex + '{}'.format() + num_TeX
    # plot.fioritures(ax, fig, title=None, label=False, grid=decade_grid, save=save)
    # for i in range(nbclasses):
    #     patchs[i] = mpatches.Patch(color=colorpred[i], label='pred = {}'.format(i))
    # plot.plt.legend(prop=legend_properties,handles=patchs)
    # plot.plt.show()

    # print('#---- on verif avec metric standart')
    # for i in range(nbclasses):
    #     print('precision class {} = {} '.format(i, pred_target_metric['precision'][i]))
    #     sum_pred = np.round(np.sum(p_decade_target_sachant_pred['target_class_{}_pred_class_{}'.format(i, i)]), 2)
    #     print('sum P((decade & target)|pred) class {} = {} \n'.format(i, sum_pred))
    #
    #     print('recall class {} = {}'.format(i, pred_target_metric['recall'][i]))
    #     sum_recall = np.round(np.sum(p_decade_pred_sachant_target['target_class_{}_pred_class_{}'.format(i, i)]), 2)
    #     print('sum P((decade & pred)|target) class {} = {} \n'.format(i, sum_recall))
    #
    #

    print('accuracy standart = {} \n'.format(pred_target_metric['acc']))

    # print('precision decade = {}'.format(np.round(pred_target_decades_metric['precision_decade'], 2)))
    print('recall decade = {} \n'.format(np.round(pred_target_decades_metric['recall_decade'], 2)))

    # print('micro precision decade = {}'.format(pred_target_decades_metric['micro_precision_decade']))
    # print('micro recall decade = {} \n'.format(pred_target_decades_metric['micro_recall_decade']))

    # print('macro precision decade = {}'.format(pred_target_decades_metric['macro_precision_decade']))
    print('macro recall decade = {}'.format(pred_target_decades_metric['macro_recall_decade']))
    # print('macro f1_score decade = {} \n'.format(pred_target_decades_metric['macro_f1_score_decade']))

    # ----------- scores
    scores = dictdata()
    mean_value_in_decade = np.zeros(decade.size-1)
    proportion_decades = np.round(norm_decade / norm_all, 2)
    for i in range(decade.size-1):
        if i == decade.size-2:
            where = np.where((y_value >= decade[i]) & (y_value <= decade[i+1]))
        else:
            where = np.where((y_value >= decade[i]) & (y_value < decade[i + 1]))
        mean_value_in_decade[i] = np.mean(y_value[where])

    for i in range(nbclasses):
        r_p = np.sum(conf_decade_on_pred_target[:, j, i] for j in range(i+1, nbclasses)) / norm_decade
        scores.add('r_pred_{}'.format(i), np.round(r_p, 2))
        scores.add('R_max_pred_{}'.format(i), np.round(np.sum([r_p[j] * decade[j+1] for j in range(decade.size-1)]), 2))
        scores.add('R_mean_pred_{}'.format(i), np.round(np.sum([r_p[j] * mean_value_in_decade[j] for j in range(decade.size - 1)]), 2))

        c_p = np.sum(conf_decade_on_pred_target[:, j, i] for j in range(i)) / norm_decade
        scores.add('c_pred_{}'.format(i), np.round(c_p, 2))
        scores.add('C_proportion_pred_{}'.format(i), np.round(np.sum([c_p[j] * proportion_decades[j] for j in range(decade.size - 1)]), 2))

        c_t = np.sum(conf_decade_on_pred_target[:, i, j] for j in range(i + 1, nbclasses)) / norm_decade
        scores.add('c_target_{}'.format(i), np.round(c_t, 2))
        scores.add('C_proportion_target_{}'.format(i),
                   np.round(np.sum([c_t[j] * proportion_decades[j] for j in range(decade.size - 1)]), 2))

        s_c = conf_decade_on_pred_target[:, i, i] / conf[i, i]
        s_p = conf_decade_on_pred_target[:, i, i] / norm_pred[i]
        s_t = conf_decade_on_pred_target[:, i, i] / norm_target[i]

    scores.add('R_mean_pred', np.array([scores['R_mean_pred_{}'.format(i)] for i in range(nbclasses)]))
    scores.add('C_proportion_pred', np.array([scores['C_proportion_pred_{}'.format(i)] for i in range(nbclasses)]))
    scores.add('C_proportion_target', np.array([scores['C_proportion_target_{}'.format(i)] for i in range(nbclasses)]))

    print('#---- scores')
    for i in range(nbclasses):
        print('risk pred={} : {}'.format(i, np.round(scores['r_pred_{}'.format(i)], 2)))
        print('R max pred={} : {}'.format(i, np.round(scores['R_max_pred_{}'.format(i)], 2)))
        print('R mean pred={} : {} \n'.format(i, np.round(scores['R_mean_pred_{}'.format(i)], 2)))

    print('\n')

    for i in range(nbclasses):
        print('cost pred={} : {}'.format(i, np.round(scores['c_pred_{}'.format(i)], 2)))
        print('C prop pred={} : {} \n'.format(i, np.round(scores['C_proportion_pred_{}'.format(i)], 3)))
        print('cost target={} : {}'.format(i, np.round(scores['c_target_{}'.format(i)], 2)))
        print('C prop target={} : {} \n'.format(i, np.round(scores['C_proportion_target_{}'.format(i)], 3)))

    ## save

    metric_supp = dictdata()

    metric_supp.add('pred_target_metric', pred_target_metric)
    metric_supp.add('pred_decades_metric', pred_decades_metric)
    metric_supp.add('pred_target_decades_metric', pred_target_decades_metric)
    metric_supp.add('scores', scores)


    ### retrospective acc

    # ------- change N classes to new N
    def replace(y_target, y_pred, a, b):
        where_target = np.where(y_target == a)[0]
        where_pred = np.where(y_pred == a)[0]

        y_target[where_target] = b
        y_pred[where_pred] = b

        return y_target, y_pred

    def back_classes(y_target, y_pred, old, new):
        if old == 5:
            y_target, y_pred = replace(y_target, y_pred, 4, 3)
            if new == old - 1:
                return y_target, y_pred
            else:
                old = old - 1
        if old == 4:
            y_target, y_pred = replace(y_target, y_pred, 2, 1)

            y_target, y_pred = replace(y_target, y_pred, 3, 2)
            if new == old - 1:
                return y_target, y_pred
            else:
                old = old - 1
        if old == 3:
            y_target, y_pred = replace(y_target, y_pred, 2, 1)
            return y_target, y_pred

    for i in range(nbclasses-1, 1, -1):
        new_nb_classes = i
        y_target, y_pred = back_classes(y_target, y_pred, config_pred.output_shape, new_nb_classes)

        # ----------- Class pour training
        nbclasses = new_nb_classes
        log_classes = classes_edges
        print('#---- training class')
        print(log_classes)
        print('---- le seuil interclasse est {}'.format(log_classes[1:-1]))

        conf = np.asarray(confusion_matrix(y_target, y_pred))
        print('--- count')
        print(conf)

        # %% # ----------- Normalisations
        norm_all = dict_all_value['test'].size
        norm_pred = np.asarray([np.sum(conf[:, i]) for i in range(nbclasses)])
        norm_target = np.asarray([np.sum(conf[i, :]) for i in range(nbclasses)])
        norm_decade, _ = np.histogram(dict_all_value['test'], bins=decade, density=False)

        # %% # ----------- standart stats

        acc_retro = dictdata()

        acc_retro.add('conf', conf)

        acc_retro.add('acc', np.round(np.sum(conf[i, i] / norm_all for i in range(nbclasses)), 2))

        acc_retro.add('precision', np.round([conf[i, i] / norm_pred[i] for i in range(nbclasses)], 2))
        acc_retro.add('micro_precision', np.round(
            np.sum(conf[i, i] for i in range(nbclasses)) / np.sum(norm_pred[i] for i in range(nbclasses)), 2))
        acc_retro.add('macro_precision',
                               np.round(np.sum(conf[i, i] / norm_pred[i] for i in range(nbclasses)) / nbclasses, 2))

        acc_retro.add('recall', np.round([conf[i, i] / norm_target[i] for i in range(nbclasses)], 2))
        acc_retro.add('micro_recall', np.round(
            np.sum(conf[i, i] for i in range(nbclasses)) / np.sum(norm_target[i] for i in range(nbclasses)), 2))
        acc_retro.add('macro_recall',
                               np.round(np.sum(conf[i, i] / norm_target[i] for i in range(nbclasses)) / nbclasses, 2))

        acc_retro.add('f1-score', 2 * acc_retro['precision'] * acc_retro['recall'] / (
                acc_retro['precision'] + acc_retro['recall']))
        acc_retro.add('micro_f1-score',
                               2 * acc_retro['micro_precision'] * acc_retro['micro_recall'] / (
                                       acc_retro['micro_precision'] + acc_retro['micro_recall']))
        acc_retro.add('macro_f1-score',
                               2 * acc_retro['macro_precision'] * acc_retro['macro_recall'] / (
                                       acc_retro['macro_precision'] + acc_retro['macro_recall']))

        metric_supp.add('acc_retr_{}'.format(i), acc_retro)

    np.save(save_metric_supp, metric_supp)

for i in range(np.size(date)):
    rpz_sub(date[i], version_pred[i], '') #sub_versions[i])


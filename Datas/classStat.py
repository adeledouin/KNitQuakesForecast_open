import numpy as np
import torch
# ------------------------------------------
def pearson_corr_coef(signal1, signal2):
    # print(signal1.shape, signal1.dtype)
    # print(signal1.shape, signal2.dtype)
    cov = 1 / signal1.size(0) * torch.sum((signal1 - torch.mean(signal1)) * (signal2 - torch.mean(signal2)))
    P = cov / (torch.sqrt(torch.var(signal1)) * torch.sqrt(torch.var(signal2)))

    return P

# ------------------------------------------
class Shape():
    '''  '''

    # ---------------------------------------------------------#
    def __init__(self, array):

        self.dim = np.size(np.shape(array))
        if self.dim == 1:
            self.size = np.size(array)
        elif self.dim == 2:
            self.cycles = np.shape(array)[0]
            self.tps = np.shape(array)[1]
            # print('shape array', self.cycles, self.tps)
        else:
            self.size_w, self.size_c, self.nb_pict = self.shape(array)
            # print('shape field', self.size_w, self.size_c, self.nb_pict)

    # ------------------------------------------
    def shape(self, array):
        w = np.shape(array)[0]
        c = np.shape(array)[1]
        nb_pict = np.shape(array)[2]
        return w, c, nb_pict

    # ------------------------------------------
    def ndim_to_1dim(self, array):
        if self.dim == 2:
            return array.reshape(self.cycles * self.tps)
        else:
            return array.reshape(self.size_w * self.size_c * self.nb_pict)


# ---------------------------------------------------------------------------------------------------------------------#
class Stat():
    '''  '''

    # ---------------------------------------------------------#
    def __init__(self, config, signal, display_figure=False, saving_step=True):
        ''' nb_processes : nombre de coeurs à utiilisé pour le mutiprocessing
            display_figure : affiche les figures pour vérifier pas de pb dans l'analyse
            saving step : pêrmet de sauver les étapes dans cas de analyse signel set, si multi set alors ne sauve pas'''

        ## Config
        self.config = config

        self.nb_processes = config.nb_process
        self.display_figure = display_figure
        self.saving_step = saving_step

        self.min = np.min(signal)
        self.max = np.max(signal)
        self.mean = np.mean(signal)
        self.var = np.var(signal, ddof=1)
        self.maxlikelihood = np.var(signal)
        self.sigma = np.sqrt(self.var)
        self.m2 = np.mean(signal * signal) / (2 * self.mean)

# ------------------------------------------
class Histo():
    '''  '''

    # ---------------------------------------------------------#
    def __init__(self, config, display_figure=False, saving_step=True):
        ''' nb_processes : nombre de coeurs à utiilisé pour le mutiprocessing
            display_figure : affiche les figures pour vérifier pas de pb dans l'analyse
            saving step : pêrmet de sauver les étapes dans cas de analyse signel set, si multi set alors ne sauve pas'''

        self.config = config

        self.nb_processes = config.nb_process
        self.display_figure = display_figure
        self.saving_step = saving_step

    # ------------------------------------------
    def regression(self, linx, liny, minreg, maxreg, x_axis='log', y_axis='log'):
        print()
        if x_axis == 'log':
            x = np.log10(linx)
        else:
            x = linx

        if minreg is None:
            minreg = np.min(x)
        if maxreg is None:
            maxreg = np.max(x)

        if y_axis == 'log':
            y = np.log10(liny)
        else:
            y = liny

        # x = np.isfinite(x)
        # y = y[np.isfinite(x)]

        min = np.where(x > minreg)[0][0]
        max = np.where(x < maxreg)[0][-1]

        coef_distri = np.polyfit(x[min:max], y[min:max], 1)

        return coef_distri, x, y

    # ------------------------------------------
    def pearson_corr_coef(self, signal1, signal2):
        cov = 1 / np.size(signal1) * np.sum((signal1 - np.mean(signal1)) * (signal2 - np.mean(signal2)))

        P = cov / (np.sqrt(np.var(signal1)) * np.sqrt(np.var(signal2)))

        coef_distri = np.polyfit(signal1, signal2, 1)

        # fig, ax = Fc.belleFigure(labelx, labely, nfigure=None)
        # ax.plot(signal1, signal2, 'b.', label='Pearson correlation coefficient = %f' % (P))
        # ax.plot(signal1, coef_distri[1]+coef_distri[0]*signal1, 'r-', label='coeff polifit = %f' % (coef_distri[0]))
        # plt.legend(loc='lower left')
        # # plt.savefig('savedmodels/model accuracy -- ' + m_type + 'png')
        # plt.show()

        return P, coef_distri

    # ------------------------------------------
    def gliss_mean(self, x, y, window):
        cut_size = np.round(window / 2)
        new_size = np.size(y) - window
        mean = np.zeros(new_size)
        x_mean = np.zeros(new_size)
        for i in range(new_size):
            mean[i] = np.mean(y[i:i + window])
            x_mean[i] = x[i]

        return x_mean, mean

    # ------------------------------------------
    def np_histo(self, y, nbbin, min_val, max_val):

        hist, bin_edges = np.histogram(y, nbbin, min_val, max_val)

        return hist, bin_edges

    # ------------------------------------------
    def my_histo(self, y, min_val, max_val, x_axis, y_axis, density, binwidth, nbbin):

        if min_val is None:
            min_val = np.min(y)

        if max_val is None:
            max_val = np.max(y)

        # print('nb de points', np.size(y))
        # print('minval', min_val, 'et maxval', max_val)

        if nbbin is None:
            bin_edges = np.arange(min_val, max_val + binwidth, binwidth)
            hist, _ = np.histogram(y, bins=bin_edges, density=True)

        if binwidth is None:
            if x_axis == 'lin':
                binwidth = (max_val - min_val) / nbbin
                bin_edges = np.arange(min_val, max_val + binwidth, binwidth)
            else:
                bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), nbbin)

            count, _ = np.histogram(y, bins=bin_edges, density=False)

            if density == 0:
                hist = count
            elif density == 1:
                hist = count / np.diff(bin_edges)
            else:
                hist = count / np.diff(bin_edges) / np.sum(count)

        x_axis_array = bin_edges[:-1] + np.diff(bin_edges) / 2

        return hist, x_axis_array
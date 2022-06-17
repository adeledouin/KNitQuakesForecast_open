import numpy as np
import matplotlib
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logging
logging.basicConfig(format='| %(levelname)s | %(asctime)s | %(message)s', level=logging.INFO)

# ---------------------------------------------------------#
class ClassPlot():

    # ---------------------------------------------------------#
    def __init__(self, remote, histo):

        self.remote = remote
        self.histo = histo

        self.plt = self.version_pylib()

        self.plt.rc('text', usetex=False)
        self.plt.rcParams['text.latex.preamble'] = [r'\boldmath']
        self.plt.rcParams.update({'font.size': 20})
        self.plt.rcParams.update({'font.weight': 'normal'})

        self.plt.rcParams.update({'axes.titleweight': 'normal'})
        self.plt.rcParams.update({'axes.labelweight': 'bold'})

        self.plt.rc('legend', fontsize=15)
        self.legend_properties = {'weight' : 'normal'}

        self.plt.rcParams['figure.figsize'] = (8, 6)

        # ---------------------------------------------------------#
    def version_pylib(self):
        if self.remote:
            matplotlib.use('pdf')
            import matplotlib.pyplot as plt
        else:
            # matplotlib.use('Qt5Agg')
            import matplotlib.pyplot as plt

        return plt

    # ---------------------------------------------------------#
    def belleFigure(self, ax1, ax2, figsize=None, nfigure=None):
        if figsize is None:
            fig = self.plt.figure(nfigure)
        else:
            fig = self.plt.figure(nfigure, figsize)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(ax1, fontdict=dict(weight='normal'))
        ax.set_ylabel(ax2, fontdict=dict(weight='normal'))
        ax.tick_params(axis='both', which='major', width=1)
        for tick in ax.xaxis.get_ticklabels():
            tick.set_weight('bold')
        for tick in ax.yaxis.get_ticklabels():
            tick.set_weight('bold')
        fig.set_tight_layout(True)
        return fig, ax

    # ---------------------------------------------------------#
    def make_colors(self, size):

        return [cm.rainbow(i) for i in np.linspace(0, 1, size)]

    # ---------------------------------------------------------#
    def make_grid(self, config):
        j = 0
        grid = np.zeros(config.nb_set)
        for i in range(config.nb_set):
            grid[i] = config.nbcycle[i] + j
            j = j + config.nbcycle[i]
        return grid

    # ---------------------------------------------------------#
    def fioritures(self, ax, fig, title, label, grid, save, major=None):
        if title is not None:
            self.plt.title(title)
        if label is not None:
            self.plt.legend(prop=self.legend_properties)
        if grid is not None:
            grid_x_ticks_minor = grid
            ax.set_xticks(grid_x_ticks_minor, minor=True)
            for tick in ax.get_xticklabels():
                tick.set_weight('bold')
            ax.grid(axis='x', which='minor', linestyle='-', alpha=3)
        if major is not None:
            grid_x_ticks_minor = grid
            ax.set_xticks(major)
            ax.set_xticks(grid_x_ticks_minor, minor=True)
            for tick in ax.get_xticklabels():
                tick.set_weight('bold')
            ax.grid(axis='x', which='minor', linestyle='-', alpha=3)
        if save is not None:
            # print(save)
            fig.set_tight_layout(True)
            self.plt.savefig(save + '.pdf')
            self.plt.savefig(save + '.png')
        if not self.remote:
            self.plt.show()
        else:
            self.plt.close(fig)

# %% ####### Normal Plot
    def plot_y(self, y, xname, yname, xsave, ysave, save=None, title=None, label=None, grid=None, pts='.'):
        fig, ax = self.belleFigure('${}$'.format(xname), '${}$'.format(yname), nfigure=None)
        ax.plot(np.arange(np.size(y)), y, pts, label=label)
        if save is not None:
            save = save + ysave + '_vs_' + xsave 
        self.fioritures(ax, fig, title, label, grid, save)

    # ---------------------------------------------------------#
    def plot_x_y(self, x, y, xname, yname, xsave, ysave, save=None, title=None, label=None, grid=None, pts='.'):
        fig, ax = self.belleFigure('${}$'.format(xname), '${}$'.format(yname), nfigure=None)
        ax.plot(x, y, pts, label=label)
        if save is not None:
            save = save + ysave + '_vs_' + xsave 
        self.fioritures(ax, fig, title, label, grid, save)

    # ---------------------------------------------------------#
    def plot_x_y_multiliste(self, size, x, y, xname, yname, xsave, ysave, save=None, title=None,
                            label=None, colors=None, grid=None, pts='.'):
        fig, ax = self.belleFigure('${}$'.format(xname), '${}$'.format(yname), nfigure=None)
        for i in range(size):
            if label is not None:
                ax.plot(x[i], y[i], pts, label=label[i])
            elif colors is not None:
                ax.plot(x[i], y[i], pts, color=colors[i])
            else:
                ax.plot(x[i], y[i], pts)
        if save is not None:
            save = save + ysave + '_vs_' + xsave 
        self.fioritures(ax, fig, title, label, grid, save)

    # ---------------------------------------------------------#
    def plot_x_y_multiarray(self, size, x, y, xname, yname, xsave, ysave, save=None, title=None, label=None,
                            colors=None, grid=None, pts='.'):
        fig, ax = self.belleFigure('${}$'.format(xname), '${}$'.format(yname), nfigure=None)
        for i in range(size):
            if label is not None:
                ax.plot(x[i, :], y[i, :], pts, label=label[i])
            elif colors is not None:
                ax.plot(x[i, :], y[i, :], pts, color=colors[i])
            else:
                ax.plot(x[i, :], y[i, :], pts)
        if save is not None:
            save = save + ysave + '_vs_' + xsave 
        self.fioritures(ax, fig, title, label, grid, save)

# %% ####### Pdf Plot
    def plot_pdf(self, x_Pdf, y_Pdf, yname, ysave, save, label=None, grid=None, xaxis='log', yaxis='log'):

        fig, ax = self.belleFigure('${}$'.format(yname), '$Pdf({})$'.format(yname), nfigure=None)
        ax.plot(x_Pdf, y_Pdf, '.', label=label)
        if xaxis == 'log':
            self.plt.xscale('log')
        if yaxis == 'log':
            self.plt.yscale('log')
        if save is not None:
            save = save + 'pdf_' + ysave 
        self.fioritures(ax, fig, title='Pdf sur {}'.format(ysave), label=label, grid=grid, save=save)

    # ---------------------------------------------------------#
    def plot_pdf_loglog_multicycles(self, size, y, ymin, ymax, yname, ysave, save, label=None, colors=None, nbbin=100,
                                    binwidth=None):
        fig, ax = self.belleFigure('${}$'.format(yname), '$Pdf({})$'.format(yname), nfigure=None)
        for i in range(size):
            y_Pdf, x_Pdf = self.histo.my_histo(y[i], ymin, ymax,
                                               'log', 'log', density=2, binwidth=binwidth, nbbin=nbbin)

            ax.plot(x_Pdf, y_Pdf, '.', color=colors[i])
        self.plt.xscale('log')
        self.plt.yscale('log')
        if save is not None:
            save = save + 'pdf_' + ysave 
        self.fioritures(ax, fig, title='Pdf sur {}'.format(ysave), label=label, grid=None, save=save)

    # ---------------------------------------------------------#
    def plot_pdf_linlin_multicycles(self, size, y, ymin, ymax, yname, ysave, save, label=None, colors=None, nbbin=100,
                                    binwidth=None):
        fig, ax = self.belleFigure('${}$'.format(yname), '$Pdf({})$'.format(yname), nfigure=None)
        for i in range(size):
            y_Pdf, x_Pdf = self.histo.my_histo(y[i, :], ymin, ymax,
                                               'lin', 'lin', density=2, binwidth=binwidth, nbbin=nbbin)

            ax.plot(x_Pdf, y_Pdf, '.', color=colors[i])
        if save is not None:
            save = save + 'pdf_' + ysave 
        self.fioritures(ax, fig, title='Pdf sur {}'.format(ysave), label=label, grid=None, save=save)

    # ---------------------------------------------------------#
    def plot_pdf_linlog_multicycles(self, size, y, ymin, ymax, yname, ysave, save, label=None, colors=None, nbbin=100,
                                    binwidth=None):
        fig, ax = self.belleFigure('${}$'.format(yname), '$Pdf({})$'.format(yname), nfigure=None)
        for i in range(size):
            if ymin != ymax:
                y_Pdf, x_Pdf = self.histo.my_histo(y[i], ymin, ymax,
                                                   'lin', 'log', density=2, binwidth=binwidth, nbbin=nbbin)

                ax.plot(x_Pdf, y_Pdf, '.', color=colors[i])
        self.plt.yscale('log')
        if save is not None:
            save = save + 'pdf_' + ysave 
        self.fioritures(ax, fig, title='Pdf sur {}'.format(ysave), label=label, grid=None, save=save)

    # ---------------------------------------------------------#
    def plot_pdf_loglog_multiseuil(self, size, y, yname, ysave, save, seuils, exposants, nbbin=100, binwidth=None):
        fig, ax = self.belleFigure('${}$'.format(yname), '$Pdf({})$'.format(yname), nfigure=None)
        for i in range(size):
            print(i)
            ymin = np.min(y[i].f[y[i].f != 0])
            ymax = y[i].stats_f.max
            if ymin != ymax:
                y_Pdf, x_Pdf = self.histo.my_histo(y[i].f, ymin, ymax,
                                                   'log', 'log', density=2, binwidth=binwidth, nbbin=nbbin)

                ax.plot(x_Pdf, y_Pdf, '.')
                        # label='seuil = {}e-{}'.format(seuils[i] * 10 ** exposants[i], exposants[i]))
        self.plt.xscale('log')
        self.plt.yscale('log')
        if save is not None:
            save = save + 'pdf_' + ysave 
        self.fioritures(ax, fig, title='Pdf sur {}'.format(ysave), label=True, grid=None, save=save)

    # ---------------------------------------------------------#
    def plot_pdf_linlog_multiseuil(self, size, y, yname, ysave, save, seuils, exposants, nbbin=100, binwidth=None):
        fig, ax = self.belleFigure('${}$'.format(yname), '$Pdf({})$'.format(yname), nfigure=None)
        for i in range(size):
            ymin = y[i].stats_f.min
            ymax = y[i].stats_f.max
            if ymin != ymax:
                y_Pdf, x_Pdf = self.histo.my_histo(y[i].f, ymin, ymax,
                                                   'lin', 'log', density=2, binwidth=binwidth, nbbin=nbbin)

                ax.plot(x_Pdf, y_Pdf, '.',
                        label='seuil = {}e-{}'.format(seuils[i] * 10 ** exposants[i], exposants[i]))
        self.plt.yscale('log')
        if save is not None:
            save = save + 'pdf_' + ysave 
        self.fioritures(ax, fig, title='Pdf sur {}'.format(ysave), label=True, grid=None, save=save)

    # ---------------------------------------------------------#
    def Pdf_loglog(self, y, ymin, ymax, yname, ysave, save, label=None, histo=True, nbbin=100, binwidth=None, x=None,
                   defreg=False):
        if histo:
            if binwidth is None:
                y_Pdf, x_Pdf = self.histo.my_histo(y, ymin, ymax,
                                                   'log', 'log', density=2, binwidth=None, nbbin=nbbin)
            else:
                y_Pdf, x_Pdf = self.histo.my_histo(y, ymin, ymax,
                                                   'log', 'log', density=2, binwidth=binwidth, nbbin=None)

            if defreg:
                grid = np.arange(np.round(np.min(np.log10(x_Pdf))), np.round(np.max(np.log10(x_Pdf))), 0.5)

                fig, ax = self.belleFigure('${}$ (log)'.format(yname), '$Pdf({})$ (log)'.format(yname), nfigure=None)
                ax.plot(np.log10(x_Pdf), np.log10(y_Pdf), '.', label=label)
                if save is not None:
                    save = save + 'pdf_' + ysave + '_defreg' 
                self.fioritures(ax, fig, title='Pdf sur {}'.format(ysave), label=label, grid=grid, save=save)

            else:
                self.plot_pdf(x_Pdf, y_Pdf, yname, ysave, save, label=label)

        else:
            if defreg:
                grid = np.arange(np.round(np.min(np.log10(x))), np.round(np.max(np.log10(x))), 0.5)

                fig, ax = self.belleFigure('${}$ (log)'.format(yname), '$Pdf({})$ (log)'.format(yname), nfigure=None)
                ax.plot(np.log10(x), np.log10(y), '.', label=label)
                if save is not None:
                    save = save + 'pdf_' + ysave + '_defreg' 
                self.fioritures(ax, fig, title='Pdf sur {}'.format(ysave), label=label, grid=grid, save=save)
            else:
                self.plot_pdf(x, y, yname, ysave, save, label=label)

    # ---------------------------------------------------------#
    def Pdf_linlin(self, y, ymin, ymax, yname, ysave, save, label=None, histo=True, nbbin=100, binwidth=None, x=None,
                   defreg=False):
        if histo:
            if binwidth is None:
                y_Pdf, x_Pdf = self.histo.my_histo(y, ymin, ymax,
                                                   'lin', 'lin', density=2, binwidth=None, nbbin=nbbin)
            else:
                y_Pdf, x_Pdf = self.histo.my_histo(y, ymin, ymax,
                                                   'lin', 'lin', density=2, binwidth=binwidth, nbbin=None)

            if defreg:
                grid = np.linspace(np.round(np.min(x_Pdf)), np.round(np.max(x_Pdf)), 5)

                fig, ax = self.belleFigure('${}$'.format(yname), '$Pdf({})$'.format(yname), nfigure=None)
                ax.plot(x_Pdf, y_Pdf, '.', label=label)
                if save is not None:
                    save = save + 'pdf_' + ysave + '_defreg' 
                self.fioritures(ax, fig, title='Pdf sur {}'.format(ysave), label=label, grid=grid, save=save)

            else:
                self.plot_pdf(x_Pdf, y_Pdf, yname, ysave, save, label=label, xaxis='lin', yaxis='lin')

        else:
            if defreg:
                grid = np.linspace(np.round(np.min(x)), np.round(np.max(x)), 5)

                fig, ax = self.belleFigure('${}$'.format(yname), '$Pdf({})$'.format(yname), nfigure=None)
                ax.plot(x, y, '.', label=label)
                if save is not None:
                    save = save + 'pdf_' + ysave + '_defreg' 
                self.fioritures(ax, fig, title='Pdf sur {}'.format(ysave), label=label, grid=grid, save=save)
            else:
                self.plot_pdf(x, y, yname, ysave, save, label=label, xaxis='lin', yaxis='lin')

    # ---------------------------------------------------------#
    def Pdf_linlog(self, y, ymin, ymax, yname, ysave, save, label=None, histo=True, nbbin=100, binwidth=None, x=None,
                   defreg=False):
        if histo:
            if binwidth is None:
                y_Pdf, x_Pdf = self.histo.my_histo(y, ymin, ymax,
                                                   'lin', 'log', density=2, binwidth=None, nbbin=nbbin)
            else:
                y_Pdf, x_Pdf = self.histo.my_histo(y, ymin, ymax,
                                                   'lin', 'log', density=2, binwidth=binwidth, nbbin=None)

            if defreg:
                grid = np.linspace(np.round(np.min(x_Pdf)), np.round(np.max(x_Pdf)), 5)

                fig, ax = self.belleFigure('${}$'.format(yname), '$Pdf({})$ (log)'.format(yname), nfigure=None)
                ax.plot(x_Pdf, np.log10(y_Pdf), '.', label=label)
                if save is not None:
                    save = save + 'pdf_' + ysave + '_defreg' 
                self.fioritures(ax, fig, title='Pdf sur {}'.format(ysave), label=label, grid=grid, save=save)

            else:
                grid = None

            self.plot_pdf(x_Pdf, y_Pdf, yname, ysave, save, label=label, grid=grid, xaxis='lin')

        else:
            if defreg:
                grid = np.linspace(np.round(np.min(x)), np.round(np.max(x)), 5)

                fig, ax = self.belleFigure('${}$'.format(yname), '$Pdf({})$ (log)'.format(yname), nfigure=None)
                ax.plot(x, np.log(y), '.', label=label)
                if save is not None:
                    save = save + 'pdf_' + ysave + '_defreg' 
                self.fioritures(ax, fig, title='Pdf sur {}'.format(ysave), label=label, grid=grid, save=save)

            else:
                self.plot_pdf(x, y, yname, ysave, save, label=label, xaxis='lin')

    # ---------------------------------------------------------#
    def Pdf_loglin(self, y, ymin, ymax, yname, ysave, save, label=None, histo=True, nbbin=100, binwidth=None,
                   x=None,
                   defreg=False):
        if histo:
            if binwidth is None:
                y_Pdf, x_Pdf = self.histo.my_histo(y, ymin, ymax,
                                                   'log', 'lin', density=2, binwidth=None, nbbin=nbbin)
            else:
                y_Pdf, x_Pdf = self.histo.my_histo(y, ymin, ymax,
                                                   'log', 'lin', density=2, binwidth=binwidth, nbbin=None)

            if defreg:
                grid = np.linspace(np.round(np.min(np.log10(x_Pdf))), np.round(np.max(np.log10(x_Pdf))), 5)

                fig, ax = self.belleFigure('${}$ (log)'.format(yname), '$Pdf({})$'.format(yname), nfigure=None)
                ax.plot(np.log10(x_Pdf), y_Pdf, '.', label=label)
                if save is not None:
                    save = save + 'pdf_' + ysave + '_defreg' 
                self.fioritures(ax, fig, title='Pdf sur {}'.format(ysave), label=label, grid=grid, save=save)

            else:
                grid = None

            self.plot_pdf(x_Pdf, y_Pdf, yname, ysave, save, label=label, grid=grid, yaxis='lin')

        else:
            if defreg:
                grid = np.linspace(np.round(np.min(np.log(x))), np.round(np.max(np.log(x))), 5)

                fig, ax = self.belleFigure('${}$ (log)'.format(yname), '$Pdf({})$'.format(yname), nfigure=None)
                ax.plot(np.log(x), y, '.', label=label)
                if save is not None:
                    save = save + 'pdf_' + ysave + '_defreg' 
                self.fioritures(ax, fig, title='Pdf sur {}'.format(ysave), label=label, grid=grid, save=save)

            else:
                self.plot_pdf(x, y, yname, ysave, save, label=label, yaxis='lin')


    # %% ####### plot reg
    def plot_reg(self, y, ymin, ymax, xname, yname, xsave, ysave, minreg=None, maxreg=None,
                 histo=True, nbbin=100, binwidth=None, x=None, xaxis='log', yaxis='log',
                 save=None, label=None, powerlaw=None):

        if histo:
            if binwidth is None:
                y_Pdf, x_Pdf = self.histo.my_histo(y, ymin, ymax,
                                                   xaxis, yaxis, density=2, binwidth=None, nbbin=nbbin)
            else:
                y_Pdf, x_Pdf = self.histo.my_histo(y, ymin, ymax,
                                                   xaxis, yaxis, density=2, binwidth=binwidth, nbbin=None)

            linx = x_Pdf[y_Pdf != 0]
            liny = y_Pdf[y_Pdf != 0]

        else:
            linx = x[y != 0]
            liny = y[y != 0]

        coef_distri, x, y = self.histo.regression(linx, liny, minreg, maxreg, xaxis, yaxis)

        polynomial = np.poly1d(coef_distri)
        ys = polynomial(x)
        if powerlaw is not None:
            ypower = powerlaw * x + coef_distri[1]  ##-1.3
        fig, ax = self.belleFigure('{} ({})'.format(xname, xaxis), '{} ({})'.format(yname, yaxis), nfigure=None)
        ax.plot(x, y, 'b.', label=label)
        ax.plot(x, ys, 'r-', label='coeff polifit = {}'.format(coef_distri[0]))
        if powerlaw is not None:
            ax.plot(x, ypower, 'g-', label='coeff powerlaw = {}'.format(powerlaw))
        if save is not None:
            save = save + ysave + '_' + xsave + '_reg' 
        self.fioritures(ax, fig, title='regression linéaire {} et {}'.format(ysave, xsave), label=True, grid=None,
                        save=save)

# %% ####### Variations plot
    def plot_variation_cycle(self, variations, ysave, save=None, label=None, grid=None,
                             multi_seuils=False, nb_seuils=None, seuils=None, exposants=None):
        fig, ax = self.belleFigure('c', '$f(c) = <f(c,t)>_t$', nfigure=None)
        if not multi_seuils:
            ax.plot(np.arange(variations.nbcycle),
                    [variations.stats_f_cycle[i].mean for i in range(variations.nbcycle)], '.',
                    label=label)
        else:
            for s in range(nb_seuils):
                ax.plot(np.arange(variations[s].nbcycle),
                        [variations[s].stats_f_cycle[i].mean for i in range(variations[s].nbcycle)], '.',
                        label='seuil = {}e-{}'.format(np.str(seuils[s] * 10 ** exposants[s]), np.str(exposants[s])))
            label = True
        if save is not None:
            save = save + 'stats_cycle_mean_' + ysave 
        self.fioritures(ax, fig, title='moyenne de cycle en cycle', label=label, grid=grid, save=save)

        fig, ax = self.belleFigure('c', '$f(c) = Var(f(c,t))_t$', nfigure=None)
        if not multi_seuils:
            ax.plot(np.arange(variations.nbcycle),
                    [variations.stats_f_cycle[i].var for i in range(variations.nbcycle)], '.',
                    label=label)
        else:
            for s in range(nb_seuils):
                ax.plot(np.arange(variations[s].nbcycle),
                        [variations[s].stats_f_cycle[i].var for i in range(variations[s].nbcycle)], '.',
                        label='seuil = {}e-{}'.format(np.str(seuils[s] * 10 ** exposants[s]), np.str(exposants[s])))
            label = True
        if save is not None:
            save = save + 'stats_cycle_var_' + ysave 
        self.fioritures(ax, fig, title='var de cycle en cycle', label=label, grid=grid, save=save)

        # fig, ax = self.belleFigure('c', '$f(c) = S_m(f(c,t))_t$', nfigure=None)
        # ax.plot(np.arange(variations.nbcycle),
        #         [variations.stats_f_cycle[i].m2 for i in range(variations.nbcycle)], '.',
        #         label=label)
        # if save is not None:
        #     save = save + 'stats_cycle_Sm_' + ysave 
        # self.fioritures(ax, fig, title='Sm de cycle en cycle', label=label, grid=grid, save=save)

    # ---------------------------------------------------------#
    def plot_variation_tps_cycle(self, variations, ysave, save=None, label=None, grid=None, colors=None):
        fig, ax = self.belleFigure('$t$', '$f(c,t) = <f(c,t)>_{\Delta t}$', nfigure=None)
        for i in range(variations.nbcycle):
            stat = variations.stats_f_cycle_tps[i]
            ax.plot(np.arange(np.size(stat)), [stat[j].mean for j in range(np.size(stat))], '.', color=colors[i])
        if save is not None:
            save = save + 'stats_cycle_tps_mean_' + ysave 
        self.fioritures(ax, fig, title='moyenne sur fenetre de temps par cycle', label=label, grid=grid, save=save)

        fig, ax = self.belleFigure('$t$', '$f(c,t) = <f(c,t)>_{\Delta t}$', nfigure=None)
        for i in range(variations.nbcycle):
            stat = variations.stats_f_cycle_tps[i]
            ax.plot(np.arange(np.size(stat)), [stat[j].var for j in range(np.size(stat))], '.', color=colors[i])
        if save is not None:
            save = save + 'stats_cycle_tps_var_' + ysave 
        self.fioritures(ax, fig, title='var sur fenetre de temps par cycle', label=label, grid=grid, save=save)

        # fig, ax = self.belleFigure('$t$', '$f(c,t) = <f(c,t)>_{\Delta t}$', nfigure=None)
        # for i in range(variations.nbcycle):
        #     stat = variations.stats_f_cycle_tps[i]
        #     ax.plot(np.arange(np.size(stat)), [stat[j].m2 for j in range(np.size(stat))], '.', color=colors[i])
        # if save is not None:
        #     save = save + 'stats_cycle_tps_Sm_' + ysave 
        # self.fioritures(ax, fig, title='Sm sur fenetre de temps par cycle', label=label, grid=grid, save=save)

    # ---------------------------------------------------------#
    def plot_coefficients(self, variations, ysave, save=None, label=None, grid=None):
        fig, ax = self.belleFigure('$c$', 'pente de $f(c,t) = <f(c,t)>_{\Delta t}$', nfigure=None)
        ax.plot(np.arange(variations.nbcycle), variations.pente_stats_f_cycle_tps[0, :], 'b.', label=label)
        if save is not None:
            save = save + 'pente_stats_cycle_tps_mean_' + ysave 
        self.fioritures(ax, fig, title='pente de mean sur fenetre de temps par cycle',
                        label=label, grid=grid, save=save)

        fig, ax = self.belleFigure('$c$', 'ordonnée de $f(c,t) = <f(c,t)>_{\Delta t}$', nfigure=None)
        ax.plot(np.arange(variations.nbcycle), variations.ordonnee_stats_f_cycle_tps[0, :], 'b.', label=label)
        if save is not None:
            save = save + 'ordonnee_stats_cycle_tps_mean_' + ysave 
        self.fioritures(ax, fig, title='ordonnee de mean sur fenetre de temps par cycle',
                        label=label, grid=grid, save=save)

        fig, ax = self.belleFigure('$c$', 'pente de $f(c,t) = Var(f(c,t))_{\Delta t}$', nfigure=None)
        ax.plot(np.arange(variations.nbcycle), variations.pente_stats_f_cycle_tps[1, :], 'b.', label=label)
        if save is not None:
            save = save + 'pente_stats_cycle_tps_var_' + ysave 
        self.fioritures(ax, fig, title='pente de var sur fenetre de temps par cycle',
                        label=label, grid=grid, save=save)

        fig, ax = self.belleFigure('$c$', 'ordonnée de $f(c,t) = Var(f(c,t))_{\Delta t}$', nfigure=None)
        ax.plot(np.arange(variations.nbcycle), variations.ordonnee_stats_f_cycle_tps[1, :], 'b.', label=label)
        if save is not None:
            save = save + 'ordonnee_stats_cycle_tps_var_' + ysave 
        self.fioritures(ax, fig, title='ordonnee de var sur fenetre de temps par cycle',
                        label=label, grid=grid, save=save)

    # ---------------------------------------------------------#
    def plot_variation_tps(self, variations, ysave, save=None, label=None, grid=None, colors=None):
        fig, ax = self.belleFigure('t', '$f(t) = <f(c,t)>_{c,\Delta t}$', nfigure=None)
        ax.plot(np.arange(variations.nbcycle),
                [variations.stats_f_tps[i].mean for i in range(variations.nbcycle)], '.',
                label=label)
        if save is not None:
            save = save + 'stats_tps_mean_' + ysave 
        self.fioritures(ax, fig, title='moyenne sur tps', label=label, grid=grid, save=save)

        fig, ax = self.belleFigure('c', '$f(c) = Var(f(c,t))_t$', nfigure=None)
        ax.plot(np.arange(variations.nbcycle),
                [variations.stats_f_tps[i].var for i in range(variations.nbcycle)], '.',
                label=label)
        if save is not None:
            save = save + 'stats_tps_var_' + ysave 
        self.fioritures(ax, fig, title='var sur tps', label=label, grid=grid, save=save)

        # fig, ax = self.belleFigure('c', '$f(c) = S_m(f(c,t))_t$', nfigure=None)
        # ax.plot(np.arange(variations.nbcycle),
        #         [variations.stats_f_tps[i].m2 for i in range(variations.nbcycle)], '.',
        #         label=label)
        # if save is not None:
        #     save = save + 'stats_tps_Sm_' + ysave 
        # self.fioritures(ax, fig, title='Sm sur ps', label=label, grid=grid, save=save)

# %% ####### Field Plot
    def plot_scalar_field(self, f, k, fname, size="5%", pad=0.1, save=None, title=None, label=None, grid=None):
        fig, ax = self.belleFigure('stich $L_{c}$', 'stich $L_{w}$', figsize=(15, 5), nfigure=None)
        v = ax.imshow(f, cmap=cm.jet, interpolation='none')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=size, pad=pad)
        cbar = fig.colorbar(v, cax=cax)
        cbar.minorticks_on()
        if save is not None:
            save = save + 'scalar_field_' + fname + '_img_{}'.format(k) 
        self.fioritures(ax, fig, title=title, label=label, grid=grid, save=save)

    # ---------------------------------------------------------#
    def plot_vector_field(self, X, Y, x, y, norm, k, fname, scale=3, save=None, title=None, label=None, grid=None):
        fig, ax = self.belleFigure('stich $L_{c}$', 'stich $L_{w}$', nfigure=None)
        q = self.plt.quiver(X, np.flipud(Y), x, y, norm, cmap=cm.jet, angles='xy', scale=scale)
        if save is not None:
            save = save + 'vector_field_' + fname + '_img_{}'.format(k) 
        self.fioritures(ax, fig, title=title, label=label, grid=grid, save=save)

    # ---------------------------------------------------------#
    def plot_field(self, config_plot, f, X, Y, x, y, k, fname, scale=3, save=None, title=None, label=None, grid=None):
        fig, ax = self.belleFigure('stich $L_{c}$', 'stich $L_{w}$', figsize=(15, 5), nfigure=None)
        v = ax.imshow(f, cmap=cm.jet, interpolation='none', vmin=-config_plot['imgevent_seuil_plot_field'],
                      vmax=config_plot['imgevent_seuil_plot_field'])
        cbar = fig.colorbar(v, ax=ax)
        q = self.plt.quiver(X, Y, x, y, angles='xy', scale=scale)
        if save is not None:
            save = save + 'vector_field_' + fname + '_img_{}'.format(k) 
        self.fioritures(ax, fig, title=title, label=label, grid=grid, save=save)

# %% ####### Event Comparaison Plot
    def scatter_plot(self, x, y, xname, yname, xsave, ysave, save=None):
        P, coef_distri = self.histo.pearson_corr_coef(x, y)

        print(xname, yname, xsave, ysave)
        fig, ax = self.belleFigure('${}$'.format(xname), '$Pdf({}$)'.format(yname), nfigure=None)
        ax.plot(x, y, 'b*', label='Pearson correlation coefficient = {}'.format(P))
        ax.plot(x, coef_distri[1] + coef_distri[0] * x, 'r-',
                label='coeff polifit = {}'.format(coef_distri[0]))
        if save is not None:
            save = save + 'scatter_plot' ### _' + ysave + '_' + xsave 
        self.fioritures(ax, fig, title='scatter plot entre {} et {}'.format(ysave, xsave), label=True, grid=None,
                        save=save)
    #
    # # ---------------------------------------------------------#
    # def comparaison_plot_loglog(self, save, linx, liny, namex, namey):
    #     coef_distri, x, y = self.histo.regression(linx, liny, np.min(np.log10(linx)), np.max(np.log10(linx)))
    #
    #     polynomial = np.poly1d(coef_distri)
    #     ys = polynomial(x)
    #     fig, ax = self.belleFigure('{} (log)'.format(namex), '{} (log)'.format(namey), nfigure=None)
    #     ax.plot(x, y, 'b.')
    #     ax.plot(x, ys, 'r-', label='coeff polifit = %f' % (coef_distri[0]))
    #     self.plt.title('compare {} et {}'.format(namex, namey))
    #     self.plt.legend(prop=self.legend_properties,loc='lower left')
    #     self.plt.savefig(save + namex + '_' + namey + '.png')
    #     if not self.remote:
    #         self.plt.show()
    #     else:
    #         self.plt.close()
    #
    # # ---------------------------------------------------------#
    # def comparaison_plot_loglin(self, save, linx, liny, namex, namey):
    #     coef_distri, x, y = self.histo.regression(linx, liny, np.min(np.log10(linx)), np.max(np.log10(linx)),
    #                                               y_axis='lin')
    #
    #     polynomial = np.poly1d(coef_distri)
    #     ys = polynomial(x)
    #     fig, ax = self.belleFigure('{} (log)'.format(namex), namey, nfigure=None)
    #     ax.plot(x, y, 'b.')
    #     ax.plot(x, ys, 'r-', label='coeff polifit = %f' % (coef_distri[0]))
    #     self.plt.title('compare {} et {}'.format(namex, namey))
    #     self.plt.legend(prop=self.legend_properties,loc='lower left')
    #     self.plt.savefig(save + namex + '_' + namey + '.png')
    #     if not self.remote:
    #         self.plt.show()
    #     else:
    #         self.plt.close()
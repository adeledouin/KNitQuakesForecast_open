import numpy as np


# ------------------------------------------
def plot_pdf_event(plot, signalevent, df, variations_df, savename, seuil, exposant, minreg, maxreg, display_figure_reg):
    plot.Pdf_loglog(df, seuil, variations_df.stats_f.max,
                    '\Delta \delta f', 'df' + savename,
                    save=signalevent.to_save_fig,
                    label='seuil =' + np.str(seuil * 10 ** exposant) + 'e-' + np.str(exposant) + 'N')

    plot.Pdf_loglog(df, seuil, variations_df.stats_f.max,
                    '\Delta \delta f', 'df' + savename,
                    save=signalevent.to_save_fig,
                    label='seuil =' + np.str(seuil * 10 ** exposant) + 'e-' + np.str(exposant) + 'N',
                    defreg=True)

    if display_figure_reg:
        plot.plot_reg(df, seuil, variations_df.stats_f.max,
                      '\Delta \delta f', 'Pdf(\Delta \delta f)',
                      'pdf', 'df' + savename,
                      minreg=minreg,
                      maxreg=maxreg,
                      save=signalevent.to_save_fig,
                      label='seuil =' + np.str(seuil * 10 ** exposant) + 'e-' + np.str(
                          exposant) + 'N',
                      powerlaw=-1.5)

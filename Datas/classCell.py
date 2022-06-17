import numpy as np


class Cell():
    """
    Classe qui permet de save des tableau contenue dans liste - same type as matlab cell - dans des array.

    Attributes:
        data (list(array) ou array) : data à sauver
        namesave (str) : chemin pour save
        extension (str) : format du fichier save
        nbfichier (int) : nb de fichier saved - taille de la liste ou nombre de lignes
        namext (list) : liste des sous noms de fichiers si nécéssaire

    """

    # ---------------------------------------------------------#
    def __init__(self, namesave, nbfichier, data=None, extension=None, namext=None):
        """
        The constructor for SignalImg.

        Parameters:
            data (list(array) ou array) : data à sauver
            namesave (str) : chemin pour save
            extension (str) : format du fichier save
            nbfichier (int) : nb de fichier saved - taille de la liste ou nombre de lignes
            namext (list) : liste des sous noms de fichiers si nécéssaire

        """
        self.data = data
        self.namesave = namesave
        self.nbfichier = nbfichier
        self.namext = namext

        self.extension = extension

        if self.extension == 'cell':
            self.save_cell()
        elif self.extension == 'csv':
            self.save_csv()

    # ------------------------------------------
    def save_cell(self):
        """
        Fonction qui sauve une liste d'array en np array numérotés par l'indice de la liste ou par sous nom
        de fichier.

        """
        for i in range(self.nbfichier):
            if self.namext is None:
                np.save(self.namesave + '_%d' % (i+1), self.data[i])
            else:
                np.save(self.namesave + '_' + self.namext[i], self.data[i])

    # ------------------------------------------
    def save_csv(self):
        """
        Fonction qui sauve une liste d'array en fichiers csv : soit en 2D csv, soit en 1D csv numéroté par
        l'indice de la liste ou par sous nom de fichier.

        """
        if self.nbfichier == 1:
            if self.namext is None:
                np.savetxt(self.namesave + '.csv', self.data, delimiter=',')
            else:
                np.savetxt(self.namesave + '_' + self.namext + '.csv', self.data, delimiter=',')
        else:
            for i in range(self.nbfichier):
                if self.namext is None:
                    np.savetxt(self.namesave + '_%d.csv' % (i+1), self.data[i], delimiter=',')
                else:
                    np.savetxt(self.namesave + '_' + self.namext[i] + '.csv', self.data[i], delimiter=',')

    # ------------------------------------------
    def reco_cell(self):
        """
        Fonction qui recrée une liste d'array à partir des fichier save.

         """
        newdata = [0 for i in range(self.nbfichier)]

        for i in range(self.nbfichier):
            if self.namext is None:
                newdata[i] = np.load(self.namesave + '_%d.npy' %(i+1))
            else:
                newdata[i] = np.load(self.namesave + '_' + self.namext[i] + '.npy')

        return newdata

    # ------------------------------------------
    def reco_array(self, i_size, j_size, dim):
        """
        Fonction qui recrée un seul array à partir de plusieur fichier npy.

         """
        newdata = np.zeros((i_size, j_size))

        for k in range(self.nbfichier):
            a = np.load(self.namesave + '_%d.npy' % (k))
            if dim == 0:
                newdata[k, :] = a
            else:
                newdata[:, k] = a

        return newdata

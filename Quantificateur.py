import numpy as np


def quant_scal_unif(x, val_min, val_max, n_bits):
    """
    Quantificateur scalaire uniforme.

    Paramètres
    ----------
    x : numpy.ndarray
        Signal d'entrée.
    val_min : float
        Valeur minimale représentable.
    val_max : float
        Valeur maximale représentable.
    n_bits : int
        Nombre de bits par échantillon.

    Retourne
    --------
    y : numpy.ndarray
        Signal quantifié.
    ind : numpy.ndarray
        Indices des niveaux de quantification.
    """

    n_niveaux = 2 ** n_bits

    delta = (val_max - val_min) / n_niveaux
    un_sur_delta = 1 / delta

    ind = np.round((x - val_min) * un_sur_delta)

    # Saturation en bas
    ind = np.maximum(ind, 0)

    # Saturation en haut
    ind = np.minimum(ind, n_niveaux - 1)

    y = val_min + ind * delta

    return y, ind.astype(int)
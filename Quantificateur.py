"""Quantification scalaire uniforme à n_bits, pleine échelle [val_min, val_max]."""

import numpy as np


def quant_scal_unif(x, val_min, val_max, n_bits):
    n_niveaux = 2 ** n_bits
    delta = (val_max - val_min) / n_niveaux
    ind = np.round((x - val_min) / delta)
    ind = np.clip(ind, 0, n_niveaux - 1)
    y = val_min + ind * delta
    return y, ind.astype(int)

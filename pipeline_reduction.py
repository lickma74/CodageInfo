"""
Réduction de débit du scaphandre : anti-repliement, sous-échantillonnage ×3,
quantification uniforme (6 ou 8 bits), optionnellement encadrée par un SAW.

SAW : |X|^p avant le quantificateur, |Y|^{1/p} après (p = 1/2).
Le SQNR global baisse ; le bruit est mis en forme (masquage), ce n'est pas un bug.
"""

import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import firwin, filtfilt
from scipy.signal.windows import hann

from Quantificateur import quant_scal_unif

SAW_P = 0.5
SAW_L = 512
FACTEUR = 3
DOSSIER = "outputs/saw"


def charger_wav(chemin):
    fs, x = wavfile.read(chemin)
    x = x[:, 0] if x.ndim > 1 else x
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float64)
    return fs, x


def sauvegarder_wav(chemin, fs, x):
    wavfile.write(chemin, fs, (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16))


def filtre_antirepliement(x, fs, facteur, marge=0.9, numtaps=201):
    """Passe-bas FIR avant décimation. fc = 0.9 × Nyquist réduite (ici ~6615 Hz)."""
    fc = marge * (fs / facteur) / 2
    taps = firwin(numtaps, cutoff=fc, fs=fs, window="hamming")
    return filtfilt(taps, [1.0], x), taps


def sous_echantillonner(x, facteur):
    return x[::facteur]


def suréchantillonner(x, facteur, taps):
    """Zéros + même passe-bas (anti-imagerie) + gain ×facteur, pour réécoute à fs d'origine."""
    x_zeros = np.zeros(len(x) * facteur)
    x_zeros[::facteur] = x
    return filtfilt(taps, [1.0], x_zeros) * facteur


def appliquer_saw(x, exposant, L=SAW_L):
    """STFT : |X|^exposant, phase inchangée, OLA 50 % √Hann."""
    hop = L // 2
    w = np.sqrt(hann(L, sym=False))
    pad = L - hop
    x_pad = np.concatenate([np.zeros(pad), x, np.zeros(L)])
    n_trames = 1 + (len(x_pad) - L) // hop
    y_pad = np.zeros(len(x_pad))
    w_sum = np.zeros(len(x_pad))
    for i in range(n_trames):
        deb = i * hop
        X = np.fft.rfft(x_pad[deb:deb + L] * w)
        mag = np.maximum(np.abs(X), 1e-12)
        Y = (mag ** exposant) * np.exp(1j * np.angle(X))
        y_pad[deb:deb + L] += np.fft.irfft(Y, n=L) * w
        w_sum[deb:deb + L] += w * w
    ok = w_sum > 1e-8
    y_pad[ok] /= w_sum[ok]
    return y_pad[pad:pad + len(x)]


def pipeline_reduction(x, fs, facteur=FACTEUR, n_bits=8, pleine_echelle=1.0,
                       utiliser_saw=False, p_saw=SAW_P):
    """Décime à fs/3, quantifie ; si SAW : warp p puis 1/p autour du quantificateur."""
    x_filtre, taps = filtre_antirepliement(x, fs, facteur)
    x_decime = sous_echantillonner(x_filtre, facteur)
    if utiliser_saw:
        x_q, ind = quant_scal_unif(
            appliquer_saw(x_decime, p_saw), -pleine_echelle, pleine_echelle, n_bits
        )
        x_reduit = appliquer_saw(x_q, 1.0 / p_saw)
    else:
        x_reduit, ind = quant_scal_unif(x_decime, -pleine_echelle, pleine_echelle, n_bits)
    return x_reduit, fs / facteur, ind, taps


def calculer_sqnr(x_ref, x_test):
    n = min(len(x_ref), len(x_test))
    bruit = np.mean((x_ref[:n] - x_test[:n]) ** 2)
    if bruit < 1e-20:
        return np.inf
    return 10 * np.log10(np.mean(x_ref[:n] ** 2) / bruit)


def main():
    os.makedirs(DOSSIER, exist_ok=True)
    fs, x = charger_wav("inputs/scaphandre.wav")
    print(f"scaphandre  fs={fs}  {len(x)/fs:.2f}s")
    for n_bits in (8, 6):
        for avec_saw, suffixe in ((False, "sans_saw"), (True, "saw")):
            x_reduit, fs_r, _, taps = pipeline_reduction(
                x, fs, n_bits=n_bits, utiliser_saw=avec_saw
            )
            y = suréchantillonner(x_reduit, FACTEUR, taps)
            out = os.path.join(DOSSIER, f"scaphandre_{n_bits}bits_{suffixe}.wav")
            sauvegarder_wav(out, fs, y)
            print(f"  {n_bits} bits {'+ SAW' if avec_saw else 'sans SAW'}  "
                  f"fs'={fs_r:.0f} Hz  SQNR={calculer_sqnr(x, y):.1f} dB  -> {out}")


if __name__ == "__main__":
    main()

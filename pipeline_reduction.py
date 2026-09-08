"""
Réduction de débit : décimation x3 + quantification uniforme, avec ou sans SAW.

    44.1 kHz / 16 bits
        → anti-repliement + sous-échantillonnage x3  (14.7 kHz)
        → [pré-SAW] |X|^p
        → quantification scalaire uniforme (8 ou 6 bits)
        → [post-SAW] |Y|^{1/p}

Réutilise quant_scal_unif() de Quantificateur.py (inchangé).
"""

import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import firwin, filtfilt
from scipy.signal.windows import hann
import matplotlib.pyplot as plt

from Quantificateur import quant_scal_unif

SAW_P = 0.5
SAW_L = 512


def charger_wav(chemin):
    fs, x = wavfile.read(chemin)
    if x.ndim > 1:
        x = x[:, 0]
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float64)
    return fs, x


def sauvegarder_wav(chemin, fs, x):
    x_clip = np.clip(x, -1.0, 1.0)
    wavfile.write(chemin, fs, (x_clip * 32767).astype(np.int16))


def concevoir_filtre(fs, fc, numtaps=201):
    return firwin(numtaps, cutoff=fc, fs=fs, window="hamming")


def filtre_antirepliement(x, fs, facteur, marge=0.9, numtaps=201):
    fs_reduit = fs / facteur
    fc = marge * (fs_reduit / 2)
    taps = concevoir_filtre(fs, fc, numtaps)
    return filtfilt(taps, [1.0], x), taps, fc


def sous_echantillonner(x, facteur):
    return x[::facteur]


def suréchantillonner(x, facteur, taps):
    x_zeros = np.zeros(len(x) * facteur)
    x_zeros[::facteur] = x
    return filtfilt(taps, [1.0], x_zeros) * facteur


def appliquer_saw(x, exposant, L=SAW_L):
    """STFT : |X| ** exposant, phase inchangée, overlap-add 50 %."""
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

    mask = w_sum > 1e-8
    y_pad[mask] /= w_sum[mask]
    return y_pad[pad:pad + len(x)]


def pipeline_reduction(x, fs, facteur=3, n_bits=8, pleine_echelle=1.0,
                       utiliser_saw=False, p_saw=SAW_P):
    x_filtre, taps, _fc = filtre_antirepliement(x, fs, facteur)
    x_decime = sous_echantillonner(x_filtre, facteur)
    fs_reduit = fs / facteur

    if utiliser_saw:
        x_pre = appliquer_saw(x_decime, p_saw)
        x_q, ind = quant_scal_unif(x_pre, -pleine_echelle, pleine_echelle, n_bits)
        x_decode = appliquer_saw(x_q, 1.0 / p_saw)
    else:
        x_q, ind = quant_scal_unif(x_decime, -pleine_echelle, pleine_echelle, n_bits)
        x_decode = x_q

    return x_q, fs_reduit, ind, taps, x_decode


def reconstruire_pour_ecoute(x_reduit, facteur, taps):
    return suréchantillonner(x_reduit, facteur, taps)


def calculer_sqnr(x_ref, x_test):
    n = min(len(x_ref), len(x_test))
    bruit = np.mean((x_ref[:n] - x_test[:n]) ** 2)
    if bruit < 1e-20:
        return np.inf
    return 10 * np.log10(np.mean(x_ref[:n] ** 2) / bruit)


def main():
    fichiers = ["inputs/parole.wav", "inputs/parole_2.wav"]
    dossier_sortie = "outputs"
    os.makedirs(dossier_sortie, exist_ok=True)
    facteur = 3

    for chemin_wav in fichiers:
        if not os.path.exists(chemin_wav):
            print(f"Fichier introuvable : {chemin_wav}")
            continue

        nom = os.path.splitext(os.path.basename(chemin_wav))[0]
        fs, x = charger_wav(chemin_wav)
        print(f"\n=== {nom} (fs={fs} Hz, durée={len(x)/fs:.2f} s) ===")

        resultats = {}
        for n_bits in [8, 6]:
            for suffixe, saw in [("sans_saw", False), ("saw", True)]:
                _tx, _fs_r, ind, taps, x_decode = pipeline_reduction(
                    x, fs, facteur=facteur, n_bits=n_bits, utiliser_saw=saw
                )
                x_ecoute = reconstruire_pour_ecoute(x_decode, facteur, taps)
                sqnr = calculer_sqnr(x, x_ecoute)
                chemin = os.path.join(
                    dossier_sortie, f"{nom}_{n_bits}bits_{suffixe}.wav"
                )
                sauvegarder_wav(chemin, fs, x_ecoute)
                print(f"{n_bits} bits {suffixe:10s}  SQNR={sqnr:.1f} dB  "
                      f"niveaux={len(np.unique(ind))}/{2**n_bits}  -> {chemin}")
                resultats[(n_bits, suffixe)] = x_ecoute

        n_aff = min(2000, len(x))
        plt.figure(figsize=(10, 5))
        plt.plot(x[:n_aff], label="original 16 bits", color="k", alpha=0.8)
        plt.plot(resultats[(6, "sans_saw")][:n_aff], label="6 bits sans SAW", alpha=0.7)
        plt.plot(resultats[(6, "saw")][:n_aff], label="6 bits + SAW", alpha=0.7)
        plt.xlabel("échantillon (44.1 kHz)")
        plt.ylabel("amplitude")
        plt.title(f"{nom} — original vs 6 bits, avec / sans SAW")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = os.path.join(dossier_sortie, f"{nom}_comparaison_quantification.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Figure -> {fig_path}")


if __name__ == "__main__":
    main()

"""
Partie B : réduction du débit du signal du scaphandre, sans et avec SAW.

    44.1 kHz / 16 bits  (ou 16 kHz pour parole.wav)
          |
          v
    Filtre anti-repliement (passe-bas)
          |
          v
    Sous-échantillonnage x3
          |
          v
    [optionnel] pré-SAW : |X|^p
          |
          v
    Quantification scalaire uniforme (8 ou 6 bits)
          |
          v
    [optionnel] post-SAW : |Y|^{1/p}  → mise en forme du bruit

Réutilise quant_scal_unif() de Quantificateur.py (inchangé).
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import firwin, filtfilt, welch
from scipy.signal.windows import hann
import matplotlib.pyplot as plt
import os

from Quantificateur import quant_scal_unif

SAW_P = 0.5
SAW_L = 512


# --------------------------------------------------------------------------
# 1) Lecture / écriture de fichiers WAV
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# 2) Filtre anti-repliement / anti-imagerie (FIR, phase nulle via filtfilt)
# --------------------------------------------------------------------------

def concevoir_filtre(fs, fc, numtaps=201):
    """
    Filtre passe-bas FIR (fenêtre de Hamming). fc doit être < Nyquist de la
    fréquence d'échantillonnage RÉDUITE (7350 Hz pour une décimation x3 à
    partir de 44.1 kHz), avec une marge de sécurité.
    """
    return firwin(numtaps, cutoff=fc, fs=fs, window="hamming")


def filtre_antirepliement(x, fs, facteur, marge=0.9, numtaps=201):
    """
    fc = marge * (fs_reduit / 2), avec fs_reduit = fs / facteur.
    marge < 1 laisse de la place pour la pente de transition du filtre FIR.
    """
    fs_reduit = fs / facteur
    fc = marge * (fs_reduit / 2)
    taps = concevoir_filtre(fs, fc, numtaps)
    return filtfilt(taps, [1.0], x), taps, fc


# --------------------------------------------------------------------------
# 3) Sous-échantillonnage / suréchantillonnage
# --------------------------------------------------------------------------

def sous_echantillonner(x, facteur):
    """Ne garde qu'un échantillon sur `facteur` (le filtrage doit être fait AVANT)."""
    return x[::facteur]


def suréchantillonner(x, facteur, taps):
    """
    Insertion de zéros + filtrage passe-bas (même filtre que l'anti-repliement,
    il sert ici d'anti-imagerie) + compensation de gain, pour ramener le
    signal réduit à la fréquence d'origine (utile pour l'écoute comparative).
    """
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


# --------------------------------------------------------------------------
# 4) Pipeline complet : anti-repliement -> décimation -> quantification
# --------------------------------------------------------------------------

def pipeline_reduction(x, fs, facteur=3, n_bits=8, pleine_echelle=1.0,
                       utiliser_saw=False, p_saw=SAW_P):
    """
    Retourne :
      x_reduit   : signal à fs/facteur à reconstruire (après post-SAW si activé)
      fs_reduit  : nouvelle fréquence d'échantillonnage
      ind        : indices de quantification
      taps       : coefficients du filtre
    """
    x_filtre, taps, fc = filtre_antirepliement(x, fs, facteur)
    x_decime = sous_echantillonner(x_filtre, facteur)
    fs_reduit = fs / facteur

    if utiliser_saw:
        x_pre = appliquer_saw(x_decime, p_saw)
        x_q, ind = quant_scal_unif(x_pre, -pleine_echelle, pleine_echelle, n_bits)
        x_reduit = appliquer_saw(x_q, 1.0 / p_saw)
    else:
        x_reduit, ind = quant_scal_unif(x_decime, -pleine_echelle, pleine_echelle, n_bits)

    return x_reduit, fs_reduit, ind, taps


def reconstruire_pour_ecoute(x_reduit, facteur, taps):
    """Remonte le signal réduit à la fréquence d'origine, pour comparaison à l'oreille."""
    return suréchantillonner(x_reduit, facteur, taps)


# --------------------------------------------------------------------------
# 5) Évaluation : bruit de quantification / SQNR
# --------------------------------------------------------------------------

def calculer_sqnr(x_ref, x_test):
    """
    SQNR = 10*log10(puissance_signal / puissance_bruit), en dB.
    À comparer à la règle empirique ~6.02 dB/bit (+1.76 dB pour un sinus
    plein échelle).
    """
    n = min(len(x_ref), len(x_test))
    erreur = x_ref[:n] - x_test[:n]
    puissance_signal = np.mean(x_ref[:n] ** 2)
    puissance_bruit = np.mean(erreur ** 2)
    if puissance_bruit < 1e-20:
        return np.inf
    return 10 * np.log10(puissance_signal / puissance_bruit)


def calculer_spectre_bruit(x_ref, x_test, fs, nperseg=2048):
    """
    Densité spectrale de puissance du bruit de quantification (méthode de
    Welch), en dB. Utile pour vérifier si le bruit est blanc (plat) --
    référence AVANT mise en forme (SAW), à comparer plus tard avec le bruit
    mis en forme.
    """
    n = min(len(x_ref), len(x_test))
    erreur = x_ref[:n] - x_test[:n]
    nperseg = min(nperseg, n)
    freqs, psd = welch(erreur, fs=fs, nperseg=nperseg)
    psd_db = 10 * np.log10(psd + 1e-20)
    return freqs, psd_db


# --------------------------------------------------------------------------
# 6) Graphiques d'évaluation
# --------------------------------------------------------------------------

def tracer_snr_vs_bits(x, fs, facteur, chemin_sortie, bits_range=range(2, 13),
                      utiliser_saw=False):
    """
    SQNR mesuré en fonction du nombre de bits par échantillon.
    """
    bits_liste = list(bits_range)
    sqnr_mesure = []

    for n_bits in bits_liste:
        x_reduit, fs_reduit, ind, taps = pipeline_reduction(
            x, fs, facteur=facteur, n_bits=n_bits, utiliser_saw=utiliser_saw
        )
        x_ecoute = reconstruire_pour_ecoute(x_reduit, facteur, taps)
        sqnr_mesure.append(calculer_sqnr(x, x_ecoute))

    plt.figure(figsize=(8, 5))
    plt.plot(bits_liste, sqnr_mesure, "o-", label="SQNR mesuré")
    plt.xlabel("Nombre de bits par échantillon")
    plt.ylabel("SQNR (dB)")
    titre = "SQNR en fonction du nombre de bits"
    if utiliser_saw:
        titre += " (quantification uniforme + SAW)"
    else:
        titre += " (quantification uniforme)"
    plt.title(titre)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(chemin_sortie, dpi=150)
    plt.close()

    return bits_liste, sqnr_mesure


def tracer_spectre_bruit(x, fs, facteur, chemin_sortie, n_bits_liste=(8, 6),
                         utiliser_saw=False):
    """Spectre (PSD, méthode de Welch) du bruit de quantification."""
    plt.figure(figsize=(9, 5))

    for n_bits in n_bits_liste:
        x_reduit, fs_reduit, ind, taps = pipeline_reduction(
            x, fs, facteur=facteur, n_bits=n_bits, utiliser_saw=utiliser_saw
        )
        x_ecoute = reconstruire_pour_ecoute(x_reduit, facteur, taps)
        freqs, psd_db = calculer_spectre_bruit(x, x_ecoute, fs)
        plt.plot(freqs, psd_db, label=f"{n_bits} bits")

    plt.axvline(fs / (2 * facteur), color="gray", linestyle=":", alpha=0.6,
                label=f"Nyquist réduite ({fs/(2*facteur):.0f} Hz)")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("PSD du bruit (dB/Hz)")
    if utiliser_saw:
        plt.title("Spectre du bruit de quantification (avec mise en forme SAW)")
    else:
        plt.title("Spectre du bruit de quantification (sans mise en forme)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, fs / 2)
    plt.tight_layout()
    plt.savefig(chemin_sortie, dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# 7) Programme principal
# --------------------------------------------------------------------------

def main():
    chemin_wav = "inputs/scaphandre.wav"
    dossier_sortie = "outputs/comp"
    os.makedirs(dossier_sortie, exist_ok=True)

    if os.path.exists(chemin_wav):
        fs, x = charger_wav(chemin_wav)
        print(f"Fichier chargé : {chemin_wav} (fs = {fs} Hz, durée = {len(x)/fs:.2f} s)")
    else:
        print("Fichier introuvable : signal synthétique de démonstration (sinus + parole simulée).")
        fs = 44100
        t = np.arange(int(fs * 1.0)) / fs
        x = 0.7 * np.sin(2 * np.pi * 1000 * t)

    facteur = 3
    resultats = {}

    for n_bits in [8, 6]:
        for avec_saw, suffixe in ((False, "sans_saw"), (True, "saw")):
            x_reduit, fs_reduit, ind, taps = pipeline_reduction(
                x, fs, facteur=facteur, n_bits=n_bits, utiliser_saw=avec_saw
            )
            x_ecoute = reconstruire_pour_ecoute(x_reduit, facteur, taps)
            nom = f"scaphandre_{n_bits}bits_{suffixe}.wav"
            chemin_sortie = os.path.join(dossier_sortie, nom)
            sauvegarder_wav(chemin_sortie, fs, x_ecoute)
            sauvegarder_wav(os.path.join("outputs", nom), fs, x_ecoute)

            sqnr = calculer_sqnr(x, x_ecoute)
            print(f"\n--- {n_bits} bits {'+ SAW' if avec_saw else 'sans SAW'} ---")
            print(f"fs réduite  : {fs_reduit:.0f} Hz")
            print(f"SQNR        : {sqnr:.1f} dB")
            print(f"Sortie      : {chemin_sortie}")
            if avec_saw:
                resultats[n_bits] = (x_reduit, fs_reduit, x_ecoute, sqnr)

    n_aff = min(2000, len(x))
    plt.figure(figsize=(10, 5))
    plt.plot(x[:n_aff], label="Original (scaphandre)", alpha=0.8)
    for n_bits, (_, _, x_ecoute, _) in resultats.items():
        plt.plot(x_ecoute[:n_aff], label=f"SAW {n_bits} bits", alpha=0.7)
    plt.xlabel("échantillon")
    plt.ylabel("amplitude")
    plt.title("Original vs compression SAW (8 et 6 bits)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    chemin_fig = os.path.join(dossier_sortie, "comparaison_quantification_saw.png")
    plt.savefig(chemin_fig, dpi=150)
    print(f"\nGraphique comparatif -> {chemin_fig}")

    chemin_snr = os.path.join(dossier_sortie, "snr_vs_bits.png")
    tracer_snr_vs_bits(x, fs, facteur, chemin_snr, utiliser_saw=False)
    print(f"Graphique SQNR vs bits (sans SAW) -> {chemin_snr}")

    chemin_snr_saw = os.path.join(dossier_sortie, "snr_vs_bits_saw.png")
    tracer_snr_vs_bits(x, fs, facteur, chemin_snr_saw, utiliser_saw=True)
    print(f"Graphique SQNR vs bits (avec SAW) -> {chemin_snr_saw}")

    chemin_bruit = os.path.join(dossier_sortie, "spectre_bruit.png")
    tracer_spectre_bruit(x, fs, facteur, chemin_bruit, n_bits_liste=[8, 6],
                         utiliser_saw=False)
    print(f"Graphique spectre du bruit (sans SAW) -> {chemin_bruit}")

    chemin_bruit_saw = os.path.join(dossier_sortie, "spectre_bruit_saw.png")
    tracer_spectre_bruit(x, fs, facteur, chemin_bruit_saw, n_bits_liste=[8, 6],
                         utiliser_saw=True)
    print(f"Graphique spectre du bruit (avec SAW) -> {chemin_bruit_saw}")


if __name__ == "__main__":
    main()

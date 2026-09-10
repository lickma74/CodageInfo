"""
Partie B (avant SAW) : réduction du débit du signal du scaphandre.

    44.1 kHz / 16 bits
          |
          v
    Filtre anti-repliement (passe-bas, fc < 7350 Hz)
          |
          v
    Sous-échantillonnage x3
          |
          v
    14.7 kHz / 16 bits
          |
          v
    Quantification scalaire uniforme (8 ou 6 bits)

Réutilise quant_scal_unif() de Quantificateur.py (inchangé).
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import firwin, filtfilt, welch
import matplotlib.pyplot as plt
import os

from Quantificateur import quant_scal_unif


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


# --------------------------------------------------------------------------
# 4) Pipeline complet : anti-repliement -> décimation -> quantification
# --------------------------------------------------------------------------

def pipeline_reduction(x, fs, facteur=3, n_bits=8, pleine_echelle=1.0):
    """
    Retourne :
      x_reduit   : signal quantifié à fs/facteur (valeurs en amplitude, pas indices)
      fs_reduit  : nouvelle fréquence d'échantillonnage
      ind        : indices de quantification (utile pour compter les niveaux/bits réellement utilisés)
      taps       : coefficients du filtre (réutilisables pour la reconstruction)
    """
    x_filtre, taps, fc = filtre_antirepliement(x, fs, facteur)
    x_decime = sous_echantillonner(x_filtre, facteur)
    fs_reduit = fs / facteur

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

def tracer_snr_vs_bits(x, fs, facteur, chemin_sortie, bits_range=range(2, 13)):
    """
    SQNR mesuré (et théorique ~6.02*n+1.76 dB) en fonction du nombre de
    bits par échantillon, pour visualiser la règle des ~6 dB/bit et
    repérer où la qualité perceptuelle décroche (ex. sous 6 bits).
    """
    bits_liste = list(bits_range)
    sqnr_mesure = []

    for n_bits in bits_liste:
        x_reduit, fs_reduit, ind, taps = pipeline_reduction(x, fs, facteur=facteur, n_bits=n_bits)
        x_ecoute = reconstruire_pour_ecoute(x_reduit, facteur, taps)
        sqnr_mesure.append(calculer_sqnr(x, x_ecoute))

    plt.figure(figsize=(8, 5))
    plt.plot(bits_liste, sqnr_mesure, "o-", label="SQNR mesuré")
    #plt.axvline(6, color="gray", linestyle=":", alpha=0.6, label="6 bits")
    plt.xlabel("Nombre de bits par échantillon")
    plt.ylabel("SQNR (dB)")
    plt.title("SQNR en fonction du nombre de bits (quantification uniforme)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(chemin_sortie, dpi=150)
    plt.close()

    return bits_liste, sqnr_mesure


def tracer_spectre_bruit(x, fs, facteur, chemin_sortie, n_bits_liste=(8, 6)):
    """Spectre (PSD, méthode de Welch) du bruit de quantification pour chaque profondeur de bits."""
    plt.figure(figsize=(9, 5))

    for n_bits in n_bits_liste:
        x_reduit, fs_reduit, ind, taps = pipeline_reduction(x, fs, facteur=facteur, n_bits=n_bits)
        x_ecoute = reconstruire_pour_ecoute(x_reduit, facteur, taps)
        freqs, psd_db = calculer_spectre_bruit(x, x_ecoute, fs)
        plt.plot(freqs, psd_db, label=f"{n_bits} bits")

    plt.axvline(fs / (2 * facteur), color="gray", linestyle=":", alpha=0.6,
                label=f"Nyquist réduite ({fs/(2*facteur):.0f} Hz)")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("PSD du bruit (dB/Hz)")
    plt.title("Spectre du bruit de quantification (sans mise en forme)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(chemin_sortie, dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# 7) Programme principal
# --------------------------------------------------------------------------

def main():
    chemin_wav = "inputs/parole1.wav"
    dossier_sortie = "outputs/comp"
    os.makedirs(dossier_sortie, exist_ok=True)

    if os.path.exists(chemin_wav):
        fs, x = charger_wav(chemin_wav)
        print(f"Fichier chargé : {chemin_wav} (fs = {fs} Hz, durée = {len(x)/fs:.2f} s)")
    else:
        print("Fichier introuvable : signal synthétique de démonstration (sinus + parole simulée).")
        fs = 44100
        t = np.arange(int(fs * 1.0)) / fs
        x = 0.7 * np.sin(2 * np.pi * 1000 * t)  # sinus plein échelle-ish, pratique pour vérifier le SQNR théorique

    facteur = 3
    resultats = {}

    for n_bits in [8, 6]:
        x_reduit, fs_reduit, ind, taps = pipeline_reduction(x, fs, facteur=facteur, n_bits=n_bits)
        x_ecoute = reconstruire_pour_ecoute(x_reduit, facteur, taps)

        chemin_sortie = os.path.join(dossier_sortie, f"scaphandre_{n_bits}bits_reconstruit.wav")
        sauvegarder_wav(chemin_sortie, fs, x_ecoute)

        sqnr = calculer_sqnr(x, x_ecoute)
        sqnr_theorique = 6.02 * n_bits + 1.76

        print(f"\n--- {n_bits} bits ---")
        print(f"fs réduite       : {fs_reduit:.0f} Hz")
        print(f"Niveaux utilisés : {len(np.unique(ind))} / {2**n_bits}")
        print(f"SQNR mesuré      : {sqnr:.1f} dB")
        print(f"SQNR théorique   : ~{sqnr_theorique:.1f} dB (sinus plein échelle)")
        print(f"Sortie           : {chemin_sortie}")

        resultats[n_bits] = (x_reduit, fs_reduit, x_ecoute, sqnr)

    # Graphique comparatif sur un court extrait
    n_aff = min(2000, len(x))
    plt.figure(figsize=(10, 5))
    plt.plot(x[:n_aff], label="Original (44.1 kHz / 16 bits)", alpha=0.8)
    for n_bits, (_, _, x_ecoute, _) in resultats.items():
        plt.plot(x_ecoute[:n_aff], label=f"Reconstruit ({n_bits} bits)", alpha=0.7)
    plt.xlabel("échantillon (à 44.1 kHz)")
    plt.ylabel("amplitude")
    plt.title("Comparaison original vs signal réduit (8 et 6 bits)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    chemin_fig = os.path.join(dossier_sortie, "comparaison_quantification.png")
    plt.savefig(chemin_fig, dpi=150)
    print(f"\nGraphique comparatif -> {chemin_fig}")

    # SQNR en fonction du nombre de bits
    chemin_snr = os.path.join(dossier_sortie, "snr_vs_bits.png")
    tracer_snr_vs_bits(x, fs, facteur, chemin_snr)
    print(f"Graphique SQNR vs bits -> {chemin_snr}")

    # Spectre du bruit de quantification (8 et 6 bits)
    chemin_bruit = os.path.join(dossier_sortie, "spectre_bruit.png")
    tracer_spectre_bruit(x, fs, facteur, chemin_bruit, n_bits_liste=[8, 6])
    print(f"Graphique spectre du bruit -> {chemin_bruit}")


if __name__ == "__main__":
    main()

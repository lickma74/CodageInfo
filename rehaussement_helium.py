"""
Restauration de la voix "hélium" par compression cepstrale de l'enveloppe
spectrale (approche DFT/FFT).

Principe (voir le schéma bloc discuté) :
  1) Analyse par trames (overlap-add 50%, fenêtres racine de Hann
     complémentaires).
  2) Pour chaque trame : FFT -> magnitude/phase -> log|X(k)| -> IFFT
     -> cepstre.
  3) Liftering du cepstre : quéfrence basse = enveloppe spectrale
     (formants), quéfrence haute = structure fine (F0 + harmoniques).
  4) SEULE l'enveloppe est comprimée sur l'axe des fréquences (facteur
     2 à 3) ; la structure fine et la phase restent inchangées.
  5) Recombinaison (somme des log-magnitudes, puis exp), réapplication
     de la phase d'origine, IFFT, fenêtre de synthèse, overlap-add.
  6) Normalisation d'énergie par trame (pas de gain/atténuation).

Dépendances : numpy, scipy, matplotlib (installation standard).
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
from scipy.signal.windows import hann
import matplotlib.pyplot as plt
import os


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
# 2) Fenêtres d'analyse/synthèse (identique à l'exercice de filtrage FFT)
# --------------------------------------------------------------------------

def fenetres_analyse_synthese(L):
    w = np.sqrt(hann(L, sym=False))
    return w, w


# --------------------------------------------------------------------------
# 3) Liftering : séparation enveloppe (basse quéfrence) / structure fine
# --------------------------------------------------------------------------

def construire_lifters(L, n_c, taper=None):
    """
    lifter_bas = 1 pour les quéfrences proches de 0 (et leur symétrique
    proche de L) -> enveloppe spectrale.
    lifter_haut = complément -> structure fine (F0 + harmoniques).
    Un léger "taper" (rampe cosinus) adoucit la coupure pour limiter les
    artefacts de type Gibbs.
    """
    if taper is None:
        taper = max(1, n_c // 4)

    lifter_bas = np.zeros(L)
    lifter_bas[:n_c + 1] = 1.0
    lifter_bas[L - n_c:] = 1.0

    if taper > 0:
        rampe = 0.5 * (1 + np.cos(np.linspace(0, np.pi, taper)))
        idx = np.arange(n_c + 1, min(n_c + 1 + taper, L // 2))
        lifter_bas[idx] = rampe[:len(idx)]
        idx_miroir = L - idx
        lifter_bas[idx_miroir] = rampe[:len(idx_miroir)]

    lifter_haut = 1.0 - lifter_bas
    return lifter_bas, lifter_haut


# --------------------------------------------------------------------------
# 4) Compression de l'enveloppe sur l'axe des fréquences
# --------------------------------------------------------------------------

def compresser_enveloppe(log_env, fs, L, facteur):
    """
    Ramène l'enveloppe "dilatée" (hélium) à sa forme naturelle en
    échantillonnant log_env à des fréquences multipliées par `facteur`.
    new_env(f) = env_mesurée(f * facteur)
    Au-delà de fs/2/facteur (information perdue, poussée hors bande),
    np.interp maintient la dernière valeur connue (léger applatissement,
    inévitable).
    Hypothèse : L pair (cas standard, ex. L=1024).
    """
    demi = L // 2 + 1
    k = np.arange(demi)
    f = k * fs / L
    f_source = np.clip(facteur * f, 0, fs / 2)

    env_demi_origine = log_env[:demi]
    nouvel_env_demi = np.interp(f_source, f, env_demi_origine)

    nouvel_env = np.zeros(L)
    nouvel_env[:demi] = nouvel_env_demi
    nouvel_env[demi:] = nouvel_env_demi[L // 2 - 1:0:-1]  # symétrie (L pair)
    return nouvel_env


# --------------------------------------------------------------------------
# 5) Cœur de l'algorithme : analyse-synthèse cepstrale par trames
# --------------------------------------------------------------------------

def rehausser_voix_helium(x, fs, L=1024, facteur_compression=2.5,
                           quefrence_coupure_ms=2.0, hop=None):
    if hop is None:
        hop = L // 2

    w_a, w_s = fenetres_analyse_synthese(L)
    n_c = max(1, int(round(quefrence_coupure_ms * 1e-3 * fs)))
    lifter_bas, lifter_haut = construire_lifters(L, n_c)

    pad_debut = L - hop
    x_pad = np.concatenate([np.zeros(pad_debut), x])
    n_trames = int(np.ceil((len(x_pad) - L) / hop)) + 1
    pad_fin = max((n_trames - 1) * hop + L - len(x_pad), 0)
    x_pad = np.concatenate([x_pad, np.zeros(pad_fin)])

    y_pad = np.zeros(len(x_pad))
    somme_fenetres = np.zeros(len(x_pad))
    eps = 1e-8

    for i in range(n_trames):
        deb = i * hop
        trame = x_pad[deb:deb + L]

        # --- Analyse ---
        X = np.fft.fft(trame * w_a)
        magnitude = np.abs(X)
        phase = np.angle(X)
        log_mag = np.log(magnitude + eps)
        cepstre = np.real(np.fft.ifft(log_mag))

        # --- Séparation enveloppe / structure fine ---
        log_env = np.real(np.fft.fft(cepstre * lifter_bas))
        log_fine = np.real(np.fft.fft(cepstre * lifter_haut))

        # --- Modification : compression de l'enveloppe SEULEMENT ---
        log_env_comprime = compresser_enveloppe(log_env, fs, L, facteur_compression)

        # --- Synthèse ---
        nouvelle_log_mag = log_env_comprime + log_fine
        nouvelle_magnitude = np.exp(nouvelle_log_mag)
        X_nouveau = nouvelle_magnitude * np.exp(1j * phase)
        y = np.real(np.fft.ifft(X_nouveau))

        # --- Normalisation d'énergie (pas de gain/atténuation) ---
        rms_in = np.sqrt(np.mean(trame ** 2) + 1e-12)
        rms_out = np.sqrt(np.mean(y ** 2) + 1e-12)
        if rms_out > 1e-9:
            y *= (rms_in / rms_out)

        y_fenetree = y * w_s
        y_pad[deb:deb + L] += y_fenetree
        somme_fenetres[deb:deb + L] += w_a * w_s

    non_nul = somme_fenetres > 1e-8
    y_pad[non_nul] /= somme_fenetres[non_nul]

    return y_pad[pad_debut:pad_debut + len(x)]


# --------------------------------------------------------------------------
# 6) Signal synthétique de démonstration/validation (voyelle "hélium")
# --------------------------------------------------------------------------

def generer_voyelle_synthetique(fs, duree, F0, formants, largeurs=None):
    """Modèle source-filtre simple : train d'impulsions + résonateurs."""
    n = int(fs * duree)
    T0 = fs / F0
    excitation = np.zeros(n)
    indices = np.round(np.arange(0, n, T0)).astype(int)
    indices = indices[indices < n]
    excitation[indices] = 1.0

    if largeurs is None:
        largeurs = [80] * len(formants)

    y = np.zeros(n)
    for f, bw in zip(formants, largeurs):
        r = np.exp(-np.pi * bw / fs)
        theta = 2 * np.pi * f / fs
        a1 = -2 * r * np.cos(theta)
        a2 = r ** 2
        y += lfilter([1.0], [1.0, a1, a2], excitation)

    return y / np.max(np.abs(y)) * 0.8


# --------------------------------------------------------------------------
# 7) Diagnostic visuel : enveloppe avant/après compression sur une trame
# --------------------------------------------------------------------------

def tracer_diagnostic_enveloppe(x, fs, L, facteur_compression,
                                 quefrence_coupure_ms, chemin_sortie):
    hop = L // 2
    w_a, _ = fenetres_analyse_synthese(L)
    n_c = max(1, int(round(quefrence_coupure_ms * 1e-3 * fs)))
    lifter_bas, _ = construire_lifters(L, n_c)

    # on choisit la trame la plus énergétique (probablement voisée)
    n_trames = (len(x) - L) // hop
    energies = [np.sum(x[i * hop:i * hop + L] ** 2) for i in range(max(n_trames, 1))]
    i_choisi = int(np.argmax(energies)) if energies else 0
    deb = i_choisi * hop
    trame = x[deb:deb + L]

    X = np.fft.fft(trame * w_a)
    log_mag = np.log(np.abs(X) + 1e-8)
    cepstre = np.real(np.fft.ifft(log_mag))
    log_env = np.real(np.fft.fft(cepstre * lifter_bas))
    log_env_comprime = compresser_enveloppe(log_env, fs, L, facteur_compression)

    freqs = np.arange(L // 2 + 1) * fs / L
    plt.figure(figsize=(9, 5))
    plt.plot(freqs, log_env[:L // 2 + 1], label="Enveloppe mesurée (hélium)")
    plt.plot(freqs, log_env_comprime[:L // 2 + 1], label=f"Enveloppe restaurée (÷{facteur_compression})")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("log-amplitude")
    plt.title("Enveloppe spectrale avant/après compression (trame la plus énergétique)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(chemin_sortie, dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# 8) Programme principal
# --------------------------------------------------------------------------

def main():
    # --- Remplacer par le chemin de votre fichier de voix hélium ---
    chemin_wav = "inputs/hel_fr4.wav"
    dossier_sortie = "outputs"
    os.makedirs(dossier_sortie, exist_ok=True)

    L = 1024                     # ~23 ms à 44.1 kHz, ≤ 50 ms, quasi-stationnaire
    facteur_compression = 2.5    # entre 2 et 3, selon l'énoncé
    quefrence_coupure_ms = 2.0   # sépare enveloppe (formants) et structure fine

    if os.path.exists(chemin_wav):
        fs, x = charger_wav(chemin_wav)
        print(f"Fichier chargé : {chemin_wav} (fs = {fs} Hz, durée = {len(x)/fs:.2f} s)")
    else:
        print("Fichier introuvable : génération d'une voyelle synthétique "
              "'hélium' pour valider l'algorithme (formants dilatés x2.5).")
        fs = 44100
        formants_normaux = [700, 1200, 2600]
        formants_helium = [f * facteur_compression for f in formants_normaux]
        x = generer_voyelle_synthetique(fs, duree=1.5, F0=140, formants=formants_helium)

    y = rehausser_voix_helium(x, fs, L=L,
                               facteur_compression=facteur_compression,
                               quefrence_coupure_ms=quefrence_coupure_ms)

    chemin_sortie_wav = os.path.join(dossier_sortie, "voix_restauree_4.wav")
    sauvegarder_wav(chemin_sortie_wav, fs, y)
    print(f"Signal restauré -> {chemin_sortie_wav}")

    chemin_diag = os.path.join(dossier_sortie, "diagnostic_enveloppe.png")
    tracer_diagnostic_enveloppe(x, fs, L, facteur_compression, quefrence_coupure_ms, chemin_diag)
    print(f"Diagnostic (enveloppe avant/après) -> {chemin_diag}")

    print(f"\nAmplitude max entrée  : {np.max(np.abs(x)):.4f}")
    print(f"Amplitude max sortie  : {np.max(np.abs(y)):.4f}")


if __name__ == "__main__":
    main()

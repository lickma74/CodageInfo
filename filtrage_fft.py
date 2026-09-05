"""
Filtrage par FFT d'un signal audio, par blocs analyse-synthèse (overlap-add 50%).

Principe :
  1) On découpe x[n] en trames de L échantillons, avec un recouvrement de 50 %
     (hop = L/2).
  2) Chaque trame est fenêtrée (fenêtre d'analyse), transformée par FFT.
  3) Le spectre est multiplié par un masque H(k) qui définit le filtre désiré
     (passe-bas demi-bande, passe-haut demi-bande, passe-bande).
  4) On revient au domaine temporel par IFFT, on refenêtre (fenêtre de
     synthèse), puis on recombine les trames traitées par overlap-add.
  5) Les fenêtres d'analyse et de synthèse sont choisies "complémentaires"
     (racine de Hann pour les deux) de sorte que leur PRODUIT redonne une
     fenêtre de Hann classique, qui satisfait la condition COLA
     (Constant OverLap-Add) à 50 % de recouvrement -> reconstruction exacte
     quand H(k) = 1 (pas de filtrage).

Dépendances : numpy, scipy, matplotlib (installation standard).
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal.windows import hann
import matplotlib.pyplot as plt
import os


# --------------------------------------------------------------------------
# 1) Lecture / écriture de fichiers WAV
# --------------------------------------------------------------------------

def charger_wav(chemin):
    """Charge un fichier WAV et retourne (fs, x) avec x en float64 dans [-1, 1]."""
    fs, x = wavfile.read(chemin)
    if x.ndim > 1:
        x = x[:, 0]  # on ne garde qu'un canal si le fichier est stéréo
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float64)
    return fs, x


def sauvegarder_wav(chemin, fs, x):
    """Sauvegarde un signal float (~[-1, 1]) en WAV 16 bits."""
    x_clip = np.clip(x, -1.0, 1.0)
    wavfile.write(chemin, fs, (x_clip * 32767).astype(np.int16))


# --------------------------------------------------------------------------
# 2) Conception des masques fréquentiels H(k)
# --------------------------------------------------------------------------

def concevoir_filtres(fs, L):
    """
    Construit les masques fréquentiels (brick-wall) pour les 3 filtres demandés.
    Utilise np.fft.fftfreq pour obtenir des fréquences signées, ce qui donne
    automatiquement un masque à symétrie conjuguée (nécessaire pour que
    l'IFFT reste réelle).
    """
    freqs = np.fft.fftfreq(L, d=1.0 / fs)
    f_abs = np.abs(freqs)

    fc_demi_bande = fs / 4.0  # coupure du filtre "demi-bande" = moitié de Nyquist

    H_passe_bas = (f_abs <= fc_demi_bande).astype(float)
    H_passe_haut = 1.0 - H_passe_bas  # complément exact du passe-bas

    H_passe_bande = ((f_abs >= 300.0) & (f_abs <= 3400.0)).astype(float)

    return {
        "passe_bas_demi_bande": H_passe_bas,
        "passe_haut_demi_bande": H_passe_haut,
        "passe_bande_300_3400": H_passe_bande,
    }


# --------------------------------------------------------------------------
# 3) Coeur de l'algorithme : filtrage par blocs avec overlap-add
# --------------------------------------------------------------------------

def fenetres_analyse_synthese(L):
    """
    Fenêtres complémentaires : racine de Hann pour l'analyse ET la synthèse.
    Leur produit redonne une fenêtre de Hann classique, qui satisfait la
    condition COLA à 50 % de recouvrement (hop = L/2).
    """
    w = np.sqrt(hann(L, sym=False))
    return w, w  # w_analyse, w_synthese


def stft_filter(x, H, L=1024, hop=None):
    """
    Filtre le signal x par blocs FFT/IFFT avec le masque fréquentiel H
    (tableau de longueur L), en utilisant l'overlap-add à 50 % par défaut.
    """
    if hop is None:
        hop = L // 2

    w_a, w_s = fenetres_analyse_synthese(L)

    # Zero-padding : marge au début pour que la première trame se construise
    # progressivement, et à la fin pour couvrir un nombre entier de trames.
    pad_debut = L - hop
    x_pad = np.concatenate([np.zeros(pad_debut), x])
    n_trames = int(np.ceil((len(x_pad) - L) / hop)) + 1
    pad_fin = max((n_trames - 1) * hop + L - len(x_pad), 0)
    x_pad = np.concatenate([x_pad, np.zeros(pad_fin)])

    y_pad = np.zeros(len(x_pad))
    somme_fenetres = np.zeros(len(x_pad))  # pour vérifier/normaliser la COLA

    for i in range(n_trames):
        deb = i * hop
        trame = x_pad[deb:deb + L]

        X = np.fft.fft(trame * w_a)          # analyse : fenêtrage + FFT
        Y = X * H                             # filtrage dans le domaine fréquentiel
        y = np.real(np.fft.ifft(Y))           # synthèse : IFFT
        y_fenetree = y * w_s                  # fenêtrage de synthèse

        y_pad[deb:deb + L] += y_fenetree
        somme_fenetres[deb:deb + L] += w_a * w_s

    # Normalisation par la somme réelle des produits de fenêtres (garde-fou
    # aux bords, où la somme théoriquement constante n'est pas encore atteinte)
    non_nul = somme_fenetres > 1e-8
    y_pad[non_nul] /= somme_fenetres[non_nul]


    # Retrait du padding pour retrouver une longueur proche de l'original
    y = y_pad[pad_debut:pad_debut + len(x)]
    return y

# --------------------------------------------------------------------------
# 4) Mesure de la réponse impulsionnelle "par mesure"
# --------------------------------------------------------------------------

def mesurer_reponse_impulsionnelle(H, L, n_total=4096, demi_fenetre=None):
    """
    Envoie une impulsion pure dans le pipeline complet et récupère la sortie
    autour de l'impulsion : c'est la réponse impulsionnelle EFFECTIVE de
    l'algorithme (fenêtrage inclus), pas seulement la réponse théorique idéale.
    """
    if demi_fenetre is None:
        demi_fenetre = L

    impulsion = np.zeros(n_total)
    centre = n_total // 2
    impulsion[centre] = 1.0

    h = stft_filter(impulsion, H, L)

    deb = max(centre - demi_fenetre, 0)
    fin = min(centre + demi_fenetre, len(h))
    axe_n = np.arange(deb, fin) - centre
    return axe_n, h[deb:fin]


# --------------------------------------------------------------------------
# 5) Programme principal
# --------------------------------------------------------------------------

def generer_signal_demo(fs, duree=2.0):
    """Signal synthétique simple pour tester le pipeline si aucun WAV n'est disponible."""
    t = np.arange(int(fs * duree)) / fs
    x = (0.3 * np.sin(2 * np.pi * 220 * t)
         + 0.2 * np.sin(2 * np.pi * 1500 * t)
         + 0.1 * np.sin(2 * np.pi * 6000 * t)
         + 0.02 * np.random.randn(len(t)))
    return x / np.max(np.abs(x)) * 0.8


def verifier_cola(fs, L=1024):
    """
    Test de sanité recommandé : filtre 'identité' (H=1 partout). Si la COLA
    est correcte, la sortie doit être quasi identique à l'entrée.
    """
    x = generer_signal_demo(fs, duree=0.5)
    H_identite = np.ones(L)
    y = stft_filter(x, H_identite, L)
    erreur = np.max(np.abs(y - x))
    print(f"[Vérification COLA] erreur max (sans filtrage) = {erreur:.2e}")


def main():
    # --- Remplacer par le chemin réel de yellow_48k.wav ou eagles_48k.wav ---
    chemin_wav = "inputs/eagles_48k.wav"
    dossier_sortie = "outputs"
    os.makedirs(dossier_sortie, exist_ok=True)

    if os.path.exists(chemin_wav):
        fs, x = charger_wav(chemin_wav)
        print(f"Fichier chargé : {chemin_wav} (fs = {fs} Hz, durée = {len(x)/fs:.2f} s)")
    else:
        print("Fichier audio introuvable : utilisation d'un signal synthétique de démonstration.")
        fs = 48000
        x = generer_signal_demo(fs)

    L = 1024

    # Test de sanité (pipeline identité)
    verifier_cola(fs, L)

    # Conception des 3 filtres et filtrage du signal
    filtres = concevoir_filtres(fs, L)
    for nom, H in filtres.items():
        y = stft_filter(x, H, L)
        chemin_sortie = os.path.join(dossier_sortie, f"sortie_{nom}.wav")
        sauvegarder_wav(chemin_sortie, fs, y)
        print(f"Filtre '{nom}' -> {chemin_sortie}")

    # Mesure et affichage des réponses impulsionnelles
    fig, axes = plt.subplots(len(filtres), 1, figsize=(9, 8), sharex=True)
    for ax, (nom, H) in zip(axes, filtres.items()):
        n, h = mesurer_reponse_impulsionnelle(H, L, demi_fenetre=200)
        ax.plot(n, h)
        ax.set_title(nom)
        ax.set_ylabel("amplitude")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("n (échantillons, relatif à l'impulsion)")
    fig.suptitle("Réponses impulsionnelles mesurées des 3 filtres")
    fig.tight_layout()
    chemin_fig = os.path.join(dossier_sortie, "reponses_impulsionnelles.png")
    fig.savefig(chemin_fig, dpi=150)
    print(f"Figure des réponses impulsionnelles -> {chemin_fig}")


if __name__ == "__main__":
    main()

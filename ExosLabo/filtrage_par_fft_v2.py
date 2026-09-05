"""
filtrage_par_FFT.py

Démonstration de filtrage par transformée (ici, la FFT).
Approche par trames, avec fenêtrage complémentaire à 50 %.
"""

import numpy as np
import soundfile as sf


def hanning_octave(N):
    """
    Reproduit la fenêtre de Hanning telle que calculée par Octave/Matlab.
    """
    n = np.arange(1, N + 1)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / (N + 1))


def main():

    # Ajustez ce chemin selon l'emplacement du fichier audio
    chemin_audio = r'inputs/eagles_48k.wav'

    # Lecture du fichier audio
    sig, Fe = sf.read(chemin_audio)

    # Nombre d'échantillons
    N = len(sig)

    # Longueur d'une trame (20 ms à Fe = 48 kHz)
    L = 480 * 2

    # Longueur de la fenêtre
    LW = 2 * L

    # Nombre de trames
    N_trames = int(np.floor(N / L)) - 1

    # Fenêtre d'analyse et de synthèse
    w = np.sqrt(hanning_octave(LW))


    # ============================================================
    # MASQUES POUR LE FILTRAGE FRÉQUENTIEL
    # ============================================================

    mask_LP = np.concatenate((
        [1],
        np.ones(LW // 4 - 1),
        [0],
        np.zeros(LW // 2),
        np.ones(LW // 4 - 1),
    ))

    mask_HP = 1 - mask_LP

    # Fréquences de coupure du filtre passe-bande
    ind_min = int(np.floor((300 / Fe) * LW))
    ind_max = int(np.floor((3400 / Fe) * LW))

    n_ones = ind_max - ind_min + 1

    mask_BP = np.zeros(LW)

    # Première partie du spectre
    mask_BP[ind_min - 1 : ind_max] = np.ones(n_ones)

    # Partie symétrique du spectre
    mask_BP[LW - ind_max - 1 : LW - ind_min] = np.ones(n_ones)


    # ============================================================
    # INITIALISATION
    # ============================================================

    ptr = 0

    mem_synthese = np.zeros(L)

    bloc_avant_fft = np.zeros(LW)

    bloc_apres_ifft = np.zeros(LW)

    signal_filtre = np.zeros(N)


    # ============================================================
    # BOUCLE PRINCIPALE
    # ============================================================

    for trame in range(N_trames):

        # Extraction d'une nouvelle trame
        new_frame = sig[ptr : ptr + L]

        # Placement de la trame dans le bloc
        bloc_avant_fft[LW // 2 :] = new_frame

        # Fenêtrage avant FFT
        xw = bloc_avant_fft * w

        # FFT
        Xf = np.fft.fft(xw)

        # Application du filtre passe-bande
        Xf_mod = Xf * mask_HP

        # FFT inverse
        y = np.real(np.fft.ifft(Xf_mod))

        # Fenêtrage après FFT inverse
        yw = y * w

        # Overlap-Add
        trame_OLA = yw[0 : LW // 2] + mem_synthese

        # Sauvegarde de la trame filtrée
        signal_filtre[ptr : ptr + L] = trame_OLA

        # Mémorisation de la deuxième moitié
        mem_synthese = yw[LW // 2 :]

        # Décalage du bloc
        bloc_avant_fft[0 : LW // 2] = bloc_avant_fft[LW // 2 :]

        # Avancement du pointeur
        ptr = ptr + L


    # ============================================================
    # ÉCRITURE DU SIGNAL FILTRÉ
    # ============================================================

    # Saturation dans [-1, 1]
    signal_filtre_sat = np.clip(signal_filtre, -1.0, 1.0)

    # Fichier de sortie
    chemin_audio_sortie = r'outputs/eagles_48k_filtre_HP.wav'

    # Écriture
    sf.write(
        chemin_audio_sortie,
        signal_filtre_sat,
        Fe
    )

    print(f"Signal filtré écrit dans : {chemin_audio_sortie}")


# ================================================================
# POINT D'ENTRÉE DU PROGRAMME
# ================================================================

if __name__ == "__main__":
    main()
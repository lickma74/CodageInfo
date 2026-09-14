"""
Restauration de voix hélium par compression cepstrale de l'enveloppe (FFT).

log|X| = enveloppe + structure fine. Liftering bas → enveloppe, qu'on
rééchantillonne en fréquence (f → α f). La phase et les hautes quéfrences
sont conservées. Reconstruction IFFT + overlap-add 50 % (√Hann).
"""

import os
import numpy as np
from scipy.io import wavfile
from scipy.signal.windows import hann

L_FFT = 1024
QUEFRENCE_MS = 2.0
ALPHA0 = {"hel_fr1": 2.0, "hel_fr2": 2.0, "hel_fr4": 2.5}
DOSSIER = "outputs/fft"


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


def construire_lifters(L, n_c, taper=None):
    """lifter_bas ≈ enveloppe (quéfrence ~ 0) ; lifter_haut = complément (F0, harmoniques)."""
    if taper is None:
        taper = max(1, n_c // 4)
    lifter_bas = np.zeros(L)
    lifter_bas[: n_c + 1] = 1.0
    lifter_bas[L - n_c:] = 1.0
    if taper > 0:
        rampe = 0.5 * (1 + np.cos(np.linspace(0, np.pi, taper)))
        idx = np.arange(n_c + 1, min(n_c + 1 + taper, L // 2))
        lifter_bas[idx] = rampe[: len(idx)]
        lifter_bas[L - idx] = rampe[: len(idx)]
    return lifter_bas, 1.0 - lifter_bas


def compresser_enveloppe(log_env, fs, L, facteur):
    """H'(f) = H(α f) par interpolation ; au-delà de Nyquist on sature à fs/2."""
    demi = L // 2 + 1
    f = np.arange(demi) * fs / L
    nouvel = np.interp(np.clip(facteur * f, 0, fs / 2), f, log_env[:demi])
    env = np.zeros(L)
    env[:demi] = nouvel
    env[demi:] = nouvel[L // 2 - 1:0:-1]
    return env


def rehausser_voix_helium(x, fs, L=L_FFT, facteur_compression=2.5,
                          quefrence_coupure_ms=QUEFRENCE_MS, hop=None):
    hop = L // 2 if hop is None else hop
    w = np.sqrt(hann(L, sym=False))
    n_c = max(1, int(round(quefrence_coupure_ms * 1e-3 * fs)))
    lifter_bas, lifter_haut = construire_lifters(L, n_c)

    pad_debut = L - hop
    x_pad = np.concatenate([np.zeros(pad_debut), x])
    n_trames = int(np.ceil((len(x_pad) - L) / hop)) + 1
    x_pad = np.concatenate([x_pad, np.zeros(max((n_trames - 1) * hop + L - len(x_pad), 0))])

    y_pad = np.zeros(len(x_pad))
    somme_w = np.zeros(len(x_pad))
    eps = 1e-8
    for i in range(n_trames):
        deb = i * hop
        trame = x_pad[deb:deb + L]
        X = np.fft.fft(trame * w)
        log_mag = np.log(np.abs(X) + eps)
        cepstre = np.real(np.fft.ifft(log_mag))
        log_env = np.real(np.fft.fft(cepstre * lifter_bas))
        log_fine = np.real(np.fft.fft(cepstre * lifter_haut))
        log_env_c = compresser_enveloppe(log_env, fs, L, facteur_compression)
        y = np.real(np.fft.ifft(np.exp(log_env_c + log_fine) * np.exp(1j * np.angle(X))))
        rms_in = np.sqrt(np.mean(trame ** 2) + 1e-12)
        rms_out = np.sqrt(np.mean(y ** 2) + 1e-12)
        if rms_out > 1e-9:
            y *= rms_in / rms_out
        y_pad[deb:deb + L] += y * w
        somme_w[deb:deb + L] += w * w

    ok = somme_w > 1e-8
    y_pad[ok] /= somme_w[ok]
    return y_pad[pad_debut:pad_debut + len(x)]


def main():
    os.makedirs(DOSSIER, exist_ok=True)
    for nom, alpha in ALPHA0.items():
        chemin = f"inputs/{nom}.wav"
        if not os.path.exists(chemin):
            print(f"manquant : {chemin}")
            continue
        fs, x = charger_wav(chemin)
        y = rehausser_voix_helium(x, fs, facteur_compression=alpha)
        out = os.path.join(DOSSIER, f"{nom}.wav")
        sauvegarder_wav(out, fs, y)
        print(f"{nom}  α={alpha}  peak {np.max(np.abs(y)):.3f}  -> {out}")


if __name__ == "__main__":
    main()

"""
Étape 6 — Synthèse LPC complète (toutes les trames + overlap-add).

Méthode (pôles, forme directe) :
  1) fenêtrage √Hann
  2) LPC → a, excitation e = A(z)·x  (lfilter causal)
  3) compression des pôles : θ → θ/α (+ re-stabilisation)
  4) synthèse : y = (1/A'(z)) · e   (forme directe, ordre 20)
  5) √Hann synthèse + overlap-add
  6) normalisation RMS ≈ entrée
"""

import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
from scipy.signal.windows import hann
import matplotlib.pyplot as plt


DUREE_TRAME_MS = 20
RECOUVREMENT = 0.5
ORDRE_LPC = 20
ALPHAS = (1.5, 2.0, 2.5, 3.0)
R_MAX = 0.985


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
    x = np.clip(x, -1.0, 1.0)
    wavfile.write(chemin, fs, (x * 32767.0).astype(np.int16))


def parametres_trame(fs):
    L = int(round(fs * DUREE_TRAME_MS / 1000.0))
    if L % 2 == 1:
        L += 1
    hop = int(L * (1.0 - RECOUVREMENT))
    return L, hop


def autocorrelation(x, ordre):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(2 * n - 1)))
    X = np.fft.rfft(x, n=nfft)
    r = np.fft.irfft(np.abs(X) ** 2, n=nfft)
    return r[: ordre + 1]


def levinson_durbin(r, ordre):
    a = np.zeros(ordre)
    e = r[0]
    if e <= 1e-12:
        return a, 0.0
    for i in range(1, ordre + 1):
        acc = r[i] + np.dot(a[: i - 1], r[i - 1:0:-1])
        k = -acc / e
        a_prev = a[: i - 1].copy()
        a[i - 1] = k
        if i > 1:
            a[: i - 1] = a_prev + k * a_prev[::-1]
        e *= (1.0 - k * k)
        if e <= 1e-18:
            break
    return a, float(max(e, 0.0))


def lpc(x, ordre):
    r = autocorrelation(x, ordre)
    a, e = levinson_durbin(r, ordre)
    gain = np.sqrt(e) if e > 0 else 0.0
    return a, gain


def residual(x, a):
    return lfilter(np.concatenate([[1.0], a]), [1.0], x)


def compresser_poles_lpc(a, alpha, r_max=R_MAX):
    """θ → θ/α, retourne les nouveaux coeffs a (forme directe)."""
    if abs(alpha - 1.0) < 1e-12:
        return np.asarray(a, dtype=np.float64).copy()

    poles = np.roots(np.concatenate([[1.0], np.asarray(a, dtype=np.float64)]))
    poles_c = []
    for p in poles:
        r = min(float(np.abs(p)), r_max)
        theta = float(np.angle(p)) / alpha
        poles_c.append(r * np.exp(1j * theta))
    A_new = np.real(np.poly(poles_c))
    A_new = A_new / A_new[0]

    poles2 = np.roots(A_new)
    poles_s = []
    for p in poles2:
        mag = np.abs(p)
        if mag >= r_max:
            p = p * (r_max / mag)
        poles_s.append(p)
    A_new = np.real(np.poly(poles_s))
    A_new = A_new / A_new[0]
    return A_new[1:].astype(np.float64)


def synthetiser_trame(excitation, a):
    """Filtre tout-pôle 1/A(z) en forme directe."""
    A = np.concatenate([[1.0], np.asarray(a, dtype=np.float64)])
    if not np.all(np.isfinite(A)):
        return np.zeros_like(excitation)
    y = lfilter([1.0], A, excitation)
    if not np.all(np.isfinite(y)):
        return np.zeros_like(excitation)
    return y


def restaurer_lpc(x, fs, alpha, ordre=ORDRE_LPC):
    L, hop = parametres_trame(fs)
    w = np.sqrt(hann(L, sym=False))

    pad = L - hop
    x_pad = np.concatenate([np.zeros(pad), x, np.zeros(L)])
    n_trames = 1 + (len(x_pad) - L) // hop

    y_pad = np.zeros(len(x_pad))
    w_sum = np.zeros(len(x_pad))
    prod = w * w

    for i in range(n_trames):
        deb = i * hop
        trame = x_pad[deb:deb + L]
        trame_w = trame * w
        rms_trame = np.sqrt(np.mean(trame_w ** 2))

        if rms_trame < 1e-4:
            y_trame = trame_w
        else:
            a, _gain = lpc(trame_w, ordre)
            exc = residual(trame_w, a)
            a_c = compresser_poles_lpc(a, alpha)
            y_hat = synthetiser_trame(exc, a_c)
            rms_y = np.sqrt(np.mean(y_hat ** 2))
            if not np.isfinite(rms_y) or rms_y < 1e-12:
                y_hat = trame_w
            else:
                y_hat = y_hat * (rms_trame / rms_y)
            y_trame = y_hat * w

        y_pad[deb:deb + L] += y_trame
        w_sum[deb:deb + L] += prod

    mask = w_sum > 1e-8
    y_pad[mask] /= w_sum[mask]
    y = y_pad[pad:pad + len(x)]

    rms_in = np.sqrt(np.mean(x ** 2)) + 1e-12
    rms_out = np.sqrt(np.mean(y ** 2)) + 1e-12
    y *= rms_in / rms_out

    peak = np.max(np.abs(y))
    if peak > 0.99:
        y *= 0.99 / peak

    return y


def stats(nom, x, y):
    print(f"  [{nom}]  peak in={np.max(np.abs(x)):.3f} out={np.max(np.abs(y)):.3f} | "
          f"RMS in={np.sqrt(np.mean(x**2)):.4f} out={np.sqrt(np.mean(y**2)):.4f}")


def main():
    dossier = "outputs/etape6"
    os.makedirs(dossier, exist_ok=True)

    fichiers = [
        "inputs/hel_fr1.wav",
        "inputs/hel_fr2.wav",
        "inputs/hel_fr4.wav",
    ]

    alpha_par_fichier = {
        "hel_fr1": 2.0,
        "hel_fr2": 2.5,
        "hel_fr4": 3.0,
    }

    print("=== Restauration LPC (pôles θ/α, forme directe) + OLA ===")
    print(f"Trame {DUREE_TRAME_MS} ms, recouvrement {100*RECOUVREMENT:.0f} %, "
          f"ordre LPC={ORDRE_LPC}\n")

    for chemin in fichiers:
        if not os.path.exists(chemin):
            print(f"Manquant : {chemin}")
            continue
        nom = os.path.splitext(os.path.basename(chemin))[0]
        fs, x = charger_wav(chemin)
        print(f"{nom} (fs={fs}, durée={len(x)/fs:.2f}s)")

        for alpha in ALPHAS:
            y = restaurer_lpc(x, fs, alpha)
            out = os.path.join(dossier, f"{nom}_lpc_a{alpha:.1f}.wav")
            sauvegarder_wav(out, fs, y)
            stats(f"α={alpha}", x, y)
            print(f"    → {out}")

        alpha0 = alpha_par_fichier[nom]
        y0 = restaurer_lpc(x, fs, alpha0)
        out0 = os.path.join(dossier, f"output_lpc_{nom}.wav")
        sauvegarder_wav(out0, fs, y0)
        print(f"    sortie principale (α={alpha0}) → {out0}\n")

    fs, x = charger_wav("inputs/hel_fr2.wav")
    y = restaurer_lpc(x, fs, alpha_par_fichier["hel_fr2"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, sig, titre in zip(
        axes,
        [x, y],
        ["hel_fr2 original (hélium)", f"hel_fr2 restauré LPC α={alpha_par_fichier['hel_fr2']}"],
    ):
        _, _, _, im = ax.specgram(sig, NFFT=1024, Fs=fs, noverlap=512, cmap="magma")
        ax.set_ylim(0, 6000)
        ax.set_ylabel("Hz")
        ax.set_title(titre)
        fig.colorbar(im, ax=ax, label="dB")
    axes[-1].set_xlabel("temps (s)")
    fig.tight_layout()
    fig_path = os.path.join(dossier, "spectrogramme_avant_apres_hel_fr2.png")
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    print(f"Figure : {fig_path}")


if __name__ == "__main__":
    main()

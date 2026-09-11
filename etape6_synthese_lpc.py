"""
Étape 6 — LPC source–filtre, OLA 50 % (sans microcoupures).

  H(ω) = |G / A(e^{jω})|     enveloppe LPC (toute la bande)
  e implicite : X / H         excitation (F0, structure fine)
  H'(ω) = H(α(ω)·ω)   α≈1 sous ~700 Hz, α plein dès ~1.6 kHz
  y = IFFT( X · H'/H )

On ne coupe pas l'aigu. On évite seulement d'entasser F1 sous ~700 Hz.
"""

import os
import numpy as np
from scipy.io import wavfile
from scipy.signal.windows import hann
import matplotlib.pyplot as plt


DUREE_TRAME_MS = 20
RECOUVREMENT = 0.5
ORDRE_LPC = 32
PREEMPH = 0.97
F_PROTECT_HZ = 700.0
F_PLEIN_HZ = 1600.0
ALPHAS = (2.0, 2.5, 3.0)


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
    r = np.correlate(x, x, mode="full")
    mid = len(x) - 1
    return r[mid:mid + ordre + 1].astype(np.float64)


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


def preaccentuer(x, mu=PREEMPH):
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - mu * x[:-1]
    return y


def enveloppe_lpc(a, gain, nfft):
    A = np.concatenate([[1.0], a])
    return np.abs(gain / np.fft.rfft(A, n=nfft))


def ratio_enveloppes(H, alpha, fs, nfft):
    """
    H'(f) = H(α(f)·f). α(f) → 1 sous 700 Hz (sinon F1 trop bas = étouffé),
    α(f) = α à partir de ~1.6 kHz.
    """
    n = len(H)
    k = np.arange(n, dtype=np.float64)
    f = k * fs / nfft
    a = np.ones(n)
    rampe = (f > F_PROTECT_HZ) & (f < F_PLEIN_HZ)
    a[rampe] = 1.0 + (alpha - 1.0) * (f[rampe] - F_PROTECT_HZ) / (F_PLEIN_HZ - F_PROTECT_HZ)
    a[f >= F_PLEIN_HZ] = alpha
    Hw = np.interp(np.minimum(k * a, n - 1), k, H)
    ratio = Hw / np.maximum(H, 1e-6 * np.max(H))
    return np.clip(ratio, 0.05, 12.0)


def restaurer_lpc(x, fs, alpha, ordre=ORDRE_LPC):
    L, hop = parametres_trame(fs)
    w = np.sqrt(hann(L, sym=False))
    nfft = L

    pad = L - hop
    x_pad = np.concatenate([np.zeros(pad), x, np.zeros(L)])
    n_trames = 1 + (len(x_pad) - L) // hop

    y_pad = np.zeros(len(x_pad))
    w_sum = np.zeros(len(x_pad))
    prod = w * w

    for i in range(n_trames):
        deb = i * hop
        trame_w = x_pad[deb:deb + L] * w
        rms_trame = np.sqrt(np.mean(trame_w ** 2))

        if rms_trame < 1e-4:
            y_hat = trame_w
        else:
            a, gain = lpc(preaccentuer(trame_w), ordre)
            if gain < 1e-12 or not np.all(np.isfinite(a)):
                y_hat = trame_w
            elif abs(alpha - 1.0) < 1e-12:
                y_hat = trame_w
            else:
                X = np.fft.rfft(trame_w, n=nfft)
                H = enveloppe_lpc(a, gain, nfft)
                y_hat = np.fft.irfft(X * ratio_enveloppes(H, alpha, fs, nfft), n=nfft)[:L]
                rms_y = np.sqrt(np.mean(y_hat ** 2))
                if np.isfinite(rms_y) and rms_y > 1e-12:
                    y_hat *= rms_trame / rms_y

        y_pad[deb:deb + L] += y_hat * w
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
        "hel_fr2": 2.0,
        "hel_fr4": 2.5,
    }

    print("=== LPC enveloppe + excitation (OLA 50 %) ===")
    print(f"Trame {DUREE_TRAME_MS} ms, recouvrement {100*RECOUVREMENT:.0f} %, "
          f"ordre LPC={ORDRE_LPC}\n")

    fs, x = charger_wav("inputs/hel_fr1.wav")
    y1 = restaurer_lpc(x, fs, 1.0)
    corr = np.corrcoef(x, y1)[0, 1]
    print(f"Reconstruction α=1 : corr={corr:.4f}\n")

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

    collegue = "/Users/mshamdaoui/Downloads/hel_fr1_reconstruit_LPC 1.wav"
    fs, x = charger_wav("inputs/hel_fr1.wav")
    y = restaurer_lpc(x, fs, alpha_par_fichier["hel_fr1"])
    panneaux = [(x, "hel_fr1 original (hélium)"), (y, "notre LPC α=2, OLA 50 %")]
    if os.path.exists(collegue):
        _, yc = charger_wav(collegue)
        panneaux.append((yc, "LPC collègue (microcoupures)"))
    fig, axes = plt.subplots(len(panneaux), 1, figsize=(11, 3.2 * len(panneaux)), sharex=True)
    if len(panneaux) == 1:
        axes = [axes]
    for ax, (sig, titre) in zip(axes, panneaux):
        _, _, _, im = ax.specgram(sig, NFFT=1024, Fs=fs, noverlap=512, cmap="magma")
        ax.set_ylim(0, 6000)
        ax.set_ylabel("Hz")
        ax.set_title(titre)
        fig.colorbar(im, ax=ax, label="dB")
    axes[-1].set_xlabel("temps (s)")
    fig.tight_layout()
    fig_path = os.path.join(dossier, "spectrogramme_avant_apres_hel_fr1.png")
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    print(f"Figure : {fig_path}")

    fs, x = charger_wav("inputs/hel_fr2.wav")
    y = restaurer_lpc(x, fs, alpha_par_fichier["hel_fr2"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, sig, titre in zip(
        axes, [x, y],
        ["hel_fr2 original (hélium)", "hel_fr2 LPC α=2, OLA 50 %"],
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

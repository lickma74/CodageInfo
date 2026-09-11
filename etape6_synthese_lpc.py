"""
Étape 6 — LPC classique, mémoire du filtre (pas d'OLA).

  Trame ~20 ms (quasi-stationnaire), hop = L :
    LPC → H,  H'(f) = H(α·f),  A' par Levinson
  Synthèse continue : y = lfilter(A, A', x) avec zi. Pas de Hann, pas d'OLA.
"""

import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
from scipy.signal.windows import hamming
import matplotlib.pyplot as plt


ORDRE_LPC = 32
NFFT_ENV = 8192
PREEMPH = 0.97
DUREE_LPC_MS = 20.0
ALPHAS = (1.0, 2.0, 2.5, 3.0)
ALPHA_PAR_FICHIER = {"hel_fr1": 2.0, "hel_fr2": 2.0, "hel_fr4": 2.5}


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


def autocorrelation(x, ordre):
    x = np.asarray(x, dtype=np.float64)
    nfft = 1 << int(np.ceil(np.log2(max(2 * len(x), 2))))
    r = np.fft.irfft(np.abs(np.fft.rfft(x, n=nfft)) ** 2, n=nfft)
    return r[: ordre + 1].astype(np.float64)


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
    a, e = levinson_durbin(autocorrelation(x, ordre), ordre)
    return a, np.sqrt(e) if e > 0 else 0.0


def enveloppe_lpc(a, gain, nfft):
    return np.abs(gain / np.fft.rfft(np.concatenate([[1.0], a]), n=nfft))


def compresser_enveloppe(H, alpha):
    """H'(f) = H(α·f) sur toute la bande. Au-delà de Nyquist : dernier bin."""
    n = len(H)
    k = np.arange(n, dtype=np.float64)
    return np.interp(np.minimum(k * alpha, n - 1), k, H)


def lpc_depuis_enveloppe(H, ordre, nfft):
    r = np.fft.irfft(np.square(np.maximum(H, 1e-12)), n=nfft)
    return levinson_durbin(r[: ordre + 1].astype(np.float64), ordre)


def longueur_trame(fs):
    return max(ORDRE_LPC + 1, int(round(DUREE_LPC_MS * 1e-3 * fs)))


def normaliser_rms(x, y):
    y = y * ((np.sqrt(np.mean(x ** 2)) + 1e-12) / (np.sqrt(np.mean(y ** 2)) + 1e-12))
    peak = np.max(np.abs(y))
    if peak > 0.99:
        y *= 0.99 / peak
    return y


def coeffs_trame(frame, ordre, alpha):
    """Hamming uniquement pour l'estimation LPC (méthode d'autocorrélation)."""
    a, gain = lpc(frame * hamming(len(frame)), ordre)
    if gain < 1e-12 or not np.all(np.isfinite(a)):
        return None, None
    H = enveloppe_lpc(a, gain, NFFT_ENV)
    a2, _ = lpc_depuis_enveloppe(compresser_enveloppe(H, alpha), ordre, NFFT_ENV)
    if not np.all(np.isfinite(a2)):
        return None, None
    return a, a2


def restaurer_lpc(x, fs, alpha, ordre=ORDRE_LPC):
    x = np.asarray(x, dtype=np.float64)
    if abs(alpha - 1.0) < 1e-12:
        return x.copy()

    L = longueur_trame(fs)
    xp = lfilter([1.0, -PREEMPH], [1.0], x)
    yp = np.empty_like(xp)
    zi = np.zeros(ordre)
    a_mem = a2_mem = None

    for i in range(0, len(xp), L):
        frame = xp[i:i + L]
        a, a2 = a_mem, a2_mem
        if len(frame) >= ordre + 1:
            est = coeffs_trame(frame, ordre, alpha)
            if est[0] is not None:
                a, a2 = est
                a_mem, a2_mem = a, a2

        if a is None:
            yp[i:i + len(frame)] = frame
            continue

        yf, zi = lfilter(
            np.concatenate([[1.0], a]),
            np.concatenate([[1.0], a2]),
            frame,
            zi=zi,
        )
        if not np.all(np.isfinite(yf)):
            yf = frame
            zi = np.zeros(ordre)
        yp[i:i + len(frame)] = yf

    y = lfilter([1.0], [1.0, -PREEMPH], yp)
    return normaliser_rms(x, y)


def tracer_diagnostic_lpc(x, fs, alpha, chemin, ordre=ORDRE_LPC, n0=30208):
    L = longueur_trame(fs)
    xp = lfilter([1.0, -PREEMPH], [1.0], x)
    n0 = max(0, min(n0, len(xp) - L))

    bloc = xp[n0:n0 + L]
    fen = hamming(len(bloc))
    a, gain = lpc(bloc * fen, ordre)
    X = np.fft.rfft(bloc * fen, n=NFFT_ENV)
    H = enveloppe_lpc(a, gain, NFFT_ENV)
    Hp = compresser_enveloppe(H, alpha)
    f = np.fft.rfftfreq(NFFT_ENV, 1.0 / fs)

    mag_x = 20 * np.log10(np.maximum(np.abs(X), 1e-12))
    mag_h = 20 * np.log10(np.maximum(H, 1e-12))
    mag_hp = 20 * np.log10(np.maximum(Hp, 1e-12))
    a2, _ = lpc_depuis_enveloppe(Hp, ordre, NFFT_ENV)
    yf = lfilter(
        np.concatenate([[1.0], a]),
        np.concatenate([[1.0], a2]),
        bloc,
    )
    Y = np.fft.rfft(yf * fen, n=NFFT_ENV)
    mag_y = 20 * np.log10(np.maximum(np.abs(Y), 1e-12))
    off = np.median(mag_x[: len(mag_x) // 8]) - np.median(mag_h[: len(mag_h) // 8])

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
    fig.suptitle(
        f"Diagnostic LPC — trame à l'échantillon {n0} (α={alpha}, ordre={ordre})",
        fontsize=12,
    )
    plots = [
        (axes[0], [(mag_x, "0.6", 0.8, "|X(k)|"),
                   (mag_h + off, "C0", 2, "Enveloppe LPC H")],
         "1) Spectre + enveloppe LPC"),
        (axes[1], [(mag_h + off, "C0", 2, "H (hélium)"),
                   (mag_hp + off, "C1", 2, f"H' = H(αf), α={alpha}")],
         "2) Enveloppe avant / après"),
        (axes[2], [(mag_x, "0.6", 0.9, "|X(k)| (hélium)"),
                   (mag_y, "C2", 0.9, "|Y(k)| (filtre A/A')"),
                   (mag_hp + off, "C1", 2, "H' (cible)")],
         "3) Spectre après A/A'"),
    ]
    for ax, courbes, titre in plots:
        for mag, couleur, lw, label in courbes:
            ax.plot(f, mag, color=couleur, lw=lw, label=label)
        ax.set_xlim(0, 8000)
        ax.set_ylim(-40, 40)
        ax.set_xlabel("Fréquence (Hz)")
        ax.set_ylabel("Amplitude (dB)")
        ax.set_title(titre)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)
    print(f"Diagnostic LPC -> {chemin}")


def tracer_spectrogrammes(x, y, fs, titres, chemin, suptitle=None):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    images = []
    for ax, sig, titre in zip(axes, [x, y], titres):
        _, _, _, im = ax.specgram(sig, NFFT=1024, Fs=fs, noverlap=512, cmap="magma")
        images.append(im)
        ax.set_ylim(0, 6000)
        ax.set_ylabel("Fréquence (Hz)")
        ax.set_title(titre)
    vmin, vmax = images[0].get_clim()
    for im, ax in zip(images, axes):
        im.set_clim(vmin, vmax)
        fig.colorbar(im, ax=ax, label="dB")
    axes[-1].set_xlabel("Temps (s)")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(chemin, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure : {chemin}")


def main():
    dossier = "outputs/etape6"
    os.makedirs(dossier, exist_ok=True)
    fichiers = ["inputs/hel_fr1.wav", "inputs/hel_fr2.wav", "inputs/hel_fr4.wav"]

    print("=== LPC classique : lfilter(A, A', x) + zi (sans OLA) ===")
    print(f"ordre={ORDRE_LPC}, trame={DUREE_LPC_MS:.0f} ms, hop=L\n")

    fs, x = charger_wav("inputs/hel_fr1.wav")
    print(f"Reconstruction α=1 : corr={np.corrcoef(x, restaurer_lpc(x, fs, 1.0))[0, 1]:.4f}\n")

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
            print(f"  α={alpha}  peak={np.max(np.abs(y)):.3f}  → {out}")
        alpha0 = ALPHA_PAR_FICHIER[nom]
        out0 = os.path.join(dossier, f"output_lpc_{nom}.wav")
        sauvegarder_wav(out0, fs, restaurer_lpc(x, fs, alpha0))
        print(f"  sortie principale (α={alpha0}) → {out0}\n")

    fs, x = charger_wav("inputs/hel_fr1.wav")
    y = restaurer_lpc(x, fs, ALPHA_PAR_FICHIER["hel_fr1"])
    tracer_diagnostic_lpc(
        x, fs, ALPHA_PAR_FICHIER["hel_fr1"],
        os.path.join(dossier, "diagnostic_lpc_hel_fr1.png"),
    )
    tracer_spectrogrammes(
        x, y, fs,
        ["Original (hélium)", "Restauré LPC (α = 2)"],
        os.path.join(dossier, "spectrogramme_avant_apres_hel_fr1.png"),
        suptitle="Spectrogramme avant / après — LPC",
    )
    fs, x = charger_wav("inputs/hel_fr2.wav")
    y = restaurer_lpc(x, fs, ALPHA_PAR_FICHIER["hel_fr2"])
    tracer_spectrogrammes(
        x, y, fs,
        ["Original (hélium)", "Restauré LPC (α = 2)"],
        os.path.join(dossier, "spectrogramme_avant_apres_hel_fr2.png"),
        suptitle="Spectrogramme avant / après — LPC",
    )


if __name__ == "__main__":
    main()

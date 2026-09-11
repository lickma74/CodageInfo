"""
Étape 6 — LPC source–filtre, OLA 50 %.

  LPC → enveloppe H = |G/A|
  H'(f) = H(α(f)·f)     α≈1 sous ~700 Hz, α plein dès ~1.6 kHz
  y = IFFT( X · H'/H )  (le résidu 1/A'·e sonne très mal : on ne l'utilise pas)

On ne touche pas à F0 : seule l'enveloppe LPC est compressée.
"""

import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, sosfilt
from scipy.signal.windows import hann
import matplotlib.pyplot as plt


DUREE_TRAME_MS = 20
RECOUVREMENT = 0.5
ORDRE_LPC = 32
PREEMPH = 0.97
F_PROTECT_HZ = 700.0
F_PLEIN_HZ = 1600.0
F_MAX_ENV_HZ = 10000.0
R_MAX = 0.98
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


def compresser_enveloppe(H, alpha, fs, nfft):
    """H'(f) = H(α(f)·f), en ne lisant H que jusqu'à F_MAX_ENV_HZ.

    Au-delà, l'enveloppe LPC colle du bruit / des faux formants : les ramener
    dans la bande utile rend H' infidèle à l'enveloppe mesurée.
    """
    n = len(H)
    k = np.arange(n, dtype=np.float64)
    f = k * fs / nfft
    af = np.ones(n)
    rampe = (f > F_PROTECT_HZ) & (f < F_PLEIN_HZ)
    af[rampe] = 1.0 + (alpha - 1.0) * (f[rampe] - F_PROTECT_HZ) / (F_PLEIN_HZ - F_PROTECT_HZ)
    af[f >= F_PLEIN_HZ] = alpha
    k_max = min(n - 1, F_MAX_ENV_HZ * nfft / fs)
    return np.interp(np.minimum(k * af, k_max), k, H)


def ratio_enveloppes(H, alpha, fs, nfft):
    Hp = compresser_enveloppe(H, alpha, fs, nfft)
    ratio = Hp / np.maximum(H, 1e-6 * np.max(H))
    return np.clip(ratio, 0.05, 12.0)


def facteur_alpha(f, alpha):
    f = abs(float(f))
    if f <= F_PROTECT_HZ:
        return 1.0
    if f >= F_PLEIN_HZ:
        return float(alpha)
    return 1.0 + (alpha - 1.0) * (f - F_PROTECT_HZ) / (F_PLEIN_HZ - F_PROTECT_HZ)


def _stabiliser_poles(poles, r_max=R_MAX):
    out = np.empty(len(poles), dtype=complex)
    for i, p in enumerate(poles):
        r = abs(p)
        if r > r_max:
            p = p * (r_max / r)
        out[i] = p
    return out


def poles_lpc(a):
    return np.roots(np.concatenate([[1.0], np.asarray(a, dtype=np.float64)]))


def warper_poles(poles, alpha, fs, r_max=R_MAX):
    """θ → θ/α(f). Pôles réels inchangés. Pas de poly() d'ordre 32."""
    if abs(alpha - 1.0) < 1e-12:
        return _stabiliser_poles(poles, r_max)
    poles_c = []
    for p in poles:
        r = min(float(abs(p)), r_max)
        theta = float(np.angle(p))
        if abs(np.sin(theta)) < 1e-8:
            poles_c.append(complex(r * np.sign(np.cos(theta) or 1.0)))
            continue
        f = abs(theta) * fs / (2.0 * np.pi)
        af = facteur_alpha(f, alpha)
        theta2 = theta / af
        r2 = min(r ** af, r_max)
        poles_c.append(r2 * np.exp(1j * theta2))
    return _stabiliser_poles(poles_c, r_max)


def enveloppe_poles(poles, gain, nfft):
    omega = np.linspace(0.0, np.pi, nfft // 2 + 1)
    zinv = np.exp(-1j * omega)
    A = np.ones_like(zinv, dtype=complex)
    for p in poles:
        A *= (1.0 - p * zinv)
    return np.abs(gain / np.maximum(np.abs(A), 1e-12))


def synthese_poles(excitation, poles, r_max=R_MAX):
    """y = 1/A'(z) · e, cascade SOS (stable, pas de poly d'ordre 32)."""
    y = np.asarray(excitation, dtype=np.float64)
    poles = list(_stabiliser_poles(poles, r_max))
    sos = []
    used = np.zeros(len(poles), dtype=bool)
    for i, p in enumerate(poles):
        if used[i]:
            continue
        used[i] = True
        if abs(np.imag(p)) < 1e-8:
            r = float(np.clip(np.real(p), -r_max, r_max))
            sos.append([1.0, 0.0, 0.0, 1.0, -r, 0.0])
            continue
        for k in range(i + 1, len(poles)):
            if not used[k] and abs(poles[k] - np.conj(p)) < 1e-5:
                used[k] = True
                break
        r = min(abs(p), r_max)
        th = np.angle(p)
        sos.append([1.0, 0.0, 0.0, 1.0, -2.0 * r * np.cos(th), r * r])
    if not sos:
        return y.copy()
    y = sosfilt(np.asarray(sos, dtype=np.float64), y)
    if not np.all(np.isfinite(y)):
        return np.zeros_like(excitation)
    return y


def residual(x, a):
    """e = A(z) · x, filtre FIR causal."""
    A = np.concatenate([[1.0], a])
    return lfilter(A, [1.0], x)


def synthetiser_trame(excitation, a):
    """y = (1/A(z)) · e. Pas de gain G : le résidu porte déjà l'énergie."""
    A = np.concatenate([[1.0], np.asarray(a, dtype=np.float64)])
    if not np.all(np.isfinite(A)):
        return np.zeros_like(excitation)
    y = lfilter([1.0], A, excitation)
    if not np.all(np.isfinite(y)):
        return np.zeros_like(excitation)
    if np.max(np.abs(y)) > 1e4 * (np.max(np.abs(excitation)) + 1e-12):
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
        trame_w = x_pad[deb:deb + L] * w
        rms_trame = np.sqrt(np.mean(trame_w ** 2))

        if rms_trame < 1e-4:
            y_hat = trame_w
        else:
            a, gain = lpc(trame_w, ordre)
            if gain < 1e-12 or not np.all(np.isfinite(a)):
                y_hat = trame_w
            elif abs(alpha - 1.0) < 1e-12:
                y_hat = trame_w
            else:
                nfft = L
                X = np.fft.rfft(trame_w, n=nfft)
                H = enveloppe_lpc(a, gain, nfft)
                y_hat = np.fft.irfft(
                    X * ratio_enveloppes(H, alpha, fs, nfft), n=nfft
                )[:L]
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


def tracer_diagnostic_lpc(x, fs, alpha, chemin, ordre=ORDRE_LPC, n0=None):
    """3 cases : spectre + enveloppe, enveloppe avant/après, spectre restauré."""
    L, hop = parametres_trame(fs)
    w = np.sqrt(hann(L, sym=False))
    nfft = L
    if n0 is None:
        best_i, best_e = 0, 0.0
        for i in range(0, len(x) - L, hop):
            e = np.mean((x[i:i + L] * w) ** 2)
            if e > best_e:
                best_e, best_i = e, i
        n0 = best_i

    trame_w = x[n0:n0 + L] * w
    a, gain = lpc(trame_w, ordre)
    X = np.fft.rfft(trame_w, n=nfft)
    H = enveloppe_lpc(a, gain, nfft)
    ratio = ratio_enveloppes(H, alpha, fs, nfft)
    Xp = X * ratio
    Hp = H * ratio
    f = np.fft.rfftfreq(nfft, 1.0 / fs)

    mag_x = 20 * np.log10(np.maximum(np.abs(X), 1e-12))
    mag_h = 20 * np.log10(np.maximum(H, 1e-12))
    mag_hp = 20 * np.log10(np.maximum(Hp, 1e-12))
    mag_xp = 20 * np.log10(np.maximum(np.abs(Xp), 1e-12))
    off = np.median(mag_x[: len(mag_x) // 8]) - np.median(mag_h[: len(mag_h) // 8])

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
    fig.suptitle(
        f"Diagnostic LPC — trame à l'échantillon {n0} (α={alpha}, ordre={ordre})",
        fontsize=12,
    )

    ax = axes[0]
    ax.plot(f, mag_x, color="0.6", lw=0.8, label="|X(k)| (original)")
    ax.plot(f, mag_h + off, color="C0", lw=2, label="Enveloppe LPC")
    ax.set_xlim(0, 8000)
    ax.set_ylim(-40, 40)
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("Amplitude (dB)")
    ax.set_title("1) Spectre original + enveloppe LPC")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(f, mag_h + off, color="C0", lw=2, label="Enveloppe mesurée (hélium)")
    ax.plot(f, mag_hp + off, color="C1", lw=2, label=f"Enveloppe restaurée (α={alpha})")
    ax.set_xlim(0, 8000)
    ax.set_ylim(-40, 40)
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("Amplitude (dB)")
    ax.set_title("2) Enveloppe avant / après")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(f, mag_x, color="0.6", lw=0.9, label="|X(k)| (original)")
    ax.plot(f, mag_xp, color="C2", lw=0.9, label="|X'(k)| (restauré)")
    ax.plot(f, mag_hp + off, color="C1", lw=2, label="Enveloppe restaurée")
    ax.set_xlim(0, 8000)
    ax.set_ylim(-40, 40)
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("Amplitude (dB)")
    ax.set_title("3) Spectre final")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)
    print(f"Diagnostic LPC -> {chemin}")


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

    print("=== LPC classique : résidu + 1/A' (OLA 50 %) ===")
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

    fs, x = charger_wav("inputs/hel_fr1.wav")
    y = restaurer_lpc(x, fs, alpha_par_fichier["hel_fr1"])
    tracer_diagnostic_lpc(
        x, fs, alpha_par_fichier["hel_fr1"],
        os.path.join(dossier, "diagnostic_lpc_hel_fr1.png"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, sig, titre in zip(
        axes,
        [x, y],
        ["hel_fr1 original (hélium)", "hel_fr1 restauré LPC α=2"],
    ):
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

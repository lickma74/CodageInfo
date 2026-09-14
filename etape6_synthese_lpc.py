"""LPC pôles + zéros, mémoire zi, sans OLA. Warp θ/α (bande parole), aigus conservés."""

import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, sosfilt, zpk2sos
from scipy.signal.windows import hamming
import matplotlib.pyplot as plt

P, Q = 64, 16
N_SOS = max((P + 1) // 2, (Q + 1) // 2)
NFFT, MU, T_MS = 8192, 0.97, 20.0
R_P, R_Z = 0.985, 0.960
F_LO, F_HI, F_AIR = 150.0, 6500.0, 7000.0
ALPHAS = (1.0, 2.0, 2.5, 3.0)
ALPHA0 = {"hel_fr1": 2.0, "hel_fr2": 2.0, "hel_fr4": 2.5}
DOSSIER = "outputs/etape6"
IDENT = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])


def charger_wav(chemin):
    fs, x = wavfile.read(chemin)
    x = x[:, 0] if x.ndim > 1 else x
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    return fs, x.astype(np.float64)


def sauver_wav(chemin, fs, x):
    wavfile.write(chemin, fs, (np.clip(x, -1, 1) * 32767).astype(np.int16))


def levinson(r, ordre):
    a, e = np.zeros(ordre), r[0]
    if e <= 1e-12:
        return a, 0.0
    for i in range(1, ordre + 1):
        k = -(r[i] + np.dot(a[: i - 1], r[i - 1:0:-1])) / e
        a_prev = a[: i - 1].copy()
        a[i - 1] = k
        if i > 1:
            a[: i - 1] = a_prev + k * a_prev[::-1]
        e *= 1.0 - k * k
        if e <= 1e-18:
            break
    return a, float(max(e, 0.0))


def lpc(x, ordre):
    nfft = 1 << int(np.ceil(np.log2(max(2 * len(x), 2))))
    r = np.fft.irfft(np.abs(np.fft.rfft(x, n=nfft)) ** 2, n=nfft)[: ordre + 1]
    a, e = levinson(r, ordre)
    return a, np.sqrt(e) if e > 0 else 0.0


def racines(a):
    z = np.roots(np.concatenate([[1.0], a]))
    return z[np.isfinite(z)]


def clip_r(z, rmax):
    r = np.abs(z)
    trop = (r > rmax) & (r > 0)
    z = np.array(z, dtype=complex, copy=True)
    z[trop] *= rmax / r[trop]
    return z


def warper(z, alpha, rmax, fs):
    z = clip_r(z, rmax)
    out = []
    for p in z:
        th, r = np.angle(p), abs(p)
        if abs(np.sin(th)) < 1e-8:
            out.append(complex(np.clip(p.real, -rmax, rmax)))
            continue
        if F_LO < abs(th) * fs / (2 * np.pi) < F_HI:
            th /= alpha
        out.append(r * np.exp(1j * th))
    return clip_r(out, rmax)


def enveloppe_cepstre(x, fs):
    c = np.fft.irfft(np.log(np.maximum(np.abs(np.fft.rfft(x, n=NFFT)), 1e-12)), n=NFFT)
    n = min(max(2, int(round(0.002 * fs))), NFFT // 2 - 1)
    lift = np.zeros(NFFT)
    lift[0] = 1.0
    lift[1:n] = 1.0
    lift[n] = lift[-n] = 0.5
    lift[-n + 1:] = 1.0
    return np.exp(np.real(np.fft.rfft(c * lift, n=NFFT)))


def sos_de(zeros, poles, rz, rp):
    try:
        sos = np.atleast_2d(zpk2sos(clip_r(zeros, rz), clip_r(poles, rp), 1.0))
    except ValueError:
        return None
    sos = np.asarray(sos, float)
    if len(sos) < N_SOS:
        sos = np.vstack([sos, np.repeat(IDENT, N_SOS - len(sos), axis=0)])
    return sos[:N_SOS] if np.all(np.isfinite(sos)) else None


def modele(frame, fs, alpha):
    fen = frame * hamming(len(frame))
    a, g = lpc(fen, P)
    if g < 1e-12 or not np.all(np.isfinite(a)):
        return None
    poles = racines(a)
    if len(poles) == 0:
        return None
    b, e = levinson(np.fft.irfft(1.0 / np.maximum(enveloppe_cepstre(fen, fs), 1e-10) ** 2)[: Q + 1], Q)
    zeros = racines(b) if e > 1e-18 and np.all(np.isfinite(b)) else np.zeros(0, complex)
    sos_ab = sos_de(poles, zeros, R_P, R_Z)
    sos_ba = sos_de(warper(zeros, alpha, R_Z, fs), warper(poles, alpha, R_P, fs), R_Z, R_P)
    if sos_ab is None or sos_ba is None:
        return None
    return sos_ab, sos_ba


def restaurer_lpc(x, fs, alpha):
    if abs(alpha - 1) < 1e-12:
        return x.copy()
    L = max(P + Q + 1, int(round(T_MS * 1e-3 * fs)))
    xp = lfilter([1, -MU], [1], x)
    yp = np.empty_like(xp)
    zi_ab = np.zeros((N_SOS, 2))
    zi_ba = np.zeros((N_SOS, 2))
    sos_lp = butter(2, F_AIR, btype="low", fs=fs, output="sos")
    zi_lp = np.zeros((len(sos_lp), 2))
    mem = None
    for i in range(0, len(xp), L):
        fr = xp[i:i + L]
        if len(fr) >= P + 1:
            est = modele(fr, fs, alpha)
            if est is not None:
                mem = est
        if mem is None:
            yp[i:i + len(fr)] = fr
            continue
        y, zi_ab = sosfilt(mem[0], fr, zi=zi_ab)
        y, zi_ba = sosfilt(mem[1], y, zi=zi_ba)
        if not np.all(np.isfinite(y)):
            y, zi_ab, zi_ba = fr, np.zeros((N_SOS, 2)), np.zeros((N_SOS, 2))
        d, zi_lp = sosfilt(sos_lp, y - fr, zi=zi_lp)
        yp[i:i + len(fr)] = fr + d
    y = lfilter([1], [1, -MU], yp)
    y *= (np.sqrt(np.mean(x ** 2)) + 1e-12) / (np.sqrt(np.mean(y ** 2)) + 1e-12)
    pk = np.max(np.abs(y))
    return y * (0.99 / pk) if pk > 0.99 else y


def _db(v):
    return 20 * np.log10(np.maximum(v, 1e-12))


def tracer_diagnostic_lpc(x, fs, alpha, chemin, n0=30208):
    L = max(P + Q + 1, int(round(T_MS * 1e-3 * fs)))
    xp = lfilter([1, -MU], [1], x)
    n0 = max(0, min(n0, len(xp) - L))
    fr, fen = xp[n0:n0 + L], hamming(L)
    a, g = lpc(fr * fen, P)
    poles, zeros = racines(a), racines(levinson(
        np.fft.irfft(1.0 / np.maximum(enveloppe_cepstre(fr * fen, fs), 1e-10) ** 2)[: Q + 1], Q)[0])
    mem = modele(fr, fs, alpha)
    if mem is None:
        return
    y, _ = sosfilt(mem[0], fr, zi=np.zeros((N_SOS, 2)))
    y, _ = sosfilt(mem[1], y, zi=np.zeros((N_SOS, 2)))
    y = fr + sosfilt(butter(2, F_AIR, btype="low", fs=fs, output="sos"), y - fr)
    f = np.fft.rfftfreq(NFFT, 1 / fs)
    w = np.exp(-1j * np.linspace(0, np.pi, NFFT // 2 + 1))

    def env(z, p):
        num, den = np.ones_like(w), np.ones_like(w)
        for zi in z:
            num *= 1 - zi * w
        for pi in p:
            den *= 1 - pi * w
        return np.abs(g * num / np.maximum(np.abs(den), 1e-12))

    mx, mh, mhp, my = (
        _db(np.abs(np.fft.rfft(fr * fen, n=NFFT))),
        _db(env(zeros, poles)),
        _db(env(warper(zeros, alpha, R_Z, fs), warper(poles, alpha, R_P, fs))),
        _db(np.abs(np.fft.rfft(y * fen, n=NFFT))),
    )
    off = np.median(mx[: len(mx) // 8]) - np.median(mh[: len(mh) // 8])
    fig, ax = plt.subplots(1, 3, figsize=(14.2, 4.2))
    fig.suptitle(f"Diagnostic pôles+zéros — trame {n0} (α={alpha})")
    for a_, courbes, titre in (
        (ax[0], [(mx, "0.6", "|X|"), (mh + off, "C0", "B/A")], "1) Spectre + enveloppe"),
        (ax[1], [(mh + off, "C0", "H"), (mhp + off, "C1", "H'")], "2) Avant / après"),
        (ax[2], [(mx, "0.6", "|X|"), (my, "C2", "|Y|"), (mhp + off, "C1", "H'")], "3) Après warp"),
    ):
        for mag, c, lab in courbes:
            a_.plot(f, mag, color=c, lw=1.2, label=lab)
        a_.set(xlim=(0, 12000), ylim=(-40, 40), title=titre, xlabel="Hz")
        a_.legend(fontsize=8)
        a_.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


def tracer_spectrogrammes(x, y, fs, titres, chemin, suptitle=None):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ims = []
    for ax, sig, titre in zip(axes, (x, y), titres):
        _, _, _, im = ax.specgram(sig, NFFT=1024, Fs=fs, noverlap=512, cmap="magma")
        ims.append(im)
        ax.set(ylim=(0, 6000), ylabel="Hz", title=titre)
    for im, ax in zip(ims, axes):
        im.set_clim(*ims[0].get_clim())
        fig.colorbar(im, ax=ax, label="dB")
    axes[-1].set_xlabel("s")
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(chemin, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    os.makedirs(DOSSIER, exist_ok=True)
    print(f"p={P} q={Q}  warp {F_LO:.0f}–{F_HI:.0f} Hz  aigus > {F_AIR:.0f} Hz")
    cache = {}
    for path in ("inputs/hel_fr1.wav", "inputs/hel_fr2.wav", "inputs/hel_fr4.wav"):
        nom = os.path.splitext(os.path.basename(path))[0]
        fs, x = charger_wav(path)
        print(f"\n{nom}")
        ys = {a: restaurer_lpc(x, fs, a) for a in ALPHAS}
        if nom == "hel_fr1":
            print(f"  α=1 corr={np.corrcoef(x, ys[1.0])[0, 1]:.4f}")
        for a, y in ys.items():
            sauver_wav(os.path.join(DOSSIER, f"{nom}_lpc_a{a:.1f}.wav"), fs, y)
        sauver_wav(os.path.join(DOSSIER, f"output_lpc_{nom}.wav"), fs, ys[ALPHA0[nom]])
        cache[nom] = (fs, x, ys[ALPHA0[nom]])
    fs, x, y = cache["hel_fr1"]
    tracer_diagnostic_lpc(x, fs, 2.0, os.path.join(DOSSIER, "diagnostic_lpc_hel_fr1.png"))
    tracer_spectrogrammes(x, y, fs, ["Hélium", "Restauré α=2"],
                          os.path.join(DOSSIER, "spectrogramme_avant_apres_hel_fr1.png"))
    fs, x, y = cache["hel_fr2"]
    tracer_spectrogrammes(x, y, fs, ["Hélium", "Restauré α=2"],
                          os.path.join(DOSSIER, "spectrogramme_avant_apres_hel_fr2.png"))


if __name__ == "__main__":
    main()

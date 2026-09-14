"""
Restauration de voix hélium par LPC pôles + zéros.

Le conduit est un filtre B(z)/A(z). On warpe θ → θ/α dans la bande parole,
puis on resynthétise en temporel (IIR + mémoire zi, sans overlap-add).
Les aigus (> 7 kHz) sont repris au signal d'origine pour éviter un passe-bas.
"""

import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, sosfilt, zpk2sos
from scipy.signal.windows import hamming

# p = formants, q = anti-formants. Ordre élevé à 44,1 kHz ; réalisé en SOS.
P, Q = 64, 16
N_SOS = max((P + 1) // 2, (Q + 1) // 2)
NFFT = 8192
MU = 0.97          # préaccentuation 1 − μ z^{-1}
T_MS = 20.0        # trame quasi-stationnaire, hop = L (pas d'OLA)
R_P, R_Z = 0.985, 0.960  # bornes de stabilité (|pôles| / |zéros|)
F_LO, F_HI = 150.0, 6500.0
F_AIR = 7000.0     # au-delà : mélange avec x (aigus conservés)
ALPHAS = (1.0, 2.0, 2.5, 3.0)
ALPHA0 = {"hel_fr1": 2.0, "hel_fr2": 2.0, "hel_fr4": 2.5}
DOSSIER = "outputs/lpc"
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
    """Yule–Walker via Levinson–Durbin. Retourne a[1..p] (a0 = 1 implicite) et l'erreur."""
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
    """Autocorrélation (FFT) + Levinson. gain = sqrt(erreur de prédiction)."""
    nfft = 1 << int(np.ceil(np.log2(max(2 * len(x), 2))))
    r = np.fft.irfft(np.abs(np.fft.rfft(x, n=nfft)) ** 2, n=nfft)[: ordre + 1]
    a, e = levinson(r, ordre)
    return a, np.sqrt(e) if e > 0 else 0.0


def racines(a):
    z = np.roots(np.concatenate([[1.0], a]))
    return z[np.isfinite(z)]


def clip_r(z, rmax):
    """Ramène |z| ≤ rmax (filtre causal stable)."""
    r = np.abs(z)
    trop = (r > rmax) & (r > 0)
    z = np.array(z, dtype=complex, copy=True)
    z[trop] *= rmax / r[trop]
    return z


def warper(z, alpha, rmax, fs):
    """Compression d'enveloppe : θ → θ/α dans [F_LO, F_HI], rayon inchangé."""
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
    """Enveloppe lisse (lifter ~2 ms) pour placer les zéros sur 1/H², pas sur les harmoniques."""
    c = np.fft.irfft(np.log(np.maximum(np.abs(np.fft.rfft(x, n=NFFT)), 1e-12)), n=NFFT)
    n = min(max(2, int(round(0.002 * fs))), NFFT // 2 - 1)
    lift = np.zeros(NFFT)
    lift[0] = 1.0
    lift[1:n] = 1.0
    lift[n] = lift[-n] = 0.5
    lift[-n + 1:] = 1.0
    return np.exp(np.real(np.fft.rfft(c * lift, n=NFFT)))


def sos_de(zeros, poles, rz, rp):
    """Biquads plutôt qu'un poly d'ordre 64 (instable en float64)."""
    try:
        sos = np.atleast_2d(zpk2sos(clip_r(zeros, rz), clip_r(poles, rp), 1.0))
    except ValueError:
        return None
    sos = np.asarray(sos, float)
    if len(sos) < N_SOS:
        sos = np.vstack([sos, np.repeat(IDENT, N_SOS - len(sos), axis=0)])
    return sos[:N_SOS] if np.all(np.isfinite(sos)) else None


def modele(frame, fs, alpha):
    """Une trame → (A/B, B'/A') en SOS. None si LPC / racines invalides."""
    fen = frame * hamming(len(frame))
    a, g = lpc(fen, P)
    if g < 1e-12 or not np.all(np.isfinite(a)):
        return None
    poles = racines(a)
    if len(poles) == 0:
        return None
    b, e = levinson(
        np.fft.irfft(1.0 / np.maximum(enveloppe_cepstre(fen, fs), 1e-10) ** 2)[: Q + 1], Q
    )
    zeros = racines(b) if e > 1e-18 and np.all(np.isfinite(b)) else np.zeros(0, complex)
    sos_ab = sos_de(poles, zeros, R_P, R_Z)
    sos_ba = sos_de(warper(zeros, alpha, R_Z, fs), warper(poles, alpha, R_P, fs), R_Z, R_P)
    if sos_ab is None or sos_ba is None:
        return None
    return sos_ab, sos_ba


def restaurer_lpc(x, fs, alpha):
    """y = (B'/A')(A/B) x, zi reporté ; puis y ← x + LPF(y−x). α=1 → identité."""
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


def main():
    os.makedirs(DOSSIER, exist_ok=True)
    for path in ("inputs/hel_fr1.wav", "inputs/hel_fr2.wav", "inputs/hel_fr4.wav"):
        nom = os.path.splitext(os.path.basename(path))[0]
        fs, x = charger_wav(path)
        print(f"{nom}  fs={fs}  {len(x)/fs:.2f}s")
        ys = {a: restaurer_lpc(x, fs, a) for a in ALPHAS}
        if nom == "hel_fr1":
            print(f"  α=1 corr={np.corrcoef(x, ys[1.0])[0, 1]:.4f}")
        for a, y in ys.items():
            sauver_wav(os.path.join(DOSSIER, f"{nom}_a{a:.1f}.wav"), fs, y)
        sauver_wav(os.path.join(DOSSIER, f"{nom}.wav"), fs, ys[ALPHA0[nom]])
        print(f"  principal α={ALPHA0[nom]} -> {DOSSIER}/{nom}.wav")


if __name__ == "__main__":
    main()

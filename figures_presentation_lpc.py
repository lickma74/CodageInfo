"""Figures pour la présentation LPC (français), parallèle à la partie FFT."""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from etape6_synthese_lpc import (
    charger_wav,
    restaurer_lpc,
    tracer_diagnostic_lpc,
    tracer_spectrogrammes,
)

DOSSIER = "outputs/etape6"


def _boite(ax, cx, cy, w, h, texte, taille=10):
    pad = 0.08
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size=0.28",
        facecolor="white",
        edgecolor="black",
        linewidth=1.35,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, texte, ha="center", va="center", fontsize=taille, zorder=3, linespacing=1.25)
    return {
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "top": cy + h / 2 + pad,
        "bottom": cy - h / 2 - pad,
        "left": cx - w / 2 - pad,
        "right": cx + w / 2 + pad,
    }


def _fleche(ax, x0, y0, x1, y1):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.25,
            color="black",
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )


def _chemin(ax, points, fleche=True):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color="black", lw=1.25, zorder=1, solid_capstyle="butt")
    if fleche:
        _fleche(ax, xs[-2], ys[-2], xs[-1], ys[-1])


def schema_processus(chemin):
    fig, ax = plt.subplots(figsize=(7.2, 10.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    w, h = 3.55, 1.35
    trame = _boite(ax, 3.15, 14.7, w, h, "x[n]\ntrames 20 ms, hop = L")
    lpc = _boite(ax, 3.15, 11.85, w, h, "LPC ordre 32\nA(z), enveloppe H")
    fine = _boite(ax, 7.55, 11.85, 3.7, h, "x[n] en continu\n(pas d'OLA)")
    comp = _boite(
        ax, 3.15, 8.7, w, 1.55,
        "H'(f) = H(α f)\nA'(z) par Levinson",
        taille=10,
    )
    rec = _boite(ax, 3.15, 5.45, w, 1.45, "Filtre A/A' + zi\ny = lfilter(A, A', x)")
    out = _boite(ax, 3.15, 2.35, w, h, "y[n]\nvoix restaurée")

    y_split = (trame["bottom"] + lpc["top"]) / 2
    _chemin(ax, [(trame["cx"], trame["bottom"]), (trame["cx"], y_split)])
    _chemin(ax, [(trame["cx"], y_split), (lpc["cx"], lpc["top"])])
    _chemin(ax, [(trame["cx"], y_split), (fine["cx"], y_split), (fine["cx"], fine["top"])])
    _chemin(ax, [(lpc["cx"], lpc["bottom"]), (comp["cx"], comp["top"])])
    _chemin(ax, [(comp["cx"], comp["bottom"]), (rec["cx"], rec["top"])])
    _chemin(
        ax,
        [
            (fine["cx"], fine["bottom"]),
            (fine["cx"], rec["cy"]),
            (rec["right"], rec["cy"]),
        ],
    )
    _chemin(ax, [(rec["cx"], rec["bottom"]), (out["cx"], out["top"])])

    fig.tight_layout()
    fig.savefig(chemin, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"→ {chemin}")


def schema_pipeline(chemin):
    fig, ax = plt.subplots(figsize=(13.2, 5.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    w, h = 3.15, 1.45
    y_haut, y_bas = 4.85, 1.55
    xs = [2.05, 5.85, 9.65, 13.45]

    b0 = _boite(ax, xs[0], y_haut, w, h, "x[n]\nentrée")
    b1 = _boite(ax, xs[1], y_haut, w, h, "Trames 20 ms\nhop = L (pas d'OLA)")
    b2 = _boite(ax, xs[2], y_haut, w, h, "LPC ordre 32\nA(z), enveloppe H")
    b3 = _boite(ax, xs[3], y_haut, w, h, "H'(f) = H(α f)\nα constant")
    b4 = _boite(ax, xs[0], y_bas, w, h, "Nouveau filtre\nA'(z) Levinson")
    b5 = _boite(ax, xs[1], y_bas, w, h, "Filtre A/A'\ny = lfilter(A, A', x)")
    b6 = _boite(ax, xs[2], y_bas, w, h, "zi → trame suivante\n(pas d'OLA)")
    b7 = _boite(ax, xs[3], y_bas, w, h, "y[n]\nvoix restaurée")

    for a, b in ((b0, b1), (b1, b2), (b2, b3), (b4, b5), (b5, b6), (b6, b7)):
        _chemin(ax, [(a["right"], a["cy"]), (b["left"], b["cy"])])

    y_coude = (b3["bottom"] + b4["top"]) / 2 + 0.35
    _chemin(
        ax,
        [
            (b3["cx"], b3["bottom"]),
            (b3["cx"], y_coude),
            (b4["cx"], y_coude),
            (b4["cx"], b4["top"]),
        ],
    )

    fig.tight_layout()
    fig.savefig(chemin, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"→ {chemin}")


def diagnostic(chemin):
    fs, x = charger_wav("inputs/hel_fr1.wav")
    tracer_diagnostic_lpc(x, fs, 2.0, chemin)


def spectrogramme(chemin, wav="inputs/hel_fr1.wav", alpha=2.0):
    fs, x = charger_wav(wav)
    y = restaurer_lpc(x, fs, alpha)
    tracer_spectrogrammes(
        x, y, fs,
        ["Original (hélium)", f"Restauré LPC (α = {alpha:g})"],
        chemin,
        suptitle="Spectrogramme avant / après — LPC",
    )


def main():
    os.makedirs(DOSSIER, exist_ok=True)
    schema_processus(os.path.join(DOSSIER, "schema_processus_lpc.png"))
    schema_pipeline(os.path.join(DOSSIER, "schema_pipeline_lpc.png"))
    diagnostic(os.path.join(DOSSIER, "diagnostic_lpc_hel_fr1.png"))
    spectrogramme(os.path.join(DOSSIER, "spectrogramme_avant_apres_hel_fr1.png"))
    spectrogramme(
        os.path.join(DOSSIER, "spectrogramme_avant_apres_hel_fr2.png"),
        wav="inputs/hel_fr2.wav",
        alpha=2.0,
    )


if __name__ == "__main__":
    main()

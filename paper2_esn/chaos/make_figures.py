"""Figures of the ESN paper (Chaos, AIP) from paper2_esn/chaos/numbers/*.json|csv.

    python paper2_esn/chaos/make_figures.py     ->  paper2_esn/chaos/figures/fig_*.pdf (+ .png previews)

Palette: Okabe-Ito (color-blind safe), one fixed color and marker per model class in every figure; 8 pt minimum font,
TrueType fonts (src/hsearch/viz.setup_en, shared with paper 1); widths 3.37 in (one column) and 6.69 in (two columns).
"""
import csv
import json
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from hsearch import viz  # noqa: E402

NUM, FIG = HERE / "numbers", HERE / "figures"
W1, W2 = 3.37, 6.69
CLASS = {  # fixed color and marker per class
    "ESN": ("#0072B2", "o", "echo state network"),
    "mlp": ("#E69F00", "s", "one-layer perceptron"),
    "mlp2": ("#009E73", "^", "two-layer perceptron"),
    "res": ("#D55E00", "D", "residual network"),
}
REGIME = {  # fixed color per regime (Okabe-Ito order); LE1 (hidden only in a weak sense) drawn dashed where it appears
    "se3": ("#0072B2", "SE3"), "se11": ("#56B4E9", "SE11"), "le1": ("#009E73", "LE1 (weakly hidden)"),
    "wangchen2": ("#E69F00", "WC, a=0.006"), "wangchen2_a0.03": ("#F0E442", "WC, a=0.03"),
    "chua_I": ("#D55E00", "Chua I"), "chua_II": ("#CC79A7", "Chua II"),
}
INK2 = "#555555"


def load(name):
    return json.loads((NUM / f"{name}.json").read_text(encoding="utf-8"))


def rows(name):
    with open(NUM / f"{name}.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def panel(ax, letter):
    ax.text(-0.16, 1.04, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", dpi=400)
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", name)


def survival_curve(ax, pairs, labels, color, marker, label, offset=0.0, ls="-"):
    x = np.arange(len(pairs)) + offset
    ok = [i for i, (k, n) in enumerate(pairs) if n >= 5]
    sh = np.array([pairs[i][0] / pairs[i][1] for i in ok])
    lo, hi = np.array([wilson(*pairs[i]) for i in ok]).T
    ax.errorbar(x[ok], sh, yerr=[sh - lo, hi - sh], color=color, marker=marker, ms=4, lw=1.2, ls=ls, capsize=2,
                elinewidth=0.8, label=label)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)


def fig_theory():
    P = rows("theory_points")
    fig, a = plt.subplots(figsize=(W1, 2.6))
    for src, keep, col, mk, lab in (("X", lambda r: r["exact"] == "1", "#0072B2", "o", "hard-tanh, exact sets"),
                                    ("I", lambda r: r["index_sum"] == "1", "#E69F00", "s", "trained tanh networks")):
        S = [r for r in P if r["source"] == src and keep(r)]
        K = np.array([int(r["K"]) for r in S]) + np.random.default_rng(0).uniform(-0.25, 0.25, len(S))
        U = np.array([int(r["U"]) for r in S]) + np.random.default_rng(1).uniform(-0.25, 0.25, len(S))
        a.scatter(K, U, s=5, alpha=0.35, color=col, marker=mk, lw=0, label=f"{lab} ({len(S)})")
    kk = np.linspace(1, 41, 2)
    a.plot(kk, (kk - 1) / 2, color="black", lw=1, label=r"$U=(K-1)/2$")
    a.plot(kk, kk, color=INK2, lw=0.8, ls=":", label=r"$U=K$")
    a.set_xlabel("number of equilibria $K$")
    a.set_ylabel("unstable equilibria $U$")
    a.set_xlim(0, 41)
    a.set_ylim(-0.5, 41)
    a.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_theory")


def fig_esn_law():
    e = load("esn_law")
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.5))
    nb = e["N_sweep"]["bins"]
    cb = e["stage_C_full_protocol"]["bins"]
    labels = [x["U_bin"].replace(">=", r"$\geq$") for x in nb]
    survival_curve(a, [(x["k"], x["n"]) for x in nb], labels, "#0072B2", "o", f"size sweep, light protocol ({sum(x['n'] for x in nb)})",
                   offset=-0.08)
    survival_curve(a, [(x["k"], x["n"]) for x in cb], labels, "#CC79A7", "v",
                   f"full-protocol ensemble ({sum(x['n'] for x in cb)})", offset=0.08, ls="--")
    a.set_xlabel("unstable equilibria $U$")
    a.set_ylabel("share still hidden")
    a.set_ylim(-0.03, 1.05)
    a.legend(loc="upper right", fontsize=7.5, title="six regimes (LE1 excluded)", title_fontsize=7.5)
    panel(a, "a")
    UN = rows("esn_UN")
    for reg, (col, lab) in REGIME.items():
        S = sorted((int(r["N"]), float(r["U_mean"])) for r in UN if r["regime"] == reg and r["set"] == "HE1")
        if S:
            N, U = np.array(S).T
            weak = reg == "le1"
            b.plot(N, U, color=col, marker="o", ms=3, lw=1.1, ls="--" if weak else "-", mfc="white" if weak else col,
                   label=lab)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlabel("reservoir size $N$")
    b.set_ylabel("mean $U$ (set HE1)")
    b.legend(loc="upper left", ncol=2, fontsize=7, frameon=True, framealpha=0.92, edgecolor="none")
    b.set_ylim(0.6, 120)
    panel(b, "b")
    fig.tight_layout()
    save(fig, "fig_esn_law")


def fig_classes():
    c, e = load("classes"), load("esn_law")
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.5))
    nb = e["N_sweep"]["bins"]
    esn = [(x["k"], x["n"]) for x in nb[:5]] + [(sum(x["k"] for x in nb[5:]), sum(x["n"] for x in nb[5:]))]
    labels = ["0", "1", "2", "3-4", "5-8", r"$\geq$9"]
    survival_curve(a, esn, labels, *CLASS["ESN"][:2], CLASS["ESN"][2], offset=-0.12)
    for i, cl in enumerate(("mlp", "mlp2", "res")):
        pairs = [(x["not_excited"], x["n"]) for x in c["survival_by_U_in"][cl]]
        survival_curve(a, pairs, labels, *CLASS[cl][:2], CLASS[cl][2], offset=-0.04 + 0.08 * i)
    a.set_xlabel(r"unstable equilibria in the region $U_{\mathrm{in}}$")
    a.set_ylabel("share still hidden")
    a.set_ylim(-0.03, 1.05)
    a.legend(loc="upper right", fontsize=7.5, title="six regimes (LE1 excluded)", title_fontsize=7.5)
    panel(a, "a")
    widths = [8, 16, 32, 64, 128, 256, 512, 1024]
    for cl in ("mlp", "mlp2", "res"):
        col, mk, lab = CLASS[cl]
        b.plot(widths, [c["mean_U_in_by_width"][cl]["HE1"][str(h)] for h in widths], color=col, marker=mk, ms=3.5, lw=1.2,
               label=f"{lab}, in region")
    b.plot(widths, [c["mean_U_by_width"]["res"]["HE1"][str(h)] for h in widths], color=CLASS["res"][0], marker="D", ms=3.5,
           lw=1.0, ls="--", mfc="white", label="residual network, all")
    b.set_xscale("log", base=2)
    b.set_yscale("log")
    b.set_xticks(widths)
    b.set_xticklabels([str(h) for h in widths])
    b.set_xlabel("hidden width $H$")
    b.set_ylabel("mean unstable equilibria (set HE1)")
    b.set_ylim(0.5, 1e3)
    b.legend(loc="upper left", fontsize=7.5, title="six regimes", title_fontsize=7.5)
    panel(b, "b")
    fig.tight_layout()
    save(fig, "fig_classes")


def fig_intervention():
    v = load("data_effects")["stage_V"]["first_hidden_attractor_only"]["by_variant"]
    order = ["H1", "E5", "E10", "E20", "eqc"]
    labs = ["H1", "+5", "+10", "+20", "H1 + exact\nequilibria"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(W1, 3.6), sharex=True)
    x = np.arange(len(order))
    sh = np.array([v[k]["share_hidden"] for k in order])
    lo, hi = np.array([v[k]["share_hidden_wilson95"] for k in order]).T
    cols = ["#0072B2"] * 4 + ["#999999"]
    for i in range(len(order)):
        a.errorbar(x[i], sh[i], yerr=[[sh[i] - lo[i]], [hi[i] - sh[i]]], color=cols[i], marker="o", ms=4, capsize=2,
                   elinewidth=0.8)
    a.plot(x[:4], sh[:4], color="#0072B2", lw=1.0)
    a.set_ylabel("share still hidden")
    a.set_ylim(0, 0.65)
    a.text(0.02, 0.95, "six regimes (LE1 excluded)", transform=a.transAxes, fontsize=7.5, va="top", color=INK2)
    panel(a, "a")
    U = [v[k]["U_mean"] for k in order]
    b.bar(x, U, color=cols, width=0.55)
    for i, u in enumerate(U):
        b.text(x[i], u + 0.15, f"{u:.1f}", ha="center", fontsize=8)
    b.set_ylabel("mean $U$")
    b.set_ylim(0, 9.5)
    b.set_xticks(x)
    b.set_xticklabels(labs)
    b.set_xlabel("readout trained on H1 plus transients")
    panel(b, "b")
    fig.tight_layout()
    save(fig, "fig_intervention")


def _cartoon(ax, model):
    """Phase-plane sketch: a stable equilibrium inside an unstable cycle (basin boundary) and a hidden attractor outside;
    in the model, a spurious saddle-node pair appears and one branch of the saddle's unstable manifold reaches the
    attractor."""
    from matplotlib.patches import Circle
    t = np.linspace(0, 2 * np.pi, 400)
    ax.add_patch(Circle((0, 0), 2.6, color="#DCEBF7", lw=0, zorder=0))            # basin of the hidden attractor
    ax.add_patch(Circle((-0.2, 0), 0.75, color="white", lw=0, zorder=1))          # basin of the equilibrium
    ax.plot(-0.2 + 0.75 * np.cos(t), 0.75 * np.sin(t), color=INK2, lw=0.8, ls="--", zorder=2)
    r = 1.75 + 0.12 * np.sin(5 * t) + 0.06 * np.sin(13 * t)                       # the hidden attractor
    ax.plot(r * np.cos(t), r * np.sin(t), color="#0072B2", lw=1.6, zorder=3)
    ax.plot([-0.2], [0], "o", color="black", ms=5, zorder=5)
    ax.text(-0.2, -0.12, "stable\nequilibrium", ha="center", va="top", fontsize=6.5, zorder=6)
    ax.text(0.0, 2.05, "hidden attractor", ha="center", fontsize=7.5, color="#0072B2")
    ax.text(-0.2, 0.86, "basin boundary", ha="center", fontsize=7, color=INK2)
    if model:
        S, Nd = np.array([0.95, -0.55]), np.array([0.45, -0.95])
        ax.plot(*S, "o", mfc="white", mec="#D55E00", mew=1.4, ms=6, zorder=6)
        ax.plot(*Nd, "o", color="#D55E00", ms=5, zorder=6)
        # unstable manifold of the saddle: one branch to the node, the other spiralling onto the attractor
        ax.annotate("", xy=Nd + [0.08, 0.06], xytext=S, arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.2))
        th0 = -0.35                                                            # landing point on the attractor
        r0 = 1.75 + 0.12 * np.sin(5 * th0) + 0.06 * np.sin(13 * th0)
        Pt = np.array([r0 * np.cos(th0), r0 * np.sin(th0)])
        s = np.linspace(0, 1, 100)[:, None]
        mid = (S + Pt) / 2 + np.array([0.15, -0.25])
        path = (1 - s) ** 2 * S + 2 * s * (1 - s) * mid + s ** 2 * Pt
        ax.plot(path[:, 0], path[:, 1], color="#D55E00", lw=1.3, zorder=4)
        th = np.linspace(th0, th0 + 0.5, 30)                                  # then along the attractor
        ra = 1.75 + 0.12 * np.sin(5 * th) + 0.06 * np.sin(13 * th)
        ax.annotate("", xy=(ra[-1] * np.cos(th[-1]), ra[-1] * np.sin(th[-1])), xytext=(ra[0] * np.cos(th[0]), ra[0] * np.sin(th[0])),
                    arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.3, connectionstyle="arc3,rad=0.25"), zorder=7)
        ax.annotate("spurious saddle", xy=S + [0.03, -0.1], xytext=(1.25, -2.62), fontsize=7, color="#D55E00", ha="center",
                    arrowprops=dict(arrowstyle="-", color="#D55E00", lw=0.6, shrinkA=1, shrinkB=3), zorder=7)
        ax.text(1.12, -0.32, "$W^u$", fontsize=8, color="#D55E00", zorder=7)
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2.8, 2.8)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_schematic():
    ex = Path(os.environ.get("ESN_DATA", HERE.parents[1] / "local" / "esn-gpu")) / "node_results" / "FIG" / "example_chua_I_mlp_HE1_H32.npz"
    have = ex.exists()
    fig, axs = plt.subplots(1, 3 if have else 2, figsize=(W2, 2.5 if have else 2.6),
                            gridspec_kw={"width_ratios": [1, 1, 1.25] if have else [1, 1]})
    _cartoon(axs[0], False)
    axs[0].set_title("true system: hidden", fontsize=8)
    panel(axs[0], "a")
    _cartoon(axs[1], True)
    axs[1].set_title("model: self-excited", fontsize=8)
    panel(axs[1], "b")
    if have:
        c = axs[2]
        z = np.load(ex)
        s = "s3"    # one-layer perceptron, seed 3: a spurious saddle excites the reproduced attractor
        for k in z.files:
            if k.startswith("true_att_"):
                A = z[k]
                c.plot(A[::4, 0], A[::4, 2], ".", ms=0.6, color="#C8C8C8", zorder=1, rasterized=True)
        for k in z.files:
            if k.startswith(f"{s}_att_"):
                A = z[k]
                c.plot(A[:, 0], A[:, 2], ".", ms=0.7, color="#0072B2", zorder=4, rasterized=True)
        for k in z.files:
            if k.startswith(f"{s}_branch"):
                B = z[k]
                if k.endswith("_none"):
                    c.plot(B[:, 0], B[:, 2], "--", color="#009E73", lw=1.0, zorder=3)
                else:
                    c.plot(B[:450, 0], B[:450, 2], "-", color="#D55E00", lw=0.7, zorder=2)
        E, un, sp = z[f"{s}_E"], z[f"{s}_E_unstable"], z[f"{s}_E_spurious"]
        for e, u, q in zip(E, un, sp):
            c.plot(e[0], e[2], "o", ms=5, mfc=("white" if u > 0 else "black"), mec=("#D55E00" if q else "black"),
                   mew=1.3, zorder=5)
        te = z["true_eq"]
        c.plot(te[:, 0], te[:, 2], "x", color="black", ms=5, mew=1.2, zorder=6)
        xl, yl = c.get_xlim(), c.get_ylim()
        for xb in (-10.0, 10.0):                    # boundary of the region Q of Chua's circuit, |x| <= 10
            if xl[0] < xb < xl[1]:
                c.axvline(xb, color="0.35", lw=0.8, ls=":", zorder=0)
        c.text(-10.0, yl[1], " $Q$", ha="left", va="top", fontsize=7, color="0.35")
        c.set_xlim(xl)
        c.set_ylim(yl)
        c.set_xlabel("$x$")
        c.set_ylabel("$z$")
        c.set_title("trained perceptron, Chua I", fontsize=8)
        panel(c, "c")
    fig.tight_layout()
    save(fig, "fig_schematic")


if __name__ == "__main__":
    viz.setup()
    viz.setup_en()
    plt.rcParams.update({"axes.grid": True, "axes.axisbelow": True, "grid.alpha": 0.5, "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white"})
    fig_schematic()
    fig_theory()
    fig_esn_law()
    fig_classes()
    fig_intervention()

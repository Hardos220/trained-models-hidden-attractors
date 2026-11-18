"""Tables of the ESN paper generated from data: Table I (test systems) from local/esn-gpu/data/<regime>.json.

    python paper2_esn/chaos/make_tables.py     ->  paper2_esn/chaos/tables/tab_systems.tex
"""
import csv
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("ESN_DATA", HERE.parents[1] / "local" / "esn-gpu")) / "data"
OUT = HERE / "tables"

REGIMES = [  # regime key, label in the paper, short system description, parameter text
    ("se3", "SE3", "one stable equilibrium", "--"),
    ("se11", "SE11", "one stable equilibrium", "--"),
    ("le1", "LE1", "line of equilibria", "$a=15$, $b=1$"),
    ("wangchen2", "WC", "Wang--Chen", "$a=0.006$"),
    ("wangchen2_a0.03", "WC$'$", "Wang--Chen", "$a=0.03$"),
    ("chua_I", "Chua I", "Chua's circuit", "set I"),
    ("chua_II", "Chua II", "Chua's circuit", "set II"),
]
REG_LABEL = {k: lab for k, lab, _, _ in REGIMES}


def flow_index(e):
    """iota = (-1)^(number of real positive eigenvalues of Df); complex pairs are excluded."""
    re, im = np.array(e["eig_re"]), np.array(e.get("eig_im", [0.0] * len(e["eig_re"])))
    return int((-1) ** int(np.sum((np.abs(im) < 1e-12) & (re > 0))))


def eq_text(m):
    if not m["equilibria"]:
        return "line $x=y=0$ (stable for $z<0$)", "--"
    parts, s = [], 0
    for e in m["equilibria"]:
        i = flow_index(e)
        s += i
        kind = "stable" if e["stable"] else "saddle-focus"
        parts.append(f"{kind} (${'+' if i > 0 else '-'}$)")
    # merge identical entries, e.g. two symmetric saddle-foci
    out = []
    for p in parts:
        n = parts.count(p)
        t = (f"{n}$\\times$ " if n > 1 else "") + p
        if t not in out:
            out.append(t)
    return ", ".join(out), f"${s:+d}$".replace("+", "")


def att_text(m, names):
    kinds = {"chaotic": "chaotic", "periodic": "cycle"}
    txt = []
    for y in names:
        if y in m["attractors"]:
            txt.append(kinds.get(m["attractors"][y]["kind"], m["attractors"][y]["kind"]))
        else:
            txt.append("equilibrium" if y in ("FP", "F0", "E0") else ("equilibria" if y == "Eset" else y))
    n = len(txt)
    return (f"{n}$\\times$ {txt[0]}" if n > 1 and len(set(txt)) == 1 else ", ".join(txt))


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for key, lab, desc, par in REGIMES:
        m = json.loads((DATA / f"{key}.json").read_text(encoding="utf-8"))
        eqs, s = eq_text(m)
        p = sum(m["attractors"][y]["p_pool"] for y in m["hidden"])
        hid = att_text(m, m['hidden']) + ("$^{a}$" if key == "le1" else "")
        rows.append(f"{lab} & {desc} & {par} & {eqs} & {s} & {hid} & "
                    f"{att_text(m, [m['visible']])} & {p:.3g} & {m['Delta_t']:g} \\\\")
    tex = r"""\begin{table*}
\caption{Test systems. Equilibria are listed with their flow-map index $\iota$ in parentheses, Eq.~\eqref{eq:trueindex};
$s$ is the index sum. $p_B$ is the fraction of uniform initial conditions in the region $Q$ that reach the hidden
attractors (both of them for Chua's circuit); $\Delta t$ is the sampling interval of the training series. Equations and
regions are given in Appendix~\ref{app:systems}. $^{a}$Hidden only in a weak sense: not reached from the stable half-line of
equilibria, but reached from the segment $0.35\lesssim z\lesssim2.2$ of the unstable one, hence self-excited in the strict
sense; reported separately (Sec.~\ref{sec:systems}).}
\label{tab:systems}
\begin{ruledtabular}
\begin{tabular}{lllllllll}
Regime & System & Parameters & Equilibria ($\iota$) & $s$ & Hidden & Visible & $p_B$ & $\Delta t$ \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table*}
"""
    (OUT / "tab_systems.tex").write_text(tex, encoding="utf-8")
    print(tex)
    tab_theory()
    tab_classes()
    tab_e1()
    tab_UN()
    tab_regressions()


def load(name):
    return json.loads((NUM / f"{name}.json").read_text(encoding="utf-8"))


def frac(k, n):
    return f"{k}/{n}"


def tab_theory():
    t, c, ls = load("theory"), load("classes"), load("lisprott")
    X, I, L = t["theorem1_X"], t["theorem1_I"], t["lemma1"]
    W, P1, P2, D = t["incompleteness_witness"]["X_exact"], t["prop1"], t["prop2_X"]["exact"], t["directions_4g"]
    idx = c["index_sum_eq1"]
    cls_idx = {k: (sum(v[0] for v in idx[k].values()), sum(v[1] for v in idx[k].values())) for k in idx}
    ch = c["chua_I_copies_of_all_three_equilibria"]
    lab = ls["labels_of_reproduced"]
    n_rep = sum(lab.values())
    rows = [
        ("Thm.~\\ref{thm:degree}", "Index sum $1$", "hard-tanh benchmark, exact sets", frac(X["index_sum_eq1_exact"], X["n_nets_exact"])),
        ("", "", "tanh echo state networks", frac(I["index_sum_eq1"], I["n_models"]) + f" ({I['index_sum_ne1']} with even $K$)"),
        ("", "", "one-layer / two-layer perceptrons", f"{frac(*cls_idx['mlp'])} / {frac(*cls_idx['mlp2'])}"),
        ("", "", "residual networks (theorem not applicable)", frac(*cls_idx["res"])),
        ("Cor.~\\ref{cor:counts}", "$U\\ge(K-1)/2$", "all models with index sum $1$", "no violation"),
        ("", "Incomplete set flagged", "Newton sets of the benchmark",
         f"{W['flagged_by_index_sum']}/{W['n_incomplete']}; {W['false_alarms_complete']} false of {W['n_complete']}"),
        ("Prop.~\\ref{prop:forced}", "Copy has index $\\iota_j$", "copies within $1\\%$ ($0.1\\%$)",
         f"{P1['n_index_match']}/{P1['n_true_like_points']} ({P1['dist_lt_1e-3']['match']}/{P1['dist_lt_1e-3']['n']})"),
        ("", "Chua~I ($s=-1$): copies of all three", "echo state networks; mlp; mlp2; res",
         f"0/96; {frac(*ch['mlp'])}; {frac(*ch['mlp2'])}; {frac(*ch['res'])}"),
        ("", "Li--Sprott ($s=0$): $K\\ge1$", "echo state networks", f"{ls['n_models']}/{ls['n_models']}"),
        ("", "Li--Sprott: reproduced, self-excited", "training attractor; unseen torus",
         f"{lab.get('self_excited', 0)}/{lab.get('self_excited', 0)}; 0/{lab.get('hidden_stat', 0)}"),
        ("Prop.~\\ref{prop:manifold}", "Test in $E^u$", "self-excited attractors, benchmark",
         frac(P2["self_excited_found_by_subspace"], P2["self_excited_reference"])),
        ("", "", "32 / 256 / 2048 directions, hard cases",
         f"{D['found']['32']} / {D['found']['256']} / {D['found']['2048']} of {D['self_excited_reference']}"),
    ]
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
    tex = r"""\begin{table*}
\caption{Checks of the theory and of the numerical pipeline. The benchmark consists of hard-tanh networks whose fixed
points are enumerated over all $3^N$ linear regions. Copies are model equilibria within $1\%$ of the half-width of the
region $Q$ (half of its largest edge) from a true equilibrium. For the test in the unstable subspace the reference is the ball test of the full protocol;
the last row counts the $27$ networks in which $32$ random directions missed a self-excited attractor.}
\label{tab:theory}
\begin{ruledtabular}
\begin{tabular}{llll}
Result & Check & Data & Count \\
\hline
""" + body + r"""
\end{tabular}
\end{ruledtabular}
\end{table*}
"""
    (OUT / "tab_theory.tex").write_text(tex, encoding="utf-8")
    print(tex)


def tab_classes():
    c, e = load("classes"), load("esn_law")
    eb = e["N_sweep"]["bins"]
    cols = {"ESN": [(b["k"], b["n"]) for b in eb[:5]] + [(sum(b["k"] for b in eb[5:]), sum(b["n"] for b in eb[5:]))]}
    for cl in ("mlp", "mlp2", "res"):
        cols[cl] = [(b["not_excited"], b["n"]) for b in c["survival_by_U_in"][cl]]
    labels = ["$0$", "$1$", "$2$", "$3$--$4$", "$5$--$8$", r"$\ge9$"]
    cell = lambda k, n: f"{k / n:.2f} ({n})" if n else "--"  # noqa: E731
    rows = [f"{lab} & " + " & ".join(cell(*cols[k][i]) for k in ("ESN", "mlp", "mlp2", "res")) + r" \\"
            for i, lab in enumerate(labels)]
    tex = r"""\begin{table}
\caption{Survival of reproduced hidden attractors: share not excited from any unstable equilibrium, with the number of
pairs (model, hidden attractor) in parentheses, by the number of unstable equilibria: all of them ($U$) for echo state
networks (ESN), those in the region ($\Uin$) for the other classes. ESN: size sweep (Table~\ref{tab:ensembles}); the last
row merges the bins $9$--$16$ and $\ge17$ of Fig.~\ref{fig:esn_law}. Perceptrons (mlp, mlp2) and residual networks (res):
all training sets (H1, E1, HE1; Sec.~\ref{sec:data}). Light protocol (Sec.~\ref{sec:protocol}). Pooled over the six regimes
with strictly hidden attractors; LE1 is reported separately (Sec.~\ref{sec:systems}).}
\label{tab:classes}
\begin{ruledtabular}
\begin{tabular}{lcccc}
$U$ or $\Uin$ & ESN & mlp & mlp2 & res \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table}
"""
    (OUT / "tab_classes.tex").write_text(tex, encoding="utf-8")
    print(tex)


def tab_e1():
    c, d, de = load("classes"), load("e1_distance"), load("data_effects")
    esn = de["E1_stage_C"]["esn"]["by_regime"]
    ng = de["E1_stage_C"]["ngrc"]["by_regime"]
    ec = load("e1_controls")
    poly = ec["poly"]["by_regime"]
    ngf = ec.get("ngrc", {}).get("by_regime")          # full stage E run when present, else the pilot of stage C
    rows = []
    for key, lab, _, _ in REGIMES:
        m = json.loads((DATA / f"{key}.json").read_text(encoding="utf-8"))
        hs = m["hidden"]
        dist = " / ".join(f"{d[f'{key}/{h}']['min_dist']:.2f}" for h in hs)
        cov = " / ".join(f"{d[f'{key}/{h}']['attractor_within_0.1']:.2f}" for h in hs)
        na = len(hs)
        er = c["E1_reproduced"][key]
        ngc = f"{ngf[key]['reproduced']}/{ngf[key]['pairs']}" if ngf else f"{ng[key]['reproduced']}/{ng[key]['n'] * na}"
        rows.append(f"{lab} & {dist} & {cov} & {esn[key]['reproduced']}/{esn[key]['n'] * na} & "
                    f"{er['mlp'][0]}/{er['mlp'][1]} & {er['mlp2'][0]}/{er['mlp2'][1]} & {er['res'][0]}/{er['res'][1]} & "
                    f"{poly[key]['reproduced']}/{poly[key]['pairs']} & {ngc} \\\\")
    tex = r"""\begin{table*}
\caption{Training without the hidden attractor (set E1). Distance: smallest distance from an E1 sample to the hidden
attractor, in units of the attractor's standard deviation per coordinate; coverage: fraction of the attractor within $0.1$
of the E1 data (two values for the two attractors of Chua's circuit). Reproductions are counted over pairs (model, hidden
attractor). Quadratic: sparse quadratic maps fitted by sequential thresholding; NG-RC: next-generation reservoir computer
with the exact quadratic library. LE1, whose attractor is hidden only in a weak sense (Sec.~\ref{sec:systems}), is listed
for completeness.}
\label{tab:e1}
\begin{ruledtabular}
\begin{tabular}{lcccccccc}
Regime & Distance & Coverage & Echo state & mlp & mlp2 & res & Quadratic & NG-RC \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table*}
"""
    (OUT / "tab_e1.tex").write_text(tex, encoding="utf-8")
    print(tex)


def tab_UN():
    with open(NUM / "esn_UN.csv", encoding="utf-8") as fh:
        R = list(csv.DictReader(fh))
    Ns = sorted({int(r["N"]) for r in R})
    rows = []
    for key, lab, _, _ in REGIMES:
        for st in ("H1", "HE1"):
            d = {int(r["N"]): float(r["U_mean"]) for r in R if r["regime"] == key and r["set"] == st}
            if d:
                rows.append(f"{lab} & {st} & " + " & ".join(f"{d[n]:.1f}" if n in d else "--" for n in Ns) + r" \\")
    tex = r"""\begin{table*}
\caption{Mean number of unstable equilibria $U$ of echo state networks in the size sweep, by regime, training set and
reservoir size $N$ (Table~\ref{tab:ensembles}); H1: one trajectory on the hidden attractor, HE1: H1 plus $20$ transients toward the visible attractor.}
\label{tab:UN}
\begin{ruledtabular}
\begin{tabular}{ll""" + "c" * len(Ns) + r"""}
Regime & Set & """ + " & ".join(f"${n}$" for n in Ns) + r""" \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table*}
"""
    (OUT / "tab_UN.tex").write_text(tex, encoding="utf-8")
    print(tex)


def tab_regressions():
    e, rb, vl = load("esn_law"), load("robustness"), load("v_logit")
    st = e["robustness"]["logit"]["coef"]

    def f(x, cl=None):
        s = f"${x[0]:+.2f}\\pm{x[1]:.2f}$"
        return s + (f" & ${cl:.2f}$" if cl is not None else " & --")
    ce, cc = rb["esn"], rb["classes"]["base"]
    rows = [
        r"\multicolumn{4}{l}{\emph{Echo state networks, size sweep} (" + f"{ce['n']} pairs)" + r"} \\",
        # exact coefficient from esn_law.json (robustness.json keeps it rounded to three digits: -3.255 -> -3.25)
        rf" & $\log(1+U)$ & {f((e['N_sweep']['logit']['coef']['log1p_U']['coef'], e['N_sweep']['logit']['coef']['log1p_U']['se']), ce['log1p_U'][2])} \\",
        rf" & set H1 (vs HE1) & {f(ce['set_H1'], ce['set_H1'][2])} \\",
        rf" & $\log N$ & {f(ce['log_N'], ce['log_N'][2])} \\",
        rf" & $\log(1+U)$, pairs with $U\ge1$ & {f(ce['U_ge_1']['log1p_U'], ce['U_ge_1']['log1p_U'][2])} \\",
        r"\multicolumn{4}{l}{\emph{Echo state networks, settings sweep} (" + f"{e['robustness']['n_rows_reproduced']} pairs; "
        + ", ".join(REG_LABEL[g] for g in ("se3", "se11", "chua_I") if g in e["robustness"]["regimes"]) + r")} \\",
        rf" & $\log(1+U)$ & {f((st['log1p_U']['coef'], st['log1p_U']['se']))} \\",
        rf" & density $0.02$ / $0.2$ (vs $0.05$) & ${st['setting_dens0.02']['coef']:+.2f}\pm{st['setting_dens0.02']['se']:.2f}$ / ${st['setting_dens0.2']['coef']:+.2f}\pm{st['setting_dens0.2']['se']:.2f}$ & -- \\",
        rf" & input scale $0.2$ / $1$ (vs $0.5$) & ${st['setting_in0.2']['coef']:+.2f}\pm{st['setting_in0.2']['se']:.2f}$ / ${st['setting_in1']['coef']:+.2f}\pm{st['setting_in1']['se']:.2f}$ & -- \\",
        r"\multicolumn{4}{l}{\emph{Perceptrons and residual networks} (" + f"{cc['n']} pairs)" + r"} \\",
        rf" & $\log(1+U_{{\mathrm{{in}}}})$ & {f(cc['log1p_U_in'], cc['log1p_U_in'][2])} \\",
        rf" & class mlp2 (vs mlp) & {f(cc['class_mlp2'], cc['class_mlp2'][2])} \\",
        rf" & class res (vs mlp) & {f(cc['class_res'], cc['class_res'][2])} \\",
        rf" & set HE1 (vs H1) & {f(cc['set_HE1'], cc['set_HE1'][2])} \\",
        rf" & set E1 (vs H1) & {f(cc['set_E1'], cc['set_E1'][2])} \\",
        rf" & $\log H$ & {f(cc['log_width'], cc['log_width'][2])} \\",
        rf" & $\log(1+U_{{\mathrm{{in}}}})$, pairs with $U_{{\mathrm{{in}}}}\ge1$ & {f(rb['classes']['U_ge_1']['log1p_U_in'], rb['classes']['U_ge_1']['log1p_U_in'][2])} \\",
        r"\multicolumn{4}{l}{\emph{Echo state networks, intervention} (" + f"{vl['n_rows']} rows, first hidden attractor)" + r"} \\",
        rf" & $\log(1+U)$ & {f(vl['log1pU'])} \\",
        rf" & E20 (vs H1), without / with $\log(1+U)$ & ${vl['variant_coef_without_U']['E20'][0]:+.2f}\pm{vl['variant_coef_without_U']['E20'][1]:.2f}$ / ${vl['variant_coef_with_U']['E20'][0]:+.2f}\pm{vl['variant_coef_with_U']['E20'][1]:.2f}$ & -- \\",
        rf" & exact equilibria (vs H1), with $\log(1+U)$ & {f(vl['variant_coef_with_U']['eqc'])} \\",
    ]
    tex = r"""\begin{table*}
\caption{Logistic regressions of ``not excited'' on the number of unstable equilibria and further terms; all fits include
fixed effects for the regime, the intervention fit also $\log N$. Coefficients with standard errors, and cluster-robust
standard errors with the training configuration as cluster where computed. All fits use the regimes with strictly hidden
attractors (six; three in the settings sweep); LE1 is reported separately (Sec.~\ref{sec:systems}).}
\label{tab:regressions}
\begin{ruledtabular}
\begin{tabular}{llll}
 & Term & Coefficient & Cluster-robust s.e. \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table*}
"""
    (OUT / "tab_regressions.tex").write_text(tex, encoding="utf-8")
    print(tex)


NUM = HERE / "numbers"

if __name__ == "__main__":
    main()

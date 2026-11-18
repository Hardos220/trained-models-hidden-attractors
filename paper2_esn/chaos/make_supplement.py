"""Supplementary material of the ESN paper: tables from numbers/{hetero,fidelity}.json, prose, figures/sm_*.pdf.

    python -X utf8 paper2_esn/chaos/make_supplement.py      ->  supplementary.tex, supplementary.pdf
"""
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUM = HERE / "numbers"

H = json.loads((NUM / "hetero.json").read_text(encoding="utf-8"))
F = json.loads((NUM / "fidelity.json").read_text(encoding="utf-8"))
C = json.loads((NUM / "completeness.json").read_text(encoding="utf-8"))
REG = [("se3", "SE3"), ("se11", "SE11"), ("wangchen2", "WC"), ("wangchen2_a0.03", "WC$'$"), ("chua_I", "Chua~I"),
       ("chua_II", "Chua~II")]


def f2(x, d=2):
    return "--" if x is None else f"{x:.{d}f}"


def slope_cell(r):
    if r is None:
        return "--"
    if r.get("separation"):
        return rf"$-\infty$ (Firth ${r['slope_firth']:.1f}$)"
    return rf"${r['slope_ML']:.2f}\pm{r['se_model']:.2f}$"


def tab_slopes():
    ens = [("esn_sweep", r"ESN, size sweep, $U$"), ("esn_full_U_in", r"ESN, full protocol, $\Uin$"),
           ("classes", r"mlp, mlp2, res, $\Uin$")]
    rows = []
    for key, lab in REG:
        cells = [slope_cell(H["interaction"][e]["per_regime"].get(key)) for e, _ in ens]
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    com = [H["interaction"][e]["common_slope"] for e, _ in ens]
    rows.append(r"\hline")
    rows.append("common slope & " + " & ".join(rf"${c['coef']:.2f}\pm{c['se_model']:.2f}$" for c in com) + r" \\")
    lr = [H["interaction"][e]["LR_interaction"] for e, _ in ens]
    rows.append(r"interaction LR $\chi^2$ (5 df) & " + " & ".join(rf"${x['chi2']:.1f}$, $p={x['p']:.1g}$" for x in lr)
                + r" \\")
    return (r"""\begin{table*}
\caption{Slope of $\log(1+U)$ (or $\log(1+\Uin)$) per regime, logistic regression with the covariates of the main fits
(maximum likelihood, model standard errors). In the Wang--Chen regime at $a=0.006$ no echo state network with $U\ge1$
keeps the attractor hidden, so the slope diverges (complete separation); the Firth estimate is given instead. The last rows
give the common slope and the likelihood-ratio test of regime-specific slopes.}
\label{tab:sm_slopes}
\begin{ruledtabular}
\begin{tabular}{llll}
Regime & """ + " & ".join(l for _, l in ens) + r""" \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table*}
""")


def tab_links():
    names = [("logit_log1pU", r"logit, $\log(1+U)$"), ("logit_U", r"logit, $U$"), ("log_U", r"log link, $U$"),
             ("cloglog_exc_U", r"cloglog, $U$"), ("cloglog_exc_log1pU", r"cloglog, $\log(1+U)$")]
    ens = [("esn_sweep", "ESN sweep"), ("esn_full_U", "ESN full"), ("classes", "classes")]
    rows = []
    for nm, lab in names:
        cells = []
        for e, _ in ens:
            for block in ("all_rows", "U_ge_1", "U_in_ge_1"):
                if block in H["links"][e]:
                    m = H["links"][e][block]["models"].get(nm)
                    cells.append(f2(m["delta_AIC"], 1) if m and m.get("delta_AIC") is not None else "--")
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    return (r"""\begin{table}
\caption{Difference of AIC to the best model for five links and transformations of the count, on all reproduced pairs and on
those with at least one unstable equilibrium (second column of each ensemble). Covariates as in the main fits.}
\label{tab:sm_links}
\begin{ruledtabular}
\begin{tabular}{lcccccc}
 & \multicolumn{2}{c}{ESN sweep} & \multicolumn{2}{c}{ESN full} & \multicolumn{2}{c}{classes} \\
Model & all & $\ge1$ & all & $\ge1$ & all & $\ge1$ \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table}
""")


def tab_pB():
    rows = []
    for key, lab in REG + [("le1", "LE1")]:
        t = F["1i_pB_table"][key]
        true = t.get("true_pB_census", {})
        tp = sum(true.values()) / max(1, len(true)) if true else None
        cells = []
        for cls in ("mlp", "mlp2", "res"):
            c = t.get("classes", {}).get(cls)
            if not c:
                cells.append("--")
                continue
            m, rb = c["pB_model"], c["pB_ratio_bounded"]
            cells.append(rf"${m[0]:.3g}$ (${rb[0]:.2f}$)")
        e = t.get("classes", {}).get("esn")
        cells.append(rf"${e['pB_model'][0]:.3g}$ (${e['pB_ratio_bounded'][0]:.2f}$)" if e else "--")
        rows.append(f"{lab} & ${tp:.3g}$ & " + " & ".join(cells) + r" \\")
    return (r"""\begin{table*}
\caption{Basin fraction of the reproduced hidden attractor in $Q$: true value (per attractor) and median over reproductions
for each class; in parentheses the median ratio of model to true basin restricted to points that stay bounded in the true
system. Large ratios on all of $Q$ with ratios near one on bounded points (Chua~I) mean that the model basin absorbs the
region from which the true system escapes.}
\label{tab:sm_pB}
\begin{ruledtabular}
\begin{tabular}{llllll}
Regime & true & mlp & mlp2 & res & ESN (full protocol) \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table*}
""")


def tab_labels():
    rows = []
    for b in H["esn_rho_labels"]["labels_by_U_bin"]:
        lab = b["bin"].replace(">=", r"$\ge$")
        rows.append(rf"{lab} & {b['n']} & {b['hidden_stat']} & {b['self_excited']} & "
                    rf"{b['excited_r']} & {b['undetermined']} \\")
    return (r"""\begin{table}
\caption{Labels of the full protocol by the number $U$ of unstable equilibria, echo state networks of the full-protocol
ensemble, six regimes (one row per network).}
\label{tab:sm_labels}
\begin{ruledtabular}
\begin{tabular}{lrrrrr}
$U$ & $n$ & stat.\ hidden & self-excited & excited at $r>10^{-6}$ & undetermined \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table}
""")


def tab_sat():
    s = C["esn_saturation"]
    rows = []
    for key, v in s["by_N_rho"].items():
        N, rho = key[1:].split("_rho")
        cells = []
        for x in ("x2", "x4", "x8"):
            w = v.get(x)
            cells.append(f"{w['more_equilibria']['k']}" if w else "--")
        last = [x for x in ("x8", "x4", "x2") if x in v][0]
        u0 = v["x2"]["mean_U_base"]
        rows.append(rf"{N} & {rho} & {v['n_networks']} & ${u0:.1f}$ & " + " & ".join(cells)
                    + rf" & ${v[last]['mean_dU']:.2f}$ & {v['reproduced_pairs']} & {v['not_excited_x1']} & "
                    + rf"{v['not_excited_x1_becomes_excited_at_max']} \\")
    return (r"""\begin{table*}
\caption{Saturation of the equilibrium search of echo state networks. Networks rebuilt from the size sweep ($\rho=0.9$) and
from the full-protocol ensemble ($\rho=1.2$), six regimes. $\bar U$: mean number of unstable equilibria found by the original
search; $\times2$, $\times4$, $\times8$: networks in which two, four or eight times as many Newton starts find equilibria
that the original search does not; $\Delta\bar U$: mean increase of $U$ with the largest number of starts; reproduced: reproduced
hidden attractors; not excited: of these, attractors that no unstable equilibrium found by the original search excites
(light protocol); changed: of these, attractors excited by an equilibrium found with more starts.}
\label{tab:sm_sat}
\begin{ruledtabular}
\begin{tabular}{rrrrrrrrrrr}
$N$ & $\rho$ & networks & $\bar U$ & $\times2$ & $\times4$ & $\times8$ & $\Delta\bar U$ & reproduced & not excited & changed \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table*}
""")


def tfmt(t):
    return f"{t:.0f}" if t >= 100 else f"{t:.3g}"


def tab_cert():
    m = C["certified_3d"]["by_class_width"]
    names = {"mlp": "mlp", "mlp2": "mlp2", "res": r"res\footnotemark[1]"}
    rows = []
    for key, v in m.items():
        cls, H = key.split("_H")
        rows.append(rf"{names[cls]} & {H} & {v['n_models']} & ${v['mean_K_cert']:.2f}$ & ${v['mean_U_cert']:.2f}$ & "
                    rf"${v['mean_U_in_cert']:.2f}$ & {v['K_differs']['k']} & {v['missed_total']} & "
                    rf"{v['undecided_models']} & ${tfmt(v['time_s']['median'])}$ & ${tfmt(v['time_s']['max'])}$ \\")
    return (r"""\begin{table*}
\caption{Certified enumeration of the fixed points of perceptrons and residual networks by the interval Krawczyk method,
compared with Newton's method from the starting points of the main text. $\bar K$, $\bar U$, $\bar U_{\mathrm{in}}$: mean
numbers of certified equilibria, unstable equilibria and unstable equilibria in $Q$; differs: models whose certified set
differs from the Newton set; missed: equilibria missed by Newton's method; undecided: models with boxes that the method
could not decide; time: wall-clock time per model in one worker process.}
\label{tab:sm_cert}
\begin{ruledtabular}
\begin{tabular}{lrrrrrrrrrr}
Class & $H$ & models & $\bar K$ & $\bar U$ & $\bar U_{\mathrm{in}}$ & differs & missed & undecided & median (s) & max (s) \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\footnotetext[1]{Residual networks are unbounded; their fixed points are enumerated in $Q$ only.}
\end{table*}
""")


def main(argv):
    tabs = {"sm_slopes": tab_slopes(), "sm_links": tab_links(), "sm_pB": tab_pB(), "sm_labels": tab_labels(),
            "sm_sat": tab_sat(), "sm_cert": tab_cert()}
    for k, v in tabs.items():
        (HERE / "tables" / f"{k}.tex").write_text(v, encoding="utf-8")
    body = (HERE / "supplement_body.tex").read_text(encoding="utf-8")
    body = re.sub(r"\\input\{tables/(sm_[a-zA-Z]+)\}", lambda m: tabs[m.group(1)], body)
    (HERE / "supplementary.tex").write_text(
        "%% Generated by make_supplement.py from supplement_body.tex and numbers/*.json; edit those, not this file.\n"
        + body, encoding="utf-8")
    if "--no-pdf" in argv:
        return
    sys.path.insert(0, str(HERE.parents[1] / "paper1_method" / "chaos"))
    import build_chaos as bc  # Tectonic of paper 1
    r = subprocess.run([str(bc.TECTONIC), "--keep-logs", "supplementary.tex"], cwd=HERE, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit("tectonic failed")
    log = (HERE / "supplementary.log").read_text(encoding="utf-8", errors="replace")
    und = re.findall(r"LaTeX Warning: Reference [^\n]*undefined", log)
    ovf = [w for w in re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log) if float(w) > 1.0]
    print(f"supplementary.pdf built; undefined {len(und)}, overfull {len(ovf)}")


if __name__ == "__main__":
    main(sys.argv[1:])

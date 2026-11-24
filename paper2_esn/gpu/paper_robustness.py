"""Robustness of the survival law (checks), from the CSV outputs of paper_classes.py and paper_esn_law.py.

1. Fits on U >= 1 only (the U = 0 point is guaranteed by Corollary 2 when the equilibrium set is complete).
2. Class x log(1 + U_in) interaction (likelihood-ratio test) and a joint fit with echo state networks.
3. Cluster-robust (sandwich) standard errors, clusters = training configuration (regime, set, class, width) or
   (regime, set, N) for echo state networks.
4. Fit-quality covariates (log Delta_att, log d_H, log training MSE).
5. Label sensitivity: "excited at radius 1e-6 < r <= 1e-3" counted as hidden.
6. Single equilibria: probability that an unstable equilibrium excites the reproduced attractor, by inside/outside the
   region, copy/spurious and dimension of the unstable manifold; and the survival predicted from these probabilities if
   equilibria acted independently, against the observed survival; sensitivity of the copy/spurious shares to the copy
   threshold (0.005, 0.01, 0.02 of the half-width L of Q).
All fits and shares use the six main regimes (paper_common.MAIN_REGIMES); LE1 is reported under "le1_separate".
Output: paper2_esn/results/numbers/robustness.json.

    python -X utf8 paper2_esn/gpu/paper_robustness.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_common import COPY_TOL, POOL_NOTE, SEPARATE  # noqa: E402

NUM = Path(__file__).resolve().parents[1] / "results" / "numbers"


def rows(name, regime_key=None, keep=None):
    """Rows of a CSV in numbers/; with regime_key, keep = 'main' (six regimes) or a regime name."""
    with open(NUM / name, encoding="utf-8") as fh:
        R = list(csv.DictReader(fh))
    if regime_key is None or keep is None:
        return R
    return [r for r in R if (r[regime_key] not in SEPARATE if keep == "main" else r[regime_key] == keep)]


def fit(X, y, clusters=None):
    f = lambda w: np.sum(np.logaddexp(0, X @ w) - y * (X @ w)) + 1e-6 * w @ w  # noqa: E731
    r = optimize.minimize(f, np.zeros(X.shape[1]), method="BFGS", options={"gtol": 1e-7, "maxiter": 5000})
    w = r.x
    p = 1 / (1 + np.exp(-X @ w))
    H = (X * (p * (1 - p))[:, None]).T @ X
    Hi = np.linalg.pinv(H)
    se = np.sqrt(np.diag(Hi))
    se_cl = None
    if clusters is not None:
        g = X * (y - p)[:, None]
        G = defaultdict(lambda: np.zeros(X.shape[1]))
        for gi, c in zip(g, clusters):
            G[c] += gi
        M = sum(np.outer(v, v) for v in G.values())
        nc = len(G)
        se_cl = np.sqrt(np.diag(Hi @ M @ Hi) * nc / max(nc - 1, 1))
    return w, se, r.fun, se_cl


def coef(w, se, j, se_cl=None):
    # four digits: three-digit values like 0.145 would round again to 0.14 in the tables
    out = [round(float(w[j]), 4), round(float(se[j]), 4)]
    if se_cl is not None:
        out.append(round(float(se_cl[j]), 4))
    return out


def classes_part(keep="main"):
    R = [r for r in rows("classes_rows.csv", "reg", keep) if r["rep"] == "True"]
    regs = sorted({r["reg"] for r in R})
    y = np.array([r["ne"] == "True" for r in R], float)
    Un = np.array([float(r["Un"]) for r in R])
    lu = np.log1p(Un)[:, None]
    cls_d = np.array([[float(r["cls"] == "mlp2"), float(r["cls"] == "res")] for r in R])
    set_d = np.array([[float(r["set"] == "HE1"), float(r["set"] == "E1")] for r in R])
    lw = np.array([[np.log(float(r["H"]))] for r in R])
    Rg = np.array([[float(r["reg"] == g) for g in regs] for r in R])
    cl = [f"{r['reg']}|{r['set']}|{r['cls']}|{r['H']}" for r in R]
    out = {}
    X = np.hstack([lu, cls_d, set_d, lw, Rg])
    w, se, f_full, sc = fit(X, y, cl)
    out["base"] = {"n": len(R), "log1p_U_in": coef(w, se, 0, sc), "class_mlp2": coef(w, se, 1, sc), "class_res": coef(w, se, 2, sc),
                   "set_HE1": coef(w, se, 3, sc), "set_E1": coef(w, se, 4, sc), "log_width": coef(w, se, 5, sc),
                   "note": "[coef, se, cluster-robust se]"}
    # interaction class x log(1+U)
    Xi = np.hstack([X, lu * cls_d])
    wi, sei, f_int, _ = fit(Xi, y)
    lr = 2 * (f_full - f_int)
    out["interaction"] = {"slope_mlp": coef(wi, sei, 0), "extra_slope_mlp2": coef(wi, sei, X.shape[1]),
                          "extra_slope_res": coef(wi, sei, X.shape[1] + 1),
                          "LR": round(lr, 2), "df": 2, "p": float(stats.chi2.sf(lr, 2))}
    # U >= 1 only
    k = Un >= 1
    w1, se1, _, sc1 = fit(X[k], y[k], [c for c, kk in zip(cl, k) if kk])
    _, _, f1n, _ = fit(X[k][:, 1:], y[k])
    _, _, f1f, _ = fit(X[k], y[k])
    out["U_ge_1"] = {"n": int(k.sum()), "log1p_U_in": coef(w1, se1, 0, sc1), "LR_U": round(2 * (f1n - f1f), 1),
                     "p": float(stats.chi2.sf(2 * (f1n - f1f), 1)),
                     "share_by_U": {str(u): [int(y[Un == u].sum()), int((Un == u).sum())] for u in range(1, 9)}}
    # fit-quality covariates
    q = np.array([[np.log(max(float(r["dA"]), 1e-6)), np.log(max(float(r["dH"]), 1e-6)), np.log(max(float(r["mse"]), 1e-12))]
                  for r in R])
    wq, seq, _, scq = fit(np.hstack([X, q]), y, cl)
    j0 = X.shape[1]
    out["fit_quality"] = {"log1p_U_in": coef(wq, seq, 0, scq), "log_delta_att": coef(wq, seq, j0, scq),
                          "log_d_H": coef(wq, seq, j0 + 1, scq), "log_train_mse": coef(wq, seq, j0 + 2, scq)}
    # bounded classes restricted to complete equilibrium sets (index sum 1); residual networks kept as they are
    kk = np.array([r["cls"] == "res" or r["ok"] == "True" for r in R])
    wk, sek, _, sck = fit(X[kk], y[kk], [c for c, k_ in zip(cl, kk) if k_])
    out["index_sum_1_only"] = {"n": int(kk.sum()), "dropped": int((~kk).sum()), "log1p_U_in": coef(wk, sek, 0, sck),
                               "class_mlp2": coef(wk, sek, 1, sck), "class_res": coef(wk, sek, 2, sck)}
    # label sensitivity
    y2 = np.array([r["label"] in ("not_excited", "excited_r") for r in R], float)
    w2, se2, _, sc2 = fit(X, y2, cl)
    out["excited_r_as_hidden"] = {"log1p_U_in": coef(w2, se2, 0, sc2), "n_excited_r": int(sum(r["label"] == "excited_r" for r in R)),
                                  "share_by_U_bin": {b: [int(y2[(Un >= lo_) & (Un <= hi_)].sum()), int(((Un >= lo_) & (Un <= hi_)).sum())]
                                                     for b, lo_, hi_ in (("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4),
                                                                         ("5-8", 5, 8), ("9+", 9, 1e9))}}
    return out, R


def esn_part(keep="main"):
    R = [r for r in rows("esn_rows.csv", "regime", keep)
         if r["sample"] == "n_sweep" and r["reproduced"] == "1" and r["index_sum"] == "1"]
    regs = sorted({r["regime"] for r in R})
    y = np.array([float(r["not_excited"]) for r in R])
    U = np.array([float(r["U"]) for r in R])
    X = np.hstack([np.log1p(U)[:, None], np.array([[float(r["set"] == "H1"), np.log(float(r["N"]))] for r in R]),
                   np.array([[float(r["regime"] == g) for g in regs] for r in R])])
    cl = [f"{r['regime']}|{r['set']}|{r['N']}" for r in R]
    w, se, _, sc = fit(X, y, cl)
    out = {"n": len(R), "log1p_U": coef(w, se, 0, sc), "set_H1": coef(w, se, 1, sc), "log_N": coef(w, se, 2, sc)}
    lnN = np.log(np.array([float(r["N"]) for r in R]))
    ci = (w[2] - 1.96 * sc[2], w[2] + 1.96 * sc[2])
    out["log_N_effect_over_range_ci95"] = [round(float(ci[0] * (lnN.max() - lnN.min())), 2),
                                           round(float(ci[1] * (lnN.max() - lnN.min())), 2)]
    k = U >= 1
    w1, se1, _, sc1 = fit(X[k], y[k], [c for c, kk in zip(cl, k) if kk])
    out["U_ge_1"] = {"n": int(k.sum()), "log1p_U": coef(w1, se1, 0, sc1)}
    return out, R


def joint(Rc, Re):
    """Echo state networks (all equilibria counted) and the three other classes in one fit with class x log(1+U)."""
    allr = [("esn", r["regime"], float(r["U"]), float(r["not_excited"])) for r in Re] + \
           [(r["cls"], r["reg"], float(r["Un"]), float(r["ne"] == "True")) for r in Rc]
    regs = sorted({a[1] for a in allr})
    C = ["mlp", "mlp2", "res"]
    lu = np.log1p([a[2] for a in allr])[:, None]
    cd = np.array([[float(a[0] == c) for c in C] for a in allr])
    Rg = np.array([[float(a[1] == g) for g in regs] for a in allr])
    y = np.array([a[3] for a in allr])
    X = np.hstack([lu, cd, Rg])
    w, se, f0, _ = fit(X, y)
    Xi = np.hstack([X, lu * cd])
    wi, sei, f1, _ = fit(Xi, y)
    lr = 2 * (f0 - f1)
    return {"n": len(allr), "log1p_U": coef(w, se, 0), "class_vs_esn": {c: coef(w, se, 1 + i) for i, c in enumerate(C)},
            "interaction_LR": round(lr, 2), "df": 3, "p": float(stats.chi2.sf(lr, 3)),
            "slope_esn": coef(wi, sei, 0), "extra_slopes": {c: coef(wi, sei, X.shape[1] + i) for i, c in enumerate(C)}}


def single_equilibria(Rc, keep="main"):
    E = rows("classes_eq_rows.csv", "reg", keep)
    out = {"note": "[excited, n] over unstable equilibria of models with a reproduced hidden attractor (one entry per "
                   "(equilibrium, hidden attractor)); copy = dist_true_eq < 0.01 (1% of the half-width L of Q)"}
    for key, f in (("inside", lambda e: e["inside"] == "1"), ("outside", lambda e: e["inside"] == "0"),
                   ("inside_copy", lambda e: e["inside"] == "1" and e["copy"] == "1"),
                   ("inside_spurious", lambda e: e["inside"] == "1" and e["copy"] == "0"),
                   ("inside_dim1", lambda e: e["inside"] == "1" and e["dim_u"] == "1"),
                   ("inside_dim2", lambda e: e["inside"] == "1" and e["dim_u"] == "2"),
                   ("inside_dim3", lambda e: e["inside"] == "1" and e["dim_u"] == "3"),
                   # one unstable direction: index -1 = real eigenvalue above 1; index +1 = real eigenvalue below -1
                   ("inside_dim1_index_minus1_eig_gt1", lambda e: e["inside"] == "1" and e["dim_u"] == "1" and e["index"] == "-1"),
                   ("inside_dim1_index_plus1_eig_lt_minus1", lambda e: e["inside"] == "1" and e["dim_u"] == "1" and e["index"] == "1"),
                   # two unstable directions with index -1: one real eigenvalue above 1 and one below -1
                   ("inside_dim2_index_minus1", lambda e: e["inside"] == "1" and e["dim_u"] == "2" and e["index"] == "-1")):
        s = [e for e in E if f(e)]
        out[key] = [sum(int(e["excited"]) for e in s), len(s)]
    # sensitivity of the copy/spurious split to the copy threshold
    if E and "dist" in E[0]:
        sens = {}
        for t in (COPY_TOL / 2, COPY_TOL, 2 * COPY_TOL):
            ins = [e for e in E if e["inside"] == "1"]
            cp = [e for e in ins if float(e["dist"]) < t]
            sp = [e for e in ins if float(e["dist"]) >= t]
            sens[f"{t:g}"] = {"inside_copy": [sum(int(e["excited"]) for e in cp), len(cp)],
                              "inside_spurious": [sum(int(e["excited"]) for e in sp), len(sp)],
                              "share_copy": round(sum(int(e["excited"]) for e in cp) / len(cp), 4) if cp else None,
                              "share_spurious": round(sum(int(e["excited"]) for e in sp) / len(sp), 4) if sp else None}
        out["copy_threshold_sensitivity"] = sens
    # independence: per (class, regime) excitation probability of an inside unstable equilibrium
    pr = defaultdict(lambda: [0, 0])
    for e in E:
        if e["inside"] == "1":
            pr[(e["cls"], e["reg"])][0] += int(e["excited"])
            pr[(e["cls"], e["reg"])][1] += 1
    pred, obs = defaultdict(list), defaultdict(list)
    for r in Rc:
        u = int(r["Un"])
        n_ = pr.get((r["cls"], r["reg"]), [0, 0])
        if not n_[1]:
            continue
        p = n_[0] / n_[1]
        b = "0" if u == 0 else "1" if u == 1 else "2" if u == 2 else "3-4" if u <= 4 else "5-8" if u <= 8 else "9+"
        pred[b].append((1 - p) ** u)
        obs[b].append(float(r["ne"] == "True"))
    out["independence_by_U_in"] = {b: {"n": len(obs[b]), "observed": round(float(np.mean(obs[b])), 3),
                                       "predicted": round(float(np.mean(pred[b])), 3)} for b in ("0", "1", "2", "3-4", "5-8", "9+") if obs[b]}
    return out


def main():
    out = {"pooling": POOL_NOTE}
    out["classes"], Rc = classes_part("main")
    out["esn"], Re = esn_part("main")
    out["joint_with_esn"] = joint(Rc, Re)
    out["single_equilibria"] = single_equilibria(Rc, "main")
    # LE1 separately: single equilibria (copies of the line vs spurious) and the independence check within LE1
    Rc1 = [r for r in rows("classes_rows.csv", "reg", "le1") if r["rep"] == "True"]
    out["le1_separate"] = {"single_equilibria": single_equilibria(Rc1, "le1"),
                           "note": "LE1: copies = model equilibria within 0.01 L of the line x = y = 0"}
    # seven regimes, as before 2026-11-11 (comparison only)
    cA, RcA = classes_part(None)
    eA, ReA = esn_part(None)
    out["sensitivity_with_le1"] = {"classes_log1p_U_in": cA["base"]["log1p_U_in"], "classes_n": cA["base"]["n"],
                                   "classes_U_ge_1_log1p_U_in": cA["U_ge_1"]["log1p_U_in"],
                                   "esn_log1p_U": eA["log1p_U"], "esn_n": eA["n"], "esn_U_ge_1_log1p_U": eA["U_ge_1"]["log1p_U"],
                                   "independence_by_U_in": single_equilibria(RcA, None)["independence_by_U_in"],
                                   "note": "[coef, se, cluster-robust se]"}
    (NUM / "robustness.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

"""Checks of the survival law beyond in-sample fits: out-of-sample prediction, a mechanism with parameters that are not
fitted to survival, and a collapse of the regimes onto one curve.

Read-only on the row tables of paper2_esn/results/numbers (classes_rows.csv, classes_eq_rows.csv, esn_C_rows.csv,
esn_rows.csv); six main regimes (paper_common.MAIN_REGIMES), LE1 reported separately where noted. Writes
numbers/law_checks.json.

Ensembles (as in paper_hetero.py):
  classes   perceptrons and residual networks, stage M, reproduced pairs; y = not excited; x = U_in (count used in the
            paper), U_all = all unstable equilibria (rows of classes_eq_rows.csv), k_all / k_in = how many of them excite.
  esn_full  echo state networks, full protocol (stage C), one row per network; y = hidden_stat; x = U or U_in;
            k = exc_true + exc_spur (unstable equilibria that excite the attractor).
  esn_sweep echo state networks, size sweep, light protocol; y = not excited; x = U.

1 out_of_sample  the law  logit P(y) = a + b log(1 + x)  (and with class/set/size covariates) fitted on a training part
                 and evaluated on a held-out part, against baselines without x: the training share (B0) and the
                 covariates alone (Bcov). Splits: leave one regime out, leave one class out, training set H1 <-> HE1,
                 width <= 128 -> >= 256, x <= 2 -> x >= 3, and across families (classes <-> echo state networks).
                 Metrics: mean log loss, Brier score, out-of-sample skill 1 - LL/LL(B0), calibration intercept and
                 slope, observed share (Wilson 95%) against mean prediction per x bin; cluster bootstrap (networks) of the
                 log-loss difference between the law and the better baseline.
2 mechanism      each network has its own probability p that one of its unstable equilibria excites the attractor,
                 p ~ Beta(alpha, beta) over networks, so the number k of exciting equilibria among n is beta-binomial.
                 (alpha, beta) are fitted ONLY to pairs that lost hiddenness (k >= 1; zero-truncated likelihood, which
                 carries no information on the share with k = 0), and the survival P(k = 0 | n) = B(alpha, beta + n) /
                 B(alpha, beta) is then predicted and compared with the observed share. For large n it decays as
                 n^(-alpha): a power law, i.e. the logit-log(1 + U) form of the law, with exponent alpha. Same with a
                 common p (zero-truncated binomial), which predicts (1 - p)^n. Pooled, per regime, per class; cluster
                 bootstrap of the predictions.
3 collapse       z = pbar_g * U, the expected number of exciting equilibria, with pbar_g the share of single unstable
                 equilibria that excite in regime g (measured on equilibria, not fitted to survival). Logistic fits in
                 log(1 + U) and log(1 + z) with and without regime effects (likelihood ratios, AIC) and leave-one-regime-
                 out prediction with pbar of the held-out regime measured on its own equilibria.
4 law_form       the two forms of the law compared out of sample on the same splits as item 1: logit in log(1 + x)
                 against S(x) = s0 B(alpha, beta + x) / B(alpha, beta) (s0 = 1 when x counts all unstable equilibria),
                 both fitted to survival on the training part, and the mechanism prediction of item 2 fitted on the
                 excited pairs of the training part only; full-data fits of (alpha, beta) pooled, per regime, per class.

Run (light, a few minutes; through the resource queue):
    rmq run --cpu 8 --ram 3 --time 20m -- python -X utf8 paper2_esn/gpu/paper_law_checks.py
Options: --boot B (default 1000), --out PATH.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import optimize, special  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_common import MAIN_REGIMES, SEPARATE  # noqa: E402
from paper_hetero import (CLS_BINS, ESN_BINS, NUM, ClusterBoot, bin_of, blabel, clean, codes, glm,  # noqa: E402
                          wilson)

REGS = tuple(sorted(MAIN_REGIMES))
CLS = ("mlp", "mlp2", "res")
EPS = 1e-12


# ============================================================================================== data
def load(include_le1=False):
    keep = set(MAIN_REGIMES) | (set(SEPARATE) if include_le1 else set())
    k = pd.read_csv(NUM / "classes_rows.csv")
    k = k[(k.rep == True) & k.reg.isin(keep)].reset_index(drop=True)  # noqa: E712
    k["mid"] = k.cls + "|" + k.set + "|" + k.reg + "|" + k.H.astype(str) + "|" + k.seed.astype(str)
    k["pid"] = k.mid + "|" + k.att
    q = pd.read_csv(NUM / "classes_eq_rows.csv")
    q = q[q.reg.isin(keep)]
    q["pid"] = q.cls + "|" + q.set + "|" + q.reg + "|" + q.H.astype(str) + "|" + q.seed.astype(str) + "|" + q.att
    g = q.groupby("pid")
    agg = pd.DataFrame({"n_all": g.size(), "k_all": g.excited.sum(),
                        "n_in": g.inside.sum(), "k_in": g.apply(lambda d: int((d.excited * d.inside).sum()))})
    k = k.merge(agg, left_on="pid", right_index=True, how="left")
    for c in ("n_all", "k_all", "n_in", "k_in"):
        k[c] = k[c].fillna(0).astype(int)
    K = pd.DataFrame({"y": k["ne"].astype(bool).astype(float), "x": k.Un.astype(int), "U_all": k.n_all,
                      "k": k.k_all, "n_in": k.n_in, "k_in": k.k_in, "reg": k.reg, "cl": k.mid, "cls": k.cls,
                      "set": k.set, "size": k.H.astype(float), "U_table": k.U.astype(int)})
    c = pd.read_csv(NUM / "esn_C_rows.csv")
    c = c[c.regime.isin(keep)].reset_index(drop=True)
    C = pd.DataFrame({"y": c.hidden.astype(float), "x": c.U.astype(int), "U_in": c.U_in.astype(int),
                      "U_all": c.U.astype(int), "k": (c.exc_true + c.exc_spur).astype(int), "reg": c.regime,
                      "cl": c.model, "cls": "esn", "set": c.set, "size": c.N.astype(float), "rho": c.rho})
    s = pd.read_csv(NUM / "esn_rows.csv")
    s = s[(s["sample"] == "n_sweep") & (s.reproduced == 1) & (s.index_sum == 1) & s.regime.isin(keep)].reset_index(drop=True)
    S = pd.DataFrame({"y": s.not_excited.astype(float), "x": s.U.astype(int), "reg": s.regime, "cl": s.model,
                      "cls": "esn", "set": s.set, "size": s.N.astype(float)})
    return {"classes": K, "esn_full": C, "esn_sweep": S}


def covariates(d):
    """Class, training-set and size covariates (dummies against the first level present in d)."""
    cols, names = [], []
    for col in ("cls", "set"):
        lv = sorted(set(d[col]))
        for v in lv[1:]:
            cols.append((d[col] == v).to_numpy(float))
            names.append(f"{col}_{v}")
    cols.append(np.log(d["size"].to_numpy(float)))
    names.append("log_size")
    return np.column_stack(cols), names


# ============================================================================================== check 1
def design(d, spec, xcol="x", ref=None):
    """Design matrix of a model spec. ref = training frame (levels of the dummies are those of the training part)."""
    ref = d if ref is None else ref
    n = len(d)
    parts, names = [np.ones(n)], ["const"]
    if spec in ("L", "Lcov"):
        parts.append(np.log1p(d[xcol].to_numpy(float)))
        names.append("log1p_x")
    if spec in ("Bcov", "Lcov"):
        for col in ("cls", "set"):
            lv = sorted(set(ref[col]))
            for v in lv[1:]:
                parts.append((d[col] == v).to_numpy(float))
                names.append(f"{col}_{v}")
        if ref["size"].nunique() > 1:
            parts.append(np.log(d["size"].to_numpy(float)))
            names.append("log_size")
    return np.column_stack(parts), names


def fit_spec(tr, spec, xcol="x"):
    X, names = design(tr, spec, xcol)
    keep = [j for j in range(X.shape[1]) if j == 0 or np.ptp(X[:, j]) > 0]
    f = glm(X[:, keep], tr.y.to_numpy(), ridge=1e-6)
    return {"spec": spec, "xcol": xcol, "keep": keep, "beta": f.beta, "names": [names[j] for j in keep], "ref": tr}


def predict(m, te):
    X, _ = design(te, m["spec"], m["xcol"], ref=m["ref"])
    return special.expit(X[:, m["keep"]] @ m["beta"])


def logloss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def calib(y, p, x, bins):
    b = bin_of(x, bins)
    rows = []
    for j, (lo, hi) in enumerate(bins):
        m = b == j
        n = int(m.sum())
        if n == 0:
            continue
        o = int(y[m].sum())
        ci = wilson(o, n)
        pr = float(p[m].mean())
        rows.append({"bin": blabel(lo, hi), "n": n, "observed": o / n, "wilson95": ci, "predicted": pr,
                     "inside_ci": bool(ci[0] - 1e-9 <= pr <= ci[1] + 1e-9),
                     "z": (o - p[m].sum()) / math.sqrt(max(float((p[m] * (1 - p[m])).sum()), 1e-9))})
    return rows


def calib_line(y, p):
    lp = special.logit(np.clip(p, 1e-6, 1 - 1e-6))
    if np.ptp(lp) < 1e-9 or y.min() == y.max():
        return None
    f = glm(np.c_[np.ones(len(y)), lp], y)
    return {"intercept": float(f.beta[0]), "slope": float(f.beta[1]),
            "slope_se": float(math.sqrt(max(f.cov[1, 1], 0)))}


def evaluate(tr, te, xcol="x", specs=("B0", "Bcov", "L", "Lcov"), bins=CLS_BINS, boot=0, rng=None):
    y = te.y.to_numpy()
    out = {"n_train": int(len(tr)), "n_test": int(len(te)), "n_test_networks": int(te.cl.nunique()),
           "test_share": float(y.mean()), "models": {}}
    P = {}
    for sp in specs:
        m = fit_spec(tr, sp, xcol)
        p = predict(m, te)
        P[sp] = p
        ll = logloss(y, p)
        out["models"][sp] = {"coef": dict(zip(m["names"], map(float, m["beta"]))), "logloss": float(ll.mean()),
                             "brier": float(np.mean((y - p) ** 2)), "calibration_line": calib_line(y, p)}
    llB0 = out["models"]["B0"]["logloss"]
    for sp in specs:
        out["models"][sp]["skill_vs_B0"] = 1 - out["models"][sp]["logloss"] / llB0
    # in-sample reference: the law refitted on the test part itself (upper bound for a 2-parameter law)
    mi = fit_spec(te, "L", xcol)
    out["in_sample_L_on_test"] = {"logloss": float(logloss(y, predict(mi, te)).mean()),
                                  "coef": dict(zip(mi["names"], map(float, mi["beta"])))}
    out["in_sample_L_on_test"]["skill_vs_B0"] = 1 - out["in_sample_L_on_test"]["logloss"] / llB0
    out["calibration_L"] = calib(y, P["L"], te[xcol].to_numpy(), bins)
    if "Lcov" in P:
        out["calibration_Lcov"] = calib(y, P["Lcov"], te[xcol].to_numpy(), bins)
    base = min((b for b in ("B0", "Bcov") if b in P), key=lambda b: out["models"][b]["logloss"])
    best = min((b for b in ("L", "Lcov") if b in P), key=lambda b: out["models"][b]["logloss"])
    d = logloss(y, P[base]) - logloss(y, P[best])
    out["delta_logloss"] = {"law": best, "baseline": base, "mean": float(d.mean())}
    if boot and len(te) > 1:
        cb = ClusterBoot(te.reg.to_numpy(), te.cl.to_numpy())
        bs = np.array([d[cb.draw(rng)].mean() for _ in range(boot)])
        out["delta_logloss"]["boot95"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
        out["delta_logloss"]["share_boot_le_0"] = float(np.mean(bs <= 0))
    out["_pred"] = P
    return out


def pooled_folds(D, folds, key, xcol="x", bins=CLS_BINS, specs=("B0", "Bcov", "L", "Lcov"), boot=0, rng=None):
    """Each fold predicts its held-out rows once; pooled metrics over all held-out rows plus per-fold results."""
    res, ys, xs, regs, cls_, Ps = {}, [], [], [], [], {s: [] for s in specs}
    for name, (mtr, mte) in folds.items():
        r = evaluate(D[mtr], D[mte], xcol, specs, bins, boot, rng)
        for s in specs:
            Ps[s].append(r["_pred"][s])
        del r["_pred"]
        res[name] = r
        ys.append(D[mte].y.to_numpy())
        xs.append(D[mte][xcol].to_numpy())
        regs.append(D[mte].reg.to_numpy())
        cls_.append(D[mte].cl.to_numpy())
    y = np.concatenate(ys)
    x = np.concatenate(xs)
    P = {s: np.concatenate(Ps[s]) for s in specs}
    pooled = {"n": int(len(y)), "models": {}}
    llB0 = float(logloss(y, P["B0"]).mean())
    for s in specs:
        ll = float(logloss(y, P[s]).mean())
        pooled["models"][s] = {"logloss": ll, "brier": float(np.mean((y - P[s]) ** 2)), "skill_vs_B0": 1 - ll / llB0,
                               "calibration_line": calib_line(y, P[s])}
    pooled["calibration_L"] = calib(y, P["L"], x, bins)
    if "Lcov" in P:
        pooled["calibration_Lcov"] = calib(y, P["Lcov"], x, bins)
    base = min((b for b in ("B0", "Bcov") if b in P), key=lambda b: pooled["models"][b]["logloss"])
    best = min((b for b in ("L", "Lcov") if b in P), key=lambda b: pooled["models"][b]["logloss"])
    d = logloss(y, P[base]) - logloss(y, P[best])
    pooled["delta_logloss"] = {"law": best, "baseline": base, "mean": float(d.mean())}
    if boot:
        cb = ClusterBoot(np.concatenate(regs), np.concatenate(cls_))
        bs = np.array([d[cb.draw(rng)].mean() for _ in range(boot)])
        pooled["delta_logloss"]["boot95"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
    return {"split": key, "pooled": pooled, "folds": res}


def check1(E, boot, rng):
    K, C, S = E["classes"], E["esn_full"], E["esn_sweep"]
    out = {}
    out["classes_leave_one_regime_out"] = pooled_folds(
        K, {g: (K.reg != g, K.reg == g) for g in REGS}, "regime", boot=boot, rng=rng)
    out["classes_leave_one_class_out"] = pooled_folds(
        K, {c: (K.cls != c, K.cls == c) for c in CLS}, "class", boot=boot, rng=rng)
    out["classes_training_set"] = pooled_folds(
        K, {"H1->HE1": (K.set == "H1", K.set == "HE1"), "HE1->H1": (K.set == "HE1", K.set == "H1")}, "set",
        boot=boot, rng=rng)
    out["classes_width_extrapolation"] = pooled_folds(
        K, {"H<=128 -> H>=256": (K["size"] <= 128, K["size"] >= 256)}, "width", boot=boot, rng=rng)
    out["classes_U_extrapolation"] = pooled_folds(
        K, {"U_in<=2 -> U_in>=3": (K.x <= 2, K.x >= 3)}, "U", boot=boot, rng=rng)
    out["esn_full_leave_one_regime_out"] = pooled_folds(
        C, {g: (C.reg != g, C.reg == g) for g in REGS}, "regime", bins=ESN_BINS, boot=boot, rng=rng)
    out["esn_full_U_in_leave_one_regime_out"] = pooled_folds(
        C, {g: (C.reg != g, C.reg == g) for g in REGS}, "regime", xcol="U_in", bins=ESN_BINS, boot=boot, rng=rng)
    out["esn_sweep_leave_one_regime_out"] = pooled_folds(
        S, {g: (S.reg != g, S.reg == g) for g in REGS}, "regime", bins=ESN_BINS, boot=boot, rng=rng)
    # across families: one family trains, the other is predicted (no class/size terms: they do not transfer)
    sp = ("B0", "L")
    Cin = C.assign(x=C.U_in)
    both = pd.concat([K.assign(fam="classes"), Cin.assign(fam="esn_full")], ignore_index=True)
    out["classes_to_esn_full_U_in"] = pooled_folds(
        both, {"classes -> esn_full": (both.fam == "classes", both.fam == "esn_full")}, "family", bins=ESN_BINS,
        specs=sp, boot=boot, rng=rng)
    out["esn_full_U_in_to_classes"] = pooled_folds(
        both, {"esn_full -> classes": (both.fam == "esn_full", both.fam == "classes")}, "family", bins=CLS_BINS,
        specs=sp, boot=boot, rng=rng)
    both = pd.concat([K.assign(fam="classes"), S.assign(fam="esn_sweep")], ignore_index=True)
    out["classes_to_esn_sweep_U"] = pooled_folds(
        both, {"classes -> esn_sweep": (both.fam == "classes", both.fam == "esn_sweep")}, "family", bins=ESN_BINS,
        specs=sp, boot=boot, rng=rng)
    return out


# ============================================================================================== check 2
def log_bb0(n, a, b):
    """log P(k = 0 | n) of the beta-binomial."""
    return special.betaln(a, b + n) - special.betaln(a, b)


def log_bb(k, n, a, b):
    return (special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1)
            + special.betaln(k + a, n - k + b) - special.betaln(a, b))


def fit_ztbb(k, n):
    """Zero-truncated beta-binomial MLE on rows with k >= 1 (rows with n = 1 carry no information)."""
    m = (k >= 1) & (n >= 2)
    k, n = k[m].astype(float), n[m].astype(float)
    if len(k) < 3:
        return None

    def nll(t):
        a, b = math.exp(t[0]), math.exp(t[1])
        l0 = log_bb0(n, a, b)
        return -float(np.sum(log_bb(k, n, a, b) - np.log1p(-np.exp(np.minimum(l0, -1e-15)))))

    best = None
    for t0 in ((0.0, 0.0), (1.0, -1.0), (-1.0, 1.0), (2.0, 0.5)):
        r = optimize.minimize(nll, t0, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    a, b = math.exp(best.x[0]), math.exp(best.x[1])
    return {"alpha": a, "beta": b, "mean_p": a / (a + b), "icc": 1 / (a + b + 1), "n_rows_used": int(len(k)),
            "nll": float(best.fun), "converged": bool(best.success),
            "at_bound": bool(abs(best.x[0]) > 9 or abs(best.x[1]) > 9)}


def fit_ztbin(k, n):
    m = (k >= 1) & (n >= 2)
    k, n = k[m].astype(float), n[m].astype(float)
    if len(k) < 3:
        return None

    def nll(t):
        p = special.expit(t)
        return -float(np.sum(k * np.log(p) + (n - k) * np.log1p(-p) - np.log1p(-(1 - p) ** n)))

    r = optimize.minimize_scalar(nll, bounds=(-12, 12), method="bounded")
    return {"p": float(special.expit(r.x)), "nll": float(r.fun), "n_rows_used": int(len(k))}


def mech_part(D, bins, xbin, boot, rng):
    k, n, y = D.k.to_numpy(), D.U_all.to_numpy(), D.y_mech.to_numpy()
    bb, bi = fit_ztbb(k, n), fit_ztbin(k, n)
    if bb is None:
        return {"n_pairs": int(len(D)), "note": "too few excited pairs with n >= 2"}
    S_bb = np.exp(log_bb0(n, bb["alpha"], bb["beta"]))
    S_bi = (1 - bi["p"]) ** n
    # bootstrap: refit on resampled networks, recompute the predicted share per bin
    xb = bin_of(D[xbin].to_numpy(), bins)
    reps = []
    if boot:
        cb = ClusterBoot(D.reg.to_numpy(), D.cl.to_numpy())
        for _ in range(boot):
            ix = cb.draw(rng)
            f = fit_ztbb(k[ix], n[ix])
            if f is None:
                continue
            Sb = np.exp(log_bb0(n, f["alpha"], f["beta"]))
            reps.append([Sb[xb == j].mean() if np.any(xb == j) else np.nan for j in range(len(bins))]
                        + [Sb.mean(), f["alpha"]])
    reps = np.array(reps) if reps else None
    rows = []
    for j, (lo, hi) in enumerate(bins):
        m = xb == j
        nn = int(m.sum())
        if nn == 0:
            continue
        o = int(y[m].sum())
        r = {"bin": blabel(lo, hi), "n": nn, "observed": o / nn, "wilson95": wilson(o, nn),
             "beta_binomial": float(S_bb[m].mean()), "binomial": float(S_bi[m].mean())}
        if reps is not None:
            r["beta_binomial_boot95"] = [float(np.nanquantile(reps[:, j], 0.025)), float(np.nanquantile(reps[:, j], 0.975))]
        ci = r["wilson95"]
        lo_, hi_ = r.get("beta_binomial_boot95", [r["beta_binomial"]] * 2)
        r["beta_binomial_overlaps"] = bool(lo_ <= ci[1] and hi_ >= ci[0])
        r["binomial_inside_ci"] = bool(ci[0] <= r["binomial"] <= ci[1])
        rows.append(r)
    o_all = float(y.mean())
    res = {"n_pairs": int(len(D)), "n_networks": int(D.cl.nunique()), "n_excited_pairs_n_ge_2": bb["n_rows_used"],
           "beta_binomial": bb, "binomial": bi, "bins_by": xbin, "bins": rows,
           "total": {"observed": o_all, "beta_binomial": float(S_bb.mean()), "binomial": float(S_bi.mean())},
           "z_total": {"beta_binomial": float((y.sum() - S_bb.sum()) / math.sqrt(max((S_bb * (1 - S_bb)).sum(), 1e-9))),
                       "binomial": float((y.sum() - S_bi.sum()) / math.sqrt(max((S_bi * (1 - S_bi)).sum(), 1e-9)))},
           "tail_exponent_alpha": bb["alpha"],
           "asymptote": "P(k = 0 | n) ~ Gamma(alpha + beta) / Gamma(beta) * n^(-alpha) for n >> beta"}
    if reps is not None:
        res["total"]["beta_binomial_boot95"] = [float(np.nanquantile(reps[:, -2], 0.025)),
                                                float(np.nanquantile(reps[:, -2], 0.975))]
        res["beta_binomial"]["alpha_boot95"] = [float(np.nanquantile(reps[:, -1], 0.025)),
                                                float(np.nanquantile(reps[:, -1], 0.975))]
    # logistic slope in log(1 + n) on the same pairs, for comparison with alpha
    if D.y_mech.nunique() > 1:
        f = glm(np.c_[np.ones(len(D)), np.log1p(n)], y)
        res["logit_slope_log1p_n"] = {"coef": float(f.beta[1]), "se": float(math.sqrt(max(f.cov[1, 1], 0)))}
    return res


def check2(E, boot, rng):
    K = E["classes"].copy()
    K["y_mech"] = (K.k == 0).astype(float)
    C = E["esn_full"].copy()
    C["y_mech"] = (C.k == 0).astype(float)
    out = {"classes_consistency": {
        "pairs": int(len(K)),
        "label_not_excited_vs_no_equilibrium_excites_mismatch": int(np.sum(K.y.to_numpy() != K.y_mech.to_numpy())),
        "U_table_vs_equilibrium_rows_mismatch": int(np.sum(K.U_table.to_numpy() != K.U_all.to_numpy()))},
        "esn_full_consistency": {
        "networks": int(len(C)), "k_gt_U": int(np.sum(C.k > C.U_all)),
        "hidden_stat_with_k_ge_1": int(np.sum((C.y == 1) & (C.k >= 1))),
        "note": "hidden_stat counts attractors excited only at a finite radius (excited_r) as hidden; the mechanism "
                "uses k = 0"}}
    out["classes_pooled"] = mech_part(K, CLS_BINS, "U_all", boot, rng)
    out["classes_pooled_bins_by_U_in"] = mech_part(K, CLS_BINS, "x", 0, rng)
    out["classes_per_regime"] = {g: mech_part(K[K.reg == g], CLS_BINS, "U_all", boot // 4, rng) for g in REGS}
    out["classes_per_class"] = {c: mech_part(K[K.cls == c], CLS_BINS, "U_all", boot // 4, rng) for c in CLS}
    out["esn_full_pooled"] = mech_part(C, ESN_BINS, "U_all", boot, rng)
    out["esn_full_per_regime"] = {g: mech_part(C[C.reg == g], ESN_BINS, "U_all", boot // 4, rng) for g in REGS}
    return out


# ============================================================================================== check 4
def fit_bb_law(x, y, s0free):
    """Survival law S(x) = s0 * B(alpha, beta + x) / B(alpha, beta) fitted by maximum likelihood to the survival outcomes
    (s0 = 1 when x counts all unstable equilibria: Corollary 1 makes S(0) = 1)."""
    def S(t, xx):
        s0 = special.expit(t[2]) if s0free else 1.0
        return np.clip(s0 * np.exp(log_bb0(xx, math.exp(t[0]), math.exp(t[1]))), EPS, 1 - EPS)

    def nll(t):
        p = S(t, x)
        return -float(np.sum(y * np.log(p) + (1 - y) * np.log1p(-p)))

    best = min((optimize.minimize(nll, t0, method="Nelder-Mead",
                                  options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-9})
                for t0 in ((0.0, 0.0, 3.0), (1.0, -1.0, 3.0), (-1.0, 1.0, 2.0))), key=lambda r: r.fun)
    t = best.x
    return {"alpha": math.exp(t[0]), "beta": math.exp(t[1]), "s0": float(special.expit(t[2])) if s0free else 1.0,
            "loglik": -float(best.fun), "S": (lambda xx, t=t: S(t, xx))}


def law_cv(D, xcol, folds, bins, s0free, boot, rng, mech=False):
    """Out-of-sample comparison of the two forms of the law, both with two parameters (three with s0):
    logit  logit S = a + b log(1 + x);   BB  S = s0 B(alpha, beta + x) / B(alpha, beta);
    mech   BB with (alpha, beta) from the zero-truncated fit to the excited pairs of the training part only (never sees a
           survival outcome; x must count all unstable equilibria, k = how many excite)."""
    ys, xs, regs, cls_, P, folds_out = [], [], [], [], {"B0": [], "logit": [], "BB": []}, {}
    if mech:
        P["mech"] = []
    for name, (mtr, mte) in folds.items():
        tr, te = D[mtr], D[mte]
        x, y = tr[xcol].to_numpy(float), tr.y.to_numpy()
        xt, yt = te[xcol].to_numpy(float), te.y.to_numpy()
        f = glm(np.c_[np.ones(len(tr)), np.log1p(x)], y)
        bb = fit_bb_law(x, y, s0free)
        p = {"B0": np.full(len(te), y.mean()), "logit": special.expit(f.beta[0] + f.beta[1] * np.log1p(xt)),
             "BB": bb["S"](xt)}
        fo = {"n_test": int(len(te)), "logit_coef": [float(f.beta[0]), float(f.beta[1])],
              "bb": {k: v for k, v in bb.items() if k != "S"}}
        if mech:
            z = fit_ztbb(tr.k.to_numpy(), tr.U_all.to_numpy())
            p["mech"] = np.clip(np.exp(log_bb0(xt, z["alpha"], z["beta"])), EPS, 1 - EPS)
            fo["mech_alpha_beta"] = [z["alpha"], z["beta"]]
        llB = float(logloss(yt, p["B0"]).mean())
        fo["skill_vs_B0"] = {k: 1 - float(logloss(yt, v).mean()) / llB for k, v in p.items()}
        folds_out[name] = fo
        for k in P:
            P[k].append(p[k])
        ys.append(yt)
        xs.append(xt)
        regs.append(te.reg.to_numpy())
        cls_.append(te.cl.to_numpy())
    y, x = np.concatenate(ys), np.concatenate(xs)
    P = {k: np.concatenate(v) for k, v in P.items()}
    LL = {k: logloss(y, v) for k, v in P.items()}
    out = {"x": xcol, "n": int(len(y)), "s0_free": s0free,
           "skill_vs_B0": {k: 1 - float(v.mean()) / float(LL["B0"].mean()) for k, v in LL.items()},
           "logloss": {k: float(v.mean()) for k, v in LL.items()},
           "calibration_line": {k: calib_line(y, v) for k, v in P.items() if k != "B0"},
           "calibration": {k: calib(y, v, x, bins) for k, v in P.items() if k != "B0"},
           "bins_inside_wilson95": {k: [int(sum(r["inside_ci"] for r in calib(y, v, x, bins))), len(calib(y, v, x, bins))]
                                    for k, v in P.items() if k != "B0"},
           "folds": folds_out}
    if boot:
        cb = ClusterBoot(np.concatenate(regs), np.concatenate(cls_))
        idx = [cb.draw(rng) for _ in range(boot)]
        for a_, b_ in (("logit", "BB"), ("B0", "BB"), ("B0", "logit")) + ((("B0", "mech"), ("logit", "mech")) if mech else ()):
            d = LL[a_] - LL[b_]
            bs = np.array([d[i].mean() for i in idx])
            out[f"delta_logloss_{a_}_minus_{b_}"] = {"mean": float(d.mean()),
                                                     "boot95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}
    return out


def check4(E, boot, rng):
    K, C, S = E["classes"], E["esn_full"], E["esn_sweep"]
    out = {}
    for xcol, s0free, mech in (("U_all", False, True), ("x", True, False)):
        key = "U_all" if xcol == "U_all" else "U_in"
        out[f"classes_{key}_leave_one_regime_out"] = law_cv(
            K, xcol, {g: (K.reg != g, K.reg == g) for g in REGS}, CLS_BINS, s0free, boot, rng, mech)
        out[f"classes_{key}_leave_one_class_out"] = law_cv(
            K, xcol, {c: (K.cls != c, K.cls == c) for c in CLS}, CLS_BINS, s0free, boot, rng, mech)
        out[f"classes_{key}_width_extrapolation"] = law_cv(
            K, xcol, {"H<=128 -> H>=256": (K["size"] <= 128, K["size"] >= 256)}, CLS_BINS, s0free, boot, rng, mech)
        out[f"classes_{key}_U_extrapolation"] = law_cv(
            K, xcol, {"x<=2 -> x>=3": (K[xcol] <= 2, K[xcol] >= 3)}, CLS_BINS, s0free, boot, rng, mech)
        out[f"classes_{key}_training_set"] = law_cv(
            K, xcol, {"H1->HE1": (K.set == "H1", K.set == "HE1"), "HE1->H1": (K.set == "HE1", K.set == "H1")},
            CLS_BINS, s0free, boot, rng, mech)
    out["esn_full_U_leave_one_regime_out"] = law_cv(
        C, "U_all", {g: (C.reg != g, C.reg == g) for g in REGS}, ESN_BINS, False, boot, rng, True)
    out["esn_full_U_in_leave_one_regime_out"] = law_cv(
        C, "U_in", {g: (C.reg != g, C.reg == g) for g in REGS}, ESN_BINS, True, boot, rng, False)
    out["esn_sweep_U_leave_one_regime_out"] = law_cv(
        S, "x", {g: (S.reg != g, S.reg == g) for g in REGS}, ESN_BINS, False, boot, rng, False)
    both = pd.concat([K.assign(fam="classes"), C.assign(fam="esn_full")], ignore_index=True)
    out["classes_to_esn_full_U_all"] = law_cv(
        both, "U_all", {"classes -> esn_full": (both.fam == "classes", both.fam == "esn_full")}, ESN_BINS, False,
        boot, rng, True)
    out["esn_full_to_classes_U_all"] = law_cv(
        both, "U_all", {"esn_full -> classes": (both.fam == "esn_full", both.fam == "classes")}, CLS_BINS, False,
        boot, rng, True)
    # full-data fits of the BB law (exponent alpha of the power-law tail), pooled, per class, per regime
    fits = {}
    for name, D, xcol, s0 in (("classes_U_all", K, "U_all", False), ("classes_U_in", K, "x", True),
                              ("esn_full_U_all", C, "U_all", False), ("esn_sweep_U", S, "x", False)):
        f = fit_bb_law(D[xcol].to_numpy(float), D.y.to_numpy(), s0)
        g = glm(np.c_[np.ones(len(D)), np.log1p(D[xcol].to_numpy(float))], D.y.to_numpy())
        fits[name] = {"pooled": {**{k: v for k, v in f.items() if k != "S"}, "logit_loglik": float(g.ll)}}
        for col, lv in (("reg", REGS), ("cls", CLS if name.startswith("classes") else ())):
            for v in lv:
                d = D[D[col] == v]
                fv = fit_bb_law(d[xcol].to_numpy(float), d.y.to_numpy(), s0)
                fits[name][v] = {k: w for k, w in fv.items() if k != "S"}
        if boot:
            cb = ClusterBoot(D.reg.to_numpy(), D.cl.to_numpy())
            al = []
            for _ in range(max(boot // 5, 50)):
                ix = cb.draw(rng)
                al.append(fit_bb_law(D[xcol].to_numpy(float)[ix], D.y.to_numpy()[ix], s0)["alpha"])
            fits[name]["pooled"]["alpha_boot95"] = [float(np.quantile(al, 0.025)), float(np.quantile(al, 0.975))]
    out["bb_law_fits"] = fits
    return out


# ============================================================================================== check 3
def pbar(D, count="U_all", exc="k"):
    """Share of single unstable equilibria that excite, per regime."""
    g = D.groupby("reg")
    return (g[exc].sum() / g[count].sum()).to_dict()


def fit_ll(X, y):
    f = glm(X, y)
    return f, float(f.ll), X.shape[1]


def collapse_part(D, xcol, count, exc, bins):
    pb = pbar(D, count, exc)
    D = D.assign(pb=D.reg.map(pb))
    D = D.assign(z=D.pb * D[xcol])
    y = D.y.to_numpy()
    regs = sorted(set(D.reg))
    R = (D.reg.to_numpy()[:, None] == np.array(regs)[None, :]).astype(float)
    lu, lz = np.log1p(D[xcol].to_numpy(float)), np.log1p(D.z.to_numpy(float))
    models = {
        "univ_log1p_U": np.c_[np.ones(len(D)), lu],
        "regimeFE_log1p_U": np.c_[R, lu],
        "regimeFE_regime_slopes_U": np.c_[R, R * lu[:, None]],
        "univ_log1p_z": np.c_[np.ones(len(D)), lz],
        "regimeFE_log1p_z": np.c_[R, lz],
        "regimeFE_regime_slopes_z": np.c_[R, R * lz[:, None]],
    }
    fits = {}
    for nm, X in models.items():
        f, ll, p = fit_ll(X, y)
        fits[nm] = {"loglik": ll, "n_par": p, "aic": 2 * p - 2 * ll}
        if nm.startswith("univ"):
            fits[nm]["coef"] = [float(f.beta[0]), float(f.beta[1])]
            fits[nm]["se"] = [float(math.sqrt(f.cov[0, 0])), float(math.sqrt(f.cov[1, 1]))]
        if nm == "regimeFE_log1p_z" or nm == "regimeFE_log1p_U":
            fits[nm]["regime_intercepts"] = dict(zip(regs, map(float, f.beta[:len(regs)])))
            fits[nm]["slope"] = float(f.beta[-1])
    from scipy import stats
    dfR = len(regs) - 1

    def lrt(a, b):
        c2 = 2 * (fits[b]["loglik"] - fits[a]["loglik"])
        df = fits[b]["n_par"] - fits[a]["n_par"]
        return {"chi2": c2, "df": df, "p": float(stats.chi2.sf(max(c2, 0), df))}

    tests = {"regime_effects_beyond_U": lrt("univ_log1p_U", "regimeFE_log1p_U"),
             "regime_effects_beyond_z": lrt("univ_log1p_z", "regimeFE_log1p_z"),
             "regime_slopes_beyond_U": lrt("regimeFE_log1p_U", "regimeFE_regime_slopes_U"),
             "regime_slopes_beyond_z": lrt("regimeFE_log1p_z", "regimeFE_regime_slopes_z"),
             "share_of_regime_deviance_removed_by_z":
                 1 - (fits["regimeFE_regime_slopes_z"]["loglik"] - fits["univ_log1p_z"]["loglik"])
                 / max(fits["regimeFE_regime_slopes_U"]["loglik"] - fits["univ_log1p_U"]["loglik"], 1e-9)}
    del dfR
    # leave one regime out: the universal law in U against the universal law in z (pbar of the held-out regime is
    # measured on its own equilibria)
    llU, llZ, llB, rows = [], [], [], {}
    for g in regs:
        tr, te = D[D.reg != g], D[D.reg == g]
        yt = te.y.to_numpy()
        fU = glm(np.c_[np.ones(len(tr)), np.log1p(tr[xcol].to_numpy(float))], tr.y.to_numpy())
        fZ = glm(np.c_[np.ones(len(tr)), np.log1p(tr.z.to_numpy(float))], tr.y.to_numpy())
        pU = special.expit(fU.beta[0] + fU.beta[1] * np.log1p(te[xcol].to_numpy(float)))
        pZ = special.expit(fZ.beta[0] + fZ.beta[1] * np.log1p(te.z.to_numpy(float)))
        pB = np.full(len(te), tr.y.mean())
        a, b_, c = logloss(yt, pU), logloss(yt, pZ), logloss(yt, pB)
        llU.append(a)
        llZ.append(b_)
        llB.append(c)
        rows[g] = {"n": int(len(te)), "pbar": float(pb[g]), "logloss_U": float(a.mean()), "logloss_z": float(b_.mean()),
                   "logloss_B0": float(c.mean()), "calibration_z": calib(yt, pZ, te[xcol].to_numpy(), bins),
                   "calibration_U": calib(yt, pU, te[xcol].to_numpy(), bins)}
    a, b_, c = (np.concatenate(v) for v in (llU, llZ, llB))
    loro = {"logloss_U": float(a.mean()), "logloss_z": float(b_.mean()), "logloss_B0": float(c.mean()),
            "skill_U": float(1 - a.mean() / c.mean()), "skill_z": float(1 - b_.mean() / c.mean()), "folds": rows}
    # binned curves for the figure: survival against U and against z per regime
    curves = {}
    zb = [(0, 0), (1e-9, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 8), (8, INFZ)]
    for g in regs:
        d = D[D.reg == g]
        cu = calib(d.y.to_numpy(), d.y.to_numpy(), d[xcol].to_numpy(), bins)
        zz = d.z.to_numpy()
        cz = []
        for lo, hi in zb:
            m = (zz >= lo) & (zz <= hi) if lo == hi else (zz > lo) & (zz <= hi)
            if m.sum():
                o = int(d.y.to_numpy()[m].sum())
                cz.append({"z_lo": lo, "z_hi": hi, "z_mean": float(zz[m].mean()), "n": int(m.sum()),
                           "observed": o / int(m.sum()), "wilson95": wilson(o, int(m.sum()))})
        curves[g] = {"by_U": [{k: r[k] for k in ("bin", "n", "observed", "wilson95")} for r in cu], "by_z": cz}
    return {"x": xcol, "pbar_count": count, "pbar_by_regime": pb, "fits": fits, "tests": tests,
            "leave_one_regime_out": loro, "curves": curves}


INFZ = 1e9


def check3(E):
    K, C = E["classes"], E["esn_full"]
    return {"classes_U_in_pbar_inside": collapse_part(K, "x", "n_in", "k_in", CLS_BINS),
            "classes_U_all_pbar_all": collapse_part(K, "U_all", "U_all", "k", CLS_BINS),
            "esn_full_U_pbar_all": collapse_part(C, "U_all", "U_all", "k", ESN_BINS)}


# ============================================================================================== figure
def figure(R):
    """fig_law: (a) held-out regimes, (b) mechanism without survival outcomes, (c) across model families."""
    from paper_hetero import W2, fig_setup, save
    plt = fig_setup()
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.6))
    c4 = R["check4_law_form"]
    styles = {"BB": ("#0072B2", "-", r"$B(\alpha,\beta+U)/B(\alpha,\beta)$"),
              "mech": ("#009E73", "--", "mechanism, excited pairs only"),
              "logit": ("#E69F00", "-.", r"logit in $\log(1+U)$"),
              "binomial": ("#999999", ":", r"independent, $(1-p)^U$")}

    def obs(ax, rows, x):
        sh = np.array([r["observed"] for r in rows])
        lo, hi = np.array([r["wilson95"] for r in rows], float).T
        ax.errorbar(x, sh, yerr=[np.clip(sh - lo, 0, None), np.clip(hi - sh, 0, None)], color="black", marker="o",
                    ms=3.2, ls="none", capsize=1.5, elinewidth=0.7, label="observed", zorder=5)

    def ticks(ax, rows, xl):
        ax.set_xticks(np.arange(len(rows)))
        ax.set_xticklabels([r["bin"].replace(">=", r"$\geq$") for r in rows], fontsize=7)
        ax.set_xlabel(xl)

    def panel(ax, block, keys, title, logy):
        cal = block["calibration"]
        rows = cal["BB"]
        x = np.arange(len(rows))
        for k in keys:
            col, ls, lab = styles[k]
            ax.plot(x, [r["predicted"] for r in cal[k]], color=col, ls=ls, lw=1.2, marker=".",
                    label=f"{lab} ({block['skill_vs_B0'][k]:.2f})")
        obs(ax, rows, x)
        ticks(ax, rows, "unstable equilibria $U$")
        ax.set_title(title, fontsize=8)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(2e-3, 1.5)
        else:
            ax.set_ylim(-0.04, 1.06)

    panel(axs[0], c4["classes_U_all_leave_one_regime_out"], ("BB", "mech", "logit"),
          "(a) held-out regimes", True)
    axs[0].set_ylabel("share still hidden")
    axs[0].legend(loc="lower left", fontsize=5.5, title="prediction (out-of-sample skill)", title_fontsize=5.5)
    # (b) mechanism on all data: fitted to excited pairs only
    m = R["check2_mechanism"]["classes_pooled"]
    rows = m["bins"]
    x = np.arange(len(rows))
    lo, hi = np.array([r.get("beta_binomial_boot95", [r["beta_binomial"]] * 2) for r in rows], float).T
    axs[1].fill_between(x, lo, hi, color=styles["mech"][0], alpha=0.2, lw=0)
    bb = m["beta_binomial"]
    axs[1].plot(x, [r["beta_binomial"] for r in rows], color=styles["mech"][0], ls="--", lw=1.2, marker=".",
                label=rf"heterogeneous $p$: $\alpha={bb['alpha']:.2f}$, $\beta={bb['beta']:.2f}$")
    axs[1].plot(x, [r["binomial"] for r in rows], color=styles["binomial"][0], ls=":", lw=1.2, marker=".",
                label=rf"common $p={m['binomial']['p']:.2f}$")
    obs(axs[1], rows, x)
    ticks(axs[1], rows, "unstable equilibria $U$")
    axs[1].set_yscale("log")
    axs[1].set_ylim(2e-3, 1.5)
    axs[1].set_title("(b) predicted from excited pairs", fontsize=8)
    axs[1].legend(loc="lower left", fontsize=5.5)
    panel(axs[2], c4["classes_to_esn_full_U_all"], ("BB", "mech", "logit"),
          "(c) perceptrons, res. nets → ESN", False)
    axs[2].legend(loc="upper right", fontsize=5.5)
    fig.tight_layout()
    save(plt, fig, "fig_law")


# ============================================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default=str(NUM / "law_checks.json"))
    ap.add_argument("--fig-only", action="store_true", help="redraw the figure from the existing JSON")
    a = ap.parse_args()
    if a.fig_only:
        figure(json.loads(Path(a.out).read_text(encoding="utf-8")))
        return
    rng = np.random.default_rng(20261123)
    t0 = time.time()
    E = load()
    out = {"description": __doc__.split("\n\n")[0], "boot": a.boot,
           "ensembles": {k: {"rows": int(len(v)), "networks": int(v.cl.nunique())} for k, v in E.items()}}
    out["check1_out_of_sample"] = check1(E, a.boot, rng)
    print(f"check 1 done {time.time() - t0:.0f}s", flush=True)
    out["check2_mechanism"] = check2(E, a.boot, rng)
    print(f"check 2 done {time.time() - t0:.0f}s", flush=True)
    out["check3_collapse"] = check3(E)
    out["check4_law_form"] = check4(E, a.boot, rng)
    print(f"check 4 done {time.time() - t0:.0f}s", flush=True)
    E7 = load(include_le1=True)
    K7 = E7["classes"]
    K7 = K7[K7.reg == "le1"].assign(y_mech=lambda d: (d.k == 0).astype(float))
    out["le1_separate"] = {"mechanism_classes": mech_part(K7, CLS_BINS, "U_all", 0, rng)}
    out["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(clean(out), indent=1), encoding="utf-8")
    print("written", a.out, out["seconds"], "s")
    figure(json.loads(Path(a.out).read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()

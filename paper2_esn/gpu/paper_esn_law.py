"""Survival law of hiddenness in closed-loop echo state networks: numbers of the article.

Reproduces, on the CURRENT data, the numbers of theory_index.md §4, §4a and §4v:
  1. stage C (full protocol): hidden share by U among models that reproduce the hidden attractor, and where the
     losses of hiddenness come from (true vs spurious unstable equilibria);
  2. lite (stage N) vs full (stage C) protocol on the same models;
  3. stage N sweep over N: share "not excited" by U, logistic regression with regime fixed effects
     (log(1+U), set, log N), likelihood-ratio tests, geometric model q^U vs (1-p)^2, U(N);
  4. robustness to reservoir settings (input scale, density) at N = 300 (same logic as gpu/analyze_robust.py).

Data: ROOT/{gpuq-results, local_results, node_results}/**/<stage>/<regime>/<model>.json.
node_results/N_old_cpu_bug (buggy CPU Newton solver, recomputed later) is excluded, and so are the CPU stage N
files at N = 150 in node_results/N (pre-fix, never recomputed on the CPU; the A100 recompute is used).
Duplicate model names: the first file in the order gpuq-results (by job id) > local_results > node_results wins.

Pooled statistics (bins, fits, loss origin, lite vs full) use the six main regimes of paper_common.MAIN_REGIMES; LE1,
whose attractor is hidden only in the weak sense, is reported separately (keys "le1_separate"); per-regime outputs
(geometric model, U(N), settings table) keep LE1. Checks added on 2026-11-11: survival of the full-protocol
ensemble by U_in (unstable equilibria whose physical image lies in Q), with the logit coefficient of log(1 + U_in); U_in
bounds for the size sweep (stage N records do not store the coordinates of the equilibria); the U = 1 vs U = 2
comparison by regime; the sensitivity of the loss origin to the copy threshold.

Outputs (paper2_esn/results/numbers/):
  esn_law.json  - all numbers with Wilson 95% intervals per U bin;
  esn_rows.csv  - one row per (model, hidden attractor) of the N sweep and of the robustness runs (all seven regimes);
  esn_UN.csv    - mean U by (regime, set, N) of the N sweep, for a U(N) figure;
  esn_C_rows.csv - one row per reproducing model of the full-protocol ensemble (stage C), with U, U_in and the label.

Run: python -X utf8 paper2_esn/gpu/paper_esn_law.py      (light: ~2000 small JSON files)
"""
import csv
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
import os
from pathlib import Path

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_common import COPY_TOL, MAIN_REGIMES, POOL_NOTE, SEPARATE, main_only, only  # noqa: E402

ROOT = Path(os.environ.get("ESN_DATA", Path(__file__).resolve().parents[2] / "local" / "esn-gpu"))
OUT = Path(__file__).resolve().parents[1] / "results" / "numbers"
SOURCES = ("gpuq-results", "local_results", "node_results")   # priority order for duplicate model names
EXCLUDED_DIRS = {"N_old_cpu_bug"}
# CPU stage N results at N = 150 were computed before the CPU Newton-solver fix (22.10.2026) and never recomputed
# on the CPU (only N = 50, 100, 200 were); all 90 were recomputed on the A100 (gpuq job 3124), which is used instead.
SUPERSEDED_N = re.compile(r"_N150_")
BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 10 ** 9)]
Z95 = stats.norm.ppf(0.975)
ROBUST_REGIMES = ("se3", "se11", "le1", "chua_I")
ROBUST_PAT = re.compile(r"HE1_N300_rho0\.9_eta(1e-07|0\.0001)_s(\d)(?:_in([\d.]+))?(?:_dens([\d.]+))?\.json$")
SUFFIX_PAT = re.compile(r"_(in|dens)[\d.]+")

# values written in theory_index.md (21.10-06.11.2026), printed next to the current ones
REF = {
    "C_n_models": 669, "C_U0": "23/23", "C_U1_4": "32-45%", "C_U5_8": "14%", "C_U_ge9": "1 of 106",
    "C_loss_only_spurious": "81%",
    "lite_full": "24/24 agree; 18 not_excited -> hidden_stat",
    "N_rows": 345, "N_bins": [1.00, 0.50, 0.56, 0.33, 0.19, 0.06, 0.04],
    "N_logU": "-3.26 +- 0.44 (p = 6e-29)", "N_logN": "+0.09 +- 0.20",
    "N_geom": {"chua_I": (0.79, 0.76), "le1": (0.85, 0.87), "se3": (0.79, 0.80), "se11": (0.84, 0.94),
               "wangchen2": (0.0, 0.083), "wangchen2_a0.03": (0.0, 0.41)},
    "R_logU": "-3.25 +- 0.65 (z = -5.0), 170 rows", "R_LR": "chi2 = 10.0, df 4, p = 0.04",
    "R_dens0.02": "+1.52 +- 0.59", "R_in1": "+1.73 +- 0.69",
}


# ----------------------------------------------------------------------------------------------- helpers
def wilson(k, n, z=Z95):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def bin_label(lo, hi):
    return f"{lo}" if lo == hi else (f">={lo}" if hi >= 10 ** 9 else f"{lo}-{hi}")


def bin_table(rows, key_U, key_y, bins=BINS):
    out = []
    for lo, hi in bins:
        s = [x for x in rows if lo <= x[key_U] <= hi]
        k = int(sum(bool(x[key_y]) for x in s))
        n = len(s)
        lo95, hi95 = wilson(k, n)
        out.append({"U_bin": bin_label(lo, hi), "U_lo": lo, "U_hi": None if hi >= 10 ** 9 else hi, "n": n, "k": k,
                    "share": (k / n) if n else None, "ci95_lo": lo95 if n else None, "ci95_hi": hi95 if n else None,
                    "U_mean_in_bin": float(np.mean([x[key_U] for x in s])) if n else None})
    return out


def src_sort_key(f):
    """Deterministic order: source priority, then numeric gpuq job id, then path."""
    rel = f.relative_to(ROOT)
    src = SOURCES.index(rel.parts[0])
    job = int(rel.parts[1]) if rel.parts[0] == "gpuq-results" and rel.parts[1].isdigit() else -1
    return (src, job, str(rel))


SKIPPED = Counter()


def stage_files(stage):
    """All result JSONs of a stage: <source>/**/<stage>/<regime>/<name>.json, excluding the buggy backup and the
    superseded pre-fix CPU N = 150 results."""
    files = []
    for src in SOURCES:
        for f in (ROOT / src).rglob(f"{stage}/*/*.json"):
            if f.parent.parent.name != stage or ".failed" in f.name or EXCLUDED_DIRS & set(f.parts):
                continue
            if stage == "N" and src == "node_results" and SUPERSEDED_N.search(f.stem):
                SKIPPED["node_results/N/*_N150_* (pre-fix CPU, superseded by A100)"] += 1
                continue
            files.append(f)
    return sorted(files, key=src_sort_key)


def load_dedup(stage):
    """key regime/name -> (record, path, list of duplicate paths); first file by src_sort_key wins."""
    out = {}
    for f in stage_files(stage):
        key = f.parent.name + "/" + f.stem
        if key in out:
            out[key][2].append(f)
            continue
        out[key] = (json.loads(f.read_text(encoding="utf-8")), f, [])
    return out


def source_of(f):
    return f.relative_to(ROOT).parts[0]


META = {}


def meta(regime):
    if regime not in META:
        META[regime] = json.loads((ROOT / "data" / f"{regime}.json").read_text(encoding="utf-8"))
    return META[regime]


def in_Q(regime, x):
    """Physical point x inside the region Q = [box_lo, box_hi] of the true system."""
    m = meta(regime)
    x = np.asarray(x, float)
    return bool(np.all(x >= np.asarray(m["box_lo"])) and np.all(x <= np.asarray(m["box_hi"])))


def uin_bounds(regime, fps):
    """Stage N records drop the coordinates of the equilibria (stage_n.py: info.pop("x")); only the distance to the
    nearest true equilibrium, dist_true_eq = |x - x*| / L with L = max(box_hi - box_lo) / 2, is kept. An unstable
    equilibrium is surely in Q if the ball of that radius around x* lies in Q, surely outside if the ball contains Q;
    otherwise undetermined. Returns (U_in lower bound, U_in upper bound). For LE1 (distance to the line x = y = 0, z
    unknown) only the 'surely outside' test is possible."""
    m = meta(regime)
    lo, hi = np.asarray(m["box_lo"], float), np.asarray(m["box_hi"], float)
    L = float(np.max(hi - lo)) / 2
    E = [np.asarray(e["point"], float) for e in m["equilibria"]]
    n_lo = n_hi = 0
    for x in fps:
        if x.get("n_unstable", 0) <= 0:
            continue
        d = x.get("dist_true_eq")
        if d is None:
            n_hi += 1
            continue
        d *= L
        if not E:                                    # LE1: distance to the z axis
            if d <= float(np.hypot(max(abs(lo[0]), abs(hi[0])), max(abs(lo[1]), abs(hi[1])))):
                n_hi += 1
            continue
        e = E[int(x.get("nearest_true_eq", 0))]
        far = float(np.linalg.norm(np.maximum(np.abs(e - lo), np.abs(hi - e))))
        if np.all(e >= lo) and np.all(e <= hi):
            near = float(min(np.min(e - lo), np.min(hi - e)))
            if d < near:
                n_lo += 1
                n_hi += 1
            elif d <= far:
                n_hi += 1
        else:
            near = float(np.linalg.norm(np.maximum(0, np.maximum(lo - e, e - hi))))
            if near <= d <= far:
                n_hi += 1
    return n_lo, n_hi


def best_match_C(r, targets):
    """Stage C: closest model attractor (same kind, delta_att < 0.2, d_H < 0.2) to one of the true hidden
    attractors in `targets`; returns (delta_att, model attractor name, target) or None (as in survival.py)."""
    m = meta(r["regime"])
    best = None
    for at in r.get("attractors", []):
        for y in targets:
            mm = at["metrics"].get(y, {})
            yk = "chaos" if m["attractors"][y]["kind"] == "chaotic" else "regular"
            if "delta_att" in mm and at["kind"] == yk and mm["delta_att"] < 0.2 and mm["d_H"] < 0.2:
                if best is None or mm["delta_att"] < best[0]:
                    best = (mm["delta_att"], at["name"], y)
    return best


def logit_fit(X, y, ridge=1e-6, max_iter=500):
    """Logistic regression by Newton steps with step halving on nll + ridge*|w|^2.
    Returns coefficients, standard errors (inverse Fisher information) and the unpenalized log-likelihood."""
    w = np.zeros(X.shape[1])
    I = np.eye(X.shape[1])

    def obj(w_):
        z = X @ w_
        return float(np.sum(np.logaddexp(0, z) - y * z) + ridge * w_ @ w_)

    f = obj(w)
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + 2 * ridge * w
        H = (X * (p * (1 - p))[:, None]).T @ X + 2 * ridge * I
        step = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-8:
            f_new = obj(w - t * step)
            if f_new <= f + 1e-12:
                break
            t /= 2
        w = w - t * step
        done = abs(f - f_new) < 1e-11 and np.max(np.abs(t * step)) < 1e-7
        f = f_new
        if done:
            break
    z = X @ w
    p = 1 / (1 + np.exp(-z))
    F = (X * (p * (1 - p))[:, None]).T @ X
    se = np.sqrt(np.abs(np.diag(np.linalg.pinv(F))))
    ll = float(-np.sum(np.logaddexp(0, z) - y * z))
    return w, se, ll


def coef_dict(names, w, se):
    out = {}
    for n_, wi, si in zip(names, w, se):
        z = wi / si if si > 0 else float("nan")
        out[n_] = {"coef": float(wi), "se": float(si), "z": float(z), "p_wald": float(2 * stats.norm.sf(abs(z)))}
    return out


def lr_test(ll_full, ll_reduced, df):
    chi2 = max(0.0, 2 * (ll_full - ll_reduced))
    return {"chi2": chi2, "df": df, "p": float(stats.chi2.sf(chi2, df))}


def rows_from_N(r, key, path, sample):
    """One row per (model, true hidden attractor) of a stage N record."""
    rows = []
    cube = r.get("census_cube", {}) or {}
    ncube = sum(cube.values())
    uin_lo, uin_hi = uin_bounds(r["regime"], r.get("fixed_points", []))
    for y, m in r["hidden_match"].items():
        rep = bool(m.get("reproduced"))
        lab = m.get("label") if rep else None
        a = m.get("attractor")
        rows.append(dict(
            sample=sample, stage="N", model=key, source=source_of(path), regime=r["regime"], set=r["set"], N=int(r["N"]),
            rho=float(r["rho"]), eta=float(r["eta"]), input_scale=float(r.get("input_scale", 0.5)),
            density=float(r.get("density", 0.05)), seed=int(r["seed"]), target=y, K=int(r["K"]), U=int(r["U"]),
            index_sum=int(r["index_sum"]), reproduced=rep, label=lab, not_excited=bool(rep and lab == "not_excited"),
            p_cube=(cube.get(a, 0) / ncube) if (rep and a and ncube) else None,
            U_in_lo=uin_lo, U_in_hi=uin_hi, U_in_C=None, train_rmse=float(r["train_rmse"])))
    return rows


def fmt_bins(b):
    return "  ".join(f"U {x['U_bin']}: {x['k']}/{x['n']}" + (f" = {x['share']:.2f} [{x['ci95_lo']:.2f}, {x['ci95_hi']:.2f}]"
                                                             if x["n"] else "") for x in b)


def mantel_haenszel(rows, flag, y, stratum):
    """Common odds ratio of y between flag True and False across strata (Mantel-Haenszel), with the Robins-Breslow-
    Greenland 95% interval and the Cochran-Mantel-Haenszel test (continuity-corrected)."""
    g = defaultdict(lambda: [0, 0, 0, 0])            # a: flag & y, b: flag & not y, c: not flag & y, d: not flag & not y
    for x in rows:
        t = g[stratum(x)]
        t[(0 if x[y] else 1) if x[flag] else (2 if x[y] else 3)] += 1
    R = S = PR = PS_QR = QS = 0.0
    sa = sea = sva = 0.0
    used = 0
    for a, b, c, d in g.values():
        n = a + b + c + d
        if (a + b) == 0 or (c + d) == 0 or n < 2:
            continue
        used += 1
        r_, s_ = a * d / n, b * c / n
        p_, q_ = (a + d) / n, (b + c) / n
        R, S = R + r_, S + s_
        PR, PS_QR, QS = PR + p_ * r_, PS_QR + p_ * s_ + q_ * r_, QS + q_ * s_
        m1, n1 = a + c, a + b
        sa += a
        sea += n1 * m1 / n
        sva += n1 * (n - n1) * m1 * (n - m1) / (n * n * (n - 1)) if n > 1 else 0
    if not used or R == 0 or S == 0:
        return {"n_strata": used, "OR": None}
    OR = R / S
    v = PR / (2 * R * R) + PS_QR / (2 * R * S) + QS / (2 * S * S)
    lo, hi = math.exp(math.log(OR) - Z95 * math.sqrt(v)), math.exp(math.log(OR) + Z95 * math.sqrt(v))
    chi2 = (abs(sa - sea) - 0.5) ** 2 / sva if sva > 0 else float("nan")
    return {"n_strata": used, "OR": OR, "ci95": [lo, hi], "CMH_chi2": chi2, "CMH_p": float(stats.chi2.sf(chi2, 1))}


# ----------------------------------------------------------------------------------------------- main
def main():
    t_start = time.time()
    res = {"meta": {"script": "paper2_esn/gpu/paper_esn_law.py", "root": str(ROOT),
                    "generated": time.strftime("%Y-%m-%d %H:%M"), "excluded_dirs": sorted(EXCLUDED_DIRS),
                    "bins": [bin_label(lo, hi) for lo, hi in BINS], "ci": "Wilson score 95%"},
           "reference_theory_index": REF}
    P = print

    # ======================================================================== 1. stage C, full protocol
    Crec = load_dedup("C")
    c_dups = sum(len(v[2]) for v in Crec.values())
    crows = []
    for key, (r, path, _) in Crec.items():
        if r.get("model") != "esn" or r["set"] not in ("H1", "HE1", "H0"):
            continue
        best = best_match_C(r, meta(r["regime"])["hidden"])
        if not best:
            continue
        fps = r.get("fixed_points", [])
        U = sum(1 for x in fps if x.get("n_unstable", 0) > 0)
        no_x = sum(1 for x in fps if x.get("n_unstable", 0) > 0 and "x" not in x)
        U_in = sum(1 for x in fps if x.get("n_unstable", 0) > 0 and "x" in x and in_Q(r["regime"], x["x"]))
        hz = r["hiddenness"][best[1]]
        d_exc = []                       # dist_true_eq of the equilibria that excite the reproduced attractor
        for k in (hz.get("from") or {}):
            try:
                x = fps[int(k)]
            except (ValueError, IndexError):
                continue
            d_exc.append(x.get("dist_true_eq", 9))
        crows.append(dict(model=key, regime=r["regime"], set=r["set"], N=int(r.get("N", 300)), rho=float(r["rho"]),
                          eta=float(r["eta"]), seed=int(r["seed"]), U=U, U_in=U_in, U_out=U - U_in, U_no_x=no_x,
                          K=len(fps), label=hz["label"], hidden=hz["label"] == "hidden_stat", target=best[2],
                          d_exc=d_exc, exc_true=sum(d < COPY_TOL for d in d_exc),
                          exc_spur=sum(d >= COPY_TOL for d in d_exc), train_rmse=float(r["train_rmse"])))

    def loss_origin(rows, tol=COPY_TOL):
        ns = [x for x in rows if not x["hidden"] and x["label"] != "undetermined"]
        et = lambda x: sum(d < tol for d in x["d_exc"])    # noqa: E731
        es = lambda x: sum(d >= tol for d in x["d_exc"])   # noqa: E731
        out = {"copy_tol": tol, "n_losses": len(ns),
               "only_spurious": sum(1 for x in ns if es(x) and not et(x)),
               "only_true": sum(1 for x in ns if et(x) and not es(x)),
               "both": sum(1 for x in ns if et(x) and es(x)),
               "no_from_record": sum(1 for x in ns if not et(x) and not es(x))}
        out["share_only_spurious"] = out["only_spurious"] / len(ns) if ns else None
        return out

    def c_logit(rows, ucols):
        """hidden_stat ~ ucols + set (H1, H0 vs HE1) + log N + rho (0.6, 1.2 vs 0.9) + regime fixed effects."""
        rr = sorted({x["regime"] for x in rows})
        names = list(ucols) + ["set_H1", "set_H0", "log_N", "rho_0.6", "rho_1.2"] + [f"regime_{g}" for g in rr]
        X = np.array([[f(x) for f in ucols.values()] + [float(x["set"] == "H1"), float(x["set"] == "H0"), math.log(x["N"]),
                       float(abs(x["rho"] - 0.6) < 1e-9), float(abs(x["rho"] - 1.2) < 1e-9)]
                      + [float(x["regime"] == g) for g in rr] for x in rows])
        y = np.array([float(x["hidden"]) for x in rows])
        w, se, ll = logit_fit(X, y)
        return names, w, se, ll, X, y

    def c_summary(rows):
        cb_ = bin_table(rows, "U", "hidden")
        sh = [x["share"] for x in cb_ if x["U_lo"] in (1, 2, 3) and x["n"]]
        return {"n_models_reproduced": len(rows), "labels": dict(Counter(x["label"] for x in rows)),
                "by_set": dict(Counter(x["set"] for x in rows)),
                "by_N": {str(k): v for k, v in sorted(Counter(x["N"] for x in rows).items())},
                "bins": cb_, "bin_U_1_4": bin_table(rows, "U", "hidden", bins=[(1, 4)])[0],
                "bin_U_ge9": bin_table(rows, "U", "hidden", bins=[(9, 10 ** 9)])[0],
                "hidden_share_U1_to_U3_4_bins_range": [min(sh), max(sh)] if sh else None,
                "loss_origin": loss_origin(rows)}

    cm = main_only(crows)
    cC = c_summary(cm)
    cb, c_14, c_ge9, loss = cC["bins"], cC["bin_U_1_4"], cC["bin_U_ge9"], cC["loss_origin"]
    loss["note"] = ("loss = reproduced hidden attractor labelled self_excited or excited_r (undetermined excluded); "
                    "'true' equilibrium (copy) = model equilibrium with dist_true_eq < 0.01, i.e. within 1% of the "
                    "half-width L = max(box_hi - box_lo)/2 of the region Q from a true one")
    # check: by the number of unstable equilibria in Q (physical image W_out r* in box_lo..box_hi)
    nm_U, wU, seU, llU, _, yC = c_logit(cm, {"log1p_U": lambda x: math.log1p(x["U"])})
    nm_Ui, wUi, seUi, llUi, _, _ = c_logit(cm, {"log1p_U_in": lambda x: math.log1p(x["U_in"])})
    nm_B, wB, seB, llB, _, _ = c_logit(cm, {"log1p_U_in": lambda x: math.log1p(x["U_in"]),
                                            "log1p_U_out": lambda x: math.log1p(x["U_out"])})
    _, _, _, ll0, _, _ = c_logit(cm, {})
    cmU1 = [x for x in cm if x["U_in"] >= 1]
    nm_U1, wU1, seU1, _, _, _ = c_logit(cmU1, {"log1p_U_in": lambda x: math.log1p(x["U_in"])})
    cC["by_U_in"] = {
        "description": "same rows; U_in = unstable equilibria whose physical image W_out r* (stored as x in the stage C "
                       "record) lies in Q = [box_lo, box_hi]; U_out = U - U_in",
        "n_unstable_without_x": sum(x["U_no_x"] for x in cm),
        "U_in_mean": float(np.mean([x["U_in"] for x in cm])), "U_mean": float(np.mean([x["U"] for x in cm])),
        "share_rows_U_in_eq_U": float(np.mean([x["U_in"] == x["U"] for x in cm])),
        "bins": bin_table(cm, "U_in", "hidden"),
        "bins_U_out_given_U_in_0": bin_table([x for x in cm if x["U_in"] == 0], "U_out", "hidden"),
        "logit": {
            "model": "logit P(hidden_stat) ~ term(s) + [set H1] + [set H0] + log N + [rho 0.6] + [rho 1.2] + regime "
                     "fixed effects (no intercept; ridge 1e-6); same rows",
            "log1p_U": {"coef": coef_dict(nm_U, wU, seU)["log1p_U"], "loglik": llU, "LR": lr_test(llU, ll0, 1)},
            "log1p_U_in": {"coef": coef_dict(nm_Ui, wUi, seUi)["log1p_U_in"], "loglik": llUi, "LR": lr_test(llUi, ll0, 1),
                           "other": {k: v for k, v in coef_dict(nm_Ui, wUi, seUi).items() if not k.startswith("regime_")}},
            "log1p_U_in_and_log1p_U_out": {"coef": {k: v for k, v in coef_dict(nm_B, wB, seB).items()
                                                    if k in ("log1p_U_in", "log1p_U_out")},
                                           "LR_U_out_beyond_U_in": lr_test(llB, llUi, 1)},
            "log1p_U_in_rows_U_in_ge_1": {"n": len(cmU1), "coef": coef_dict(nm_U1, wU1, seU1)["log1p_U_in"]},
        }}
    cC["loss_origin_copy_threshold_sensitivity"] = {f"{t:g}": loss_origin(cm, t) for t in (COPY_TOL / 2, COPY_TOL, 2 * COPY_TOL)}
    cC["by_regime"] = {g: {"n": len(s), "hidden": sum(x["hidden"] for x in s),
                           "U_mean": float(np.mean([x["U"] for x in s])), "U_in_mean": float(np.mean([x["U_in"] for x in s]))}
                       for g in sorted({x["regime"] for x in crows}) for s in [only(crows, g)]}
    # U = 1 vs U = 2 by regime (the pooled shares are not monotone here either)
    t12 = {g: {str(u): [sum(x["hidden"] for x in cm if x["regime"] == g and x["U"] == u),
                        sum(1 for x in cm if x["regime"] == g and x["U"] == u)] for u in (1, 2)}
           for g in sorted({x["regime"] for x in cm})}
    den = sum(v["1"][1] for v in t12.values() if v["2"][1])
    cC["U1_vs_U2"] = {"by_regime": t12,
                      "U2_standardized_to_U1_regime_mix": (sum(v["1"][1] * v["2"][0] / v["2"][1] for v in t12.values()
                                                               if v["2"][1]) / den) if den else None,
                      "mantel_haenszel_U2_vs_U1_by_regime": mantel_haenszel(
                          [dict(x, U2=x["U"] == 2) for x in cm if x["U"] in (1, 2)], "U2", "hidden", lambda x: x["regime"])}
    cl = only(crows, "le1")
    le1C = c_summary(cl)
    le1C["by_U_in"] = bin_table(cl, "U_in", "hidden")
    le1C["note"] = ("LE1: 'true' (copy) = model equilibrium within 0.01 L of the line x = y = 0; the true attractor is "
                    "itself reached from a segment of the unstable half-line")
    res["stage_C_full_protocol"] = {
        "description": "ESN models of stage C (sets H1, HE1, H0; rho 0.6/0.9/1.2; N 300-1000) whose closest attractor "
                       "reproduces a true hidden attractor (delta_att < 0.2, d_H < 0.2, same kind); one row per model "
                       "(best-matching hidden attractor); hidden = label hidden_stat; " + POOL_NOTE,
        "regimes": sorted({x["regime"] for x in cm}),
        "n_models_files": len(Crec), "n_duplicate_files_ignored": c_dups,
        **cC,
        "ensemble_all_regimes": {"n_models_reproduced": len(crows), "labels": dict(Counter(x["label"] for x in crows))},
        "le1_separate": le1C,
    }
    P("=" * 100)
    P(f"1. STAGE C (full protocol), six regimes: {len(cm)} models with reproduced hidden attractor "
      f"(all seven: {len(crows)})   [theory: {REF['C_n_models']}]")
    P(f"   labels {cC['labels']}")
    P("   " + fmt_bins(cb))
    P(f"   U 1-4 pooled {c_14['k']}/{c_14['n']} = {c_14['share']:.2f}; bins 1, 2, 3-4: "
      + ", ".join(f"{x['share']:.0%}" for x in cb if x["U_lo"] in (1, 2, 3) and x["n"]) + f"   [theory {REF['C_U1_4']}]")
    P(f"   U >= 9: {c_ge9['k']} of {c_ge9['n']}   [theory {REF['C_U_ge9']}]")
    P(f"   losses {loss['n_losses']}: only spurious {loss['only_spurious']} ({loss['share_only_spurious']:.0%}), only true "
      f"{loss['only_true']}, both {loss['both']}, no record {loss['no_from_record']}   [theory {REF['C_loss_only_spurious']}]")
    for t, v in cC["loss_origin_copy_threshold_sensitivity"].items():
        P(f"     copy tol {t}: only spurious {v['only_spurious']}/{v['n_losses']} = {v['share_only_spurious']:.3f}, "
          f"only true {v['only_true']}, both {v['both']}")
    P("   by U_in: " + fmt_bins(cC["by_U_in"]["bins"]))
    lg = cC["by_U_in"]["logit"]
    P(f"   logit log1p(U) {lg['log1p_U']['coef']['coef']:+.2f} +- {lg['log1p_U']['coef']['se']:.2f}; log1p(U_in) "
      f"{lg['log1p_U_in']['coef']['coef']:+.2f} +- {lg['log1p_U_in']['coef']['se']:.2f}; joint U_in/U_out "
      f"{lg['log1p_U_in_and_log1p_U_out']['coef']['log1p_U_in']['coef']:+.2f}/"
      f"{lg['log1p_U_in_and_log1p_U_out']['coef']['log1p_U_out']['coef']:+.2f}")
    P(f"   LE1 separately: {len(cl)} models; " + fmt_bins(le1C["bins"]))

    # ======================================================================== 2. stage N records
    Nrec = load_dedup("N")
    sweep_models, robust_models = {}, {}
    n_dup_pairs = []
    for key, (r, path, dups) in Nrec.items():
        default = (not SUFFIX_PAT.search(path.stem) and float(r.get("input_scale", 0.5)) == 0.5
                   and float(r.get("density", 0.05)) == 0.05 and float(r["rho"]) == 0.9 and r["set"] in ("HE1", "H1"))
        if default:
            sweep_models[key] = (r, path)
            for d in dups:          # same model computed twice (A100 vs CPU): agreement check
                r2 = json.loads(d.read_text(encoding="utf-8"))
                lab = lambda q: {y: (bool(m.get("reproduced")), m.get("label") if m.get("reproduced") else None)
                                 for y, m in q["hidden_match"].items()}
                n_dup_pairs.append(dict(key=key, a=source_of(path), b=source_of(d), U_equal=r["U"] == r2["U"],
                                        K_equal=r["K"] == r2["K"], labels_equal=lab(r) == lab(r2),
                                        n_targets=len(r["hidden_match"]),
                                        n_targets_equal=sum(lab(r)[y] == lab(r2)[y] for y in r["hidden_match"])))
    # robustness sample: exactly as gpu/analyze_robust.py (node_results/N, 4 regimes, N = 300, HE1, rho 0.9)
    for reg in ROBUST_REGIMES:
        for f in sorted((ROOT / "node_results" / "N" / reg).glob("HE1_N300_rho0.9_*.json")):
            if ROBUST_PAT.search(f.name):
                robust_models[f"{reg}/{f.stem}"] = (json.loads(f.read_text(encoding="utf-8")), f)

    srows = [x for k, (r, p) in sweep_models.items() for x in rows_from_N(r, k, p, "n_sweep")]
    rrows = [x for k, (r, p) in robust_models.items() for x in rows_from_N(r, k, p, "robustness")]
    for x in rrows:
        isc, dn = x["input_scale"], x["density"]
        x["setting"] = "base" if (isc == 0.5 and dn == 0.05) else (f"in{isc:g}" if isc != 0.5 else f"dens{dn:g}")
    for x in srows:
        x["setting"] = "base"
    # exact U_in for sweep models that were also run in stage C (same model: same name and training error, same U)
    c_twin = {}
    for key, (c, path, _) in Crec.items():
        if c.get("model") == "esn":
            c_twin[f"{c['regime']}/{c['set']}_N{int(c.get('N', 300))}_rho{c['rho']:g}_eta{c['eta']:g}_s{c['seed']}"] = c
    n_twin = Counter()
    for x in srows:
        c = c_twin.get(x["model"])
        if c is None:
            continue
        if abs(c["train_rmse"] - x["train_rmse"]) > 1e-9 * max(1.0, abs(c["train_rmse"])):
            n_twin["different_training_error"] += 1
            continue
        fps = c.get("fixed_points", [])
        if sum(1 for f in fps if f.get("n_unstable", 0) > 0) != x["U"] or any("x" not in f for f in fps):
            n_twin["different_U"] += 1
            continue
        x["U_in_C"] = sum(1 for f in fps if f.get("n_unstable", 0) > 0 and in_Q(x["regime"], f["x"]))
        n_twin["used"] += 1

    # overlap of the two samples
    robust_base_keys = {k for k, (r, p) in robust_models.items() if not SUFFIX_PAT.search(p.stem)}
    robust_suffix_keys = set(robust_models) - robust_base_keys
    in_sweep_from_robust = {k for k in robust_base_keys if k in sweep_models}
    sweep_file_is_robust = {k for k in in_sweep_from_robust if sweep_models[k][1] == robust_models[k][1]}
    overlap = {
        "robustness_models": len(robust_models), "robustness_suffixed_in_dens": len(robust_suffix_keys),
        "robustness_base_setting": len(robust_base_keys),
        "suffixed_models_in_sweep": len(robust_suffix_keys & set(sweep_models)),
        "base_models_also_in_sweep": len(in_sweep_from_robust),
        "base_models_in_sweep_via_A100_file": len(in_sweep_from_robust - sweep_file_is_robust),
        "base_models_in_sweep_only_via_CPU_file": len(sweep_file_is_robust),
        "note": "suffixed runs (_in<x>, _dens<x>) are NOT in the N sweep; the 40 base-setting runs (input 0.5, density "
                "0.05, N 300) have plain names: those with an A100 twin are de-duplicated (A100 file kept), the rest "
                "(seeds 3-4) enter the sweep as extra N = 300 models",
    }

    # ======================================================================== 3. N sweep
    D_all = [x for x in srows if x["reproduced"] and x["index_sum"] == 1]
    D = main_only(D_all)                       # pooled statistics: six regimes
    D_le1 = only(D_all, "le1")
    nb = bin_table(D, "U", "not_excited")
    regs = sorted({x["regime"] for x in D})
    sets_ = sorted({x["set"] for x in srows})

    def design(rows, with_U=True, with_N=True, with_set=True, ucol="U"):
        rr = sorted({x["regime"] for x in rows})
        cols, names = [], []
        if with_U:
            cols.append([math.log1p(x[ucol]) for x in rows]); names.append("log1p_" + ucol)
        if with_set:
            cols.append([float(x["set"] == "H1") for x in rows]); names.append("set_H1")
        if with_N:
            cols.append([math.log(x["N"]) for x in rows]); names.append("log_N")
        for g in rr:
            cols.append([float(x["regime"] == g) for x in rows]); names.append(f"regime_{g}")
        return np.array(cols, float).T, names

    def sweep_law(rows, ucol="U"):
        y = np.array([float(x["not_excited"]) for x in rows])
        X, names = design(rows, ucol=ucol)
        w, se, ll = logit_fit(X, y)
        _, _, ll_noU = logit_fit(design(rows, with_U=False)[0], y)
        _, _, ll_noN = logit_fit(design(rows, with_N=False, ucol=ucol)[0], y)
        return {"n_rows": len(rows), "n_not_excited": int(y.sum()), "coef": coef_dict(names, w, se),
                "LR_log1p_U": lr_test(ll, ll_noU, 1), "LR_log_N_beyond_U": lr_test(ll, ll_noN, 1),
                "loglik": ll, "model": f"logit P(not_excited) ~ log(1+{ucol}) + [set H1] + log N + regime fixed effects "
                                      "(no intercept; ridge 1e-6)"}

    law = sweep_law(D)
    law_with_le1 = sweep_law(D_all)
    # geometric model P(not excited | U) = q^U per regime (LE1 included, per regime), p = median basin share of the
    # model attractor in the cube
    geom = {}
    for g in sorted({x["regime"] for x in D_all}):
        s = [x for x in D_all if x["regime"] == g]
        U = np.array([x["U"] for x in s], float)
        h = np.array([x["not_excited"] for x in s], float)

        def nll_q(q):
            return -np.sum(h * U * np.log(q) + (1 - h) * np.log(np.maximum(1 - q ** U, 1e-300)))
        q = float(optimize.minimize_scalar(nll_q, bounds=(1e-4, 1 - 1e-6), method="bounded").x)
        pc = float(np.nanmedian([x["p_cube"] if x["p_cube"] is not None else np.nan for x in s]))
        geom[g] = {"n_rows": len(s), "q_fit": q, "q_at_lower_bound": q < 2e-4, "p_cube_median": pc,
                   "one_minus_p_squared": (1 - pc) ** 2, "U_mean": float(U.mean())}
    # U(N): one value per model
    models_tab = defaultdict(list)
    for k, (r, p) in sweep_models.items():
        models_tab[(r["regime"], r["set"])].append((int(r["N"]), int(r["U"]), int(r["K"]), int(r["index_sum"])))
    un_rows, un_shape = [], {}
    for (g, st), v in sorted(models_tab.items()):
        Ns = sorted({x[0] for x in v})
        means = []
        for N in Ns:
            u = np.array([x[1] for x in v if x[0] == N], float)
            kk = np.array([x[2] for x in v if x[0] == N], float)
            rr = [x for x in srows if x["regime"] == g and x["set"] == st and x["N"] == N]
            un_rows.append(dict(regime=g, set=st, N=N, n_models=len(u), U_mean=u.mean(), U_sd=u.std(ddof=1) if len(u) > 1 else 0.0,
                                U_sem=(u.std(ddof=1) / math.sqrt(len(u))) if len(u) > 1 else 0.0, U_median=float(np.median(u)),
                                U_min=int(u.min()), U_max=int(u.max()), K_mean=kk.mean(),
                                n_index_sum_1=sum(1 for x in v if x[0] == N and x[3] == 1),
                                n_rows=len(rr), n_rows_reproduced=sum(x["reproduced"] for x in rr),
                                n_rows_law=sum(x["reproduced"] and x["index_sum"] == 1 for x in rr),
                                n_not_excited_law=sum(x["not_excited"] and x["index_sum"] == 1 for x in rr)))
            means.append(u.mean())
        means = np.array(means)
        i0 = int(np.argmin(means))
        rho_s, p_s = stats.spearmanr([x[0] for x in v], [x[1] for x in v])
        if 0 < i0 < len(Ns) - 1 and means[0] > means[i0] + 1 and means[-1] > means[i0] + 1:
            shape = "U-shaped"
        elif rho_s > 0 and p_s < 0.05 and means[0] <= means[i0] + 1:
            shape = "increasing"
        elif rho_s < 0 and p_s < 0.05:
            shape = "decreasing"
        else:
            shape = "flat/irregular"
        un_shape[f"{g}/{st}"] = {"N": Ns, "U_mean": [float(m) for m in means], "argmin_N": Ns[i0],
                                 "U_at_min": float(means[i0]), "spearman_rho": float(rho_s), "spearman_p": float(p_s),
                                 "shape": shape,
                                 "shape_rule": "U-shaped if the minimum of mean U is interior and both ends exceed it by "
                                               "> 1; increasing if Spearman > 0 (p < 0.05) and U(N_min) within 1 of the minimum"}
    # sensitivity: without the CPU-only N = 300 base models of the robustness experiment
    D_noextra = [x for x in D if x["model"] not in sweep_file_is_robust]
    law_noextra = sweep_law(D_noextra)

    # ---- check: U = 1 vs U = 2 (non-monotone pooled shares) and regime composition
    def by_reg(rows, u):
        return {g: {"n": len(s), "k": sum(x["not_excited"] for x in s),
                    "share": (sum(x["not_excited"] for x in s) / len(s)) if s else None}
                for g in sorted({x["regime"] for x in D_all}) for s in [[x for x in rows if x["regime"] == g and x["U"] == u]]}
    u12 = [dict(x, U2=x["U"] == 2) for x in D if x["U"] in (1, 2)]
    ref_mix = Counter(x["regime"] for x in D if x["U"] == 1)
    sh2 = {g: v["share"] for g, v in by_reg(D, 2).items()}
    std_num = sum(ref_mix[g] * sh2[g] for g in ref_mix if sh2.get(g) is not None)
    std_den = sum(ref_mix[g] for g in ref_mix if sh2.get(g) is not None)
    # logit with U-bin dummies + set + log N + regime effects; regime-adjusted (marginally standardized) share per bin
    bins_ = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 10 ** 9)]
    bi = lambda u: next(i for i, (a, b) in enumerate(bins_) if a <= u <= b)   # noqa: E731
    rr_ = sorted({x["regime"] for x in D})
    Xb = np.array([[float(bi(x["U"]) == i) for i in range(1, len(bins_))] + [float(x["set"] == "H1"), math.log(x["N"])]
                   + [float(x["regime"] == g) for g in rr_] for x in D])
    yb = np.array([float(x["not_excited"]) for x in D])
    wb, seb, _ = logit_fit(Xb, yb, ridge=1e-3)
    adj = []
    for i in range(len(bins_)):
        Xi = Xb.copy()
        Xi[:, :len(bins_) - 1] = 0.0
        if i:
            Xi[:, i - 1] = 1.0
        adj.append({"U_bin": bin_label(*bins_[i]), "adjusted_share": float(np.mean(1 / (1 + np.exp(-Xi @ wb)))),
                    "coef_vs_U0": [float(wb[i - 1]), float(seb[i - 1])] if i else None})
    X12 = np.array([[float(x["U2"]), float(x["set"] == "H1"), math.log(x["N"])]
                    + [float(x["regime"] == g) for g in sorted({z["regime"] for z in u12})] for x in u12])
    w12, se12, _ = logit_fit(X12, np.array([float(x["not_excited"]) for x in u12]))
    u1_vs_u2 = {
        "pooled_U1": bin_table(D, "U", "not_excited", bins=[(1, 1)])[0], "pooled_U2": bin_table(D, "U", "not_excited", bins=[(2, 2)])[0],
        "by_regime_U1": by_reg(D_all, 1), "by_regime_U2": by_reg(D_all, 2),
        "by_regime_set_U1_U2": {f"{g}/{st}": {str(u): [sum(x["not_excited"] for x in s), len(s)]
                                              for u in (1, 2) for s in [[x for x in D if x["regime"] == g and x["set"] == st and x["U"] == u]]}
                                for g, st in sorted({(x["regime"], x["set"]) for x in D})},
        "U2_standardized_to_U1_regime_mix": std_num / std_den if std_den else None,
        "mantel_haenszel_U2_vs_U1_by_regime": mantel_haenszel(u12, "U2", "not_excited", lambda x: x["regime"]),
        "mantel_haenszel_U2_vs_U1_by_regime_set": mantel_haenszel(u12, "U2", "not_excited", lambda x: (x["regime"], x["set"])),
        "logit_U2_vs_U1": {"coef_U2": [float(w12[0]), float(se12[0])], "z": float(w12[0] / se12[0]),
                           "model": "rows with U in {1, 2}: not_excited ~ [U = 2] + [set H1] + log N + regime fixed effects"},
        "regime_adjusted_bins": {"model": "not_excited ~ U-bin dummies (vs U = 0) + [set H1] + log N + regime fixed effects "
                                          "(ridge 1e-3); share = mean prediction over all rows with the bin set",
                                 "bins": adj},
        "note": "six regimes (pooled rows); by_regime tables include LE1 for reference",
    }

    # ---- check: U_in for the size sweep. Stage N records drop the coordinates of the equilibria, so U_in is
    # bounded from the distance to the nearest true equilibrium (uin_bounds), and exact where the same model was also run
    # in stage C (its record keeps the coordinates)
    det = [x for x in D if x["U_in_lo"] == x["U_in_hi"]]
    for x in D:
        x["U_in_det"] = x["U_in_lo"]
    exact_c = [dict(x, U_in_exact=x["U_in_C"]) for x in D if x["U_in_C"] is not None]
    uin_sweep = {
        "why_not_exact": "stage_n.py removes the coordinates x of the equilibria before writing the record "
                         "(info.pop('x')); only dist_true_eq (to the nearest true equilibrium, in units of L = half "
                         "of the largest edge of Q) and nearest_true_eq are stored, which bound U_in but do not fix it",
        "n_rows_law": len(D), "n_rows_U_in_determined": len(det),
        "share_rows_determined": len(det) / len(D) if D else None,
        "U_mean": float(np.mean([x["U"] for x in D])), "U_in_lo_mean": float(np.mean([x["U_in_lo"] for x in D])),
        "U_in_hi_mean": float(np.mean([x["U_in_hi"] for x in D])),
        "bins_U_in_lower_bound": bin_table(D, "U_in_lo", "not_excited"),
        "bins_U_in_upper_bound": bin_table(D, "U_in_hi", "not_excited"),
        "bins_U_in_determined_rows_only": bin_table(det, "U_in_det", "not_excited"),
        "logit_log1p_U_in_lower_bound": sweep_law(D, ucol="U_in_lo")["coef"]["log1p_U_in_lo"],
        "logit_log1p_U_in_upper_bound": sweep_law(D, ucol="U_in_hi")["coef"]["log1p_U_in_hi"],
        "logit_log1p_U_in_determined_rows": sweep_law(det, ucol="U_in_det")["coef"]["log1p_U_in_det"] if len(det) > 20 else None,
        "logit_log1p_U_same_determined_rows": sweep_law(det)["coef"]["log1p_U"] if len(det) > 20 else None,
        "stage_C_twins": {"rows": len(exact_c), "matching": dict(n_twin),
                          "bounds_hold": sum(x["U_in_lo"] <= x["U_in_exact"] <= x["U_in_hi"] for x in exact_c),
                          "bins_U_in_exact": bin_table(exact_c, "U_in_exact", "not_excited"),
                          "logit_log1p_U_in_exact": (sweep_law(exact_c, ucol="U_in_exact")["coef"]["log1p_U_in_exact"]
                                                     if len(exact_c) > 20 else None),
                          "logit_log1p_U_same_rows": sweep_law(exact_c)["coef"]["log1p_U"] if len(exact_c) > 20 else None,
                          "note": "sweep models (rho 0.9, N 300-1000) whose stage C twin has the same training error and "
                                  "the same U; U_in from the stage C coordinates"},
    }

    # LE1 separately (one regime, set HE1 only: logit with intercept, log(1+U), log N)
    def le1_law(rows):
        if len(rows) < 10 or len({x["not_excited"] for x in rows}) < 2:
            return None
        X = np.array([[1.0, math.log1p(x["U"]), math.log(x["N"])] for x in rows])
        y = np.array([float(x["not_excited"]) for x in rows])
        w, se, _ = logit_fit(X, y)
        return {"n_rows": len(rows), "coef": coef_dict(["const", "log1p_U", "log_N"], w, se)}

    res["N_sweep"] = {
        "description": "stage N, lite protocol: closed-loop ESN, rho 0.9, eta 1e-7 and 1e-4, input scale 0.5, density 0.05, "
                       "sets HE1 (7 regimes) and H1 (se3, se11); rows = (model, true hidden attractor); law sample = "
                       "reproduced and index sum 1; " + POOL_NOTE,
        "n_models": sum(1 for r, p in sweep_models.values() if r["regime"] not in SEPARATE),
        "n_rows": len(main_only(srows)),
        "n_rows_reproduced": sum(x["reproduced"] for x in main_only(srows)),
        "n_rows_reproduced_index_sum_ne_1": sum(x["reproduced"] and x["index_sum"] != 1 for x in main_only(srows)),
        "n_models_index_sum_ne_1": sum(1 for r, p in sweep_models.values() if r["index_sum"] != 1 and r["regime"] not in SEPARATE),
        "n_rows_law": len(D), "n_not_excited_law": sum(x["not_excited"] for x in D),
        "regimes": regs, "sets": sets_, "N_values": sorted({x["N"] for x in srows}),
        "ensemble_all_regimes": {
            "n_models": len(sweep_models), "n_rows": len(srows), "n_rows_reproduced": sum(x["reproduced"] for x in srows),
            "n_rows_law": len(D_all), "n_not_excited_law": sum(x["not_excited"] for x in D_all),
            "n_models_index_sum_ne_1": sum(1 for r, p in sweep_models.values() if r["index_sum"] != 1)},
        "le1_separate": {
            "n_models": sum(1 for r, p in sweep_models.values() if r["regime"] == "le1"),
            "n_rows": len(only(srows, "le1")), "n_rows_law": len(D_le1),
            "n_not_excited_law": sum(x["not_excited"] for x in D_le1),
            "bins": bin_table(D_le1, "U", "not_excited"), "logit": le1_law(D_le1)},
        "sensitivity_with_le1": {"n_rows_law": len(D_all), "bins": bin_table(D_all, "U", "not_excited"),
                                 "log1p_U": law_with_le1["coef"]["log1p_U"], "log_N": law_with_le1["coef"]["log_N"],
                                 "set_H1": law_with_le1["coef"]["set_H1"], "LR_log1p_U": law_with_le1["LR_log1p_U"],
                                 "LR_log_N_beyond_U": law_with_le1["LR_log_N_beyond_U"],
                                 "note": "seven regimes, as before 2026-11-11 (for comparison only)"},
        "U1_vs_U2": u1_vs_u2,
        "by_U_in": uin_sweep,
        "models_by_source": dict(Counter(source_of(p) for r, p in sweep_models.values())),
        "overlap_with_robustness": overlap,
        "duplicate_model_checks": {"n_pairs": len(n_dup_pairs),
                                   "U_equal": sum(d["U_equal"] for d in n_dup_pairs),
                                   "K_equal": sum(d["K_equal"] for d in n_dup_pairs),
                                   "all_target_labels_equal": sum(d["labels_equal"] for d in n_dup_pairs),
                                   "targets": sum(d["n_targets"] for d in n_dup_pairs),
                                   "targets_equal": sum(d["n_targets_equal"] for d in n_dup_pairs),
                                   "pairs_by_sources": dict(Counter(f"{d['a']} vs {d['b']}" for d in n_dup_pairs)),
                                   "N_values": sorted({int(re.search(r'_N(\d+)_', d['key']).group(1)) for d in n_dup_pairs})},
        "bins": nb,
        "logit": law,
        "geometric_model": geom,
        "U_of_N": un_shape,
        "sensitivity_without_cpu_only_N300_base_models": {
            "n_rows_law": len(D_noextra), "bins": bin_table(D_noextra, "U", "not_excited"),
            "log1p_U": law_noextra["coef"]["log1p_U"], "log_N": law_noextra["coef"]["log_N"],
            "LR_log_N_beyond_U": law_noextra["LR_log_N_beyond_U"]},
    }
    P("=" * 100)
    P(f"2/3. N SWEEP (lite protocol): {len(sweep_models)} models, {len(srows)} rows, law sample {len(D)} rows "
      f"(reproduced, sum ind = 1)   [theory: {REF['N_rows']}]")
    P(f"   sources {res['N_sweep']['models_by_source']}; N values {res['N_sweep']['N_values']}")
    P(f"   overlap with robustness: {overlap}")
    dc = res["N_sweep"]["duplicate_model_checks"]
    P(f"   duplicate model files (same name twice): {dc['n_pairs']} pairs ({dc['pairs_by_sources']}, N {dc['N_values']}); "
      f"U equal {dc['U_equal']}, K equal {dc['K_equal']}, target labels equal {dc['targets_equal']}/{dc['targets']}")
    P("   " + fmt_bins(nb))
    P(f"   theory shares: {REF['N_bins']}")
    for n_ in ("log1p_U", "set_H1", "log_N"):
        c = law["coef"][n_]
        P(f"   {n_:8s} {c['coef']:+.2f} +- {c['se']:.2f}  z {c['z']:+.1f}  p_wald {c['p_wald']:.1e}")
    P(f"   LR log(1+U): chi2 {law['LR_log1p_U']['chi2']:.1f}, p {law['LR_log1p_U']['p']:.1e}   [theory {REF['N_logU']}]")
    P(f"   LR log N beyond U: chi2 {law['LR_log_N_beyond_U']['chi2']:.2f}, p {law['LR_log_N_beyond_U']['p']:.2f}   "
      f"[theory log N {REF['N_logN']}]")
    P(f"   without CPU-only N=300 base models: {len(D_noextra)} rows, log1p_U {law_noextra['coef']['log1p_U']['coef']:+.2f} "
      f"+- {law_noextra['coef']['log1p_U']['se']:.2f}, log N {law_noextra['coef']['log_N']['coef']:+.2f} +- "
      f"{law_noextra['coef']['log_N']['se']:.2f}")
    lw = res["N_sweep"]["sensitivity_with_le1"]
    P(f"   with LE1 (seven regimes, {lw['n_rows_law']} rows): log1p_U {lw['log1p_U']['coef']:+.2f} +- {lw['log1p_U']['se']:.2f}")
    P(f"   LE1 separately: {len(D_le1)} rows; " + fmt_bins(res["N_sweep"]["le1_separate"]["bins"]))
    q = res["N_sweep"]["U1_vs_U2"]
    P(f"   U1 vs U2: pooled {q['pooled_U1']['share']:.2f} vs {q['pooled_U2']['share']:.2f}; U2 standardized to the U1 "
      f"regime mix {q['U2_standardized_to_U1_regime_mix']:.2f}; MH OR {q['mantel_haenszel_U2_vs_U1_by_regime']}")
    for g in q["by_regime_U1"]:
        a, b = q["by_regime_U1"][g], q["by_regime_U2"][g]
        P(f"     {g:16s} U1 {a['k']}/{a['n']}   U2 {b['k']}/{b['n']}")
    P("   regime-adjusted bins: " + ", ".join(f"{b['U_bin']}: {b['adjusted_share']:.2f}" for b in q["regime_adjusted_bins"]["bins"]))
    ui = res["N_sweep"]["by_U_in"]
    P(f"   U_in bounds: determined in {ui['n_rows_U_in_determined']}/{ui['n_rows_law']} rows; lower-bound bins "
      + fmt_bins(ui["bins_U_in_lower_bound"]))
    P(f"   stage C twins: {ui['stage_C_twins']['rows']} rows {ui['stage_C_twins']['matching']}, bounds hold "
      f"{ui['stage_C_twins']['bounds_hold']}")
    P("   geometric model q^U vs (1-p_cube)^2:")
    for g, v in geom.items():
        ref = REF["N_geom"].get(g)
        P(f"     {g:16s} n {v['n_rows']:3d}  q {v['q_fit']:.2f}  (1-p)^2 {v['one_minus_p_squared']:.2f}  p_cube {v['p_cube_median']:.3f}"
          + (f"   [theory {ref[0]:.2f}/{ref[1]:.2f}]" if ref else ""))
    P("   U(N), mean U per N:")
    for k, v in un_shape.items():
        P(f"     {k:20s} {v['shape']:15s} argmin N {v['argmin_N']:5d}  rho_s {v['spearman_rho']:+.2f}  "
          + " ".join(f"{N}:{u:.1f}" for N, u in zip(v["N"], v["U_mean"])))

    # ======================================================================== 2b. lite vs full on the same models
    def coarse_lite(m):
        if not m.get("reproduced"):
            return "not_reproduced"
        return "candidate" if m.get("label") == "not_excited" else "excited"

    def coarse_full(lab):
        if lab is None:
            return "not_reproduced"
        return "candidate" if lab in ("hidden_stat", "undetermined") else "excited"

    def lite_full(select):
        tab = Counter()
        pairs = []
        for key, (c, path, _) in Crec.items():
            if c.get("model") != "esn" or not select(c):
                continue
            N = int(c.get("N", 300))
            nkey = f"{c['regime']}/{c['set']}_N{N}_rho{c['rho']:g}_eta{c['eta']:g}_s{c['seed']}"
            if nkey not in Nrec:
                continue
            n = Nrec[nkey][0]
            same_model = abs(n["train_rmse"] - c["train_rmse"]) <= 1e-9 * max(1.0, abs(c["train_rmse"]))
            for y, m in n["hidden_match"].items():
                best = best_match_C(c, [y])
                full = c["hiddenness"][best[1]]["label"] if best else None
                lite = m.get("label") if m.get("reproduced") else None
                tab[(lite or "not_reproduced", full or "not_reproduced")] += 1
                pairs.append(dict(lite=coarse_lite(m), full=coarse_full(full), lite_label=lite, full_label=full,
                                  same_model=same_model, regime=c["regime"], N=N))
        agree = sum(p["lite"] == p["full"] for p in pairs)
        ne = [p for p in pairs if p["lite_label"] == "not_excited"]
        return {"n_pairs": len(pairs), "n_agree_coarse": agree,
                "n_lite_not_excited": len(ne),
                "n_lite_not_excited_full_hidden_stat": sum(p["full_label"] == "hidden_stat" for p in ne),
                "n_full_hidden_stat": sum(p["full_label"] == "hidden_stat" for p in pairs),
                "all_same_model_train_rmse": all(p["same_model"] for p in pairs),
                "regimes": sorted({p["regime"] for p in pairs}), "N_values": sorted({p["N"] for p in pairs}),
                "crosstab_lite_to_full": {f"{a} -> {b}": v for (a, b), v in sorted(tab.items(), key=lambda t: -t[1])},
                "coarse_classes": "candidate = lite not_excited / full hidden_stat or undetermined; excited = self_excited "
                                  "or excited_r; not_reproduced"}

    lf_orig = lite_full(lambda c: c["set"] == "HE1" and c["rho"] == 0.9 and c.get("N") in (500, 700)
                        and c["regime"] in ("se3", "se11"))
    lf_all = lite_full(lambda c: c["set"] in ("HE1", "H1") and c["rho"] == 0.9 and c["regime"] not in SEPARATE)
    lf_le1 = lite_full(lambda c: c["set"] in ("HE1", "H1") and c["rho"] == 0.9 and c["regime"] == "le1")
    res["lite_vs_full"] = {"original_check_se3_se11_HE1_N500_700": lf_orig,
                           "extended_check_all_overlapping_models": lf_all,
                           "le1_separate": lf_le1, "note": "extended check: " + POOL_NOTE}
    P("=" * 100)
    P(f"2b. LITE vs FULL protocol (same models)   [theory: {REF['lite_full']}]")
    for name, d in (("se3/se11 HE1 N 500,700", lf_orig), ("all overlapping, six regimes (rho 0.9, H1/HE1)", lf_all),
                    ("LE1 overlapping", lf_le1)):
        P(f"   {name}: pairs {d['n_pairs']}, coarse agree {d['n_agree_coarse']}, lite not_excited {d['n_lite_not_excited']} "
          f"-> hidden_stat {d['n_lite_not_excited_full_hidden_stat']}; same model {d['all_same_model_train_rmse']}; "
          f"N {d['N_values']}")
        P(f"      {d['crosstab_lite_to_full']}")

    # ======================================================================== 4. robustness (analyze_robust.py logic)
    X_all = [x for x in rrows if x["reproduced"]]
    X_ = main_only(X_all)                      # pooled fit: se3, se11, chua_I (LE1 separately)
    X_le1 = only(X_all, "le1")
    rsets = sorted({x["setting"] for x in X_} - {"base"})
    rregs = sorted({x["regime"] for x in X_})

    def rdesign(xs, with_set=True):
        names = ["const", "log1p_U"] + [f"regime_{g}" for g in rregs[1:]] + ([f"setting_{s}" for s in rsets] if with_set else [])
        M = [[1.0, math.log1p(x["U"])] + [float(x["regime"] == g) for g in rregs[1:]]
             + ([float(x["setting"] == s) for s in rsets] if with_set else []) for x in xs]
        return np.array(M), names

    yR = np.array([float(x["not_excited"]) for x in X_])
    M1, names1 = rdesign(X_)
    wR, seR, ll1 = logit_fit(M1, yR, ridge=0.0)
    _, _, ll0 = logit_fit(rdesign(X_, False)[0], yR, ridge=0.0)
    settings_tab = {}
    for (g, st) in sorted({(x["regime"], x["setting"]) for x in rrows}):
        xs = [x for x in rrows if x["regime"] == g and x["setting"] == st]
        mods = {x["model"]: x for x in xs}
        settings_tab[f"{g}/{st}"] = {"n_rows": len(xs), "n_models": len(mods),
                                     "U_mean": float(np.mean([x["U"] for x in mods.values()])),
                                     "n_index_sum_1": sum(x["index_sum"] == 1 for x in mods.values()),
                                     "n_reproduced": sum(x["reproduced"] for x in xs),
                                     "n_not_excited": sum(x["not_excited"] for x in xs)}
    # with LE1, as before 2026-11-11 (comparison only)
    rregs_all = sorted({x["regime"] for x in X_all})
    MA = np.array([[1.0, math.log1p(x["U"])] + [float(x["regime"] == g) for g in rregs_all[1:]]
                   + [float(x["setting"] == s) for s in rsets] for x in X_all])
    yA = np.array([float(x["not_excited"]) for x in X_all])
    wA, seA, llA1 = logit_fit(MA, yA, ridge=0.0)
    _, _, llA0 = logit_fit(MA[:, :len(rregs_all) + 1], yA, ridge=0.0)
    res["robustness"] = {
        "description": "stage N lite protocol, node_results/N (CPU), se3/se11/le1/chua_I, HE1, N 300, rho 0.9, eta 1e-7/1e-4, "
                       "seeds 0-4; settings base (input 0.5, density 0.05), input 0.2/1.0, density 0.02/0.2; "
                       "logic of gpu/analyze_robust.py; pooled bins and fit on se3, se11, chua_I (LE1 separately)",
        "regimes": rregs,
        "n_models": sum(1 for r, p in robust_models.values() if r["regime"] not in SEPARATE),
        "n_rows": len(main_only(rrows)),
        "n_models_index_sum_1": sum(1 for r, p in robust_models.values() if r["index_sum"] == 1 and r["regime"] not in SEPARATE),
        "n_rows_reproduced": len(X_), "n_not_excited": int(yR.sum()),
        "ensemble_all_regimes": {"n_models": len(robust_models), "n_rows": len(rrows), "n_rows_reproduced": len(X_all)},
        "settings_table": settings_tab,
        "bins": bin_table(X_, "U", "not_excited"),
        "logit": {"coef": coef_dict(names1, wR, seR), "model": "logit P(not_excited) ~ const + log(1+U) + regime + setting "
                                                               "(base = input 0.5, density 0.05; reference regime chua_I)"},
        "LR_settings_beyond_U": lr_test(ll1, ll0, len(rsets)),
        "le1_separate": {"n_models": sum(1 for r, p in robust_models.values() if r["regime"] == "le1"),
                         "n_rows_reproduced": len(X_le1), "n_not_excited": sum(x["not_excited"] for x in X_le1),
                         "bins": bin_table(X_le1, "U", "not_excited"),
                         "by_setting": {s: [sum(x["not_excited"] for x in X_le1 if x["setting"] == s),
                                            sum(1 for x in X_le1 if x["setting"] == s)] for s in ["base"] + rsets}},
        "sensitivity_with_le1": {"n_rows_reproduced": len(X_all),
                                 "coef": {k: v for k, v in coef_dict(["const", "log1p_U"] + [f"regime_{g}" for g in rregs_all[1:]]
                                                                     + [f"setting_{s}" for s in rsets], wA, seA).items()
                                          if not k.startswith("regime_")},
                                 "LR_settings_beyond_U": lr_test(llA1, llA0, len(rsets))},
    }
    RL = res["robustness"]["logit"]["coef"]
    P("=" * 100)
    P(f"4. ROBUSTNESS (se3, se11, chua_I): {res['robustness']['n_models']} models (sum ind = 1: "
      f"{res['robustness']['n_models_index_sum_1']}), {len(X_)} reproduced rows   [theory: 200 models, 170 rows, with LE1]")
    for k, v in settings_tab.items():
        P(f"   {k:16s} models {v['n_models']:2d}  U mean {v['U_mean']:5.1f}  reproduced {v['n_reproduced']:2d}  "
          f"not excited {v['n_not_excited']:2d}")
    for n_ in names1:
        c = RL[n_]
        P(f"   {n_:16s} {c['coef']:+.2f} +- {c['se']:.2f}  z {c['z']:+.1f}")
    lrS = res["robustness"]["LR_settings_beyond_U"]
    P(f"   LR settings beyond U: chi2 {lrS['chi2']:.1f}, df {lrS['df']}, p {lrS['p']:.3f}   [theory {REF['R_LR']}]")
    P(f"   [theory log(1+U) {REF['R_logU']}; dens0.02 {REF['R_dens0.02']}; in1 {REF['R_in1']}]")

    # ======================================================================== sample decision
    res["sample_decision"] = {
        "main_survival_curve": "N_sweep (stage N lite protocol, six regimes without LE1, sets HE1+H1, N 50-2000, rows "
                               "reproduced with index sum 1); LE1 under le1_separate",
        "check": "stage_C_full_protocol (independent full-protocol labels hidden_stat) and lite_vs_full",
        "robustness_in_sweep": "suffixed robustness runs (_in<x>/_dens<x>) are excluded from the sweep; the base-setting "
                               "robustness runs are plain-named N = 300 models and are merged into the sweep with "
                               "de-duplication against identical A100 models",
    }
    res["notes_vs_theory_index"] = {
        "stage_C": "theory 669 models were counted on 21.10 with N 300 and 1000 only; now +24 models at N 500/700 "
                   "(se3, se11, HE1, the lite-vs-full set) and late N 300/1000 completions",
        "N_sweep": "theory 345 rows were counted on 22.10 before the CPU Newton fix: A100 N 300-2000 for all groups plus "
                   "pre-fix CPU N 50-200 for only chua_I, le1, se3/HE1, se11/HE1 (N 150 with index sum != 1 in many "
                   "models). Now: CPU N 50/100/200 recomputed after the fix for all 9 (regime, set) groups, N 150 from "
                   "the A100 recompute, plus 16 base-setting N 300 CPU models from the robustness experiment",
        "U_of_N": "wangchen2_a0.03 had only N >= 300 on 22.10 (monotone); with N 50-200 it is U-shaped",
    }

    # ======================================================================== outputs
    OUT.mkdir(parents=True, exist_ok=True)
    res["meta"]["runtime_s"] = round(time.time() - t_start, 1)
    res["meta"]["skipped_files"] = dict(SKIPPED)
    (OUT / "esn_law.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    cols = ["sample", "stage", "regime", "set", "N", "input_scale", "density", "seed", "target", "K", "U", "index_sum",
            "reproduced", "not_excited", "label", "eta", "rho", "setting", "model", "source", "U_in_lo", "U_in_hi", "U_in_C"]
    with open(OUT / "esn_rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for x in sorted(srows, key=lambda x: (x["regime"], x["set"], x["N"], x["eta"], x["seed"], x["target"])) + \
                sorted(rrows, key=lambda x: (x["regime"], x["setting"], x["eta"], x["seed"], x["target"])):
            w.writerow({**x, "reproduced": int(x["reproduced"]), "not_excited": int(x["not_excited"]),
                        "label": x["label"] or "", "U_in_C": "" if x.get("U_in_C") is None else x["U_in_C"]})
    ccols = ["regime", "set", "N", "rho", "eta", "seed", "target", "K", "U", "U_in", "U_out", "label", "hidden",
             "exc_true", "exc_spur", "model"]
    with open(OUT / "esn_C_rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ccols, extrasaction="ignore")
        w.writeheader()
        for x in sorted(crows, key=lambda x: (x["regime"], x["set"], x["N"], x["rho"], x["eta"], x["seed"])):
            w.writerow({**x, "hidden": int(x["hidden"])})
    ucols = ["regime", "set", "N", "n_models", "U_mean", "U_sd", "U_sem", "U_median", "U_min", "U_max", "K_mean",
             "n_index_sum_1", "n_rows", "n_rows_reproduced", "n_rows_law", "n_not_excited_law"]
    with open(OUT / "esn_UN.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ucols)
        w.writeheader()
        for x in un_rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()})
    P("=" * 100)
    P(f"written: {OUT / 'esn_law.json'}, esn_rows.csv ({len(srows) + len(rrows)} rows), esn_UN.csv ({len(un_rows)} rows); "
      f"{res['meta']['runtime_s']} s")


if __name__ == "__main__":
    sys.exit(main())

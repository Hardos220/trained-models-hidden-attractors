"""Paper numbers for the model classes of stage M (mlp, mlp2, res on H1, HE1, E1; 7 regimes; widths 8-1024; 20 seeds).

Same data and definitions as analyze_m2.py (U_in = unstable equilibria inside the true region box_lo..box_hi; a row is a
(model, hidden attractor) pair; "not excited" = label not_excited), written to paper2_esn/chaos/numbers/classes.json:
  counts, index sums per class and set, survival by U_in bins with Wilson 95% intervals, mean U and U_in by class, set and
  width, logistic regression (log(1+U_in), class, set, log width, regime fixed effects) with likelihood-ratio tests,
  per-class slopes, E1 reproductions per regime and class; and classes_rows.csv (one row per pair) for the figures.

Pooled statistics (survival, regression, slopes, mean U and U_in by class, set and width, reproduction shares, E1 counts
over regimes) use the six main regimes of paper_common.MAIN_REGIMES; LE1 is reported under "le1_separate". The index
sums (a check of Theorem 1 on the models, not a statistic of hiddenness) and the Chua I checks keep all models. The row
tables classes_rows.csv, classes_eq_rows.csv and classes_models.csv keep all seven regimes; classes_eq_rows.csv carries
dist_true_eq (column dist) for the copy-threshold sensitivity of paper_robustness.py.

    python -X utf8 paper2_esn/gpu/paper_classes.py
"""
import csv
import json
import sys
from collections import Counter, defaultdict
import os
from pathlib import Path

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_common import COPY_TOL, MAIN_REGIMES, POOL_NOTE, main_only, only  # noqa: E402

ROOT = Path(os.environ.get("ESN_DATA", Path(__file__).resolve().parents[2] / "local" / "esn-gpu"))
OUT = Path(__file__).resolve().parents[1] / "chaos" / "numbers"
CLS, SETS = ("mlp", "mlp2", "res"), ("H1", "HE1", "E1")
BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 10 ** 6)]
R_MAX = 1e3   # physical units; the true regions are within |x| <= 15


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def fit(X, y):
    f = lambda w: np.sum(np.logaddexp(0, X @ w) - y * (X @ w)) + 1e-4 * w @ w  # noqa: E731
    r = optimize.minimize(f, np.zeros(X.shape[1]), method="BFGS")
    p = 1 / (1 + np.exp(-X @ r.x))
    Hs = (X * (p * (1 - p))[:, None]).T @ X
    return r.x, np.sqrt(np.diag(np.linalg.pinv(Hs))), r.fun


def main():
    meta, rows, models, seen, eq_rows = {}, [], [], set(), []
    for f in sorted((ROOT / "gpuq-results").rglob("M/*/*.json")):
        key = f.parent.name + "/" + f.stem
        if key in seen:
            continue
        seen.add(key)
        r = json.loads(f.read_text(encoding="utf-8"))
        if r["class"] not in CLS:
            continue
        reg = r["regime"]
        m = meta.setdefault(reg, json.loads((ROOT / "data" / f"{reg}.json").read_text(encoding="utf-8")))
        lo, hi = np.array(m["box_lo"]), np.array(m["box_hi"])
        # genuine equilibria: Newton runaways of residual networks (|x| ~ 1e13-1e21, tanh saturated) are dropped
        fps = [(i, x) for i, x in enumerate(r["fixed_points"]) if np.linalg.norm(x["x"]) <= R_MAX]
        n_runaway = len(r["fixed_points"]) - len(fps)
        inside = lambda x: bool(np.all(np.array(x["x"]) >= lo) and np.all(np.array(x["x"]) <= hi))  # noqa: E731
        U = sum(1 for _, x in fps if x["n_unstable"] > 0)
        Un = sum(1 for _, x in fps if x["n_unstable"] > 0 and inside(x))
        isum = sum(x["index"] for _, x in fps)
        near = len({x.get("nearest_true_eq") for _, x in fps if x.get("dist_true_eq", 9) < 0.01})
        models.append(dict(cls=r["class"], set=r["set"], reg=reg, H=r["H"], U=U, Un=Un, K=len(fps), ok=isum == 1,
                           near=near, runaway=n_runaway, seed=r["seed"],
                           n_spur=sum(1 for _, x in fps if x.get("dist_true_eq", 9) >= 0.01),
                           n_spur_stable=sum(1 for _, x in fps if x.get("dist_true_eq", 9) >= 0.01 and x["n_unstable"] == 0),
                           spur_index=sum(x["index"] for _, x in fps if x.get("dist_true_eq", 9) >= 0.01)))
        rc = r.get("radius_counts", {})
        for y, mm in r["hidden_match"].items():
            rows.append(dict(cls=r["class"], set=r["set"], reg=reg, H=r["H"], seed=r["seed"], att=y, U=U, Un=Un,
                             ok=isum == 1, rep=bool(mm.get("reproduced")), ne=mm.get("label") == "not_excited",
                             label=mm.get("label"), dA=mm.get("delta_att"), dH=mm.get("d_H"), mse=r.get("train_mse")))
            a = mm.get("attractor")
            if not mm.get("reproduced") or not a:
                continue
            for i, x in fps:
                if x["n_unstable"] == 0:
                    continue
                hits = [float(k.split("|")[1]) for k, cnt in rc.items() if k.split("|")[0] == str(i) and cnt.get(a, 0) > 0]
                eq_rows.append(dict(cls=r["class"], set=r["set"], reg=reg, H=r["H"], seed=r["seed"], att=y, inside=int(inside(x)),
                                    copy=int(x.get("dist_true_eq", 9) < COPY_TOL), dim_u=x["n_unstable"],
                                    index=x["index"], excited=int(bool(hits)), r_min=min(hits) if hits else "",
                                    dist=round(float(x.get("dist_true_eq", 9)), 6)))
    models_all, rows_all = models, rows
    models, rows = main_only(models_all, "reg"), main_only(rows_all, "reg")      # pooled statistics: six regimes
    out = {"pooling": POOL_NOTE, "n_models": len(models), "n_rows": len(rows), "regimes": sorted(MAIN_REGIMES),
           "ensemble_all_regimes": {"n_models": len(models_all), "n_rows": len(rows_all), "regimes": sorted(meta)}}

    # index sums (check of Theorem 1, all seven regimes) and mean U per class and set (per model, six regimes)
    out["index_sum_eq1"] = {c: {s: [sum(x["ok"] for x in models_all if x["cls"] == c and x["set"] == s),
                                    sum(1 for x in models_all if x["cls"] == c and x["set"] == s)] for s in SETS} for c in CLS}
    out["index_sum_eq1_note"] = "all seven regimes (check of Theorem 1 on the models)"
    out["mean_U"] = {c: {s: round(float(np.mean([x["U"] for x in models if x["cls"] == c and x["set"] == s])), 2)
                         for s in SETS} for c in CLS}
    out["mean_U_in"] = {c: {s: round(float(np.mean([x["Un"] for x in models if x["cls"] == c and x["set"] == s])), 2)
                            for s in SETS} for c in CLS}
    widths = sorted({x["H"] for x in models})
    out["mean_U_in_by_width"] = {c: {s: {str(h): round(float(np.mean([x["Un"] for x in models if x["cls"] == c
                                                                         and x["set"] == s and x["H"] == h])), 2)
                                         for h in widths} for s in SETS} for c in CLS}
    out["mean_U_by_width"] = {c: {s: {str(h): round(float(np.mean([x["U"] for x in models if x["cls"] == c
                                                                       and x["set"] == s and x["H"] == h])), 2)
                                      for h in widths} for s in SETS} for c in CLS}
    out["with_le1_reference"] = {
        "note": "seven regimes, as before 2026-11-11 (descriptive equilibrium counts, for comparison only)",
        "mean_U": {c: {s: round(float(np.mean([x["U"] for x in models_all if x["cls"] == c and x["set"] == s])), 2)
                       for s in SETS} for c in CLS},
        "mean_U_in": {c: {s: round(float(np.mean([x["Un"] for x in models_all if x["cls"] == c and x["set"] == s])), 2)
                          for s in SETS} for c in CLS},
        "mean_U_in_by_width_HE1": {c: {str(h): round(float(np.mean([x["Un"] for x in models_all if x["cls"] == c
                                                                    and x["set"] == "HE1" and x["H"] == h])), 2)
                                       for h in widths} for c in CLS},
        "mean_U_by_width_HE1": {c: {str(h): round(float(np.mean([x["U"] for x in models_all if x["cls"] == c
                                                                 and x["set"] == "HE1" and x["H"] == h])), 2)
                                    for h in widths} for c in CLS}}

    # Proposition 1 for Chua I (s = -1): models with copies (distance < 0.01) of all three true equilibria
    out["chua_I_copies_of_all_three_equilibria"] = {
        c: [sum(1 for x in models if x["cls"] == c and x["reg"] == "chua_I" and x["near"] == 3),
            sum(1 for x in models if x["cls"] == c and x["reg"] == "chua_I")] for c in CLS}

    out["runaway_points_dropped"] = {c: sum(x["runaway"] for x in models if x["cls"] == c) for c in CLS}
    out["chua_I_spurious_when_two_copies"] = {
        c: {"n_models": sum(1 for x in models if x["cls"] == c and x["reg"] == "chua_I" and x["near"] == 2),
            "spurious_index_sum": dict(__import__("collections").Counter(x["spur_index"] for x in models
                                                                           if x["cls"] == c and x["reg"] == "chua_I" and x["near"] == 2 and x["ok"])),
            "n_spurious_mean": round(float(np.mean([x["n_spur"] for x in models if x["cls"] == c and x["reg"] == "chua_I" and x["near"] == 2] or [np.nan])), 2),
            "n_spurious_stable_mean": round(float(np.mean([x["n_spur_stable"] for x in models if x["cls"] == c and x["reg"] == "chua_I" and x["near"] == 2] or [np.nan])), 2)}
        for c in CLS}

    # reproduction and survival
    out["reproduced"] = {c: {s: [sum(x["rep"] for x in rows if x["cls"] == c and x["set"] == s),
                                 sum(1 for x in rows if x["cls"] == c and x["set"] == s)] for s in SETS} for c in CLS}
    out["not_excited_among_reproduced"] = {c: {s: [sum(x["ne"] for x in rows if x["cls"] == c and x["set"] == s and x["rep"]),
                                                   sum(x["rep"] for x in rows if x["cls"] == c and x["set"] == s)]
                                               for s in SETS} for c in CLS}
    surv = {}
    for c in CLS + ("all",):
        D = [x for x in rows if x["rep"] and (c == "all" or x["cls"] == c)]
        surv[c] = []
        for a, b in BINS:
            q = [x for x in D if a <= x["Un"] <= b]
            k = sum(x["ne"] for x in q)
            surv[c].append({"bin": f"{a}-{b if b < 10 ** 6 else 'inf'}", "n": len(q), "not_excited": k,
                            "share": round(k / len(q), 4) if q else None, "wilson95": wilson(k, len(q))})
    out["survival_by_U_in"] = surv
    out["survival_by_U_in_and_set"] = {
        s: [{"bin": f"{a}-{b if b < 10 ** 6 else 'inf'}",
             "n": len([x for x in rows if x["rep"] and x["set"] == s and a <= x["Un"] <= b]),
             "not_excited": sum(x["ne"] for x in rows if x["rep"] and x["set"] == s and a <= x["Un"] <= b)} for a, b in BINS]
        for s in SETS}

    # logistic regression
    D = [x for x in rows if x["rep"]]
    y = np.array([x["ne"] for x in D], float)
    regs = sorted({x["reg"] for x in D})
    U = np.array([[np.log1p(x["Un"])] for x in D])
    extra = np.array([[float(x["cls"] == "mlp2"), float(x["cls"] == "res"), float(x["set"] == "HE1"),
                       float(x["set"] == "E1"), np.log(x["H"])] for x in D])
    cls_only = extra[:, :2]
    R = np.array([[float(x["reg"] == g) for g in regs] for x in D])
    w, se, f_full = fit(np.hstack([U, extra, R]), y)
    _, _, f_noU = fit(np.hstack([extra, R]), y)
    _, _, f_Uonly = fit(np.hstack([U, R]), y)
    _, _, f_Ucls = fit(np.hstack([U, cls_only, R]), y)
    names = ["log1p_U_in", "class_mlp2", "class_res", "set_HE1", "set_E1", "log_width"]
    lr = lambda a, b: 2 * (a - b)  # noqa: E731
    out["regression"] = {
        "n": len(D), "coef": {nm: [round(float(wi), 3), round(float(si), 3)] for nm, wi, si in zip(names, w, se)},
        "LR_U": [round(lr(f_noU, f_full), 1), float(stats.chi2.sf(lr(f_noU, f_full), 1))],
        "LR_class_set_width_beyond_U_df5": [round(lr(f_Uonly, f_full), 1), float(stats.chi2.sf(lr(f_Uonly, f_full), 5))],
        "LR_class_beyond_U_df2": [round(lr(f_Uonly, f_Ucls), 1), float(stats.chi2.sf(lr(f_Uonly, f_Ucls), 2))],
    }
    slopes = {}
    for c in CLS:
        idx = [i for i, x in enumerate(D) if x["cls"] == c]
        wc, sc, _ = fit(np.hstack([U, R])[idx], y[idx])
        slopes[c] = [round(float(wc[0]), 3), round(float(sc[0]), 3), len(idx)]
    out["slope_by_class"] = slopes

    # E1: reproductions per regime (all seven, LE1 included) and class
    out["E1_reproduced"] = {g: {c: [sum(x["rep"] for x in rows_all if x["reg"] == g and x["cls"] == c and x["set"] == "E1"),
                                    sum(1 for x in rows_all if x["reg"] == g and x["cls"] == c and x["set"] == "E1")] for c in CLS}
                            for g in sorted({x["reg"] for x in rows_all})}
    out["E1_reproduced_by_attractor_chua_I"] = {a: sum(x["rep"] for x in rows if x["reg"] == "chua_I" and x["set"] == "E1" and x["att"] == a)
                                               for a in ("HA_a", "HA_b")}
    far = ("chua_II", "wangchen2", "wangchen2_a0.03", "se11")     # at most 1% of the attractor within 0.1 of E1 (six regimes)
    out["E1_far_regimes"] = {"regimes": far,
                             "reproduced": sum(x["rep"] for x in rows if x["set"] == "E1" and x["reg"] in far),
                             "models": sum(1 for x in models if x["set"] == "E1" and x["reg"] in far)}

    # LE1 separately (weakly hidden control)
    R1 = only(rows_all, "le1", "reg")
    M1 = only(models_all, "le1", "reg")
    E1q = [x for x in eq_rows if x["reg"] == "le1" and x["inside"] == 1]
    surv1 = {}
    for c in CLS + ("all",):
        Dc = [x for x in R1 if x["rep"] and (c == "all" or x["cls"] == c)]
        surv1[c] = [{"bin": f"{a}-{b if b < 10 ** 6 else 'inf'}", "n": len(q), "not_excited": sum(x["ne"] for x in q),
                     "share": round(sum(x["ne"] for x in q) / len(q), 4) if q else None, "wilson95": wilson(sum(x["ne"] for x in q), len(q))}
                    for a, b in BINS for q in [[x for x in Dc if a <= x["Un"] <= b]]]
    out["le1_separate"] = {
        "note": "LE1: attractor reached from a segment of the unstable half-line of true equilibria (hidden only in the "
                "weak sense); copies = model equilibria within 0.01 L of the line x = y = 0",
        "n_models": len(M1), "n_rows": len(R1),
        "reproduced": {c: {s: [sum(x["rep"] for x in R1 if x["cls"] == c and x["set"] == s),
                               sum(1 for x in R1 if x["cls"] == c and x["set"] == s)] for s in SETS} for c in CLS},
        "not_excited_among_reproduced": {c: {s: [sum(x["ne"] for x in R1 if x["cls"] == c and x["set"] == s and x["rep"]),
                                                 sum(x["rep"] for x in R1 if x["cls"] == c and x["set"] == s)] for s in SETS}
                                         for c in CLS},
        "survival_by_U_in": surv1,
        "mean_U": {c: {s: round(float(np.mean([x["U"] for x in M1 if x["cls"] == c and x["set"] == s])), 2) for s in SETS} for c in CLS},
        "mean_U_in": {c: {s: round(float(np.mean([x["Un"] for x in M1 if x["cls"] == c and x["set"] == s])), 2) for s in SETS} for c in CLS},
        "runaway_points_dropped": {c: sum(x["runaway"] for x in M1 if x["cls"] == c) for c in CLS},
        "E1": {"reproduced": sum(x["rep"] for x in R1 if x["set"] == "E1"), "models": sum(1 for x in M1 if x["set"] == "E1")},
        "unstable_equilibria_in_Q": {k: {"n": len(q), "excite": sum(x["excited"] for x in q),
                                         "share": round(sum(x["excited"] for x in q) / len(q), 4) if q else None}
                                     for k, q in (("copies_of_line", [x for x in E1q if x["copy"] == 1]),
                                                  ("spurious", [x for x in E1q if x["copy"] == 0]))},
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "classes.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    with open(OUT / "classes_rows.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows_all[0]))
        wr.writeheader()
        wr.writerows(rows_all)
    with open(OUT / "classes_eq_rows.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(eq_rows[0]))
        wr.writeheader()
        wr.writerows(eq_rows)
    with open(OUT / "classes_models.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(models_all[0]))
        wr.writeheader()
        wr.writerows(models_all)
    print(json.dumps({k: out[k] for k in ("n_models", "n_rows", "index_sum_eq1", "mean_U_in", "survival_by_U_in",
                                          "regression", "slope_by_class", "E1_far_regimes", "le1_separate")}, indent=1)[:8000])


if __name__ == "__main__":
    main()

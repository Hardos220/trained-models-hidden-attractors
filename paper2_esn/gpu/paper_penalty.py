"""Paper numbers for stage P: training with a penalty on unstable equilibria (lam = 0, 0.1, 1) at fixed data.

Models: two-hidden-layer perceptrons mlp2 (H = 256) on H1 and HE1 in the seven regimes, 20 seeds (the design after the
fix of 2026-11-10; other classes and widths in the result folders are skipped); for a given
(regime, set, class, width, seed) the models with different lam share the initialization and the batches, so they are
paired. Definitions as in paper_classes.py: U_in = unstable equilibria inside the true region box_lo..box_hi (Newton
runaways |x| > 1e3 dropped), a row is a (model, hidden attractor) pair, "not excited" = light-protocol label not_excited.

Written to paper2_esn/results/numbers/penalty.json and penalty_rows.csv:
  per lam and class: mean U_in and U, share with U_in = 0, median training error, index sums, reproduction, survival
  (not excited | reproduced) with Wilson intervals; paired changes (U_in, training error, McNemar for survival);
  calibration: survival predicted by the stage M law (logistic fit on classes_rows.csv with the same covariates) from the
  new U_in versus observed survival, per lam; within-stage logistic fit with lam dummies and a likelihood-ratio test for
  lam beyond U_in.
Pooled statistics use the six main regimes (paper_common.MAIN_REGIMES); LE1 is reported under "le1_separate" and the
seven-regime values under "with_le1_reference"; penalty_rows.csv keeps all seven regimes.

    python -X utf8 paper2_esn/gpu/paper_penalty.py
"""
import csv
import json
import sys
from collections import defaultdict
import os
from pathlib import Path

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_common import POOL_NOTE, SEPARATE, main_only, only  # noqa: E402

ROOT = Path(os.environ.get("ESN_DATA", Path(__file__).resolve().parents[2] / "local" / "esn-gpu"))
OUT = Path(__file__).resolve().parents[1] / "results" / "numbers"
R_MAX = 1e3
BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 10 ** 6)]
DESIGN = {("mlp2", 256)}


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


def load_models():
    meta, models, rows, seen = {}, [], [], set()
    files = sorted((ROOT / "gpuq-results").rglob("P/*/*.json")) + sorted((ROOT / "node_results").rglob("P/*/*.json"))
    for f in files:
        if ".failed" in f.name:
            continue
        key = f.parent.name + "/" + f.stem
        if key in seen:
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        if "hidden_match" not in r:          # train-only tuning runs
            continue
        if (r["class"], r["H"]) not in DESIGN:  # mlp H = 64 groups left from the first chain (job 6040) are unpaired
            continue
        seen.add(key)
        reg = r["regime"]
        m = meta.setdefault(reg, json.loads((ROOT / "data" / f"{reg}.json").read_text(encoding="utf-8")))
        lo, hi = np.array(m["box_lo"]), np.array(m["box_hi"])
        fps = [x for x in r["fixed_points"] if np.linalg.norm(x["x"]) <= R_MAX]
        inside = lambda x: bool(np.all(np.array(x["x"]) >= lo) and np.all(np.array(x["x"]) <= hi))  # noqa: E731
        U = sum(1 for x in fps if x["n_unstable"] > 0)
        Un = sum(1 for x in fps if x["n_unstable"] > 0 and inside(x))
        rho_in = max([x["rho"] for x in fps if x["n_unstable"] > 0 and inside(x)], default=0.0)
        isum = sum(x["index"] for x in fps)
        pair = (reg, r["set"], r["class"], r["H"], r["seed"])
        mod = dict(reg=reg, set=r["set"], cls=r["class"], H=r["H"], seed=r["seed"], lam=float(r["lam"]), U=U, Un=Un,
                   K=len(fps), ok=isum == 1, mse=float(r["train_mse"]), pair=pair)
        models.append(mod)
        for y, mm in r["hidden_match"].items():
            rows.append(dict(reg=reg, set=r["set"], cls=r["class"], H=r["H"], seed=r["seed"], lam=float(r["lam"]), att=y,
                             U=U, Un=Un, rho_max_in=rho_in, ok=isum == 1, rep=bool(mm.get("reproduced")),
                             ne=mm.get("label") == "not_excited", label=mm.get("label"), mse=float(r["train_mse"])))
    return models, rows


def stage_m_law():
    """Logistic law of stage M (paper_classes.py, full model): not excited ~ log1p(U_in) + class + set + log width + regime."""
    D = []
    with open(OUT / "classes_rows.csv", encoding="utf-8") as fh:
        for x in csv.DictReader(fh):
            if x["rep"] == "True" and x["reg"] not in SEPARATE:       # six regimes
                D.append(x)
    regs = sorted({x["reg"] for x in D})

    def design(xs):
        return np.array([[np.log1p(float(x["Un"])), float(x["cls"] == "mlp2"), float(x["cls"] == "res"),
                          float(x["set"] == "HE1"), float(x["set"] == "E1"), np.log(float(x["H"]))]
                         + [float(x["reg"] == g) for g in regs] for x in xs])
    y = np.array([x["ne"] == "True" for x in D], float)
    w, se, _ = fit(design(D), y)
    return w, se, design, len(D)


def per_lam(models, rows, lams, classes):
    """Per lam and class: equilibria, training error, index sums, reproduction and survival."""
    per = {}
    for c in classes + ["all"]:
        per[c] = {}
        for l in lams:
            M = [m for m in models if m["lam"] == l and (c == "all" or m["cls"] == c)]
            R = [x for x in rows if x["lam"] == l and (c == "all" or x["cls"] == c)]
            Rr = [x for x in R if x["rep"]]
            k = sum(x["ne"] for x in Rr)
            per[c][str(l)] = dict(
                n_models=len(M), mean_U_in=round(float(np.mean([m["Un"] for m in M])), 3) if M else None,
                mean_U=round(float(np.mean([m["U"] for m in M])), 3) if M else None,
                share_U_in_0=round(float(np.mean([m["Un"] == 0 for m in M])), 4) if M else None,
                median_mse=float(np.median([m["mse"] for m in M])) if M else None,
                index_sum_1=[sum(m["ok"] for m in M), len(M)],
                reproduced=[len(Rr), len(R)], not_excited=[k, len(Rr)],
                share_not_excited=round(k / len(Rr), 4) if Rr else None, wilson95=wilson(k, len(Rr)))
    return per


def survival_bins(rows, lams):
    return {str(l): [{"bin": f"{a}-{b if b < 10 ** 6 else 'inf'}",
                      "n": len([x for x in rows if x["rep"] and x["lam"] == l and a <= x["Un"] <= b]),
                      "not_excited": sum(x["ne"] for x in rows if x["rep"] and x["lam"] == l and a <= x["Un"] <= b)}
                     for a, b in BINS] for l in lams}


def paired_vs_lam0(models, rows, lams):
    """Paired comparisons against lam = 0 (same regime, set, class, width and seed)."""
    by_pair = defaultdict(dict)
    for m in models:
        by_pair[m["pair"]][m["lam"]] = m
    row_by = {(x["reg"], x["set"], x["cls"], x["H"], x["seed"], x["att"], x["lam"]): x for x in rows}
    paired = {}
    for l in lams:
        if l == 0:
            continue
        P = [(d[0.0], d[l]) for d in by_pair.values() if 0.0 in d and l in d]
        if not P:
            continue
        dU = np.array([b["Un"] - a["Un"] for a, b in P], float)
        ratio = np.array([b["mse"] / a["mse"] for a, b in P])
        wil = stats.wilcoxon(dU) if np.any(dU != 0) else None
        # survival: pairs (model, attractor) reproduced under both lam
        both = []
        for x in rows:
            if x["lam"] != 0.0 or not x["rep"]:
                continue
            z = row_by.get((x["reg"], x["set"], x["cls"], x["H"], x["seed"], x["att"], l))
            if z is not None and z["rep"]:
                both.append((x["ne"], z["ne"]))
        b01 = sum(1 for a, b in both if not a and b)    # excited without penalty, not excited with it
        b10 = sum(1 for a, b in both if a and not b)
        mcn = float(stats.binomtest(b01, b01 + b10).pvalue) if b01 + b10 else None
        rep0 = sum(1 for x in rows if x["lam"] == 0.0 and x["rep"])
        repl = sum(1 for x in rows if x["lam"] == l and x["rep"])
        paired[str(l)] = dict(
            n_pairs=len(P), mean_dU_in=round(float(dU.mean()), 3), share_U_in_down=round(float(np.mean(dU < 0)), 4),
            share_U_in_up=round(float(np.mean(dU > 0)), 4), n_U_in_up=int(np.sum(dU > 0)), n_U_in_down=int(np.sum(dU < 0)),
            wilcoxon_p=float(wil.pvalue) if wil is not None else None,
            median_mse_ratio=round(float(np.median(ratio)), 4), q90_mse_ratio=round(float(np.quantile(ratio, 0.9)), 4),
            reproduced_rows_lam0=rep0, reproduced_rows_lam=repl,
            survival_pairs_reproduced_both=len(both), not_excited_lam0=sum(a for a, _ in both),
            not_excited_lam=sum(b for _, b in both), gained=b01, lost=b10, mcnemar_p=mcn)
    return paired


def rho_tables(rows, lams):
    """Spectral radii of the unstable equilibria in Q (largest per reproduced pair), by lam and excitation."""
    rq = defaultdict(list)
    for x in rows:
        if x["rep"] and x["Un"] >= 1:
            rq[(x["lam"], "not_excited" if x["ne"] else "excited")].append(x["rho_max_in"])
    by_rho = {
        f"{l:g}": {lab: [sum(x["ne"] for x in rows if x["rep"] and x["lam"] == l and x["Un"] >= 1 and f(x["rho_max_in"])),
                         sum(1 for x in rows if x["rep"] and x["lam"] == l and x["Un"] >= 1 and f(x["rho_max_in"]))]
                   for lab, f in (("rho_max<1.05", lambda r: r < 1.05), ("rho_max>=1.05", lambda r: r >= 1.05))}
        for l in lams}
    dist = {f"{l:g}|{e}": dict(n=len(v), q10_q50_q90=[round(float(q), 4) for q in np.quantile(v, [0.1, 0.5, 0.9])],
                               share_below_1_05=round(float(np.mean(np.array(v) < 1.05)), 3))
            for (l, e), v in sorted(rq.items(), key=str)}
    return by_rho, dist


def main():
    models_all, rows_all = load_models()
    if not models_all:
        print("no stage P results yet")
        return
    models, rows = main_only(models_all, "reg"), main_only(rows_all, "reg")     # pooled statistics: six regimes
    M1, R1 = only(models_all, "le1", "reg"), only(rows_all, "le1", "reg")
    lams = sorted({m["lam"] for m in models_all})
    classes = sorted({m["cls"] for m in models_all})
    all_regs = sorted({m["reg"] for m in models_all})
    out = {"pooling": POOL_NOTE, "n_models": len(models), "n_rows": len(rows), "lams": lams,
           "regimes": sorted({m["reg"] for m in models}),
           "ensemble_all_regimes": {"n_models": len(models_all), "n_rows": len(rows_all), "regimes": all_regs},
           "n_models_by_regime_lam": {g: {str(l): sum(1 for m in models_all if m["reg"] == g and m["lam"] == l) for l in lams}
                                      for g in all_regs}}

    out["by_lam"] = per_lam(models, rows, lams, classes)
    out["survival_by_U_in_and_lam"] = survival_bins(rows, lams)
    out["paired_vs_lam0"] = paired_vs_lam0(models, rows, lams)

    # calibration against the stage M law (fitted on the six regimes)
    w, se, design, nM = stage_m_law()
    cal = {}
    for l in lams:
        D = [x for x in rows if x["rep"] and x["lam"] == l]
        if not D:
            continue
        p = 1 / (1 + np.exp(-design(D) @ w))
        O, E = sum(x["ne"] for x in D), float(p.sum())
        V = float((p * (1 - p)).sum())
        cal[str(l)] = dict(n=len(D), observed=int(O), predicted=round(E, 1), z=round((O - E) / np.sqrt(V), 2) if V > 0 else None,
                           share_observed=round(O / len(D), 4), share_predicted=round(E / len(D), 4))
    out["calibration_stage_M_law"] = {"law_n_rows": nM, "law_coef_log1p_U_in": [round(float(w[0]), 3), round(float(se[0]), 3)],
                                      "by_lam": cal}

    # empirical calibration: the logistic law underestimates survival at U_in = 0 (0.99 observed in stage M), so compare
    # with the observed survival of stage M two-layer perceptrons in the same regime and U_in bin (pooled over the six
    # regimes when a cell has fewer than 30 rows); p = (k + 0.5) / (n + 1)
    def b_of(u):
        return next(i for i, (a, b) in enumerate(BINS) if a <= u <= b)
    cnt = defaultdict(lambda: [0, 0])
    with open(OUT / "classes_rows.csv", encoding="utf-8") as fh:
        for x in csv.DictReader(fh):
            if x["rep"] != "True" or x["cls"] != "mlp2" or x["reg"] in SEPARATE:
                continue
            bi = b_of(int(float(x["Un"])))
            for k in ((x["reg"], bi), ("*", bi)):
                cnt[k][0] += x["ne"] == "True"
                cnt[k][1] += 1
    emp = {}
    for l in lams:
        D = [x for x in rows if x["rep"] and x["lam"] == l]
        if not D:
            continue
        E = V = 0.0
        for x in D:
            k, n = cnt.get((x["reg"], b_of(x["Un"])), (0, 0))
            if n < 30:
                k, n = cnt[("*", b_of(x["Un"]))]
            p = (k + 0.5) / (n + 1)
            E, V = E + p, V + p * (1 - p)
        O = sum(x["ne"] for x in D)
        emp[str(l)] = dict(n=len(D), observed=int(O), expected=round(E, 1), z=round((O - E) / np.sqrt(V), 2) if V > 0 else None,
                           share_observed=round(O / len(D), 4), share_expected=round(E / len(D), 4))
    out["calibration_stage_M_empirical"] = {
        "stage_M_mlp2_pooled_by_bin": {f"{a}-{b if b < 10 ** 6 else 'inf'}": cnt[("*", i)] for i, (a, b) in enumerate(BINS)},
        "by_lam": emp}

    # within-stage logistic fit: lam beyond U_in (six regimes; seven with LE1 for comparison, LE1 being the regime whose
    # line of equilibria the models approximate by chains of nearly neutral equilibria)
    nz = [l for l in lams if l != 0]

    def logit_within(D):
        y = np.array([x["ne"] for x in D], float)
        regs = sorted({x["reg"] for x in D})
        # log(1 + U_in) plus a step at U_in = 0: without the step the lam dummies absorb the misfit of the logistic curve
        # at U_in = 0, where almost all penalized models lie (Corollary 2: no unstable equilibrium in Q)
        Ucol = np.array([[np.log1p(x["Un"]), float(x["Un"] == 0)] for x in D])
        Lcol = np.array([[float(x["lam"] == l) for l in nz] for x in D])
        C = np.array([[float(x["set"] == "HE1")] for x in D])
        R = np.array([[float(x["reg"] == g) for g in regs] for x in D])
        if len(D) <= 20 or len(set(y)) < 2:
            return None
        wf, sf, f_full = fit(np.hstack([Ucol, Lcol, C, R]), y)
        _, _, f_noL = fit(np.hstack([Ucol, C, R]), y)
        wl, sl, f_noU = fit(np.hstack([Lcol, C, R]), y)
        lr_L, lr_U = 2 * (f_noL - f_full), 2 * (f_noU - f_full)
        return dict(
            n=len(D), coef_log1p_U_in=[round(float(wf[0]), 3), round(float(sf[0]), 3)],
            coef_U_in_zero=[round(float(wf[1]), 3), round(float(sf[1]), 3)],
            coef_lam_with_U={str(l): [round(float(wf[2 + i]), 3), round(float(sf[2 + i]), 3)] for i, l in enumerate(nz)},
            coef_lam_without_U={str(l): [round(float(wl[i]), 3), round(float(sl[i]), 3)] for i, l in enumerate(nz)},
            LR_lam_beyond_U=[round(lr_L, 2), float(stats.chi2.sf(lr_L, len(nz)))],
            LR_U_beyond_lam=[round(lr_U, 2), float(stats.chi2.sf(lr_U, 2))])
    out["logit_within"] = logit_within([x for x in rows if x["rep"]])
    out["logit_within_with_le1"] = logit_within([x for x in rows_all if x["rep"]])

    # reproductions and survival by regime (all seven, LE1 included) and lam; survival with U_in >= 1 by lam
    out["by_regime_lam"] = {
        g: {str(l): dict(reproduced=[sum(x["rep"] for x in rows_all if x["reg"] == g and x["lam"] == l),
                                     sum(1 for x in rows_all if x["reg"] == g and x["lam"] == l)],
                         not_excited=sum(x["ne"] for x in rows_all if x["reg"] == g and x["lam"] == l and x["rep"]),
                         mean_U_in=round(float(np.mean([m["Un"] for m in models_all if m["reg"] == g and m["lam"] == l])), 3))
            for l in lams} for g in all_regs}

    def ge1(R, l):
        return [sum(x["ne"] for x in R if x["rep"] and x["lam"] == l and x["Un"] >= 1),
                sum(1 for x in R if x["rep"] and x["lam"] == l and x["Un"] >= 1)]
    out["survival_U_in_ge1"] = {str(l): {"six_regimes": ge1(rows, l), "le1": ge1(R1, l), "with_le1": ge1(rows_all, l)}
                                for l in lams}
    out["survival_U_in_ge1_by_rho"], out["rho_unstable_in_Q"] = rho_tables(rows, lams)

    # LE1 separately (weakly hidden control)
    le1_rho, le1_rho_dist = rho_tables(R1, lams)
    out["le1_separate"] = {
        "note": "LE1: the true attractor is reached from a segment of the unstable half-line of equilibria",
        "n_models": len(M1), "n_rows": len(R1),
        "by_lam": per_lam(M1, R1, lams, classes)["all"],
        "survival_by_U_in_and_lam": survival_bins(R1, lams),
        "paired_vs_lam0": paired_vs_lam0(M1, R1, lams),
        "survival_U_in_ge1_by_rho": le1_rho, "rho_unstable_in_Q": le1_rho_dist}
    # seven regimes, as before 2026-11-11 (comparison only)
    pa = paired_vs_lam0(models_all, rows_all, lams)
    out["with_le1_reference"] = {
        "note": "seven regimes (comparison only)", "n_models": len(models_all),
        "by_lam": per_lam(models_all, rows_all, lams, classes)["all"],
        "paired_vs_lam0": pa,
        "survival_U_in_ge1_by_rho": rho_tables(rows_all, lams)[0]}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "penalty.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    with open(OUT / "penalty_rows.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows_all[0]))
        wr.writeheader()
        wr.writerows(rows_all)
    print(json.dumps({k: out[k] for k in out if k not in ("survival_by_U_in_and_lam",)}, indent=1)[:12000])


if __name__ == "__main__":
    main()

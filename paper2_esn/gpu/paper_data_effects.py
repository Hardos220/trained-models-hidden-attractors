"""What the training data do: all numbers for the paper section on data effects, from the current full data.

Sections
  1. Stage V (intervention: same reservoir W, W_in, b and input scaling; only W_out retrained on different data):
     share of hidden among reproduced by variant (Wilson 95% CI), U by variant, logistic regression with and
     without log(1+U), within-model pairs H1 -> E20, per-regime effect on U, on the basin share of the reproduced
     attractor in the cube and on the share of unstable equilibria that excite it, the prediction (1 - p_cube)^(2U),
     and the effect of the equilibrium constraint (eqc) on U.
     "Hidden" in stage V = lite-protocol label "not_excited" (stage N protocol: no basin-boundary search), counted
     among rows with the hidden attractor reproduced and sum of indices = 1. Two row sets are reported:
     all (model x variant x hidden attractor) pairs, and the first hidden attractor of each regime only (the
     convention of the earlier analysis; it differs only for chua_I and chua_II, which have HA_a and HA_b).
  2. Check of the lite protocol against the full one (stage N lite vs stage C full on the same ESN).
  3. Escape redirection for ESN (stage C, full protocol, sets H1, HE1, H0; hidden = "hidden_stat"):
     share of initial conditions that escape to infinity in the true system and that the model sends to the
     reproduced hidden attractor / to its fixed points; properties of spurious equilibria.
  4. Escape redirection for the fully connected surrogates of stage M (mlp, mlp2, res; hidden = lite
     "not_excited").
  5. Training without data on the hidden attractor (set E1) in stage C: ESN and NG-RC (logic of gpu/aggregate_c.py,
     with duplicates of the same model across result folders removed).

Data: ROOT/{gpuq-results,local_results,node_results}/.../<stage>/<regime>/*.json (one JSON per model; the same model
may sit in several job folders - the first one in sorted order is used) and ROOT/data/<regime>.{json,npz}.
node_results/N_old_cpu_bug is excluded.

Pooled statistics (shares by variant, regressions, within-model pairs, pooled escape statistics, lite vs full counts, E1
totals) use the six main regimes of paper_common.MAIN_REGIMES; LE1, whose attractor is hidden only in the weak sense, is
reported under "le1_separate"; per-regime tables keep LE1. v_rows.csv and escape_rows.csv keep all seven regimes.

Outputs: paper2_esn/results/numbers/data_effects.json, v_rows.csv, escape_rows.csv; the report goes to stdout.
Run: python -X utf8 paper2_esn/gpu/paper_data_effects.py [--workers K]
(stage M is ~4.7 GB of JSON; K worker processes parse it, default $RMQ_CPU or 1).
"""
import argparse
import csv
import datetime
import json
import math
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import optimize, stats
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_common import COPY_TOL, POOL_NOTE, SEPARATE, main_only, only  # noqa: E402

ROOT = Path(os.environ.get("ESN_DATA", Path(__file__).resolve().parents[2] / "local" / "esn-gpu"))
OUT = Path(__file__).resolve().parents[1] / "results" / "numbers"
RES_DIRS = ("gpuq-results", "local_results", "node_results")
EXCLUDE_PARTS = {"N_old_cpu_bug"}
ARMS = ["H1", "eqc", "E5", "E10", "E20", "L", "E20+L"]
_META = {}


# ---------------------------------------------------------------------------------------------- helpers
def meta(reg):
    if reg not in _META:
        _META[reg] = json.loads((ROOT / "data" / f"{reg}.json").read_text(encoding="utf-8"))
    return _META[reg]


def collect(stage, dirs=RES_DIRS, name_filter=None):
    """{regime/stem: path} for all model files of a stage; first occurrence (dirs in order, sorted paths) wins."""
    out = {}
    for d in dirs:
        for f in sorted((ROOT / d).rglob(f"{stage}/*/*.json")):
            if ".failed" in f.name or EXCLUDE_PARTS & set(f.parts):
                continue
            if name_filter and not name_filter(f.stem):
                continue
            out.setdefault(f.parent.name + "/" + f.stem, f)
    return out


def load(f):
    return json.loads(Path(f).read_text(encoding="utf-8"))


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def mean(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def median(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def mwu(a, b):
    a = [x for x in a if np.isfinite(x)]
    b = [x for x in b if np.isfinite(x)]
    if len(a) < 2 or len(b) < 2:
        return None
    return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def van_elteren(rows, val, flag, stratum):
    """Stratified Wilcoxon (van Elteren) test of val between flag=True and flag=False within strata; weights
    1/(n_i + 1), tie-corrected variances. z < 0: the flagged group has smaller values. Strata with one group empty
    are skipped."""
    g = defaultdict(lambda: ([], []))
    for x in rows:
        v = x[val]
        if v is not None and np.isfinite(v):
            g[stratum(x)][0 if x[flag] else 1].append(v)
    T = ET = VT = 0.0
    used = 0
    for a, b in g.values():
        n1, n2 = len(a), len(b)
        if not n1 or not n2:
            continue
        n = n1 + n2
        rk = stats.rankdata(a + b)
        _, t = np.unique(a + b, return_counts=True)
        var = n1 * n2 / 12 * ((n + 1) - np.sum(t ** 3 - t) / (n * (n - 1)))
        w = 1 / (n + 1)
        T += w * rk[:n1].sum()
        ET += w * n1 * (n + 1) / 2
        VT += w * w * var
        used += 1
    if not used or VT <= 0:
        return None
    z = (T - ET) / math.sqrt(VT)
    return dict(z=z, p=float(2 * stats.norm.sf(abs(z))), n_strata=used)


def logit_fit(X, y):
    """Logistic regression without intercept (regime dummies play its role), tiny ridge 1e-4 as in earlier scripts.
    Returns coefficients, standard errors (inverse Hessian) and the negative log-likelihood."""
    def nll(w):
        z = X @ w
        return np.sum(np.logaddexp(0, z) - y * z) + 1e-4 * w @ w
    res = optimize.minimize(nll, np.zeros(X.shape[1]), method="BFGS")
    p = 1 / (1 + np.exp(-X @ res.x))
    H = (X * (p * (1 - p))[:, None]).T @ X
    return res.x, np.sqrt(np.diag(np.linalg.pinv(H))), float(res.fun)


def best_match(r, m):
    """Stage C: model attractor that reproduces a true hidden attractor (kind matches, delta_att < 0.2, d_H < 0.2);
    over all hidden attractors the one with the smallest delta_att. Returns (delta_att, name, y) or None.
    Same rule as gpu/aggregate_c.py."""
    best = None
    for at in r.get("attractors", []):
        for y in m["hidden"]:
            mm = at["metrics"].get(y, {})
            yk = "chaos" if m["attractors"][y]["kind"] == "chaotic" else "regular"
            if "delta_att" in mm and at["kind"] == yk and mm["delta_att"] < 0.2 and mm["d_H"] < 0.2:
                if best is None or mm["delta_att"] < best[0]:
                    best = (mm["delta_att"], at["name"], y)
    return best


def best_match_for(r, m, y):
    """Stage C: best model attractor for one given hidden attractor y (or None)."""
    best = None
    yk = "chaos" if m["attractors"][y]["kind"] == "chaotic" else "regular"
    for at in r.get("attractors", []):
        mm = at["metrics"].get(y, {})
        if "delta_att" in mm and at["kind"] == yk and mm["delta_att"] < 0.2 and mm["d_H"] < 0.2:
            if best is None or mm["delta_att"] < best[0]:
                best = (mm["delta_att"], at["name"])
    return best


def sig(x, n=6):
    return float(f"{x:.{n}g}")


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else sig(float(o))
    return o


def fmt(x, f=".2f"):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, f)


# ---------------------------------------------------------------------------------------------- 1. stage V
def v_rows_build(dirs=RES_DIRS):
    V = collect("V", dirs=dirs)
    rows = []
    for key, f in V.items():
        r = load(f)
        hid = meta(r["regime"])["hidden"]
        for arm in ARMS:
            a = r["arms"].get(arm)
            if not a:
                continue
            cube = a.get("census_cube", {})
            ncube = sum(cube.values())
            unst_keys = defaultdict(lambda: [0, 0])
            for y in hid:
                m = a["hidden_match"].get(y, {})
                att = m.get("attractor") if m.get("reproduced") else None
                fe = p_rad = p_cube = p_basin = None
                if att:
                    per_eq = defaultdict(lambda: [0, 0])
                    tot = hit = 0
                    for k, c in a.get("radius_counts", {}).items():
                        i = k.split("|")[0]
                        n_ = sum(c.values())
                        h_ = c.get(att, 0)
                        per_eq[i][0] += n_
                        per_eq[i][1] += h_
                        tot += n_
                        hit += h_
                    # share of unstable equilibria from whose neighbourhood the attractor is reached (U = 0: undefined)
                    fe = float(np.mean([v[1] > 0 for v in per_eq.values()])) if per_eq else None
                    p_rad = hit / tot if tot else None
                    p_cube = cube.get(att, 0) / ncube if ncube else None
                    bas = a.get(f"census_basin_{y}", {})
                    nb = sum(bas.values())
                    p_basin = bas.get(att, 0) / nb if nb else None
                rows.append(dict(model=key, regime=r["regime"], N=r["N"], rho=r["rho"], eta=r["eta"], seed=r["seed"],
                                 arm=arm, target=y, first=(y == hid[0]), U=a["U"], K=a["K"], isum=a["index_sum"],
                                 ok=a["index_sum"] == 1, rep=bool(m.get("reproduced")), label=m.get("label"),
                                 ne=bool(m.get("reproduced")) and m.get("label") == "not_excited",
                                 delta_att=m.get("delta_att"), p_cube=p_cube, p_basin=p_basin, fe=fe, p_rad=p_rad))
    return V, rows


def v_analysis(rows_all, models_all):
    """Pooled statistics on the six main regimes, per-regime tables on all seven, LE1 separately."""
    out = _v_core(main_only(rows_all), main_only(models_all))
    full = _v_core(rows_all, models_all)
    out["pooling"] = POOL_NOTE
    out["n_models_six_regimes"] = len({x["model"] for x in main_only(models_all)})
    out["by_regime_N"] = full["by_regime_N"]
    out["by_regime_reproduced"] = full["by_regime_reproduced"]
    for k, v in full["U_mean_all_models_H1_eqc_E20"].items():
        if k not in ("all", "paired_H1_to_eqc"):
            out["U_mean_all_models_H1_eqc_E20"][k] = v
    le1 = _v_core(only(rows_all, "le1"), only(models_all, "le1"))
    out["le1_separate"] = {"n_models": len({x["model"] for x in only(models_all, "le1")}),
                           **{k: le1[k] for k in ("by_variant", "logit", "paired")},
                           "U_mean_all_models_H1_eqc_E20": {k: v for k, v in le1["U_mean_all_models_H1_eqc_E20"].items()
                                                            if k in ("all", "paired_H1_to_eqc")}}
    out["with_le1_reference"] = {"note": "seven regimes (comparison only)",
                                 **{k: full[k] for k in ("by_variant", "logit", "paired")},
                                 "U_mean_all_models_H1_eqc_E20": {k: v for k, v in full["U_mean_all_models_H1_eqc_E20"].items()
                                                                  if k in ("all", "paired_H1_to_eqc")}}
    return out


def _v_core(rows, models_rows):
    """rows: rows of the chosen set (all pairs or first attractor only); models_rows: one row per model x arm
    (first hidden attractor) for model-level U statistics."""
    out = {}
    D = [x for x in rows if x["rep"] and x["ok"]]
    # per variant
    per_arm = {}
    for arm in ARMS:
        s = [x for x in rows if x["arm"] == arm]
        mrow = [x for x in models_rows if x["arm"] == arm]
        rep = [x for x in s if x["rep"] and x["ok"]]
        k = sum(x["ne"] for x in rep)
        per_arm[arm] = dict(n_rows=len(s), n_models=len(mrow), U_mean=mean([x["U"] for x in mrow]),
                            U_median=median([x["U"] for x in mrow]),
                            n_models_index_sum_ne_1=sum(not x["ok"] for x in mrow),
                            n_rows_reproduced_raw=sum(x["rep"] for x in s),
                            n_reproduced=len(rep), n_hidden=k, share_hidden=k / len(rep) if rep else None,
                            share_hidden_wilson95=wilson(k, len(rep)))
    out["by_variant"] = per_arm
    # per regime and N
    g = defaultdict(lambda: defaultdict(list))
    for x in rows:
        g[f"{x['regime']}|N{x['N']}"][x["arm"]].append(x)
    gm = defaultdict(lambda: defaultdict(list))
    for x in models_rows:
        gm[f"{x['regime']}|N{x['N']}"][x["arm"]].append(x)
    by_rn = {}
    for k in sorted(g):
        by_rn[k] = {}
        for arm in ARMS:
            s = g[k][arm]
            rep = [x for x in s if x["rep"] and x["ok"]]
            by_rn[k][arm] = dict(n_models=len(gm[k][arm]), U_mean=mean([x["U"] for x in gm[k][arm]]),
                                 n_reproduced=len(rep), n_hidden=sum(x["ne"] for x in rep))
    out["by_regime_N"] = by_rn
    # logistic regression with and without U
    regs = sorted({x["regime"] for x in D})
    arms_ = [a for a in ARMS if a != "H1"]
    Xb = np.array([[float(x["regime"] == r_) for r_ in regs] + [np.log(x["N"])] + [float(x["arm"] == a) for a in arms_]
                   for x in D])
    Xu = np.hstack([Xb, np.log1p([[x["U"]] for x in D])])
    y = np.array([x["ne"] for x in D], float)
    w1, s1, f1 = logit_fit(Xb, y)
    w2, s2, f2 = logit_fit(Xu, y)
    j0 = len(regs) + 1
    lr = 2 * (f1 - f2)
    out["logit"] = dict(n_rows=len(D), covariates="regime dummies (no intercept), log N, variant dummies (vs H1)",
                        variant_coef_without_U={a: [w1[j0 + i], s1[j0 + i]] for i, a in enumerate(arms_)},
                        variant_coef_with_U={a: [w2[j0 + i], s2[j0 + i]] for i, a in enumerate(arms_)},
                        log1pU_coef=[w2[-1], s2[-1]], log1pU_z=w2[-1] / s2[-1], LR_stat_U=lr,
                        LR_p_U=float(stats.chi2.sf(lr, 1)), log_N_coef_with_U=[w2[len(regs)], s2[len(regs)]])
    # within model (and hidden attractor): H1 -> E20, and H1 -> each variant
    by = defaultdict(dict)
    for x in rows:
        by[(x["model"], x["target"])][x["arm"]] = x
    paired = {}
    for arm in ("E5", "E10", "E20", "eqc", "L", "E20+L"):
        dU, dh, regs_up = [], [], []
        for (mk, t), d in by.items():
            if "H1" in d and arm in d and d["H1"]["rep"] and d[arm]["rep"] and d["H1"]["ok"] and d[arm]["ok"]:
                dU.append(d[arm]["U"] - d["H1"]["U"])
                dh.append(int(d[arm]["ne"]) - int(d["H1"]["ne"]))
                if dh[-1] > 0 and dU[-1] > 0:
                    regs_up.append(d["H1"]["regime"])
        dU, dh = np.array(dU), np.array(dh)
        rec = dict(n_pairs=len(dU), U_decreased=int(np.sum(dU < 0)), U_same=int(np.sum(dU == 0)),
                   U_increased=int(np.sum(dU > 0)), median_dU=float(np.median(dU)) if len(dU) else None,
                   hidden_appeared=int(np.sum(dh > 0)), hidden_lost=int(np.sum(dh < 0)),
                   appeared_with_dU_neg=int(np.sum((dh > 0) & (dU < 0))), appeared_with_dU_zero=int(np.sum((dh > 0) & (dU == 0))),
                   appeared_with_dU_pos=int(np.sum((dh > 0) & (dU > 0))),
                   lost_with_dU_neg=int(np.sum((dh < 0) & (dU < 0))), lost_with_dU_zero=int(np.sum((dh < 0) & (dU == 0))),
                   lost_with_dU_pos=int(np.sum((dh < 0) & (dU > 0))),
                   regimes_of_appeared_with_dU_pos=dict(Counter(regs_up)))
        if len(dU) > 3 and np.std(dh) > 0 and np.std(dU) > 0:
            sp = stats.spearmanr(dU, dh)
            rec["spearman_dU_dhidden"] = [float(sp[0]), float(sp[1])]
        paired[f"H1->{arm}"] = rec
    out["paired"] = paired
    # per regime: U, basin share in the cube, share of exciting unstable equilibria, prediction, hidden share
    reg_tab = {}
    for reg in sorted({x["regime"] for x in rows}):
        reg_tab[reg] = {}
        for arm in ARMS:
            d = [x for x in D if x["regime"] == reg and x["arm"] == arm]
            if not d:
                continue
            pred = [(1 - x["p_cube"]) ** (2 * x["U"]) for x in d if x["p_cube"] is not None]
            reg_tab[reg][arm] = dict(n=len(d), U_mean=mean([x["U"] for x in d]), p_cube=mean([x["p_cube"] for x in d]),
                                     p_true_basin_lifted=mean([x["p_basin"] for x in d]),
                                     share_unstable_eq_exciting=mean([x["fe"] for x in d]),
                                     n_with_U0=sum(x["U"] == 0 for x in d),
                                     p_radius_test=mean([x["p_rad"] for x in d]), hidden_share=mean([float(x["ne"]) for x in d]),
                                     prediction_1mp_2U=mean(pred))
    out["by_regime_reproduced"] = reg_tab
    # eqc and U: model level, all models (not only reproduced)
    eqc = {"all": dict(H1=mean([x["U"] for x in models_rows if x["arm"] == "H1"]),
                       eqc=mean([x["U"] for x in models_rows if x["arm"] == "eqc"]),
                       E20=mean([x["U"] for x in models_rows if x["arm"] == "E20"]))}
    for reg in sorted({x["regime"] for x in models_rows}):
        eqc[reg] = dict(H1=mean([x["U"] for x in models_rows if x["arm"] == "H1" and x["regime"] == reg]),
                        eqc=mean([x["U"] for x in models_rows if x["arm"] == "eqc" and x["regime"] == reg]),
                        E20=mean([x["U"] for x in models_rows if x["arm"] == "E20" and x["regime"] == reg]))
    # model-level pairs H1 -> eqc
    bm = defaultdict(dict)
    for x in models_rows:
        bm[x["model"]][x["arm"]] = x["U"]
    dd = np.array([d["eqc"] - d["H1"] for d in bm.values() if "eqc" in d and "H1" in d])
    eqc["paired_H1_to_eqc"] = dict(n=len(dd), U_increased=int(np.sum(dd > 0)), U_same=int(np.sum(dd == 0)),
                                   U_decreased=int(np.sum(dd < 0)), median_dU=float(np.median(dd)))
    out["U_mean_all_models_H1_eqc_E20"] = eqc
    return out


def print_v(name, res):
    print(f"\n--- stage V, rows: {name}")
    print("variant | models | U mean (median) | Σind≠1 | reproduced | hidden | share [Wilson 95%]")
    for arm, a in res["by_variant"].items():
        ci = a["share_hidden_wilson95"]
        print(f"  {arm:6s} | {a['n_models']:3d} | {fmt(a['U_mean'], '5.1f')} ({fmt(a['U_median'], '4.1f')}) | "
              f"{a['n_models_index_sum_ne_1']:2d} | {a['n_reproduced']:3d} | {a['n_hidden']:3d} | "
              f"{fmt(a['share_hidden'])} [{fmt(ci[0])}, {fmt(ci[1])}]")
    lg = res["logit"]
    print(f"logistic regression on {lg['n_rows']} rows: variant coef without U -> with U")
    for a in lg["variant_coef_without_U"]:
        w1, s1 = lg["variant_coef_without_U"][a]
        w2, s2 = lg["variant_coef_with_U"][a]
        print(f"  {a:6s} {w1:+.2f} ± {s1:.2f} -> {w2:+.2f} ± {s2:.2f}")
    print(f"  log(1+U) {lg['log1pU_coef'][0]:+.2f} ± {lg['log1pU_coef'][1]:.2f}, LR {lg['LR_stat_U']:.1f}, p = {lg['LR_p_U']:.1e}")
    p = res["paired"]["H1->E20"]
    print(f"paired H1 -> E20: {p['n_pairs']} pairs; U down {p['U_decreased']}, same {p['U_same']}, up {p['U_increased']}, "
          f"median dU {p['median_dU']:+.1f}; hidden appeared {p['hidden_appeared']} (dU<0 {p['appeared_with_dU_neg']}, "
          f"dU=0 {p['appeared_with_dU_zero']}, dU>0 {p['appeared_with_dU_pos']} {p['regimes_of_appeared_with_dU_pos']}), "
          f"lost {p['hidden_lost']} (dU<0 {p['lost_with_dU_neg']}, dU=0 {p['lost_with_dU_zero']}, dU>0 {p['lost_with_dU_pos']})"
          + (f"; Spearman {p['spearman_dU_dhidden'][0]:+.2f} (p {p['spearman_dU_dhidden'][1]:.1e})" if "spearman_dU_dhidden" in p else ""))
    print("regime (reproduced, Σind=1) | U H1 -> E20 | p_cube H1 -> E20 | exciting share H1 -> E20 | hidden H1 -> E20 | "
          "(1-p)^(2U) H1 -> E20 | U eqc")
    for reg, d in res["by_regime_reproduced"].items():
        h, e, q = d.get("H1", {}), d.get("E20", {}), d.get("eqc", {})
        print(f"  {reg:16s} | {fmt(h.get('U_mean'), '4.1f')} -> {fmt(e.get('U_mean'), '4.1f')} | "
              f"{fmt(h.get('p_cube'), '.3f')} -> {fmt(e.get('p_cube'), '.3f')} | "
              f"{fmt(h.get('share_unstable_eq_exciting'))} -> {fmt(e.get('share_unstable_eq_exciting'))} | "
              f"{fmt(h.get('hidden_share'))} -> {fmt(e.get('hidden_share'))} | "
              f"{fmt(h.get('prediction_1mp_2U'))} -> {fmt(e.get('prediction_1mp_2U'))} | {fmt(q.get('U_mean'), '4.1f')}")
    eq = res["U_mean_all_models_H1_eqc_E20"]
    print("U mean over all models H1 / eqc / E20: " + "; ".join(
        f"{k} {fmt(v['H1'], '.1f')}/{fmt(v['eqc'], '.1f')}/{fmt(v['E20'], '.1f')}" for k, v in eq.items() if k != "paired_H1_to_eqc")
          + f"; paired H1->eqc: {eq['paired_H1_to_eqc']}")


# ---------------------------------------------------------------------------------------------- 2. lite vs full
def lite_vs_full(C_esn):
    """Pooled counts on the six main regimes; LE1 separately."""
    res = _lite_vs_full(C_esn, lambda g: g not in SEPARATE)
    print("  LE1 separately:")
    res["le1_separate"] = _lite_vs_full(C_esn, lambda g: g == "le1")
    res["pooling"] = POOL_NOTE
    return res


def _lite_vs_full(C_esn, keep_regime):
    Nl = collect("N")
    legacy, general = Counter(), Counter()
    for key, f in C_esn.items():
        c = load(f)
        if c.get("model") != "esn" or not keep_regime(c["regime"]):
            continue
        N, rho = c.get("N", 300), c.get("rho")
        nkey = f"{c['regime']}/{c['set']}_N{N}_rho{rho}_eta{c['eta']:g}_s{c['seed']}"
        if nkey not in Nl:
            continue
        n = load(Nl[nkey])
        m = meta(c["regime"])
        for y in m["hidden"]:
            nm = n["hidden_match"].get(y, {})
            lite = nm.get("label") if nm.get("reproduced") else "not_reproduced"
            b = best_match_for(c, m, y)
            full = c["hiddenness"][b[1]]["label"] if b else "not_reproduced"
            general[(lite, full)] += 1
        # legacy check of the earlier analysis: HE1, N 500/700, rho 0.9, gpuq-results, first hidden attractor vs the
        # best-matching model attractor over all hidden attractors
        if "gpuq-results" in f.parts and c["set"] == "HE1" and N in (500, 700) and rho == 0.9:
            b = best_match(c, m)
            full = c["hiddenness"][b[1]]["label"] if b else "not_reproduced"
            nm = next(iter(n["hidden_match"].values()))
            lite = nm.get("label") if nm.get("reproduced") else "not_reproduced"
            legacy[(lite, full)] += 1
    both = {k: v for k, v in general.items() if "not_reproduced" not in k}
    agree = sum(v for (l_, f_), v in both.items() if (l_ == "not_excited") == (f_ == "hidden_stat"))
    ne_tot = sum(v for (l_, f_), v in both.items() if l_ == "not_excited")
    ne_hid = sum(v for (l_, f_), v in both.items() if l_ == "not_excited" and f_ == "hidden_stat")
    hid_tot = sum(v for (l_, f_), v in both.items() if f_ == "hidden_stat")
    res = dict(legacy_HE1_N500_700={f"{a} -> {b}": v for (a, b), v in sorted(legacy.items(), key=lambda t: -t[1])},
               legacy_n=sum(legacy.values()),
               all_overlapping_pairs={f"{a} -> {b}": v for (a, b), v in sorted(general.items(), key=lambda t: -t[1])},
               n_pairs_both_reproduced=sum(both.values()), n_agree_not_excited_eq_hidden_stat=agree,
               lite_not_excited_total=ne_tot, lite_not_excited_and_full_hidden_stat=ne_hid, full_hidden_stat_total=hid_tot)
    print("\n=== 2. lite protocol (stage N) vs full protocol (stage C) on the same ESN")
    print(f"legacy subset (HE1, N 500/700, rho 0.9): {res['legacy_n']} models: {res['legacy_HE1_N500_700']}")
    print(f"all overlapping (model x hidden attractor): {res['all_overlapping_pairs']}")
    print(f"  both reproduced {res['n_pairs_both_reproduced']}: labels agree (not_excited <=> hidden_stat) in {agree}; "
          f"lite not_excited {ne_tot}, of them full hidden_stat {ne_hid}; full hidden_stat {hid_tot}")
    return res


# ---------------------------------------------------------------------------------------------- 3. ESN escape
def esn_escape(C_esn):
    trees, tlab = {}, {}
    rows, spur, csv_rows = [], [], []
    for key, f in C_esn.items():
        r = load(f)
        if r.get("model") != "esn" or r["set"] not in ("H1", "HE1", "H0"):
            continue
        reg = r["regime"]
        m = meta(reg)
        if reg not in trees:
            z = np.load(ROOT / "data" / f"{reg}.npz")
            trees[reg] = cKDTree(z["ics_uniform"])
            tlab[reg] = np.asarray(z["labels_uniform"]).astype(str)
        best = best_match(r, m)
        if not best:
            continue
        a, y = best[1], best[2]
        lab = r["hiddenness"][a]["label"]
        conf = r["confusion"]["uL2"]

        def share(true_key, pred):
            row = conf.get(true_key, {})
            n = sum(row.values())
            return (sum(v for k, v in row.items() if pred(k)) / n) if n else np.nan
        other_true = [t for t in conf if t not in ("escape", "unknown") and t not in m["hidden"]]
        x = dict(key=key, reg=reg, N=r.get("N", 300), rho=r.get("rho"), eta=r.get("eta"), seed=r.get("seed"), set=r["set"],
                 target=y, label=lab, hid=lab == "hidden_stat",
                 n_escape=sum(conf.get("escape", {}).values()),
                 esc_to_A=share("escape", lambda k: k == a), esc_to_fp=share("escape", lambda k: k.startswith("f")),
                 HA_to_A=share(y, lambda k: k == a),
                 fp_to_A=np.nanmean([share(t, lambda k: k == a) for t in other_true]) if other_true else np.nan)
        rows.append(x)
        lo, hi = np.array(m["box_lo"]), np.array(m["box_hi"])
        for fp in r["fixed_points"]:
            if fp.get("dist_true_eq", 9) < 0.01:
                continue
            xo = np.array(fp["x"])
            inside = bool(np.all(xo >= lo) and np.all(xo <= hi))
            tl = str(tlab[reg][trees[reg].query(xo)[1]]) if inside else "outside_box"
            spur.append(dict(reg=reg, unst=fp["n_unstable"] > 0, sat=fp["share_sat"], mabs=fp["mean_abs_r"],
                             inside=inside, tl=tl, hid=lab == "hidden_stat"))
    res = {"n_models_reproduced": len(rows), "sets": ["H1", "HE1", "H0"],
           "hidden_definition": "full protocol label hidden_stat of the model attractor that best matches a true hidden attractor"}
    per = {}
    print("\n=== 3. ESN (stage C, sets H1/HE1/H0): escape redirection")
    print("regime | n (hidden) | escape->A mean hidden / not | median hidden / not | MWU p | escape->fixed points | true HA->A")
    for reg in sorted({x["reg"] for x in rows}):
        s = [x for x in rows if x["reg"] == reg]
        h = [x["esc_to_A"] for x in s if x["hid"]]
        nh = [x["esc_to_A"] for x in s if not x["hid"]]
        per[reg] = dict(n=len(s), n_hidden=sum(x["hid"] for x in s), n_with_escape=sum(x["n_escape"] > 0 for x in s),
                        esc_to_A_mean_hidden=mean(h), esc_to_A_mean_not_hidden=mean(nh),
                        esc_to_A_median_hidden=median(h), esc_to_A_median_not_hidden=median(nh), mwu_p=mwu(h, nh),
                        esc_to_fp_mean=mean([x["esc_to_fp"] for x in s]), esc_to_fp_median=median([x["esc_to_fp"] for x in s]),
                        HA_to_A_mean=mean([x["HA_to_A"] for x in s]),
                        fp_basin_to_A_mean_hidden=mean([x["fp_to_A"] for x in s if x["hid"]]),
                        fp_basin_to_A_mean_not_hidden=mean([x["fp_to_A"] for x in s if not x["hid"]]))
        p = per[reg]
        print(f"  {reg:16s} | {p['n']:3d} ({p['n_hidden']:3d}) | {fmt(p['esc_to_A_mean_hidden'])} / {fmt(p['esc_to_A_mean_not_hidden'])} | "
              f"{fmt(p['esc_to_A_median_hidden'], '.3f')} / {fmt(p['esc_to_A_median_not_hidden'], '.3f')} | "
              f"{fmt(p['mwu_p'], '.1e')} | {fmt(p['esc_to_fp_mean'])} | {fmt(p['HA_to_A_mean'])}")
    res["by_regime"] = per
    res["pooling"] = POOL_NOTE
    res["n_models_reproduced_six_regimes"] = len(main_only(rows, "reg"))
    rows_all, rows = rows, main_only(rows, "reg")               # pooled statistics: six regimes
    h = [x["esc_to_A"] for x in rows if x["hid"] and np.isfinite(x["esc_to_A"])]
    nh = [x["esc_to_A"] for x in rows if not x["hid"] and np.isfinite(x["esc_to_A"])]
    res["pooled"] = dict(n_hidden=len(h), n_not_hidden=len(nh), median_hidden=median(h), median_not_hidden=median(nh),
                         mean_hidden=mean(h), mean_not_hidden=mean(nh), mwu_p=mwu(h, nh),
                         esc_to_fp_mean=mean([x["esc_to_fp"] for x in rows]),
                         van_elteren_by_regime=van_elteren(rows, "esc_to_A", "hid", lambda x: x["reg"]),
                         van_elteren_by_regime_set=van_elteren(rows, "esc_to_A", "hid", lambda x: (x["reg"], x["set"])))
    pr = res["pooled"]
    ve, ve2 = pr["van_elteren_by_regime"], pr["van_elteren_by_regime_set"]
    print(f"pooled (six regimes): escape->A median hidden {pr['median_hidden']:.3f} (n={len(h)}) vs not hidden "
          f"{pr['median_not_hidden']:.3f} (n={len(nh)}); means {pr['mean_hidden']:.3f} vs {pr['mean_not_hidden']:.3f}; "
          f"Mann–Whitney p = {pr['mwu_p']:.1e}; stratified (van Elteren) by regime z = {ve['z']:+.2f}, p = {ve['p']:.1e}; "
          f"by regime x set z = {ve2['z']:+.2f}, p = {ve2['p']:.1e}")
    # spurious equilibria (six regimes; LE1 separately)
    spur_all, spur = spur, main_only(spur, "reg")
    sp = {"n_spurious": len(spur), "definition": "fixed points of models with a reproduced hidden attractor, farther than "
          "0.01 L (L = half of the largest edge of Q) from every true equilibrium; true outcome = label of the nearest "
          "uniform census point; six regimes"}
    sl = only(spur_all, "le1", "reg")
    sp["le1_separate"] = {"n_spurious": len(sl), "n_unstable": sum(x["unst"] for x in sl),
                          "share_inside_box_unstable": mean([float(x["inside"]) for x in sl if x["unst"]])}
    print(f"spurious equilibria: {len(spur)}")
    for u, nm in ((True, "unstable"), (False, "stable")):
        s = [x for x in spur if x["unst"] == u]
        tl = Counter(x["tl"] for x in s)
        esc_or_out = sum(1 for x in s if x["tl"] in ("escape", "outside_box"))
        sp[nm] = dict(n=len(s), share_saturated_median=median([x["sat"] for x in s]),
                      mean_abs_r_median=median([x["mabs"] for x in s]), share_inside_box=mean([float(x["inside"]) for x in s]),
                      share_escape_or_outside_box=esc_or_out / len(s) if s else None,
                      true_outcome_counts=dict(tl.most_common()))
        print(f"  {nm:8s}: {len(s)}; saturation (share |r|>0.9) median {sp[nm]['share_saturated_median']:.2f}; |r| median "
              f"{sp[nm]['mean_abs_r_median']:.2f}; inside box {sp[nm]['share_inside_box']:.2f}; in escape region or "
              f"outside box {sp[nm]['share_escape_or_outside_box']:.2f}; {dict(tl.most_common(6))}")
    sp["saturation_mwu_p_stable_vs_unstable"] = mwu([x["sat"] for x in spur if not x["unst"]], [x["sat"] for x in spur if x["unst"]])
    res["spurious_equilibria"] = sp
    for x in rows_all:
        csv_rows.append(dict(stage="C", model_class="esn", regime=x["reg"], set=x["set"], size=x["N"], rho=x["rho"],
                             eta=x["eta"], seed=x["seed"], target=x["target"], label=x["label"], hidden=int(x["hid"]),
                             hidden_definition="full_hidden_stat", n_true_escape=x["n_escape"],
                             escape_to_attractor=x["esc_to_A"], escape_to_fixed_point=x["esc_to_fp"]))
    return res, csv_rows


# ---------------------------------------------------------------------------------------------- 4. stage M escape
def m_extract(path):
    """One stage M file -> rows (model x hidden attractor); only the fields needed here."""
    r = load(path)
    if r.get("class") not in ("mlp", "mlp2", "res"):
        return []
    m = meta(r["regime"])
    lo, hi = np.array(m["box_lo"]), np.array(m["box_hi"])
    U_in = sum(1 for x in r["fixed_points"] if x["n_unstable"] > 0 and np.all(np.array(x["x"]) >= lo)
               and np.all(np.array(x["x"]) <= hi))
    esc = r.get("census_confusion", {}).get("escape", {})
    n = sum(esc.values())
    out = []
    for y, mm in r["hidden_match"].items():
        a = mm.get("attractor")
        rep = bool(mm.get("reproduced"))
        out.append(dict(reg=r["regime"], cls=r["class"], set=r["set"], H=r["H"], seed=r["seed"], target=y, U=r["U"],
                        U_in=U_in, isum=r["index_sum"], rep=rep, label=mm.get("label"),
                        ne=rep and mm.get("label") == "not_excited", n_escape=n,
                        esc_to_A=(esc.get(a, 0) / n) if (a and n) else np.nan,
                        esc_to_fp=(sum(v for k, v in esc.items() if k.startswith("f")) / n) if n else np.nan,
                        esc_to_inf=(esc.get("esc", 0) / n) if n else np.nan))
    return out


def m_escape(workers):
    files = {}
    for f in sorted((ROOT / "gpuq-results").rglob("M/*/*.json")):
        if ".failed" in f.name:
            continue
        files.setdefault(f.parent.name + "/" + f.stem, str(f))
    paths = list(files.values())
    if workers > 1:
        with Pool(workers) as pool:
            parts = pool.map(m_extract, paths, chunksize=64)
    else:
        parts = [m_extract(p) for p in paths]
    rows = [x for p in parts for x in p]
    R = [x for x in rows if x["rep"]]
    res = {"n_models": len(paths), "n_rows_model_x_hidden": len(rows), "n_reproduced_rows": len(R),
           "hidden_definition": "lite label not_excited (stage M protocol: radius test around unstable equilibria)"}
    print(f"\n=== 4. stage M: {len(paths)} models, {len(rows)} rows (model x hidden attractor), {len(R)} reproduced")
    print("class subset | n hidden / not | escape->A median hidden / not | mean hidden / not | MWU p | escape->fixed points mean | escape->inf mean")

    def summ(s):
        h = [x["esc_to_A"] for x in s if x["ne"] and np.isfinite(x["esc_to_A"])]
        nh = [x["esc_to_A"] for x in s if not x["ne"] and np.isfinite(x["esc_to_A"])]
        return dict(n_hidden=len(h), n_not_hidden=len(nh), median_hidden=median(h), median_not_hidden=median(nh),
                    mean_hidden=mean(h), mean_not_hidden=mean(nh), mwu_p=mwu(h, nh),
                    esc_to_fp_mean=mean([x["esc_to_fp"] for x in s]), esc_to_fp_mean_hidden=mean([x["esc_to_fp"] for x in s if x["ne"]]),
                    esc_to_fp_mean_not_hidden=mean([x["esc_to_fp"] for x in s if not x["ne"]]),
                    esc_to_inf_mean=mean([x["esc_to_inf"] for x in s]),
                    van_elteren_by_regime=van_elteren(s, "esc_to_A", "ne", lambda x: x["reg"]),
                    van_elteren_by_regime_set=van_elteren(s, "esc_to_A", "ne", lambda x: (x["reg"], x["set"])))
    out = {}
    for cls in ("mlp", "mlp2", "res"):
        for sub, sets in (("H1+HE1", ("H1", "HE1")), ("all_sets", ("H1", "HE1", "E1")), ("H1", ("H1",)), ("HE1", ("HE1",)),
                          ("E1", ("E1",))):
            s = [x for x in R if x["cls"] == cls and x["set"] in sets and x["reg"] not in SEPARATE]   # six regimes
            if not s:
                continue
            d = summ(s)
            out[f"{cls}|{sub}"] = d
            ve = d["van_elteren_by_regime_set"]
            print(f"  {cls:4s} {sub:8s} | {d['n_hidden']:4d} / {d['n_not_hidden']:4d} | {fmt(d['median_hidden'], '.3f')} / "
                  f"{fmt(d['median_not_hidden'], '.3f')} | {fmt(d['mean_hidden'], '.3f')} / {fmt(d['mean_not_hidden'], '.3f')} | "
                  f"{fmt(d['mwu_p'], '.1e')} | {fmt(d['esc_to_fp_mean'])} | {fmt(d['esc_to_inf_mean'])}"
                  + (f" | stratified regime x set z {ve['z']:+.1f} p {ve['p']:.1e}" if ve else ""))
    res["by_class_and_sets"] = out
    res["pooling"] = "by_class_and_sets: " + POOL_NOTE + "; by_class_regime_H1_HE1 includes LE1"
    res["le1_separate"] = {cls: summ([x for x in R if x["cls"] == cls and x["reg"] == "le1"]) for cls in ("mlp", "mlp2", "res")}
    per = {}
    for cls in ("mlp", "mlp2", "res"):
        for reg in sorted({x["reg"] for x in R}):
            s = [x for x in R if x["cls"] == cls and x["reg"] == reg and x["set"] in ("H1", "HE1")]
            if s:
                per[f"{cls}|{reg}|H1+HE1"] = summ(s)
    res["by_class_regime_H1_HE1"] = per
    csv_rows = [dict(stage="M", model_class=x["cls"], regime=x["reg"], set=x["set"], size=x["H"], rho="", eta="",
                     seed=x["seed"], target=x["target"], label=x["label"], hidden=int(x["ne"]),
                     hidden_definition="lite_not_excited", n_true_escape=x["n_escape"],
                     escape_to_attractor=x["esc_to_A"], escape_to_fixed_point=x["esc_to_fp"]) for x in R]
    return res, csv_rows


# ---------------------------------------------------------------------------------------------- 5. E1, stage C
def c_e1(C_all):
    tab = defaultdict(lambda: {"n": 0, "reproduced": 0, "labels": Counter(), "N": Counter()})
    for key, f in C_all.items():
        r = load(f)
        if r["set"] != "E1":
            continue
        m = meta(r["regime"])
        t = tab[(r["model"], r["regime"])]
        t["n"] += 1
        t["N"][r.get("N", 300) if r["model"] == "esn" else 0] += 1
        b = best_match(r, m)
        if b:
            t["reproduced"] += 1
            t["labels"][f"{b[2]}:{r['hiddenness'][b[1]]['label']}"] += 1
    res = {}
    print("\n=== 5. set E1 (only the visible attractor's basin), stage C")
    for model in ("esn", "ngrc"):
        d = {reg: dict(n=t["n"], reproduced=t["reproduced"], labels=dict(t["labels"]),
                       n_by_N=dict(t["N"]) if model == "esn" else None)
             for (mm, reg), t in sorted(tab.items()) if mm == model}
        dm = {g: v for g, v in d.items() if g not in SEPARATE}
        tot = dict(n=sum(v["n"] for v in dm.values()), reproduced=sum(v["reproduced"] for v in dm.values()),
                   hidden_stat=sum(c for v in dm.values() for k, c in v["labels"].items() if k.endswith("hidden_stat")),
                   regimes=sorted(dm))
        tot_all = dict(n=sum(v["n"] for v in d.values()), reproduced=sum(v["reproduced"] for v in d.values()))
        res[model] = {"by_regime": d, "total": tot, "total_all_regimes": tot_all, "le1_separate": d.get("le1"),
                      "pooling": "total: " + POOL_NOTE}
        print(f"  {model}: total {tot}")
        for reg, v in d.items():
            print(f"    {reg:16s} n {v['n']:3d} reproduced {v['reproduced']} {v['labels']}" + (f" N: {v['n_by_N']}" if v["n_by_N"] else ""))
    return res


# ---------------------------------------------------------------------------------------------- main
def write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for x in rows:
            w.writerow({k: ("" if (isinstance(v, float) and not np.isfinite(v)) else (sig(v) if isinstance(v, float) else v))
                        for k, v in x.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=int(os.environ.get("RMQ_CPU", "1")))
    ap.add_argument("--skip-m", action="store_true", help="skip stage M (4.7 GB of JSON); for quick checks only")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
              "script": "paper2_esn/gpu/paper_data_effects.py", "data_root": str(ROOT)}

    # 1. stage V
    V, rows = v_rows_build()
    first = [x for x in rows if x["first"]]
    n_by = Counter((load(f)["regime"]) for f in V.values())
    print(f"=== 1. stage V: {len(V)} models ({dict(sorted(n_by.items()))}), {len(rows)} rows (model x variant x hidden attractor)")
    resV = {"n_models": len(V), "n_models_by_regime": dict(n_by), "n_rows": len(rows),
            "N_values": sorted({x["N"] for x in rows}), "etas": sorted({x["eta"] for x in rows}),
            "hidden_definition": ("lite-protocol label not_excited (stage N protocol, no basin-boundary search), among rows "
                                  "with the hidden attractor reproduced (delta_att < 0.2, d_H < 0.2) and sum of indices = 1; "
                                  "the full label hidden_stat is not computed in stage V"),
            "all_hidden_attractors": v_analysis(rows, first),
            "first_hidden_attractor_only": v_analysis(first, first)}
    print_v("all hidden attractors (chua_I, chua_II: HA_a and HA_b)", resV["all_hidden_attractors"])
    print_v("first hidden attractor only (earlier convention)", resV["first_hidden_attractor_only"])
    # reconciliation with the earlier 66-model analysis (gpuq-results only; node_results adds 16 models at N = 300)
    V66, rows66 = v_rows_build(dirs=("gpuq-results",))
    f66 = [x for x in rows66 if x["first"]]
    r66 = _v_core(f66, f66)          # reconciliation with the earlier analysis: all seven regimes, as it was done then
    resV["legacy_check_gpuq_results_only_first_attractor"] = dict(
        n_models=len(V66), by_variant={k: dict(n_reproduced=v["n_reproduced"], n_hidden=v["n_hidden"], share_hidden=v["share_hidden"])
                                       for k, v in r66["by_variant"].items()},
        paired_H1_E20=r66["paired"]["H1->E20"], logit_E20_without_with_U=[r66["logit"]["variant_coef_without_U"]["E20"],
                                                                          r66["logit"]["variant_coef_with_U"]["E20"]],
        log1pU_coef=r66["logit"]["log1pU_coef"], LR_p_U=r66["logit"]["LR_p_U"])
    print(f"legacy check, gpuq-results only ({len(V66)} models, first attractor): shares " + ", ".join(
        f"{k} {fmt(v['share_hidden'])}" for k, v in r66["by_variant"].items()))
    result["stage_V"] = resV
    write_csv(OUT / "v_rows.csv",
              [dict(regime=x["regime"], N=x["N"], rho=x["rho"], eta=x["eta"], seed=x["seed"], variant=x["arm"],
                    target=x["target"], U=x["U"], K=x["K"], index_sum=x["isum"], reproduced=int(x["rep"]),
                    in_analysis=int(x["rep"] and x["ok"]), label=x["label"] or "",
                    hidden_or_not_excited=(int(x["ne"]) if x["rep"] else ""),
                    p_cube=x["p_cube"] if x["p_cube"] is not None else float("nan"),
                    share_unstable_eq_exciting=x["fe"] if x["fe"] is not None else float("nan")) for x in rows],
              ["regime", "N", "rho", "eta", "seed", "variant", "target", "U", "K", "index_sum", "reproduced", "in_analysis",
               "label", "hidden_or_not_excited", "p_cube", "share_unstable_eq_exciting"])

    # 2-3, 5. stage C
    C_all = collect("C")
    C_esn = {k: f for k, f in C_all.items() if "_esn_" in f.name}
    result["lite_vs_full_check"] = lite_vs_full(C_esn)
    resE, esc_rows = esn_escape(C_esn)
    result["escape_redirection_esn_stage_C"] = resE
    result["E1_stage_C"] = c_e1(C_all)

    # 4. stage M
    if not a.skip_m:
        resM, m_rows = m_escape(max(1, a.workers))
        result["escape_redirection_stage_M"] = resM
        esc_rows += m_rows
    write_csv(OUT / "escape_rows.csv", esc_rows,
              ["stage", "model_class", "regime", "set", "size", "rho", "eta", "seed", "target", "label", "hidden",
               "hidden_definition", "n_true_escape", "escape_to_attractor", "escape_to_fixed_point"])
    (OUT / "data_effects.json").write_text(json.dumps(clean(result), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwritten: {OUT / 'data_effects.json'}, {OUT / 'v_rows.csv'}, {OUT / 'escape_rows.csv'} ({len(esc_rows)} rows)")


if __name__ == "__main__":
    main()

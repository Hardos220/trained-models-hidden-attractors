"""Stage M in full (first queue m1 + second queue m2): classes mlp, mlp2, res on sets H1, HE1, E1 for all regimes.
Prints: per class and set — Σind = 1, reproduction, U total and inside the true region, escape, not excited among
reproduced; the per-regime tables for mlp2 and E1; the survival law by U inside the true region for each class;
a logistic regression of "not excited" on log(1 + U inside) with class, set, width and regime terms, and the
likelihood-ratio test for everything beyond U; the E1 models that reproduced a hidden attractor, one by one.
Data: local/esn-gpu/gpuq-results/<job>/out/M/<regime>/*.json (one file per model)."""
import json
import os
from pathlib import Path

import numpy as np
from scipy import optimize, stats

ROOT = Path(os.environ.get("ESN_DATA", Path(__file__).resolve().parents[2] / "local" / "esn-gpu"))
meta, rows, seen = {}, [], set()
for f in sorted((ROOT / "gpuq-results").rglob("M/*/*.json")):
    key = f.parent.name + "/" + f.stem
    if key in seen:
        continue
    seen.add(key)
    r = json.loads(f.read_text(encoding="utf-8"))
    if r["class"] not in ("mlp", "mlp2", "res"):
        continue
    reg = r["regime"]
    m = meta.setdefault(reg, json.loads((ROOT / "data" / f"{reg}.json").read_text(encoding="utf-8")))
    lo, hi = np.array(m["box_lo"]), np.array(m["box_hi"])
    inside = lambda x: bool(np.all(np.array(x) >= lo) and np.all(np.array(x) <= hi))
    Un = sum(1 for x in r["fixed_points"] if x["n_unstable"] > 0 and inside(x["x"]))
    cc = r.get("census_counts", {}); n = sum(cc.values())
    for y, mm in r["hidden_match"].items():
        rows.append(dict(cls=r["class"], set=r["set"], reg=reg, H=r["H"], seed=r["seed"], att=y, U=r["U"], Un=Un,
                         ok=r["index_sum"] == 1, rep=bool(mm.get("reproduced")), ne=mm.get("label") == "not_excited",
                         lab=mm.get("label"), mm=mm, esc=cc.get("esc", 0) / n if n else np.nan, mse=r["train_mse"]))
REGS = sorted({x["reg"] for x in rows})
CLS, SETS = ("mlp", "mlp2", "res"), ("H1", "HE1", "E1")
print(f"rows (model x hidden attractor): {len(rows)}, models: {len(seen)}, regimes: {', '.join(REGS)}")

print("\nclass set  |    n | Σind=1 | repr | U total | U inside | escape | not excited / repr")
for cls in CLS:
    for st in SETS:
        s = [x for x in rows if x["cls"] == cls and x["set"] == st]
        if not s:
            continue
        rep = [x for x in s if x["rep"]]
        print(f"  {cls:4s} {st:3s} | {len(s):4d} | {np.mean([x['ok'] for x in s]):.3f} | {len(rep):4d} | "
              f"{np.mean([x['U'] for x in s]):7.1f} | {np.mean([x['Un'] for x in s]):8.2f} | {np.nanmean([x['esc'] for x in s]):.2f} | "
              f"{sum(x['ne'] for x in rep)}/{len(rep)}" + (f" = {np.mean([x['ne'] for x in rep]):.2f}" if rep else ""))

print("\nper regime, reproduced (not excited) of n:")
cols = [("mlp", "H1"), ("mlp2", "H1"), ("mlp", "HE1"), ("mlp2", "HE1"), ("mlp", "E1"), ("mlp2", "E1"), ("res", "E1")]
print("  regime           | " + " | ".join(f"{c} {s}" for c, s in cols))
for g in REGS:
    cells = []
    for c, s in cols:
        q = [x for x in rows if x["reg"] == g and x["cls"] == c and x["set"] == s]
        cells.append(f"{sum(x['rep'] for x in q)} ({sum(x['rep'] and x['ne'] for x in q)})/{len(q)}" if q else "—")
    print(f"  {g:16s} | " + " | ".join(cells))

print("\nsurvival law: P(not excited | reproduced) by U inside the true region")
bins = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 10 ** 6)]
print("  class  | " + " | ".join(f"U {a}-{b if b < 10 ** 6 else 'inf'}" for a, b in bins))
for cls in CLS:
    D = [x for x in rows if x["cls"] == cls and x["rep"]]
    cells = []
    for a, b in bins:
        q = [x for x in D if a <= x["Un"] <= b]
        cells.append(f"{np.mean([x['ne'] for x in q]):.2f} ({len(q)})" if q else "—")
    print(f"  {cls:6s} | " + " | ".join(cells))


def fit(X, y):
    f = lambda w: np.sum(np.logaddexp(0, X @ w) - y * (X @ w)) + 1e-4 * w @ w
    r = optimize.minimize(f, np.zeros(X.shape[1]), method="BFGS")
    p = 1 / (1 + np.exp(-X @ r.x)); Hs = (X * (p * (1 - p))[:, None]).T @ X
    return r.x, np.sqrt(np.diag(np.linalg.pinv(Hs))), r.fun


D = [x for x in rows if x["rep"]]
y = np.array([x["ne"] for x in D], float)
regs = sorted({x["reg"] for x in D})
U = np.array([[np.log1p(x["Un"])] for x in D])
extra = np.array([[float(x["cls"] == "mlp2"), float(x["cls"] == "res"), float(x["set"] == "HE1"), float(x["set"] == "E1"),
                   np.log(x["H"])] for x in D])
R = np.array([[float(x["reg"] == g) for g in regs] for x in D])
w, se, f_full = fit(np.hstack([U, extra, R]), y)
_, _, f_noU = fit(np.hstack([extra, R]), y)
_, _, f_Uonly = fit(np.hstack([U, R]), y)
names = ["log(1+U inside)", "class mlp2", "class res", "set HE1", "set E1", "log width"]
print(f"\nlogistic regression on {len(D)} reproduced (mlp, mlp2, res; regime terms included):")
for nm, wi, si in list(zip(names, w, se)):
    print(f"  {nm:16s} {wi:+.2f} ± {si:.2f}  z = {wi / si:+.1f}")
print(f"  LR for U inside: {2 * (f_noU - f_full):.1f}, p = {stats.chi2.sf(2 * (f_noU - f_full), 1):.1e}")
print(f"  LR for class, set and width beyond U: {2 * (f_Uonly - f_full):.1f}, df 5, p = {stats.chi2.sf(2 * (f_Uonly - f_full), 5):.1e}")
for c in ("mlp", "mlp2", "res"):
    Dc = [i for i, x in enumerate(D) if x["cls"] == c]
    if len(Dc) > 20 and 0 < y[Dc].mean() < 1:
        wc, sc, _ = fit(np.hstack([U, R])[Dc], y[Dc])
        print(f"  {c:4s} alone: log(1+U inside) {wc[0]:+.2f} ± {sc[0]:.2f} (n = {len(Dc)})")

print("\nE1 models that reproduced a hidden attractor:")
for x in sorted((x for x in rows if x["set"] == "E1" and x["rep"]), key=lambda x: (x["reg"], x["cls"], x["H"], x["seed"])):
    mm = {k: v for k, v in x["mm"].items() if k not in ("reproduced",)}
    print(f"  {x['reg']:16s} {x['cls']:4s} H{x['H']:<5d} s{x['seed']:<3d} {x['att']:6s} U {x['U']:.0f} U inside {x['Un']} "
          f"Σind=1 {x['ok']} mse {x['mse']:.1e} {mm}")

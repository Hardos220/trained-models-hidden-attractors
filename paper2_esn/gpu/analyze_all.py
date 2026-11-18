"""Первая сводка этапов H, A, K по всем каталогам результатов (gpuq-results, local_results, node_results)."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIRS = [ROOT / "gpuq-results", ROOT / "local_results", ROOT / "node_results"]


def uniq(pattern):
    seen = {}
    for d in DIRS:
        for f in d.rglob(pattern):
            seen.setdefault(f.relative_to(f.parents[1]).as_posix() if "H" in pattern else f.name, f)
    return seen


# ---------------- H
print("=== H: hard-tanh, точное множество против протокола")
H = {}
for d in DIRS:
    for f in d.rglob("H/N*/*.json"):
        H.setdefault(f.parent.name + "/" + f.stem, f)
nE, nEf, miss_nets = [], [], 0
lab_pairs = Counter()
cov = defaultdict(lambda: [0, 0, 0])      # n_d -> [число повторов, M ≤ conservative, M ≤ refined]
by_N = defaultdict(lambda: Counter())
for key, f in sorted(H.items()):
    r = json.loads(f.read_text(encoding="utf-8"))
    nE.append(r["E"]["n"]); nEf.append(len(r["E_f"]["ids"]))
    if len(r["E_f"]["ids"]) < r["E"]["n"]:
        miss_nets += 1
    by_N[r["N"]]["nets"] += 1
    for a, v in r["attractors"].items():
        if "label_reference" not in v:
            continue
        lab_pairs[(v["label_reference"], v["label_protocol"])] += 1
        by_N[r["N"]]["attractors"] += 1
        for row in v["replicates"]:
            c = cov[row["n"]]
            c[0] += 1
            c[1] += row["M_true"] <= row["conservative"] + 1e-15
            c[2] += row["M_true"] <= row["refined"] + 1e-15
print(f"сетей {len(H)}; |E| медиана {np.median(nE):.0f}, макс {max(nE)}; сетей, где E_f неполно: {miss_nets}")
print("аттракторов по N:", {N: dict(c) for N, c in sorted(by_N.items())})
print("метки (эталон, протокол):", dict(lab_pairs))
for n_d, (tot, okc, okr) in sorted(cov.items()):
    print(f"  n_∂ = {n_d}: повторов {tot}, покрытие осторожной границы {okc / tot:.4f}, подстановочной {okr / tot:.4f}")

# ---------------- A
print("\n=== A: перепись сетей Лоренца")
A = {}
for d in DIRS:
    for f in d.rglob("A/r*.json"):
        A.setdefault(f.stem, f)
rows = []
for name, f in sorted(A.items()):
    r = json.loads(f.read_text(encoding="utf-8"))
    labs = [v["label"] for v in r["hiddenness"].values()]
    rows.append({"name": name, "b0": r["zero_bias"], "N": r["N"], "chaos": r["A_tr"]["reconstructed_chaotic"],
                 "n_att": len(r["attractors"]), "n_fp": len(r["fixed_points"]),
                 "n_stable": sum(1 for x in r["fixed_points"] if x["rho"] < 1),
                 "hidden": sum(1 for x in labs if x == "hidden_stat"), "labels": labs,
                 "collapsed": r["survival"]["fraction_collapsed"], "vpt": r["vpt"]["0.4"]["mean"],
                 "climate": min((at["metrics"]["lorenz"].get("delta_att", 9) for at in r["attractors"]
                                 if "lorenz" in at["metrics"]), default=None)})
for b0 in (False, True):
    R = [x for x in rows if x["b0"] == b0]
    if not R:
        continue
    ch = [x for x in R if x["chaos"]]
    print(f"b {'= 0' if b0 else 'случайное'}: сетей {len(R)}; обучающий аттрактор хаотический {len(ch)}; "
          f"паразитные аттракторы (кроме A_tr) в {sum(1 for x in R if x['n_att'] > (1 if x['chaos'] else 0))}; "
          f"скрытых hidden_stat всего {sum(x['hidden'] for x in R)} в {sum(1 for x in R if x['hidden'])} сетях; "
          f"устойчивые точки есть у {sum(1 for x in R if x['n_stable'])}; "
          f"обрушение A_tr за 5·10⁴ шагов (доля > 0.5) у {sum(1 for x in ch if x['collapsed'] > 0.5)} из {len(ch)}")
    lab = Counter(l for x in R for l in x["labels"])
    print("   метки аттракторов:", dict(lab))
good = [x for x in rows if x["chaos"] and x["vpt"] > 2.0]
print(f"сети с хорошим прогнозом (VPT(0.4) > 2 ляпуновских времени): {len(good)}; из них с паразитными аттракторами "
      f"{sum(1 for x in good if x['n_att'] > 1)}, с обрушением A_tr {sum(1 for x in good if x['collapsed'] > 0.5)}, "
      f"со скрытыми {sum(1 for x in good if x['hidden'])}")

# ---------------- K
print("\n=== K: множитель W")
K = {}
for d in DIRS:
    for f in d.rglob("K/*.json"):
        K.setdefault(f.stem, f)
for name, f in sorted(K.items()):
    r = json.loads(f.read_text(encoding="utf-8"))
    sel = r["selection"]
    line = []
    for row in r["rows"]:
        if "metrics" in row:
            rh = row.get("r_hat")
            tag = "S" if (rh is not None and rh <= 1e-6) else ("e" if (rh is not None and rh <= 1e-3) else "h")
            line.append(f"{row['s']:.2f}:{tag}{row['metrics']['delta_att']:.2f}")
        else:
            line.append(f"{row['s']:.2f}:{row.get('lost', '-')[:4]}")
    print(f"{name} [{sel['hidden_label_C']}]: " + " ".join(line))

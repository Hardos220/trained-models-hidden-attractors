"""Итоговая сводка по всем каталогам результатов: C (ESN N = 300 и 1000, NG-RC), H, A, K, L."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIRS = [ROOT / "gpuq-results", ROOT / "local_results", ROOT / "node_results"]


def collect(glob, key):
    out = {}
    for d in DIRS:
        for f in d.rglob(glob):
            out.setdefault(key(f), f)
    return out


meta = {k: json.loads((ROOT / "data" / f"{k}.json").read_text(encoding="utf-8"))
        for k in ["se11", "se3", "wangchen2", "wangchen2_a0.03", "le1", "chua_I", "chua_II"]}

# ---------------------------------------------------------------- C
C = collect("C/*/*.json", lambda f: f.parent.name + "/" + f.stem)
tab = defaultdict(lambda: {"n": 0, "rec": 0, "hid": 0, "eq": 0, "hid_eq": 0})
for key, f in C.items():
    r = json.loads(f.read_text(encoding="utf-8"))
    m = meta[r["regime"]]
    model = r["model"] if r["model"] == "ngrc" else f"esn{r.get('N', 300)}"
    best = None
    for at in r.get("attractors", []):
        for y in m["hidden"]:
            mm = at["metrics"].get(y, {})
            yk = "chaos" if m["attractors"][y]["kind"] == "chaotic" else "regular"
            if "delta_att" in mm and at["kind"] == yk and mm["delta_att"] < 0.2 and mm["d_H"] < 0.2:
                if best is None or mm["delta_att"] < best[0]:
                    best = (mm["delta_att"], at["name"])
    eq = any(fp["rho"] < 1 and fp.get("dist_true_eq", 9) < 0.01 for fp in r.get("fixed_points", []))
    t = tab[(model, r["regime"], r["set"])]
    t["n"] += 1
    t["eq"] += eq
    if best:
        t["rec"] += 1
        lab = r["hiddenness"][best[1]]["label"]
        t["hid"] += lab == "hidden_stat"
        t["hid_eq"] += (lab == "hidden_stat") and eq
print("=== C: n | скрытый аттрактор воспроизведён | из них скрыт в модели | верное устойчивое равновесие | скрыт и равновесие верно")
for model in ("esn300", "esn1000", "ngrc"):
    tot = Counter()
    for (mo, reg, s), t in sorted(tab.items()):
        if mo != model:
            continue
        print(f"{model:8s} {reg:16s} {s:4s} n={t['n']:3d} rec={t['rec']:3d} hid={t['hid']:3d} eq={t['eq']:3d} hid+eq={t['hid_eq']:3d}")
        for k_, v in t.items():
            tot[(s, k_)] += v
    for s in ("H1", "H0", "E1", "HE1"):
        if tot[(s, "n")]:
            print(f"   {model} ИТОГО {s}: n={tot[(s, 'n')]} rec={tot[(s, 'rec')]} hid={tot[(s, 'hid')]} eq={tot[(s, 'eq')]} hid+eq={tot[(s, 'hid_eq')]}")

# ---------------------------------------------------------------- H
H = collect("H/N*/*.json", lambda f: f.parent.name + "/" + f.stem)
pairs, cov = Counter(), defaultdict(lambda: [0, 0, 0])
nets_incomplete = 0
for f in H.values():
    r = json.loads(f.read_text(encoding="utf-8"))
    nets_incomplete += len(r["E_f"]["ids"]) < r["E"]["n"]
    for v in r["attractors"].values():
        if "label_reference" not in v:
            continue
        pairs[(v["label_reference"], v["label_protocol"])] += 1
        for row in v["replicates"]:
            c = cov[row["n"]]
            c[0] += 1
            c[1] += row["M_true"] <= row["conservative"] + 1e-15
            c[2] += row["M_true"] <= row["refined"] + 1e-15
hid_p = sum(v for (a, b), v in pairs.items() if b == "hidden_stat")
false_h = pairs[("self_excited", "hidden_stat")] + pairs[("excited_r", "hidden_stat")]
print(f"\n=== H: сетей {len(H)}, где метод Ньютона нашёл не все точки: {nets_incomplete}; аттракторов {sum(pairs.values())}")
print("   (эталон, протокол):", dict(pairs))
print(f"   протокол назвал скрытыми {hid_p}, из них на самом деле не скрыты {false_h}")
for n_d, (tot, okc, okr) in sorted(cov.items()):
    print(f"   n_∂={n_d}: повторов {tot}, покрытие осторожной {okc / tot:.4f}, подстановочной {okr / tot:.4f}")

# ---------------------------------------------------------------- A
A = collect("A/r*.json", lambda f: f.stem)
rows = []
for f in A.values():
    r = json.loads(f.read_text(encoding="utf-8"))
    labs = [v["label"] for v in r["hiddenness"].values()]
    rows.append(dict(b0=r["zero_bias"], chaos=r["A_tr"]["reconstructed_chaotic"], n_att=len(r["attractors"]),
                     hid=sum(l == "hidden_stat" for l in labs), stable=sum(x["rho"] < 1 for x in r["fixed_points"]),
                     coll=r["survival"]["fraction_collapsed"], vpt=r["vpt"]["0.4"]["mean"]))
print(f"\n=== A: сетей {len(rows)}")
for b0 in (False, True):
    R = [x for x in rows if x["b0"] == b0]
    ch = [x for x in R if x["chaos"]]
    print(f"   b{'=0' if b0 else ' случ.'}: {len(R)} сетей; хаос воспроизведён {len(ch)}; есть паразитные аттракторы "
          f"{sum(1 for x in R if x['n_att'] > (1 if x['chaos'] else 0))}; скрытых {sum(x['hid'] for x in R)}; "
          f"есть устойчивые точки {sum(1 for x in R if x['stable'])}; обрушение {sum(1 for x in ch if x['coll'] > 0.5)}/{len(ch)}")
good = [x for x in rows if x["chaos"] and x["vpt"] > 2]
print(f"   хороший прогноз (VPT>2): {len(good)}; с паразитными {sum(1 for x in good if x['n_att'] > 1)}; "
      f"с обрушением {sum(1 for x in good if x['coll'] > 0.5)}; с устойчивыми точками {sum(1 for x in good if x['stable'])}")
v_ok = [x["vpt"] for x in rows if x["chaos"] and x["coll"] <= 0.5]
v_bad = [x["vpt"] for x in rows if x["chaos"] and x["coll"] > 0.5]
print(f"   VPT медиана: без обрушения {np.median(v_ok):.2f}, с обрушением {np.median(v_bad):.2f}")

# ---------------------------------------------------------------- K fine
K = collect("K/*_fine.json", lambda f: f.stem)
ds_list = [0.0005, 0.002, 0.01]
grp = defaultdict(lambda: Counter())
for f in K.values():
    r = json.loads(f.read_text(encoding="utf-8"))
    g = "hidden" if r["selection"]["hidden_label_C"] == "hidden_stat" else "other"
    by = {round(x["s"], 4): x for x in r["rows"]}
    grp[g]["n"] += 1
    for ds in ds_list:
        keep = all(by.get(round(1 + sg * ds, 4), {}).get("metrics") for sg in (1, -1))
        stay = keep and all((by[round(1 + sg * ds, 4)].get("r_hat") or 1) > 1e-3 for sg in (1, -1))
        grp[g][f"keep{ds}"] += keep
        grp[g][f"hid{ds}"] += stay
print("\n=== K (мелкая сетка):")
for g, c in grp.items():
    print(f"   {g}: n={c['n']}; " + "; ".join(f"±{ds}: сохранён {c[f'keep{ds}']}, скрыт {c[f'hid{ds}']}" for ds in ds_list))

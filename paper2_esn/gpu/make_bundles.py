"""Компактные входы для этапов X и I из локальных результатов (чтобы не возить на другой ПК полные json/npz).

data/x_index.json + data/x_bundle.npz — сети этапа H: параметры, E_f, аттракторы с метками; массивы E, regions, reps.
data/i_index.json + data/i_bundle.npz — модели ESN этапа C: параметры и сведения о точках; массив fps.
Запуск: python gpu/make_bundles.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIRS = [ROOT / "gpuq-results", ROOT / "local_results", ROOT / "node_results"]


def collect(glob):
    out = {}
    for d in DIRS:
        for f in sorted(d.rglob(glob)):
            if ".failed" in f.name or not f.with_suffix(".npz").exists():
                continue
            out.setdefault(f.parent.name + "/" + f.stem, f)
    return out


# ---------------------------------------------------------------- X (этап H)
idx, arr = [], {}
for i, (key, f) in enumerate(sorted(collect("H/N*/*.json").items())):
    r = json.loads(f.read_text(encoding="utf-8"))
    z = np.load(f.with_suffix(".npz"))
    atts = {a: {k: v.get(k) for k in ("kind", "label_reference", "label_protocol", "r_hat_ref", "r_hat_protocol")}
            for a, v in r["attractors"].items() if "kind" in v}
    idx.append({"i": i, "key": key, "N": r["N"], "rho": r["rho"], "eta": r["eta"], "seed": r["seed"],
                "E_n": r["E"]["n"], "E_f": r["E_f"]["ids"], "attractors": atts})
    arr[f"{i}_E"], arr[f"{i}_regions"], arr[f"{i}_reps"] = z["E"], z["regions"], z["reps"]
(ROOT / "data" / "x_index.json").write_text(json.dumps(idx), encoding="utf-8")
np.savez_compressed(ROOT / "data" / "x_bundle.npz", **arr)
print("X:", len(idx), "сетей")

# ---------------------------------------------------------------- I (этап C, ESN)
idx, arr = [], {}
for i, (key, f) in enumerate(sorted(collect("C/*/*_esn_*.json").items())):
    r = json.loads(f.read_text(encoding="utf-8"))
    z = np.load(f.with_suffix(".npz"))
    idx.append({"i": i, "key": key, "regime": r["regime"], "set": r["set"], "N": r.get("N", 300), "rho": r["rho"],
                "eta": r["eta"], "seed": r["seed"],
                "fp": [{k: x.get(k) for k in ("rho", "n_unstable", "dist_true_eq", "found_by")} for x in r["fixed_points"]]})
    arr[f"{i}_fps"] = z["fps"]
(ROOT / "data" / "i_index.json").write_text(json.dumps(idx), encoding="utf-8")
np.savez_compressed(ROOT / "data" / "i_bundle.npz", **arr)
print("I:", len(idx), "моделей")

"""How close the E1 training data (transients of the visible basin) come to each hidden attractor.

Distances are measured in units of the attractor's standard deviation per coordinate: the smallest distance from an E1
sample to the attractor, and the fractions of the attractor within 0.1 and 0.3 of the E1 data. Data:
local/esn-gpu/data/<regime>.json and .npz (train_E1, att_<name>). Output: paper2_esn/results/numbers/e1_distance.json.

    python -X utf8 paper2_esn/gpu/paper_e1_distance.py
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

DATA = Path(os.environ.get("ESN_DATA", Path(__file__).resolve().parents[2] / "local" / "esn-gpu")) / "data"
OUT = Path(__file__).resolve().parents[1] / "results" / "numbers" / "e1_distance.json"
REGIMES = ["chua_I", "chua_II", "le1", "se11", "se3", "wangchen2", "wangchen2_a0.03"]


def main():
    res = {}
    for reg in REGIMES:
        m = json.loads((DATA / f"{reg}.json").read_text(encoding="utf-8"))
        z = np.load(DATA / f"{reg}.npz")
        E = z["train_E1"]
        for h in m["hidden"]:
            A = z[f"att_{h}"]
            s = np.array(m["attractors"][h]["std"])
            dE = cKDTree(A / s).query(E / s)[0]
            dA = cKDTree(E / s).query(A / s)[0]
            res[f"{reg}/{h}"] = {"min_dist": round(float(dE.min()), 3),
                                 "attractor_within_0.1": round(float(np.mean(dA < 0.1)), 3),
                                 "attractor_within_0.3": round(float(np.mean(dA < 0.3)), 3)}
            print(f"{reg:16s} {h:5s} {res[f'{reg}/{h}']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

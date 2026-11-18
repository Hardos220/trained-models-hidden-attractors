"""Stage X: completeness of the unstable-subspace test versus the number of random directions (32, 256, 2048).

Run from local/esn-gpu: python -I -X utf8 <this file>. Inputs: local_results/X (32 directions), node_results/X256,
node_results/X2048, data/x_missed_keys.txt (the 27 nets where 32 directions missed a self-excited attractor).
"""
import collections
import json
import os
from pathlib import Path

os.chdir(os.environ.get('ESN_DATA', 'local/esn-gpu'))
DIRS = {"c32": "local_results/X", "c256": "node_results/X256", "c2048": "node_results/X2048"}
rows = []
for k in [line.strip() for line in open("data/x_missed_keys.txt") if line.strip()]:
    f = k.replace("/", "_") + ".json"
    res = {c: json.load(open(Path(d) / f)) for c, d in DIRS.items()}
    for a, ra in res["c32"]["attractors"].items():
        row = dict(key=k, a=a, kind=ra["kind"], ref=ra["label_reference"],
                   nu=max([e["n_unstable"] for e in res["c32"]["equilibria"]] or [0]))
        row.update({c: bool(res[c]["attractors"].get(a, {}).get("excited_sub_E")) for c in DIRS})
        rows.append(row)
se = [r for r in rows if r["ref"] == "self_excited"]
missed = [r for r in se if not r["c32"]]
print("self-excited by reference:", len(se), " caught at 32/256/2048:", [sum(r[c] for r in se) for c in DIRS])
print("missed at 32:", len(missed), " caught later at 256/2048:", [sum(r[c] for r in missed) for c in ("c256", "c2048")])
print("caught at 32 but lost at 256/2048:", [sum(1 for r in se if r["c32"] and not r[c]) for c in ("c256", "c2048")])
still = [r for r in se if not r["c2048"]]
print("still missed at 2048:", len(still), dict(collections.Counter(r["kind"] for r in still)), "max dim E^u:", sorted(r["nu"] for r in still))
other = [r for r in rows if r["ref"] != "self_excited"]
for r in other:
    print("not self-excited:", r["key"], r["a"], r["kind"], r["ref"], "excited at 32/256/2048:", [r[c] for c in DIRS])

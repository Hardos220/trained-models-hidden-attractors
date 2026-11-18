"""Equilibrium-free control (s = 0): echo state networks of stage L trained on the Li-Sprott system (Rohm et al. 2021).

The true system has no equilibria, so all its attractors are hidden and s = 0; by Theorem 1 every bounded model has an
odd number K >= 1 of equilibria, all of them spurious. For each network: K, U, the attractors it reproduces (delta_att and
d_H below 0.2 against a true attractor) and their full-protocol labels. Output: paper2_esn/chaos/numbers/lisprott.json.

    python -X utf8 paper2_esn/gpu/paper_lisprott.py
"""
import json
from collections import Counter
import os
from pathlib import Path

ROOT = Path(os.environ.get("ESN_DATA", Path(__file__).resolve().parents[2] / "local" / "esn-gpu"))
OUT = Path(__file__).resolve().parents[1] / "chaos" / "numbers" / "lisprott.json"


def main():
    seen = {}
    for d in ("gpuq-results", "local_results", "node_results"):
        for f in (ROOT / d).rglob("L/esn_*.json"):
            seen.setdefault(f.name, f)
    models, rows = [], []
    for name, f in sorted(seen.items()):
        r = json.loads(f.read_text(encoding="utf-8"))
        fps = r.get("fixed_points", [])
        K, U = len(fps), sum(1 for x in fps if x.get("n_unstable", 0) > 0)
        models.append({"model": name, "case": r["case"], "K": K, "U": U})
        hz = r.get("hiddenness", {})
        for at in r.get("attractors", []):
            for tgt, mm in at.get("metrics", {}).items():
                if mm.get("delta_att", 9) < 0.2 and mm.get("d_H", 9) < 0.2:
                    rows.append({"model": name, "case": r["case"], "target": tgt, "attractor": at["name"], "kind": at["kind"],
                                 "K": K, "U": U, "label": hz.get(at["name"], {}).get("label")})
    out = {"n_models": len(models), "by_case": {}, "K_counts": dict(Counter(m["K"] for m in models)),
           "U_counts": dict(Counter(m["U"] for m in models)), "reproductions": rows,
           "labels_of_reproduced": dict(Counter(x["label"] for x in rows)),
           "labels_by_U": {str(u): dict(Counter(x["label"] for x in rows if x["U"] == u)) for u in sorted({x["U"] for x in rows})}}
    for case in sorted({m["case"] for m in models}):
        ms = [m for m in models if m["case"] == case]
        rs = [x for x in rows if x["case"] == case]
        out["by_case"][case] = {"n_models": len(ms), "K": dict(Counter(m["K"] for m in ms)), "U": dict(Counter(m["U"] for m in ms)),
                                "reproduced_targets": dict(Counter(x["target"] for x in rs)),
                                "labels": dict(Counter(x["label"] for x in rs))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "reproductions"}, indent=1))


if __name__ == "__main__":
    main()

"""Сводка результатов этапа C по каталогам gpuq-results/*/out/C: таблица моделей и аттракторов."""
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "gpuq-results")
rows = []
for f in sorted(ROOT.glob("*/out/C/*/*.json")):
    r = json.loads(f.read_text(encoding="utf-8"))
    true = r.get("true_attractors", {})
    best = []
    for at in r.get("attractors", []):
        m = at.get("metrics", {})
        if not m:
            continue
        nm, mm = min(((k, v) for k, v in m.items() if "delta_att" in v), key=lambda kv: kv[1]["delta_att"], default=(None, None))
        hl = r["hiddenness"].get(at["name"], {}).get("label")
        best.append(f"{at['name']}:{at['kind'][0]}→{nm}({mm['delta_att']:.3f},dH {mm['d_H']:.2f}) {hl}" if nm else f"{at['name']}:{hl}")
    fps = r.get("fixed_points", [])
    stab = [f for f in fps if f["rho"] < 1]
    near = min((f.get("dist_true_eq", 9) for f in stab), default=None)
    cen = r["census"].get("uL2") or r["census"].get("uL0")
    tot = sum(cen.values()) if cen else 0
    top = sorted(cen.items(), key=lambda kv: -kv[1])[:3] if cen else []
    rows.append((r["regime"], r["set"], r["model"], r.get("eta"), r.get("rho"), r.get("seed"), r["A_tr"]["kind"],
                 len(fps), len(stab), near, "; ".join(best), ", ".join(f"{k} {v / tot:.2f}" for k, v in top)))
for row in rows:
    reg, s, mdl, eta, rho, seed, atr, nfp, nst, near, best, top = row
    tag = f"eta={eta:g}" if mdl == "ngrc" else f"rho={rho} eta={eta:g} s{seed}"
    nr = "-" if near is None else f"{near:.3f}"
    print(f"{reg:16s} {s:4s} {mdl:5s} {tag:22s} A_tr={atr:7s} FP={nfp:2d} stab={nst} dEq={nr:6s} | {best} | uniform: {top}")

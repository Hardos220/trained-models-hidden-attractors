"""Сводка этапа C по режимам и наборам: восстановлен ли скрытый аттрактор, его метка скрытости в модели,
воспроизведено ли устойчивое истинное равновесие (или видимый цикл LC у Чуа II)."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
res = Path(sys.argv[1])
kind_filter = sys.argv[2] if len(sys.argv) > 2 else "esn"
tab = defaultdict(lambda: {"n": 0, "rec": 0, "labels": Counter(), "eq": 0, "vis": 0, "hid_and_eq": 0})
for f in sorted(res.rglob("C/*/*.json")):
    r = json.loads(f.read_text(encoding="utf-8"))
    if r["model"] != kind_filter:
        continue
    meta = json.loads((ROOT / "data" / f"{r['regime']}.json").read_text(encoding="utf-8"))
    t = tab[(r["regime"], r["set"])]
    t["n"] += 1
    best = None
    for at in r.get("attractors", []):
        for y in meta["hidden"]:
            m = at["metrics"].get(y, {})
            ykind = "chaos" if meta["attractors"][y]["kind"] == "chaotic" else "regular"
            if "delta_att" in m and at["kind"] == ykind and m["delta_att"] < 0.2 and m["d_H"] < 0.2:
                if best is None or m["delta_att"] < best[0]:
                    best = (m["delta_att"], at["name"])
    eq_ok = any(fp["rho"] < 1 and fp.get("dist_true_eq", 9) < 0.01 for fp in r.get("fixed_points", []))
    vis = meta["visible"]
    vis_ok = False
    if vis in meta["attractors"]:
        vis_ok = any(at["metrics"].get(vis, {}).get("delta_att", 9) < 0.2 and at["metrics"].get(vis, {}).get("d_H", 9) < 0.2
                     for at in r.get("attractors", []))
    if eq_ok:
        t["eq"] += 1
    if vis_ok:
        t["vis"] += 1
    if best:
        t["rec"] += 1
        lab = r["hiddenness"][best[1]]["label"]
        t["labels"][lab] += 1
        if eq_ok and lab == "hidden_stat":
            t["hid_and_eq"] += 1
print(f"{'режим':16s} {'набор':5s} {'n':>3s} {'HA восст.':>9s} {'метки восстановленных':45s} {'равн.':>5s} {'LC':>3s} {'скрыт+равн.':>11s}")
for (reg, s), t in sorted(tab.items()):
    labs = ", ".join(f"{k} {v}" for k, v in sorted(t["labels"].items()))
    print(f"{reg:16s} {s:5s} {t['n']:3d} {t['rec']:9d} {labs:45s} {t['eq']:5d} {t['vis']:3d} {t['hid_and_eq']:11d}")

"""Отбор моделей этапа K по результатам этапа C -> data/k_selection.json.

Модель ESN отбирается, если у неё есть колебательный аттрактор X того же типа, что скрытый аттрактор Y режима,
с Δatt(X, Y) < 0.2 и d_H(X, Y) < 0.2 (у цепей Чуа Y — тот из HA_a, HA_b, для которого Δatt меньше).
В каждом режиме — до 2 моделей со скрытым аттрактором (hidden_stat, undetermined) и остальные до 4 — с наименьшим Δatt.
Запуск: python gpu/select_k.py <каталог с результатами C (рекурсивно)> [--per 4]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
res_dir = Path(sys.argv[1])
per = int(sys.argv[sys.argv.index("--per") + 1]) if "--per" in sys.argv else 4
best = {}
for f in sorted(res_dir.rglob("C/*/*_esn_*.json")):
    r = json.loads(f.read_text(encoding="utf-8"))
    meta = json.loads((ROOT / "data" / f"{r['regime']}.json").read_text(encoding="utf-8"))
    hidden = meta["hidden"]
    cands = []
    for k, at in enumerate(r.get("attractors", [])):
        for y in hidden:
            m = at["metrics"].get(y)
            if not m or "delta_att" not in m:
                continue
            ykind = "chaos" if meta["attractors"][y]["kind"] == "chaotic" else "regular"
            if at["kind"] == ykind and m["delta_att"] < 0.2 and m["d_H"] < 0.2:
                cands.append((m["delta_att"], k, y))
    if not cands:
        continue
    d, k, y = min(cands)
    if not f.with_suffix(".npz").exists():      # json без npz (неполная выгрузка) — пропускаем
        continue
    z = np.load(f.with_suffix(".npz"))
    sel = {"regime": r["regime"], "set": r["set"], "rho": r["rho"], "eta": r["eta"], "seed": r["seed"],
           "name": f.stem, "target": y, "attractor": f"a{k}", "delta_att": d,
           "hidden_label_C": r["hiddenness"][f"a{k}"]["label"],
           "rep": z["reps"][k].tolist(), "fps": z["fps"].tolist()}
    best.setdefault(r["regime"], []).append(sel)
out = []
for reg, lst in sorted(best.items()):
    lst.sort(key=lambda s: s["delta_att"])
    hid = [s for s in lst if s["hidden_label_C"] in ("hidden_stat", "undetermined")][: per // 2]
    rest = [s for s in lst if s not in hid][: per - len(hid)]
    pick = hid + rest
    out += pick
    print(reg, len(lst), "qualify; selected", [(s["name"], round(s["delta_att"], 4), s["hidden_label_C"]) for s in pick])
outp = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "data/k_selection.json"
(ROOT / outp).write_text(json.dumps(out), encoding="utf-8")
print("selected", len(out))

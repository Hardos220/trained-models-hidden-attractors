"""Цепочки для А100 на 10–12 ч: N = 1000, H1/HE1 для wangchen2_a0.03, chua_I, chua_II, wangchen2 (по цепочке на режим)
и E1 для se11, se3, le1 (пятая цепочка). Команда, упавшая с ошибкой CUDA (код 3 или 4), повторяется до 3 раз:
готовые модели в каталоге задачи пропускаются.

Запуск: python gpu/make_night2.py  ->  gpu/n2_A.sh … gpu/n2_E.sh
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "python gpu/stage_c.py --kinds esn --N 1000 --etas 1e-7,1e-4 --seeds 0,1,2 --skip data/c_skip.txt"
HEAD = """#!/usr/bin/env bash
run() { for i in 1 2 3; do "$@"; rc=$?; [ $rc -ne 3 ] && [ $rc -ne 4 ] && return; echo "[chain] exit $rc, retry $i"; sleep 30; done; }
"""
CHAINS = {
    "A": [("wangchen2_a0.03", s, r) for s in ("H1", "HE1") for r in ("0.6", "0.9")],
    "B": [("chua_I", s, r) for s in ("H1", "HE1") for r in ("0.6", "0.9")],
    "C": [("chua_II", s, r) for s in ("H1", "HE1") for r in ("0.6", "0.9")],
    "D": [("wangchen2", s, r) for s in ("H1", "HE1") for r in ("0.6", "0.9")],
    "E": [(reg, "E1", r) for reg in ("se3", "le1", "se11") for r in ("0.6", "0.9")],
}
for k, cmds in CHAINS.items():
    lines = [HEAD.rstrip("\n"), "# " + ", ".join(f"{a} {b} ρ={c}" for a, b, c in cmds)]
    lines += [f"run {BASE} --regimes {reg} --sets {s} --rhos {rho}" for reg, s, rho in cmds]
    (ROOT / "gpu" / f"n2_{k}.sh").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(k, len(cmds), "команд")

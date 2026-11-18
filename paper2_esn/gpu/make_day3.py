"""Цепочки для А100 «до вечера» 21.10: хвосты цепочек n2 (с конца, с --seen-dirs их каталогов) и seed 3, 4 при N = 1000
для se3, se11, le1 (H1, HE1; сначала seed 3 по всем наборам, потом seed 4). Команда, упавшая с ошибкой CUDA, повторяется.

Запуск: python gpu/make_day3.py  ->  gpu/d3_T.sh, d3_S3.sh, d3_S11.sh, d3_L1.sh
Каталоги задач 2202, 2199, 2200 не удалять с сервера, пока d3_T не закончится.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "python gpu/stage_c.py --kinds esn --N 1000 --etas 1e-7,1e-4 --skip data/c_skip.txt"
HEAD = """#!/usr/bin/env bash
run() { for i in 1 2 3; do "$@"; rc=$?; [ $rc -ne 3 ] && [ $rc -ne 4 ] && return; echo "[chain] exit $rc, retry $i"; sleep 30; done; }"""
SEEN = {2202: "/srv/gpuq/jobs/2202/out", 2199: "/srv/gpuq/jobs/2199/out", 2200: "/srv/gpuq/jobs/2200/out"}
chains = {"T": [f"run {BASE} --seeds 0,1,2 --regimes chua_I --sets HE1 --rhos 0.9 --reverse --seen-dirs {SEEN[2202]}",
                f"run {BASE} --seeds 0,1,2 --regimes wangchen2 --sets HE1 --rhos 0.9 --reverse --seen-dirs {SEEN[2199]}"]
          + [f"run {BASE} --seeds 0,1,2 --regimes {reg} --sets E1 --rhos {rho} --reverse --seen-dirs {SEEN[2200]}"
             for reg in ("se11", "le1") for rho in ("0.9", "0.6")]}
for key, reg in (("S3", "se3"), ("S11", "se11"), ("L1", "le1")):
    chains[key] = [f"run {BASE} --seeds {sd} --regimes {reg} --sets {s} --rhos {rho}"
                   for sd in (3, 4) for s in ("H1", "HE1") for rho in ("0.6", "0.9")]
for k, cmds in chains.items():
    (ROOT / "gpu" / f"d3_{k}.sh").write_text("\n".join([HEAD] + cmds) + "\n", encoding="utf-8", newline="\n")
    print(k, len(cmds), "команд")

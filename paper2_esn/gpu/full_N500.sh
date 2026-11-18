#!/usr/bin/env bash
# Полный протокол (этап C) для моделей развёртки N: se3, se11, HE1, N = 500 — проверка метки «не возбуждён».
run() { for i in 1 2 3; do "$@"; rc=$?; [ $rc -ne 3 ] && [ $rc -ne 4 ] && return; echo "[chain] exit $rc, retry $i"; sleep 30; done; }
run python gpu/stage_c.py --regimes se3,se11 --sets HE1 --kinds esn --N 500 --rhos 0.9 --etas 1e-7,1e-4 --seeds 0,1,2

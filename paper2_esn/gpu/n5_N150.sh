#!/usr/bin/env bash
# Этап N, пересчёт N = 150 (на CPU node метод Ньютона обрывался на CPU): 7 систем HE1 и se3, se11 H1.
run() { for i in 1 2 3; do "$@"; rc=$?; [ $rc -ne 3 ] && [ $rc -ne 4 ] && return; echo "[chain] exit $rc, retry $i"; sleep 30; done; }
run python gpu/stage_n.py --regimes se3,se11,le1,chua_I,chua_II,wangchen2,wangchen2_a0.03 --sets HE1 --Ns 150 --rhos 0.9 --etas 1e-7,1e-4 --seeds 0,1,2,3,4
run python gpu/stage_n.py --regimes se3,se11 --sets H1 --Ns 150 --rhos 0.9 --etas 1e-7,1e-4 --seeds 0,1,2,3,4

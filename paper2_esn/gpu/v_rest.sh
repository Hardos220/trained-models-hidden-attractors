#!/usr/bin/env bash
# Этап V для chua_II, wangchen2, wangchen2_a0.03, N = 300.
run() { for i in 1 2 3; do "$@"; rc=$?; [ $rc -ne 3 ] && [ $rc -ne 4 ] && return; echo "[chain] exit $rc, retry $i"; sleep 30; done; }
run python gpu/stage_v.py --regimes chua_II,wangchen2,wangchen2_a0.03 --Ns 300 --rhos 0.9 --etas 1e-7,1e-4 --seeds 0,1,2

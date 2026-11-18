#!/usr/bin/env bash
# Этап V (вмешательство через данные) для le1, N = 300 и 700.
run() { for i in 1 2 3; do "$@"; rc=$?; [ $rc -ne 3 ] && [ $rc -ne 4 ] && return; echo "[chain] exit $rc, retry $i"; sleep 30; done; }
run python gpu/stage_v.py --regimes le1 --Ns 300,700 --rhos 0.9 --etas 1e-7,1e-4 --seeds 0,1,2

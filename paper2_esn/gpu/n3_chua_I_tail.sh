#!/usr/bin/env bash
# Этап N, chua_I, N = 2000: последние 4 модели после нехватки ОЗУ в #2569.
run() { for i in 1 2 3; do "$@"; rc=$?; [ $rc -ne 3 ] && [ $rc -ne 4 ] && return; echo "[chain] exit $rc, retry $i"; sleep 30; done; }
run python gpu/stage_n.py --regimes chua_I --sets HE1 --Ns 2000 --rhos 0.9 --etas 1e-7 --seeds 2
run python gpu/stage_n.py --regimes chua_I --sets HE1 --Ns 2000 --rhos 0.9 --etas 1e-4 --seeds 0,1,2

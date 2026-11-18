#!/usr/bin/env bash
# Этап N, большие N для se11 (H1), повтор после ошибки CUDA.
run() { for i in 1 2 3; do "$@"; rc=$?; [ $rc -ne 3 ] && [ $rc -ne 4 ] && return; echo "[chain] exit $rc, retry $i"; sleep 30; done; }
run python gpu/stage_n.py --regimes se11 --sets H1 --Ns 300,500,700,1000,1400,2000 --rhos 0.9 --etas 1e-7,1e-4 --seeds 0,1,2

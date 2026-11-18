#!/usr/bin/env bash
# Этап N, большие N для le1 (HE1, ρ = 0.9, η = 1e-7 и 1e-4, seed 0–2), повтор после ошибки CUDA.
run() { for i in 1 2 3; do "$@"; rc=$?; [ $rc -ne 3 ] && [ $rc -ne 4 ] && return; echo "[chain] exit $rc, retry $i"; sleep 30; done; }
run python gpu/stage_n.py --regimes le1 --sets HE1 --Ns 300,500,700,1000,1400,2000 --rhos 0.9 --etas 1e-7,1e-4 --seeds 0,1,2

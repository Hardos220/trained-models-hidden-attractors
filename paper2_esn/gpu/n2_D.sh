#!/usr/bin/env bash
run() { for i in 1 2 3; do "$@"; rc=$?; [ $rc -ne 3 ] && [ $rc -ne 4 ] && return; echo "[chain] exit $rc, retry $i"; sleep 30; done; }
# wangchen2 H1 ρ=0.6, wangchen2 H1 ρ=0.9, wangchen2 HE1 ρ=0.6, wangchen2 HE1 ρ=0.9
run python gpu/stage_c.py --kinds esn --N 1000 --etas 1e-7,1e-4 --seeds 0,1,2 --skip data/c_skip.txt --regimes wangchen2 --sets H1 --rhos 0.6
run python gpu/stage_c.py --kinds esn --N 1000 --etas 1e-7,1e-4 --seeds 0,1,2 --skip data/c_skip.txt --regimes wangchen2 --sets H1 --rhos 0.9
run python gpu/stage_c.py --kinds esn --N 1000 --etas 1e-7,1e-4 --seeds 0,1,2 --skip data/c_skip.txt --regimes wangchen2 --sets HE1 --rhos 0.6
run python gpu/stage_c.py --kinds esn --N 1000 --etas 1e-7,1e-4 --seeds 0,1,2 --skip data/c_skip.txt --regimes wangchen2 --sets HE1 --rhos 0.9

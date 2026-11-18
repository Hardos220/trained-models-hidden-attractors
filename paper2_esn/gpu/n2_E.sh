#!/usr/bin/env bash
run() { for i in 1 2 3; do "$@"; rc=$?; [ $rc -ne 3 ] && [ $rc -ne 4 ] && return; echo "[chain] exit $rc, retry $i"; sleep 30; done; }
# se3 E1 ρ=0.6, se3 E1 ρ=0.9, le1 E1 ρ=0.6, le1 E1 ρ=0.9, se11 E1 ρ=0.6, se11 E1 ρ=0.9
run python gpu/stage_c.py --kinds esn --N 1000 --etas 1e-7,1e-4 --seeds 0,1,2 --skip data/c_skip.txt --regimes se3 --sets E1 --rhos 0.6
run python gpu/stage_c.py --kinds esn --N 1000 --etas 1e-7,1e-4 --seeds 0,1,2 --skip data/c_skip.txt --regimes se3 --sets E1 --rhos 0.9
run python gpu/stage_c.py --kinds esn --N 1000 --etas 1e-7,1e-4 --seeds 0,1,2 --skip data/c_skip.txt --regimes le1 --sets E1 --rhos 0.6
run python gpu/stage_c.py --kinds esn --N 1000 --etas 1e-7,1e-4 --seeds 0,1,2 --skip data/c_skip.txt --regimes le1 --sets E1 --rhos 0.9
run python gpu/stage_c.py --kinds esn --N 1000 --etas 1e-7,1e-4 --seeds 0,1,2 --skip data/c_skip.txt --regimes se11 --sets E1 --rhos 0.6
run python gpu/stage_c.py --kinds esn --N 1000 --etas 1e-7,1e-4 --seeds 0,1,2 --skip data/c_skip.txt --regimes se11 --sets E1 --rhos 0.9

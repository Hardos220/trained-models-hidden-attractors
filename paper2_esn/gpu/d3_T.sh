#!/usr/bin/env bash
run() { for i in 1 2 3; do "$@"; rc=$?; [ $rc -ne 3 ] && [ $rc -ne 4 ] && return; echo "[chain] exit $rc, retry $i"; sleep 30; done; }
run python gpu/stage_c.py --kinds esn --N 1000 --etas 1e-7,1e-4 --skip data/c_skip.txt --seeds 0,1,2 --regimes chua_I --sets HE1 --rhos 0.9 --reverse --seen-dirs /srv/gpuq/jobs/2202/out
run python gpu/stage_c.py --kinds esn --N 1000 --etas 1e-7,1e-4 --skip data/c_skip.txt --seeds 0,1,2 --regimes wangchen2 --sets HE1 --rhos 0.9 --reverse --seen-dirs /srv/gpuq/jobs/2199/out
run python gpu/stage_c.py --kinds esn --N 1000 --etas 1e-7,1e-4 --skip data/c_skip.txt --seeds 0,1,2 --regimes se11 --sets E1 --rhos 0.9 --reverse --seen-dirs /srv/gpuq/jobs/2200/out
run python gpu/stage_c.py --kinds esn --N 1000 --etas 1e-7,1e-4 --skip data/c_skip.txt --seeds 0,1,2 --regimes se11 --sets E1 --rhos 0.6 --reverse --seen-dirs /srv/gpuq/jobs/2200/out
run python gpu/stage_c.py --kinds esn --N 1000 --etas 1e-7,1e-4 --skip data/c_skip.txt --seeds 0,1,2 --regimes le1 --sets E1 --rhos 0.9 --reverse --seen-dirs /srv/gpuq/jobs/2200/out
run python gpu/stage_c.py --kinds esn --N 1000 --etas 1e-7,1e-4 --skip data/c_skip.txt --seeds 0,1,2 --regimes le1 --sets E1 --rhos 0.6 --reverse --seen-dirs /srv/gpuq/jobs/2200/out

#!/usr/bin/env bash
# Stage E, NG-RC part: next-generation reservoir computers on random subsets of the E1 transients, all regimes.
run() { for i in 1 2 3; do "$@"; rc=$?; [ $rc -ne 3 ] && [ $rc -ne 4 ] && return; echo "[chain] exit $rc, retry $i"; sleep 30; done; }
run python gpu/stage_e.py --regimes se3,wangchen2_a0.03,se11,le1,wangchen2,chua_I,chua_II --kinds ngrc --reps 10 --etas 1e-3,1e-1,10
tar czf "$GPUQ_OUT/A_E.tgz" -C "$GPUQ_OUT" E && echo "[chain] archive $(du -h "$GPUQ_OUT/A_E.tgz" | cut -f1)"

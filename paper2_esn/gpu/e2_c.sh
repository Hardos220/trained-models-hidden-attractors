#!/usr/bin/env bash
# Stage E, NG-RC part for wangchen2,chua_I (10 subsets of E1, eta 1e-3, 1e-1, 10). Models whose json exist are skipped.
run() { for i in 1 2 3; do "$@"; rc=$?; [ $rc -ne 3 ] && [ $rc -ne 4 ] && return; echo "[chain] exit $rc, retry $i"; sleep 30; done; }
run python gpu/stage_e.py --regimes wangchen2,chua_I --kinds ngrc --reps 10 --etas 1e-3,1e-1,10
tar czf "$GPUQ_OUT/A_E.tgz" -C "$GPUQ_OUT" E && echo "[chain] archive $(du -h "$GPUQ_OUT/A_E.tgz" | cut -f1)"

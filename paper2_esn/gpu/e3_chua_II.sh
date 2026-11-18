#!/usr/bin/env bash
# Stage E, NG-RC trained on H1 and HE1 for chua_II (random 75% window of H1, plus 15 of 20 transients for HE1), 5 realizations,
# eta 1e-3, 1e-1, 10. Models whose json exist are skipped.
run() { for i in 1 2 3; do "$@"; rc=$?; [ $rc -ne 3 ] && [ $rc -ne 4 ] && return; echo "[chain] exit $rc, retry $i"; sleep 30; done; }
run python gpu/stage_e.py --regimes chua_II --kinds ngrc --sets H1,HE1 --reps 5 --etas 1e-3,1e-1,10
tar czf "$GPUQ_OUT/A_E.tgz" -C "$GPUQ_OUT" E && echo "[chain] archive $(du -h "$GPUQ_OUT/A_E.tgz" | cut -f1)"

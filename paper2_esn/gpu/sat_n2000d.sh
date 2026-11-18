#!/usr/bin/env bash
# Stage S (saturation of the ESN equilibrium search, check), design group n2000d of data/sat_design.json.
# Networks whose json exist are skipped; a network in progress resumes from its block checkpoint.
run() { for i in 1 2 3; do "$@"; rc=$?; [ $rc -ne 3 ] && [ $rc -ne 4 ] && return; echo "[chain] exit $rc, retry $i"; sleep 30; done; }
run python gpu/stage_sat.py --design data/sat_design.json --groups n2000d
tar czf "$GPUQ_OUT/A_S.tgz" -C "$GPUQ_OUT" S && echo "[chain] archive $(du -h "$GPUQ_OUT/A_S.tgz" | cut -f1)"

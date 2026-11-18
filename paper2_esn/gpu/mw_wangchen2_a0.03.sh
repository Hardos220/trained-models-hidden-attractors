#!/usr/bin/env bash
# Stage MW (stage M models retrained with saved weights, check of completeness) for wangchen2_a0.03:
# H1, HE1 x mlp H64, H256, mlp2 H64, H256, res H64 x 20 seeds. Groups whose files exist are skipped.
run() { for i in 1 2 3; do "$@"; rc=$?; [ $rc -ne 3 ] && [ $rc -ne 4 ] && return; echo "[chain] exit $rc, retry $i"; sleep 30; done; }
run python gpu/stage_mw.py --regimes wangchen2_a0.03 --sets H1,HE1 --specs mlp:64,mlp:256,mlp2:64,mlp2:256,res:64 --seeds 20
tar czf "$GPUQ_OUT/A_MW.tgz" -C "$GPUQ_OUT" MW && echo "[chain] archive $(du -h "$GPUQ_OUT/A_MW.tgz" | cut -f1)"

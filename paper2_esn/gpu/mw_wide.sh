#!/usr/bin/env bash
# Stage MW, wide classes (mlp H256, mlp2 H256; H1, HE1; 20 seeds as in stage M) for the five regimes whose narrow classes
# are trained on a CPU node (gpu/submit_mw_node.sh), plus the narrow classes of se11 (its CPU task was cancelled to save
# time); se3 has all classes in gpu/mw_se3.sh. Groups whose files exist are skipped.
run() { for i in 1 2 3; do "$@"; rc=$?; [ $rc -ne 3 ] && [ $rc -ne 4 ] && return; echo "[chain] exit $rc, retry $i"; sleep 30; done; }
run python gpu/stage_mw.py --regimes wangchen2_a0.03,chua_I,se11,wangchen2,chua_II --sets H1,HE1 --specs mlp2:256,mlp:256 --seeds 20
run python gpu/stage_mw.py --regimes se11 --sets H1,HE1 --specs mlp:64,mlp2:64,res:64 --seeds 20
tar czf "$GPUQ_OUT/A_MW.tgz" -C "$GPUQ_OUT" MW && echo "[chain] archive $(du -h "$GPUQ_OUT/A_MW.tgz" | cut -f1)"

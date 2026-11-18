#!/usr/bin/env bash
# Stage P for se11: two-hidden-layer perceptrons (H = 256) trained with and without the penalty on unstable equilibria in
# the region (lam 0, 0.1, 1), training sets HE1 and H1, 20 seeds. Groups whose json exist are skipped.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { for i in 1 2 3; do "$@"; rc=$?; [ $rc -ne 3 ] && [ $rc -ne 4 ] && return; echo "[chain] exit $rc, retry $i"; sleep 30; done; }
run python gpu/stage_p.py --regimes se11 --sets HE1,H1 --classes mlp2 --widths 256 --lams 0,0.1,1 --seeds 20
tar czf "$GPUQ_OUT/A_P.tgz" -C "$GPUQ_OUT" P && echo "[chain] archive $(du -h "$GPUQ_OUT/A_P.tgz" | cut -f1)"

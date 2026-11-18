#!/usr/bin/env bash
# Этап M, вторая очередь для se3: mlp2 на H1 и HE1, все классы на E1.
run() { for i in 1 2 3; do "$@"; rc=$?; [ $rc -ne 3 ] && [ $rc -ne 4 ] && return; echo "[chain] exit $rc, retry $i"; sleep 30; done; }
run python gpu/stage_m.py --regimes se3 --sets H1,HE1 --classes mlp2 --widths 256,64,512,128,32,16,1024,8 --seeds 20
run python gpu/stage_m.py --regimes se3 --sets E1 --classes mlp,res,mlp2 --widths 256,64,512,128,32,16,1024,8 --seeds 20
tar czf "$GPUQ_OUT/A_M.tgz" -C "$GPUQ_OUT" M && echo "[chain] архив $(du -h "$GPUQ_OUT/A_M.tgz" | cut -f1)"

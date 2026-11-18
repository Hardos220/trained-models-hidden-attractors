#!/usr/bin/env bash
# Этап M, первая очередь для wangchen2_a0.03: mlp и res на H1 и HE1.
run() { for i in 1 2 3; do "$@"; rc=$?; [ $rc -ne 3 ] && [ $rc -ne 4 ] && return; echo "[chain] exit $rc, retry $i"; sleep 30; done; }
run python gpu/stage_m.py --regimes wangchen2_a0.03 --sets H1,HE1 --classes mlp,res --widths 256,64,512,128,32,16,1024,8 --seeds 20

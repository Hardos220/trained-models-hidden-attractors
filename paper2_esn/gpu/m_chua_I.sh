#!/usr/bin/env bash
# Этап M (полносвязные суррогаты в 3D) для chua_I: H1, HE1, E1 × mlp, mlp2, res × 8 ширин × 20 seed.
run() { for i in 1 2 3; do "$@"; rc=$?; [ $rc -ne 3 ] && [ $rc -ne 4 ] && return; echo "[chain] exit $rc, retry $i"; sleep 30; done; }
run python gpu/stage_m.py --regimes chua_I --sets H1,HE1,E1 --classes mlp,mlp2,res --widths 256,1024,64,512,128,32,16,8 --seeds 20

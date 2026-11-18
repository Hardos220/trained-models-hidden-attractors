#!/usr/bin/env bash
# Ночная цепочка B: esn-C1000-le1-H1-r0.6, esn-C1000-le1-H1-r0.9, esn-C1000-le1-HE1-r0.6, esn-C1000-le1-HE1-r0.9
python gpu/stage_c.py --regimes le1 --sets H1 --kinds esn --N 1000 --rhos 0.6 --etas 1e-7,1e-4 --seeds 0,1,2 --skip data/c_skip.txt
python gpu/stage_c.py --regimes le1 --sets H1 --kinds esn --N 1000 --rhos 0.9 --etas 1e-7,1e-4 --seeds 0,1,2 --skip data/c_skip.txt
python gpu/stage_c.py --regimes le1 --sets HE1 --kinds esn --N 1000 --rhos 0.6 --etas 1e-7,1e-4 --seeds 0,1,2 --skip data/c_skip.txt
python gpu/stage_c.py --regimes le1 --sets HE1 --kinds esn --N 1000 --rhos 0.9 --etas 1e-7,1e-4 --seeds 0,1,2 --skip data/c_skip.txt

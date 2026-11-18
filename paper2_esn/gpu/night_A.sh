#!/usr/bin/env bash
# Ночная цепочка A: esn-C1000-se3-H1-r0.9, esn-C1000-se3-HE1-r0.6, esn-C1000-se3-HE1-r0.9, esn-H16-s5-10
python gpu/stage_c.py --regimes se3 --sets H1 --kinds esn --N 1000 --rhos 0.9 --etas 1e-7,1e-4 --seeds 0,1,2 --skip data/c_skip.txt
python gpu/stage_c.py --regimes se3 --sets HE1 --kinds esn --N 1000 --rhos 0.6 --etas 1e-7,1e-4 --seeds 0,1,2 --skip data/c_skip.txt
python gpu/stage_c.py --regimes se3 --sets HE1 --kinds esn --N 1000 --rhos 0.9 --etas 1e-7,1e-4 --seeds 0,1,2 --skip data/c_skip.txt
python gpu/stage_h.py --N 16 --seeds 5,6,7,8,9,10 --skip data/h_skip.txt

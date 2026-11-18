#!/usr/bin/env bash
# Ночная цепочка D: esn-H18-s11-14, esn-H20-s9-12
python gpu/stage_h.py --N 18 --seeds 11,12,13,14 --skip data/h_skip.txt
python gpu/stage_h.py --N 20 --rho 0.9,1.2 --seeds 9,10,11,12 --skip data/h_skip.txt

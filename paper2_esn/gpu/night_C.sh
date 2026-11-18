#!/usr/bin/env bash
# Ночная цепочка C: esn-H18-s5-7, esn-H18-s8-10, esn-H16-s11-15
python gpu/stage_h.py --N 18 --seeds 5,6,7 --skip data/h_skip.txt
python gpu/stage_h.py --N 18 --seeds 8,9,10 --skip data/h_skip.txt
python gpu/stage_h.py --N 16 --seeds 11,12,13,14,15 --skip data/h_skip.txt

#!/usr/bin/env bash
# Ночная цепочка E: esn-H20-s3-5, esn-H20-s6-8, esn-H18-s15-17
python gpu/stage_h.py --N 20 --rho 0.9,1.2 --seeds 3,4,5 --skip data/h_skip.txt
python gpu/stage_h.py --N 20 --rho 0.9,1.2 --seeds 6,7,8 --skip data/h_skip.txt
python gpu/stage_h.py --N 18 --seeds 15,16,17 --skip data/h_skip.txt

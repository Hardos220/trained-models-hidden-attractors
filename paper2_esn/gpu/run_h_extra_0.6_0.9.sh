#!/usr/bin/env bash
# Дополнительные сети H: N = 16, затем N = 18, ρ = 0.6,0.9; готовое — в data/h_skip.txt.
python gpu/stage_h.py --N 16 --rho 0.6,0.9 --seeds 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 --skip data/h_skip.txt
python gpu/stage_h.py --N 18 --rho 0.6,0.9 --seeds 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 --skip data/h_skip.txt

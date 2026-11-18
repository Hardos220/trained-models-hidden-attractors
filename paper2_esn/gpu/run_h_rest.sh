#!/usr/bin/env bash
# Этап H на A100 с резервом 1 ГБ: оставшиеся сети N = 18 (готовые перечислены в data/h_skip.txt), затем N = 20.
python gpu/stage_h.py --N 18 --skip data/h_skip.txt
python gpu/stage_h.py --N 20 --rho 0.9,1.2 --seeds 0,1,2 --skip data/h_skip.txt

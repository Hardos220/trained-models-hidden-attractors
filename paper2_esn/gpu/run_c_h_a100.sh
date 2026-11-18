#!/usr/bin/env bash
# Чуа I (этап C) и остаток этапа H на A100; готовое перечислено в data/c_skip.txt и data/h_skip.txt.
python gpu/stage_c.py --regimes chua_I --kinds esn,ngrc --skip data/c_skip.txt
python gpu/stage_h.py --N 18 --skip data/h_skip.txt
python gpu/stage_h.py --N 20 --rho 0.9,1.2 --seeds 0,1,2 --skip data/h_skip.txt

#!/usr/bin/env bash
# Продолжение этапов C (Чуа I) и H с конца списка; модели, готовые у задач #1266 и #1267, пропускаются.
python gpu/stage_c.py --regimes chua_I --kinds esn,ngrc --reverse --skip data/c_skip.txt --seen-dirs /srv/gpuq/jobs/1266/out
python gpu/stage_h.py --N 18 --reverse --skip data/h_skip.txt --seen-dirs /srv/gpuq/jobs/1267/out
python gpu/stage_h.py --N 20 --rho 0.9,1.2 --seeds 0,1,2 --reverse --skip data/h_skip.txt --seen-dirs /srv/gpuq/jobs/1267/out

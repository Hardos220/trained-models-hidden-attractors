#!/usr/bin/env bash
# Пункт 1: недостающие модели wangchen2_a0.03, HE1, N = 300 — на CPU node, пятью небольшими задачами.
cd "${ESN_DATA:-local/esn-gpu}" || exit 1
R=rmq   # local job queue: rmq submit <resources> -- <command>
for seeds in 3,4,5 6,7 8,9 10,11 12; do
  $R submit --where cpu-node --cpu 4 --ram 3 --time 12h --preempt requeue \
    --name "esn-C-wc03-HE1-s${seeds//,/-}" \
    --requirements requirements-node.txt --env GPUQ_OUT=node_results --results node_results/ \
    --files "gpu/*.py" --files "src/hsearch/*.py" --files "src/hsearch/systems/*.py" \
    --files "data/wangchen2_a0.03.json" --files "data/wangchen2_a0.03.npz" --files "data/wangchen2_a0.03_l1.npz" \
    --files "data/c_skip.txt" --files "requirements-node.txt" \
    -- python gpu/stage_c.py --regimes wangchen2_a0.03 --sets HE1 --kinds esn --rhos 0.6,0.9 --etas 1e-7,1e-4 \
       --seeds "$seeds" --skip data/c_skip.txt
done

#!/usr/bin/env bash
# CPU node: этап R (контроль «до обучения», остаток после #736) и этап V (вмешательство, дополнительные seed 3–4, N = 300).
cd "${ESN_DATA:-local/esn-gpu}" || exit 1
R=rmq   # local job queue: rmq submit <resources> -- <command>
COMMON=(--where cpu-node --cpu 4 --ram 3 --preempt requeue --requirements requirements-node.txt
        --env GPUQ_OUT=node_results --results node_results/
        --files "gpu/*.py" --files "src/hsearch/*.py" --files "src/hsearch/systems/*.py" --files requirements-node.txt)
for spec in "se3 2" "se11 0,1,2" "chua_I 0,1,2" "le1 0,1,2"; do
  set -- $spec
  $R submit "${COMMON[@]}" --time 6h --name "esn-R-$1" --files "data/$1.*" --files "data/${1}_l1.npz" \
    -- python gpu/stage_r.py --regimes "$1" --sets H1 --N 300 --etas 1e-4 --seeds "$2" --draws 2
done
for reg in se3 se11 le1 chua_I; do
  $R submit "${COMMON[@]}" --time 10h --name "esn-V-$reg-s34" --files "data/$reg.*" --files "data/${reg}_l1.npz" \
    -- python gpu/stage_v.py --regimes "$reg" --Ns 300 --rhos 0.9 --etas 1e-7,1e-4 --seeds 3,4
done

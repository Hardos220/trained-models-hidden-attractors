#!/usr/bin/env bash
# CPU node: sparse quadratic maps trained on H1 and HE1 (random 75% window of the H1 trajectory, plus 15 of the 20
# transients for HE1), 10 realizations x 3 thresholds per set, six regimes (LE1 is treated separately in the paper).
cd "${ESN_DATA:-local/esn-gpu}" || exit 1
R=rmq   # local job queue: rmq submit <resources> -- <command>
COMMON=(--where cpu-node --cpu 2 --ram 2 --preempt requeue --requirements requirements-node.txt
        --env GPUQ_OUT=node_results --results node_results/
        --files "gpu/*.py" --files "src/hsearch/*.py" --files "src/hsearch/systems/*.py" --files requirements-node.txt)
for reg in se3 wangchen2_a0.03 se11 wangchen2 chua_I chua_II; do
  $R submit "${COMMON[@]}" --time 3h --name "esn-E-polyH-$reg" --files "data/$reg.*" --files "data/${reg}_l1.npz" \
    -- python gpu/stage_e.py --regimes "$reg" --kinds poly --sets H1,HE1 --reps 10 --thresholds 0.003,0.01,0.03
done

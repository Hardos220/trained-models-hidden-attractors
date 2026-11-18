#!/usr/bin/env bash
# CPU node (CPU): stage MW for five regimes (stage M models retrained with saved weights, input of krawczyk_m.py).
# CPU-feasible variant of the A100 chains gpu/mw_<regime>.sh: 10 seeds per group instead of 20 and the narrow classes
# only (mlp H64, mlp2 H64, res H64; H1 and HE1; 20000 steps as in stage M); mlp H256 and mlp2 H256 of these regimes are
# trained on the A100 (gpu/mw_wide.sh). On the CPU the random numbers differ from the GPU ones, so these models are not
# the stage M models themselves (the A100 run for se3 is the reproduction check).
cd "${ESN_DATA:-local/esn-gpu}" || exit 1
R=rmq   # local job queue: rmq submit <resources> -- <command>
COMMON=(--where cpu-node --cpu 1 --ram 2 --preempt requeue --requirements requirements-node.txt
        --env GPUQ_OUT=node_results --results node_results/
        --files "gpu/*.py" --files "src/hsearch/*.py" --files "src/hsearch/systems/*.py" --files requirements-node.txt)
for reg in ${REGS:-wangchen2_a0.03 chua_I wangchen2 chua_II}; do
  $R submit "${COMMON[@]}" --time 8h --name "esn-MW-cpu-$reg" --files "data/$reg.*" --files "data/${reg}_l1.npz" \
    -- python gpu/stage_mw.py --regimes "$reg" --sets H1,HE1 --specs mlp:64,mlp2:64,res:64 --seeds 10
done

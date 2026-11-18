#!/usr/bin/env bash
# CPU node: N = 300, HE1, seed $1 (по умолчанию 3) для chua_I, chua_II, wangchen2 — шестью небольшими задачами по 2 модели (по ρ).
cd "${ESN_DATA:-local/esn-gpu}" || exit 1
R=rmq   # local job queue: rmq submit <resources> -- <command>
SEED=${1:-3}
for reg in chua_I chua_II wangchen2; do
  for rho in 0.6 0.9; do
    $R submit --where cpu-node --cpu 4 --ram 3 --time 8h --preempt requeue \
      --name "esn-C-$reg-HE1-s$SEED-r$rho" \
      --requirements requirements-node.txt --env GPUQ_OUT=node_results --results node_results/ \
      --files "gpu/*.py" --files "src/hsearch/*.py" --files "src/hsearch/systems/*.py" \
      --files "data/$reg.json" --files "data/$reg.npz" --files "data/${reg}_l1.npz" \
      --files "data/c_skip.txt" --files "requirements-node.txt" \
      -- python gpu/stage_c.py --regimes "$reg" --sets HE1 --kinds esn --rhos "$rho" --etas 1e-7,1e-4 \
         --seeds "$SEED" --skip data/c_skip.txt
  done
done

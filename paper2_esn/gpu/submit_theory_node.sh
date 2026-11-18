#!/usr/bin/env bash
# CPU node, теория: этап X (точный эталон, 2 части), этап I (индексы моделей C, 2 части),
# этап N при малых N (50–200) — по задаче на режим. Входы: data/x_*, data/i_* (gpu/make_bundles.py).
cd "${ESN_DATA:-local/esn-gpu}" || exit 1
R=rmq   # local job queue: rmq submit <resources> -- <command>
COMMON=(--where cpu-node --cpu 4 --preempt requeue --requirements requirements-node.txt
        --env GPUQ_OUT=node_results --results node_results/
        --files "gpu/*.py" --files "src/hsearch/*.py" --files "src/hsearch/systems/*.py" --files "requirements-node.txt")
for p in 0 1; do
  $R submit "${COMMON[@]}" --ram 3 --time 3h --name "esn-X-$p" --files data/x_index.json --files data/x_bundle.npz \
    --files data/lorenz_l1.npz -- python gpu/stage_x.py --part $p/2
done
REG=(--files "data/se11.*" --files "data/se3.*" --files "data/le1.*" --files "data/chua_I.*" --files "data/chua_II.*"
     --files "data/wangchen2.*" --files "data/wangchen2_a0.03.*" --files "data/*_l1.npz")
for p in 0 1; do
  $R submit "${COMMON[@]}" --ram 4 --time 4h --name "esn-I-$p" --files data/i_index.json --files data/i_bundle.npz \
    "${REG[@]}" -- python gpu/stage_i.py --part $p/2
done
for reg in se3 se11 le1 chua_I; do
  $R submit "${COMMON[@]}" --ram 3 --time 10h --name "esn-N-small-$reg" --files "data/$reg.*" --files "data/${reg}_l1.npz" \
    -- python gpu/stage_n.py --regimes $reg --sets HE1 --Ns 50,100,150,200 --rhos 0.9 --etas 1e-7,1e-4 --seeds 0,1,2,3,4
done

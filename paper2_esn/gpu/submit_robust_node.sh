#!/usr/bin/env bash
# CPU node: (1) Т2 на эталоне — тест по неустойчивому подпространству с 256 направлениями для 27 сетей, где 32 не хватило;
# (2) устойчивость закона выживания к гиперпараметрам: N = 300, HE1, ρ = 0.9, η = 1e-7, 1e-4, seed 0–4;
#     масштаб входа 0.2 / 0.5 / 1.0 (0.5 — как на А100, проверка CPU против GPU) и плотность W 0.02 / 0.2.
cd "${ESN_DATA:-local/esn-gpu}" || exit 1
R=rmq   # local job queue: rmq submit <resources> -- <command>
COMMON=(--where cpu-node --cpu 4 --ram 3 --preempt requeue --requirements requirements-node.txt
        --env GPUQ_OUT=node_results --results node_results/
        --files "gpu/*.py" --files "src/hsearch/*.py" --files "src/hsearch/systems/*.py" --files requirements-node.txt)
$R submit "${COMMON[@]}" --time 3h --name "esn-X256" --files data/x_index.json --files data/x_bundle.npz \
  --files data/lorenz_l1.npz --files data/x_missed_keys.txt \
  -- python gpu/stage_x.py --keys data/x_missed_keys.txt --rand-dirs 256 --tag 256
for reg in se3 se11 le1 chua_I; do
  F=(--files "data/$reg.*" --files "data/${reg}_l1.npz")
  $R submit "${COMMON[@]}" --time 10h --name "esn-N-input-$reg" "${F[@]}" \
    -- python gpu/stage_n.py --regimes "$reg" --sets HE1 --Ns 300 --rhos 0.9 --etas 1e-7,1e-4 --seeds 0,1,2,3,4 --input-scales 0.2,0.5,1.0
  $R submit "${COMMON[@]}" --time 8h --name "esn-N-dens-$reg" "${F[@]}" \
    -- python gpu/stage_n.py --regimes "$reg" --sets HE1 --Ns 300 --rhos 0.9 --etas 1e-7,1e-4 --seeds 0,1,2,3,4 --densities 0.02,0.2
done

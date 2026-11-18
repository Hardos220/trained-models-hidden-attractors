#!/usr/bin/env bash
# CPU node (CPU): certified enumeration of the fixed points of the stage MW models (gpu/krawczyk_m.py).
# First gathers the saved weights (stage MW .npz from the A100 jobs in gpuq-results/*/out/MW and from CPU node in
# node_results/MW) into local/esn-gpu/mw_weights/<regime>/, then submits one task per regime (and class pattern).
#   REGS="se3" PATS="*.npz" CPU=2 TIME=4h bash paper2_esn/gpu/submit_kr_node.sh
cd "${ESN_DATA:-local/esn-gpu}" || exit 1
R=rmq   # local job queue: rmq submit <resources> -- <command>
for reg in ${REGS:-se3}; do
  mkdir -p "mw_weights/$reg"
  for f in gpuq-results/*/out/MW/"$reg"/*.npz node_results/MW/"$reg"/*.npz; do
    [ -f "$f" ] || continue
    b=$(basename "$f")
    [ -f "mw_weights/$reg/$b" ] || cp "$f" "mw_weights/$reg/$b"
  done
  echo "$reg: $(ls mw_weights/$reg/*.npz 2>/dev/null | wc -l) weight files"
done
COMMON=(--where cpu-node --cpu ${CPU:-2} --ram 2 --preempt requeue --requirements requirements-node.txt
        --env GPUQ_OUT=node_results --results node_results/
        --files "gpu/krawczyk_m.py" --files requirements-node.txt)
for reg in ${REGS:-se3}; do
  for pat in ${PATS:-"*.npz"}; do
    tag=$(echo "$pat" | tr -d '*.' | sed 's/npz$//')
    $R submit "${COMMON[@]}" --time ${TIME:-4h} --name "esn-KR-$reg${tag:+-$tag}" --files "data/$reg.json" \
      --files "mw_weights/$reg/$pat" \
      -- python gpu/krawczyk_m.py --src mw_weights --regimes "$reg" --pattern "$pat" --res-start-box --workers ${CPU:-2}
  done
done

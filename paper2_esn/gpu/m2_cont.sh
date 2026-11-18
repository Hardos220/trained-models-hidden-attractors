#!/usr/bin/env bash
# Этап M, вторая очередь: продолжение цепочки m2_<режим>.sh, остановленной по лимиту времени.
#   bash gpu/m2_cont.sh РЕЖИМ ID [ID ...]
# Готовые json берутся из /srv/gpuq/jobs/<ID>/out/M/<режим> прежних задач, оборванные при остановке удаляются;
# stage_m.py пропускает посчитанные модели и досчитывает остальные. В конце архив всех 800 файлов, как в m2_<режим>.sh.
reg=$1; shift
mkdir -p "$GPUQ_OUT/M/$reg"
for old in "$@"; do
  src=/srv/gpuq/jobs/$old/out/M/$reg
  if [ -d "$src" ]; then
    cp -n "$src"/*.json "$GPUQ_OUT/M/$reg/" 2>/dev/null
    echo "[chain] из #$old: $(ls "$src" | grep -c '\.json$') json"
  else
    echo "[chain] нет доступа к $src"; ls -ld "/srv/gpuq/jobs/$old" "/srv/gpuq/jobs/$old/out" 2>&1
  fi
done
python - "$GPUQ_OUT/M/$reg" <<'EOF'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1]); bad = 0
for p in d.glob("*.json"):
    try:
        json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        p.unlink(); bad += 1
print(f"[chain] готовых json {sum(1 for _ in d.glob('*.json'))}, оборванных удалено {bad}", flush=True)
EOF
run() { for i in 1 2 3; do "$@"; rc=$?; [ $rc -ne 3 ] && [ $rc -ne 4 ] && return; echo "[chain] exit $rc, retry $i"; sleep 30; done; }
run python gpu/stage_m.py --regimes "$reg" --sets H1,HE1 --classes mlp2 --widths 256,64,512,128,32,16,1024,8 --seeds 20
run python gpu/stage_m.py --regimes "$reg" --sets E1 --classes mlp,res,mlp2 --widths 256,64,512,128,32,16,1024,8 --seeds 20
tar czf "$GPUQ_OUT/A_M.tgz" -C "$GPUQ_OUT" M && echo "[chain] архив $(du -h "$GPUQ_OUT/A_M.tgz" | cut -f1)"

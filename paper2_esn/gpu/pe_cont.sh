#!/usr/bin/env bash
# Continue a stage P or E chain stopped by the time limit: copy the finished json from the earlier jobs' output dirs on the
# server, drop broken ones, then rerun the chain script (the stage scripts skip models whose json exists).
#   bash gpu/pe_cont.sh <chain script> <subdir P|E> ID [ID ...]
script=$1; sub=$2; shift 2
mkdir -p "$GPUQ_OUT/$sub"
for old in "$@"; do
  src=/srv/gpuq/jobs/$old/out/$sub
  if [ -d "$src" ]; then
    cp -rn "$src"/. "$GPUQ_OUT/$sub/" 2>/dev/null
    echo "[chain] from #$old: $(find "$src" -name '*.json' | wc -l) json"
  else
    echo "[chain] no access to $src"
  fi
done
python - "$GPUQ_OUT/$sub" <<'EOF'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1]); bad = 0
for p in d.rglob("*.json"):
    try:
        json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        p.unlink(); bad += 1
print(f"[chain] json ready {sum(1 for _ in d.rglob('*.json'))}, broken removed {bad}", flush=True)
EOF
bash "gpu/$script"

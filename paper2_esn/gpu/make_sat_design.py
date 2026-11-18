"""Design of stage S (saturation of the equilibrium search of echo state networks; check of completeness).

Picks a stratified subsample of the trained networks of the size sweep (stage N, rho = 0.9) and of the full-protocol
ensemble at rho = 1.2 (stage C, N = 300), preferring networks that reproduce the hidden attractor, and writes
  local/esn-gpu/data/sat_design.json  - list of networks with the job group and the number of start blocks (mult);
  local/esn-gpu/data/sat_cfps.npz     - stored full-protocol equilibria (stage C, reservoir coordinates) of every
                                        network of the list that is also in stage C (key "<regime>/<stage N name>").
Strata: six main regimes (training sets of stage N: H1 and HE1 for SE3 and SE11, HE1 for the others; H1 and HE1 for
rho = 1.2), N in {100, 300, 700, 1400, 2000}. Within a stratum: reproducing networks first, then by seed and eta.

    python -X utf8 paper2_esn/gpu/make_sat_design.py
"""
import json
import re
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("ESN_DATA", Path(__file__).resolve().parents[2] / "local" / "esn-gpu"))
SOURCES = ("gpuq-results", "local_results", "node_results")
REGIMES = ("se3", "se11", "wangchen2", "wangchen2_a0.03", "chua_I", "chua_II")
# N -> (networks per stratum, start blocks, job groups (round robin))
PLAN_N = {100: (6, 8, ["n100_300"]), 300: (6, 8, ["n100_300"]), 700: (3, 4, ["n700"]),
          1400: (1, 4, ["n1400a", "n1400b"]), 2000: (1, 4, ["n2000a", "n2000b", "n2000c", "n2000d"])}
PLAN_R12 = (4, 8, ["r12"])
PAT_N = re.compile(r"^(H1|HE1)_N(\d+)_rho([\d.]+)_eta([^_]+)_s(\d+)$")
PAT_C = re.compile(r"^(H1|HE1)_esn(?:_N(\d+))?_rho([\d.]+)_eta([^_]+)_s(\d+)$")


def src_key(f):
    rel = f.relative_to(ROOT)
    job = int(rel.parts[1]) if rel.parts[0] == "gpuq-results" and rel.parts[1].isdigit() else -1
    return (SOURCES.index(rel.parts[0]), job, str(rel))


def collect(stage, pat):
    out = {}
    files = []
    for s in SOURCES:
        files += [f for f in (ROOT / s).rglob(f"{stage}/*/*.json") if f.parent.parent.name == stage
                  and "N_old_cpu_bug" not in f.parts and ".failed" not in f.name]
    for f in sorted(files, key=src_key):
        m = pat.match(f.stem)
        if not m:
            continue
        reg = f.parent.name
        if reg not in REGIMES:
            continue
        if stage == "N" and f.parts[len(ROOT.parts)] == "node_results" and "_N150_" in f.stem:
            continue
        s, N, rho, eta, seed = m.group(1), int(m.group(2) or 300), float(m.group(3)), float(m.group(4)), int(m.group(5))
        k = (reg, s, N, rho, eta, seed)
        if k not in out:
            out[k] = f
    return out


def reproduced_N(f):
    r = json.loads(f.read_text(encoding="utf-8"))
    return any(v.get("reproduced") for v in r["hidden_match"].values())


def reproduced_C(f, hidden):
    r = json.loads(f.read_text(encoding="utf-8"))
    return any(m.get("delta_att", 9) < 0.2 and m.get("d_H", 9) < 0.2
               for a in r["attractors"] for y, m in a["metrics"].items() if y in hidden)


def main():
    recN, recC = collect("N", PAT_N), collect("C", PAT_C)
    meta = {g: json.loads((ROOT / "data" / f"{g}.json").read_text(encoding="utf-8")) for g in REGIMES}
    nets, cf = [], {}
    rr = {}

    def add(k, src, f_orig, mult, groups):
        reg, s, N, rho, eta, seed = k
        g = groups[rr.setdefault(tuple(groups), 0) % len(groups)]
        rr[tuple(groups)] += 1
        e = {"regime": reg, "set": s, "N": N, "rho": rho, "eta": eta, "seed": seed, "src": src, "mult": mult,
             "group": g, "orig": str(f_orig.relative_to(ROOT)).replace("\\", "/")}
        if k in recC:
            fc = recC[k].with_suffix(".npz")
            if fc.exists():
                e["orig_C"] = str(recC[k].relative_to(ROOT)).replace("\\", "/")
                name = f"{s}_N{N}_rho{rho}_eta{eta:g}_s{seed}"
                cf[f"{reg}/{name}"] = np.load(fc)["fps"]
        nets.append(e)

    for N, (n_take, mult, groups) in PLAN_N.items():
        for reg in REGIMES:
            for s in ("H1", "HE1"):
                cand = [k for k in recN if k[0] == reg and k[1] == s and k[2] == N and k[3] == 0.9]
                if not cand:
                    continue
                cand.sort(key=lambda k: (not reproduced_N(recN[k]), k[5], k[4]))
                for k in cand[:n_take]:
                    add(k, "N", recN[k], mult, groups)
    n_take, mult, groups = PLAN_R12
    for reg in REGIMES:
        for s in ("H1", "HE1"):
            cand = [k for k in recC if k[0] == reg and k[1] == s and k[2] == 300 and k[3] == 1.2]
            cand.sort(key=lambda k: (not reproduced_C(recC[k], meta[reg]["hidden"]), k[5], k[4]))
            for k in cand[:n_take]:
                add(k, "C", recC[k], mult, groups)
    (ROOT / "data" / "sat_design.json").write_text(json.dumps({"networks": nets}, indent=1), encoding="utf-8")
    np.savez_compressed(ROOT / "data" / "sat_cfps.npz", **cf)
    from collections import Counter
    print(len(nets), "networks;", Counter(e["group"] for e in nets), "; with stored C equilibria:", len(cf))
    print(Counter((e["N"], e["rho"]) for e in nets))


if __name__ == "__main__":
    main()

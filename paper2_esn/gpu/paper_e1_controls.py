"""Paper numbers for stage E: controls for training without the hidden attractor (set E1, random 15 of 20 transients).

  poly - sparse quadratic maps (discrete SINDy, thresholds 0.003, 0.01, 0.03), stage M protocol (light labels);
  ngrc - next-generation reservoir computers (eta 1e-3, 1e-1, 10), stage C protocol (full labels, hidden_stat).

Per kind, parameter and regime: realizations, reproductions of each hidden attractor (same match rules as stages M and C),
labels among reproductions, mean number of unstable equilibria, share of escaping trajectories from the uniform census.
Written to paper2_esn/chaos/numbers/e1_controls.json.

    python -X utf8 paper2_esn/gpu/paper_e1_controls.py
"""
import json
import sys
from collections import Counter, defaultdict
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_data_effects as PD  # noqa: E402
from paper_common import POOL_NOTE, SEPARATE  # noqa: E402

ROOT = Path(os.environ.get("ESN_DATA", Path(__file__).resolve().parents[2] / "local" / "esn-gpu"))
OUT = Path(__file__).resolve().parents[1] / "chaos" / "numbers"


def fate(lab):
    return "escape" if lab == "esc" else "equilibrium" if lab.startswith("f") else "attractor" if lab.startswith("a") else "other"


def collect():
    seen, files = set(), []
    for d in ("node_results", "gpuq-results"):
        for f in sorted((ROOT / d).rglob("E/*/*.json")):
            if ".failed" in f.name:
                continue
            key = f.parent.name + "/" + f.stem
            if key in seen:
                continue
            seen.add(key)
            files.append(f)
    return files


def u_in(r, m):
    """Unstable equilibria whose physical position lies in the region Q (records with 3-dimensional x only)."""
    lo, hi = np.array(m["box_lo"]), np.array(m["box_hi"])
    n = 0
    for x in r.get("fixed_points", []):
        p = np.asarray(x.get("x", []), float)
        if x.get("n_unstable", 0) > 0 and p.shape == (3,) and np.all(p >= lo) and np.all(p <= hi):
            n += 1
    return n


def summarize(files, want):
    """Summary of the stage E records trained on set `want` (E1, H1 or HE1; records without a set are E1)."""
    tab = defaultdict(lambda: dict(n=0, rep_any=0, rep=Counter(), labels=Counter(), U=[], Uin=[], esc=[], rmse=[],
                                   fate=Counter()))
    for f in files:
        r = PD.load(f)
        if r.get("set", "E1") != want:
            continue
        reg = r["regime"]
        m = PD.meta(reg)
        if r.get("kind", "poly") == "poly" and "hidden_match" in r:
            key = ("poly", f"thr={r['thr']:g}", reg)
            t = tab[key]
            t["n"] += 1
            any_rep = False
            for y, mm in r["hidden_match"].items():
                if mm.get("reproduced"):
                    any_rep = True
                    t["rep"][y] += 1
                    t["labels"][mm.get("label")] += 1
            t["rep_any"] += any_rep
            t["U"].append(r["U"])
            t["Uin"].append(u_in(r, m))
            cc = r.get("census_counts", {})
            t["esc"].append(cc.get("esc", 0) / max(1, sum(cc.values())))
            t["rmse"].append(r["train_rmse"])
            for g, lab in r.get("census_confusion", {}).items():
                if g.startswith("HA"):           # uniform points in the basin of a true hidden attractor
                    for k, c in lab.items():
                        t["fate"][fate(k)] += c
        elif r.get("model") == "ngrc" or "hiddenness" in r:
            key = ("ngrc", f"eta={r['eta']:g}", reg)
            t = tab[key]
            t["n"] += 1
            any_rep = False
            for y in m["hidden"]:
                b = PD.best_match_for(r, m, y)
                if b:
                    any_rep = True
                    t["rep"][y] += 1
                    t["labels"][r["hiddenness"][b[1]]["label"]] += 1
            t["rep_any"] += any_rep
            t["U"].append(sum(1 for x in r["fixed_points"] if x["n_unstable"] > 0))
            t["Uin"].append(u_in(r, m))
            q = r.get("census", {}).get("Q", {})
            t["esc"].append(q.get("esc", 0) / max(1, sum(q.values())))
            t["rmse"].append(r["train_rmse"])
    out = {}
    for (kind, par, reg), t in sorted(tab.items()):
        out.setdefault(kind, {}).setdefault(par, {})[reg] = dict(
            n=t["n"], reproduced_any=t["rep_any"], reproduced=dict(t["rep"]), labels=dict(t["labels"]),
            mean_U=round(float(np.mean(t["U"])), 2), mean_U_in=round(float(np.mean(t["Uin"])), 2),
            share_escape=round(float(np.mean(t["esc"])), 3), median_train_rmse=float(np.median(t["rmse"])),
            hidden_basin_fate={k: round(v / max(1, sum(t["fate"].values())), 3) for k, v in sorted(t["fate"].items())})
    # per kind and regime over all parameter values: reproduced pairs (model, hidden attractor) out of n * hidden; hidden =
    # not excited (light protocol, quadratic maps) or statistically hidden (full protocol, NG-RC)
    for kind, d in out.items():
        byr = {}
        for par, regs in d.items():
            for reg, v in regs.items():
                b = byr.setdefault(reg, dict(n=0, pairs=0, reproduced=0, labels=Counter(), U=[], Uin=[]))
                b["n"] += v["n"]
                b["pairs"] += v["n"] * len(PD.meta(reg)["hidden"])
                b["reproduced"] += sum(v["reproduced"].values())
                b["labels"].update(v["labels"])
                b["U"] += [v["mean_U"]] * v["n"]
                b["Uin"] += [v["mean_U_in"]] * v["n"]
        out[kind]["by_regime"] = {
            k: dict(n=b["n"], pairs=b["pairs"], reproduced=b["reproduced"], labels=dict(b["labels"]),
                    hidden=b["labels"].get("not_excited", 0) + b["labels"].get("hidden_stat", 0),
                    mean_U=round(float(np.mean(b["U"])), 2), mean_U_in=round(float(np.mean(b["Uin"])), 2))
            for k, b in sorted(byr.items())}
    # totals over the six main regimes (paper_common.MAIN_REGIMES); seven regimes and LE1 alone for reference
    tot, tot_all, tot_le1 = {}, {}, {}
    for kind, d in out.items():
        d = {p: v for p, v in d.items() if p != "by_regime"}
        for T, keep in ((tot, lambda g: g not in SEPARATE), (tot_all, lambda g: True), (tot_le1, lambda g: g == "le1")):
            allr = [v for p in d.values() for g, v in p.items() if keep(g)]
            lab = Counter()
            for v in allr:
                lab.update(v["labels"])
            T[kind] = dict(n=sum(v["n"] for v in allr), reproduced_any=sum(v["reproduced_any"] for v in allr),
                           reproduced_pairs=sum(sum(v["reproduced"].values()) for v in allr),
                           hidden=lab.get("not_excited", 0) + lab.get("hidden_stat", 0),
                           labels=dict(lab), regimes=sorted({g for p in d.values() for g in p if keep(g)}))
    out["total"] = tot
    out["total_all_regimes"] = tot_all
    out["le1_separate"] = tot_le1
    out["pooling"] = "total: " + POOL_NOTE
    return out


def main():
    files = collect()
    out = summarize(files, "E1")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e1_controls.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    # the same classes trained on the hidden attractor (H1: 75% window of the trajectory; HE1: plus 15 of 20 transients):
    # can the class represent the hidden attractor, and does it keep it hidden?
    st = {s: summarize(files, s) for s in ("H1", "HE1")}
    (OUT / "structure_controls.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    for s, o in st.items():
        print(s, json.dumps(o["total"]))
        for kind in ("poly", "ngrc"):
            if kind in o:
                print("  ", kind, {g: (v["reproduced"], v["pairs"], v["hidden"], v["mean_U"], v["mean_U_in"])
                                   for g, v in o[kind]["by_regime"].items()})
    for kind, d in out.items():
        if kind in ("total", "total_all_regimes", "le1_separate", "pooling"):
            continue
        for par, regs in d.items():
            if par == "by_regime":
                continue
            for reg, v in regs.items():
                print(f"{kind:5s} {par:10s} {reg:16s} n {v['n']:3d} reproduced {v['reproduced_any']:2d} {v['labels']} "
                      f"U {v['mean_U']:.2f} esc {v['share_escape']:.2f} rmse {v['median_train_rmse']:.3g} HA basin -> {v['hidden_basin_fate']}")
    print(json.dumps(out["total"], indent=1))


if __name__ == "__main__":
    main()

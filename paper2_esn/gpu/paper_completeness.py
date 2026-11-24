"""Completeness of the equilibrium sets (check): numbers of the article.

A. Echo state networks (stage S, gpu/stage_sat.py; design data/sat_design.json from make_sat_design.py). Each network of
   a stratified subsample of the size sweep (stage N, rho = 0.9, N = 100 ... 2000) and of the full-protocol ensemble at
   rho = 1.2 (stage C, N = 300) is rebuilt; the stage N search is repeated with the original start budget x1 (the same
   random numbers: reproduction check against the stored record), x2, x4 (and x8 for N <= 300) by independent extra
   blocks of starts. Reported: reproduction of the stored K, U, index sum, training error and labels; by N and rho the
   share of networks in which x2 / x4 / x8 starts find further equilibria, further UNSTABLE equilibria, further ones in
   Q, the mean increase of U and U_in, and whether a newly found unstable equilibrium excites the reproduced hidden
   attractor (label change). For networks that are also in stage C, the same against the stored full-protocol set.
B. Perceptrons and residual networks (stage MW, gpu/stage_mw.py: stage M models retrained with saved weights; and
   gpu/krawczyk_m.py: certified enumeration by the interval Krawczyk method). Reported per class and width: share of
   models whose certified count differs from the Newton count, the missed equilibria (stable / unstable, inside / outside
   Q), undecided boxes, run time; for res in the region Q only (the residual network is unbounded). Reproduction of
   stage M by the A100 retraining (training error, K, U of the stored stage M record of the same name).

Output: paper2_esn/results/numbers/completeness.json
    python -X utf8 paper2_esn/gpu/paper_completeness.py
"""
import json
import math
import sys
from collections import Counter, defaultdict
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_common import MAIN_REGIMES  # noqa: E402

ROOT = Path(os.environ.get("ESN_DATA", Path(__file__).resolve().parents[2] / "local" / "esn-gpu"))
OUT = Path(__file__).resolve().parents[1] / "results" / "numbers"
SOURCES = ("gpuq-results", "local_results", "node_results")
R_MAX = 1e3


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def share(k, n):
    return {"k": int(k), "n": int(n), "share": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def src_key(f):
    rel = f.relative_to(ROOT)
    job = int(rel.parts[1]) if rel.parts[0] == "gpuq-results" and rel.parts[1].isdigit() else -1
    return (SOURCES.index(rel.parts[0]), job, str(rel))


def files_of(stage, pattern="*.json"):
    out = {}
    fs = []
    for s in SOURCES:
        fs += [f for f in (ROOT / s).rglob(f"{stage}/*/{pattern}") if f.parent.parent.name == stage
               and ".failed" not in f.name and "test_out" not in f.parts]
    for f in sorted(fs, key=src_key):
        out.setdefault(f.parent.name + "/" + f.stem, f)
    return out


def label_of(r):
    if r is None or not np.isfinite(r):
        return "not_excited"
    return "self_excited" if r <= 1e-6 else ("excited_r" if r <= 1e-3 else "not_excited")


def fingerprint_equal(a, b, tol=1e-3):
    """Same number of equilibria with the same numbers of unstable directions and spectral radii equal to tol
    (relative): networks rebuilt with eta = 1e-7 differ from the stored ones by rounding in the ridge solve."""
    a, b = sorted(a), sorted(b)
    return len(a) == len(b) and all(x[1] == y[1] and abs(x[0] - y[0]) <= tol * abs(y[0]) for x, y in zip(a, b))


def r_hat(fps, sel, y):
    rs = [fps[i]["r_exc"].get(y) for i in sel if fps[i]["n_unstable"] > 0 and fps[i].get("r_exc", {}).get(y) is not None]
    return min(rs) if rs else None


# ------------------------------------------------------------------------------------------------ A. ESN
def esn_part():
    design = json.loads((ROOT / "data" / "sat_design.json").read_text(encoding="utf-8"))["networks"]
    S = files_of("S")
    rows, verify = [], []
    for e in design:
        name = f"{e['set']}_N{e['N']}_rho{e['rho']}_eta{e['eta']:g}_s{e['seed']}"
        f = S.get(f"{e['regime']}/{name}")
        if f is None:
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        fps = r["fixed_points"]
        n = len(fps)
        mult = r["mult"]
        fb = [x.get("first_block") for x in fps]
        sel = {k: [i for i in range(n) if fb[i] is not None and fb[i] < k] for k in (1, 2, 4, 8) if k <= mult}
        row = {"regime": e["regime"], "set": e["set"], "N": e["N"], "rho": e["rho"], "eta": e["eta"], "seed": e["seed"],
               "src": e["src"], "mult": mult, "cum": r["cum"], "time_s": r["time_total_s"],
               "time_blocks_s": r["time_s"]["blocks"]}
        hm = r["hidden_match"]
        row["reproduced"] = {y: bool(m.get("reproduced")) for y, m in hm.items()}
        # labels by start budget (all unstable equilibria found by the first k blocks)
        row["labels"] = {y: {f"x{k}": label_of(r_hat(fps, s, y)) for k, s in sel.items()} for y in hm
                         if hm[y].get("attractor")}
        new_unst = [i for i in range(n) if fb[i] is not None and fb[i] >= 1 and fps[i]["n_unstable"] > 0]
        row["new_unstable"] = len(new_unst)
        row["new_unstable_in"] = sum(fps[i]["inside"] for i in new_unst)
        row["new_stable"] = sum(1 for i in range(n) if fb[i] is not None and fb[i] >= 1 and fps[i]["n_unstable"] == 0)
        row["new_by"] = Counter("defl" if fps[i]["hits_defl"] > 0 and fps[i]["hits_plain"] == 0 else "plain"
                                for i in range(n) if fb[i] is not None and fb[i] >= 1)
        row["new_exciting"] = {y: sum(1 for i in new_unst if fps[i].get("r_exc", {}).get(y) is not None
                                      and fps[i]["r_exc"][y] <= 1e-3) for y in row["labels"]}
        row["f1_x1"] = sum(1 for i in sel[1] if fps[i]["hits_plain"] + fps[i]["hits_defl"] == 1)
        # full-protocol (stage C) set of the same network
        if "C" in r["cum"]:
            inC = [i for i in range(n) if fps[i]["in_C"]]
            row["C_labels"] = {y: {"C": label_of(r_hat(fps, inC, y)), "C_and_all": label_of(r_hat(fps, range(n), y))}
                               for y in row["labels"]}
            notC_unst = [i for i in range(n) if not fps[i]["in_C"] and fps[i]["n_unstable"] > 0]
            row["new_vs_C_unstable"] = len(notC_unst)
            row["new_vs_C_exciting"] = {y: sum(1 for i in notC_unst if fps[i].get("r_exc", {}).get(y) is not None
                                               and fps[i]["r_exc"][y] <= 1e-3) for y in row["labels"]}
            row["C_stored"] = r.get("C_stored")
        # reproduction of the stored record
        o = json.loads((ROOT / e["orig"]).read_text(encoding="utf-8"))
        v = {"regime": e["regime"], "name": name, "src": e["src"],
             "train_rmse_rel_diff": abs(r["train_rmse"] - o["train_rmse"]) / o["train_rmse"]}
        if e["src"] == "N":
            v.update(K_stored=o["K"], U_stored=o["U"], isum_stored=o["index_sum"], K_x1=r["cum"]["x1"]["K"],
                     U_x1=r["cum"]["x1"]["U"], isum_x1=r["cum"]["x1"]["index_sum"],
                     fp_fingerprint_match=fingerprint_equal([(x["rho"], x["n_unstable"]) for x in o["fixed_points"]],
                                                            [(fps[i]["rho"], fps[i]["n_unstable"]) for i in sel[1]]),
                     reproduced_stored={y: bool(m.get("reproduced")) for y, m in o["hidden_match"].items()},
                     reproduced_now=row["reproduced"],
                     label_stored={y: o["hidden_match"][y].get("label") for y in row["labels"]},
                     label_now={y: row["labels"][y]["x1"] for y in row["labels"]})
        else:
            uo = sum(1 for x in o["fixed_points"] if x["n_unstable"] > 0)
            v.update(K_stored=len(o["fixed_points"]), U_stored=uo, K_C_rebuilt=r["cum"]["C"]["K"],
                     U_C_rebuilt=r["cum"]["C"]["U"], C_stored=r.get("C_stored"))
        verify.append(v)
        rows.append(row)

    def agg(rs, base="x1"):
        out = {"n_networks": len(rs)}
        for k in (2, 4, 8):
            key = f"x{k}"
            rk = [x for x in rs if key in x["cum"]]
            if not rk:
                continue
            dK = [x["cum"][key]["K"] - x["cum"][base]["K"] for x in rk]
            dU = [x["cum"][key]["U"] - x["cum"][base]["U"] for x in rk]
            dUi = [x["cum"][key]["U_in"] - x["cum"][base]["U_in"] for x in rk]
            out[key] = {"n": len(rk), "more_equilibria": share(sum(d > 0 for d in dK), len(rk)),
                        "more_unstable": share(sum(d > 0 for d in dU), len(rk)),
                        "more_unstable_in_Q": share(sum(d > 0 for d in dUi), len(rk)),
                        "mean_dK": round(float(np.mean(dK)), 3), "mean_dU": round(float(np.mean(dU)), 3),
                        "mean_dU_in": round(float(np.mean(dUi)), 3), "max_dU": int(max(dU)),
                        "mean_U_base": round(float(np.mean([x["cum"][base]["U"] for x in rk])), 2),
                        "index_sum_1_base": sum(x["cum"][base]["index_sum"] == 1 for x in rk),
                        "index_sum_1_after": sum(x["cum"][key]["index_sum"] == 1 for x in rk)}
        # label changes for reproduced hidden attractors
        pairs = [(x, y) for x in rs for y, ok in x["reproduced"].items() if ok and y in x["labels"]]
        kmax = {id(x): f"x{x['mult']}" for x in rs}
        ne = [(x, y) for x, y in pairs if x["labels"][y]["x1"] == "not_excited"]
        out["reproduced_pairs"] = len(pairs)
        out["not_excited_x1"] = len(ne)
        out["not_excited_x1_becomes_excited_at_max"] = sum(x["labels"][y][kmax[id(x)]] != "not_excited" for x, y in ne)
        out["new_unstable_exciting_reproduced"] = sum(x["new_exciting"][y] for x, y in pairs)
        out["new_unstable_total"] = sum(x["new_unstable"] for x in rs)
        out["new_stable_total"] = sum(x["new_stable"] for x in rs)
        return out

    by = {}
    for (N, rho), rs in sorted(group(rows, lambda x: (x["N"], x["rho"])).items()):
        by[f"N{N}_rho{rho}"] = agg(rs)
    res = {"n_networks": len(rows), "by_N_rho": by, "pooled_main_regimes": agg([x for x in rows if x["regime"] in MAIN_REGIMES])}
    # against the stored full-protocol sets (stage C)
    rc = [x for x in rows if "C" in x["cum"]]
    cmp = {}
    for (N, rho), rs in sorted(group(rc, lambda x: (x["N"], x["rho"])).items()):
        dU = [x["cum"]["C_and_all"]["U"] - x["cum"]["C"]["U"] for x in rs]
        dUi = [x["cum"]["C_and_all"]["U_in"] - x["cum"]["C"]["U_in"] for x in rs]
        dK = [x["cum"]["C_and_all"]["K"] - x["cum"]["C"]["K"] for x in rs]
        pairs = [(x, y) for x in rs for y, ok in x["reproduced"].items() if ok and y in x["labels"]]
        ch = [(x, y) for x, y in pairs if x["C_labels"][y]["C"] == "not_excited" and x["C_labels"][y]["C_and_all"] != "not_excited"]
        cmp[f"N{N}_rho{rho}"] = {"n": len(rs), "more_equilibria": share(sum(d > 0 for d in dK), len(rs)),
                                "more_unstable": share(sum(d > 0 for d in dU), len(rs)),
                                "more_unstable_in_Q": share(sum(d > 0 for d in dUi), len(rs)),
                                "mean_dU": round(float(np.mean(dU)), 3), "mean_dU_in": round(float(np.mean(dUi)), 3),
                                "reproduced_pairs": len(pairs),
                                "not_excited_with_C_set": sum(x["C_labels"][y]["C"] == "not_excited" for x, y in pairs),
                                "becomes_excited_with_all": len(ch)}
    res["vs_full_protocol_C"] = cmp
    vN = [v for v in verify if v["src"] == "N"]
    res["reproduction"] = {
        "n_N": len(vN), "K_U_isum_equal": sum(v["K_stored"] == v["K_x1"] and v["U_stored"] == v["U_x1"]
                                              and v["isum_stored"] == v["isum_x1"] for v in vN),
        "fingerprint_equal": sum(v["fp_fingerprint_match"] for v in vN),
        "reproduced_flag_equal": sum(v["reproduced_stored"] == v["reproduced_now"] for v in vN),
        "label_equal": sum(v["label_stored"] == v["label_now"] for v in vN),
        "train_rmse_max_rel_diff": max((v["train_rmse_rel_diff"] for v in verify), default=None),
        "n_C": len(verify) - len(vN),
        "C_K_U_equal": sum(v.get("K_stored") == v.get("K_C_rebuilt") and v.get("U_stored") == v.get("U_C_rebuilt")
                           for v in verify if v["src"] == "C"),
        "mismatches": [v for v in vN if not (v["K_stored"] == v["K_x1"] and v["U_stored"] == v["U_x1"])][:30]}
    res["time_s"] = {f"N{N}": {"median_total": round(float(np.median([x["time_s"] for x in rs])), 1),
                               "median_blocks": round(float(np.median([x["time_blocks_s"] for x in rs])), 1)}
                     for N, rs in sorted(group(rows, lambda x: x["N"]).items())}
    res["networks_with_new_unstable"] = [
        {k: x[k] for k in ("regime", "set", "N", "rho", "eta", "seed", "new_unstable", "new_unstable_in", "new_exciting",
                           "reproduced", "labels")} | {"cum": x["cum"], "new_by": dict(x["new_by"])}
        for x in rows if x["new_unstable"] > 0]
    return res, rows


def group(rows, key):
    g = defaultdict(list)
    for x in rows:
        g[key(x)].append(x)
    return g


# ------------------------------------------------------------------------------------------------ B. 3-D models
def models_part():
    KR = files_of("KR")
    MW = files_of("MW")
    Mst = files_of("M")
    rows = []
    for key, f in KR.items():
        r = json.loads(f.read_text(encoding="utf-8"))
        reg = key.split("/")[0]
        mw = json.loads(MW[key].read_text(encoding="utf-8")) if key in MW else {}
        cls, H = r["class"], r["H"]
        dom = "image_box" if cls != "res" else "Q"
        d = r["domains"][dom]
        row = {"regime": reg, "name": r["name"], "class": cls, "H": H, "set": r["name"].split("_")[1],
               "device": mw.get("device", "cuda"), "batch_seeds": mw.get("batch_seeds", 20), "status": d["status"],
               "n_undecided": d["n_undecided"], "n_boxes": d["n_boxes"], "time_s": d["time_s"],
               "cert": d["certified"], "newton": d["newton"], "missed": d["missed_by_newton"],
               "not_certified": d["newton_not_certified"]}
        if "start_box" in r["domains"]:
            ds = r["domains"]["start_box"]
            row["start_box"] = {"cert": ds["certified"], "newton": ds["newton"], "missed": len(ds["missed_by_newton"]),
                                "missed_unstable": sum(e["n_unstable"] > 0 for e in ds["missed_by_newton"]),
                                "n_undecided": ds["n_undecided"], "status": ds["status"]}
        if key in Mst and row["batch_seeds"] == 20 and row["device"] == "cuda" and mw:
            m0 = json.loads(Mst[key].read_text(encoding="utf-8"))
            fps0 = [x for x in m0["fixed_points"] if np.linalg.norm(x["x"]) <= R_MAX]
            fps1 = [x for x in mw["fixed_points"] if np.linalg.norm(x["x"]) <= R_MAX]
            row["stage_M"] = {"train_mse_rel_diff": abs(m0["train_mse"] - mw["train_mse"]) / m0["train_mse"],
                              "K_equal": len(fps0) == len(fps1),
                              "U_equal": sum(x["n_unstable"] > 0 for x in fps0) == sum(x["n_unstable"] > 0 for x in fps1)}
        rows.append(row)
    out = {"n_models": len(rows), "domain": {"mlp": "image box of F (all fixed points)", "mlp2": "image box of F (all fixed points)",
                                             "res": "region Q inflated by 1% of its width (completeness only in Q)"}}
    by = {}
    for (cls, H), rs in sorted(group(rows, lambda x: (x["class"], x["H"])).items()):
        n = len(rs)
        ok = [x for x in rs if x["status"] == "complete" and x["n_undecided"] == 0]
        miss = [e for x in rs for e in x["missed"]]
        by[f"{cls}_H{H}"] = {
            "n_models": n, "certified_complete": len(ok),
            "K_differs": share(sum(x["cert"]["K"] != x["newton"]["K"] for x in ok), len(ok)),
            "U_differs": share(sum(x["cert"]["U"] != x["newton"]["U"] for x in ok), len(ok)),
            "U_in_differs": share(sum(x["cert"]["U_in"] != x["newton"]["U_in"] for x in ok), len(ok)),
            "missed_total": len(miss), "missed_unstable": sum(e["n_unstable"] > 0 for e in miss),
            "missed_stable": sum(e["n_unstable"] == 0 for e in miss),
            "missed_inside_Q": sum(e["inside"] for e in miss), "missed_unstable_inside_Q": sum(e["inside"] and e["n_unstable"] > 0 for e in miss),
            "newton_not_certified": sum(len(x["not_certified"]) for x in rs),
            "mean_K_cert": round(float(np.mean([x["cert"]["K"] for x in rs])), 2),
            "mean_U_cert": round(float(np.mean([x["cert"]["U"] for x in rs])), 2),
            "mean_U_in_cert": round(float(np.mean([x["cert"]["U_in"] for x in rs])), 2),
            "index_sum_1_cert": sum(x["cert"]["index_sum"] == 1 for x in ok),
            "time_s": {"median": round(float(np.median([x["time_s"] for x in rs])), 2),
                       "max": round(float(np.max([x["time_s"] for x in rs])), 2)},
            "boxes_median": int(np.median([x["n_boxes"] for x in rs])),
            "undecided_models": sum(x["n_undecided"] > 0 or x["status"] != "complete" for x in rs),
            "by_device": dict(Counter(f"{x['device']}/{x['batch_seeds']}seeds" for x in rs))}
        if cls == "res":
            sb = [x["start_box"] for x in rs if "start_box" in x]
            if sb:
                by[f"{cls}_H{H}"]["start_box"] = {
                    "n": len(sb), "K_differs": sum(s["cert"]["K"] != s["newton"]["K"] for s in sb),
                    "missed": sum(s["missed"] for s in sb), "missed_unstable": sum(s["missed_unstable"] for s in sb),
                    "undecided_models": sum(s["n_undecided"] > 0 or s["status"] != "complete" for s in sb)}
    out["by_class_width"] = by
    rep = [x["stage_M"] for x in rows if "stage_M" in x]
    out["reproduction_stage_M"] = {"n": len(rep), "train_mse_rel_diff_max": max((x["train_mse_rel_diff"] for x in rep), default=None),
                                   "train_mse_rel_diff_median": float(np.median([x["train_mse_rel_diff"] for x in rep])) if rep else None,
                                   "K_equal": sum(x["K_equal"] for x in rep), "U_equal": sum(x["U_equal"] for x in rep)}
    out["models_with_missed"] = [{k: x[k] for k in ("regime", "name", "cert", "newton")} | {"missed": x["missed"]}
                                 for x in rows if x["missed"]][:60]
    return out, rows


def stage_c_found_by():
    """Existing data: in the full-protocol ensemble (stage C), which search phase found each equilibrium. Phase 2 starts
    (cube, training states, lifted true equilibria) are followed by segments between attractors and stable
    equilibria, 1000 deflated starts, and, for candidate hidden attractors, 2e4 Newton runs from the basin boundary."""
    out = defaultdict(lambda: {"n_networks": 0, "K": 0, "U": 0, "unstable_after_start_phase": 0,
                               "networks_with_unstable_after_start_phase": 0, "found_by": Counter(),
                               "boundary_searches": 0, "boundary_newton_runs": 0, "boundary_new_equilibria": 0})
    seen = set()
    for key, f in files_of("C").items():
        if key in seen or "_esn_" not in f.stem or key.split("/")[0] not in MAIN_REGIMES:
            continue
        seen.add(key)
        r = json.loads(f.read_text(encoding="utf-8"))
        g = out[f"N{r.get('N', 300)}_rho{r['rho']}"]
        fps = r["fixed_points"]
        g["n_networks"] += 1
        g["K"] += len(fps)
        g["U"] += sum(x["n_unstable"] > 0 for x in fps)
        late = sum(1 for x in fps if x.get("found_by") != "start" and x["n_unstable"] > 0)
        g["unstable_after_start_phase"] += late
        g["networks_with_unstable_after_start_phase"] += late > 0
        g["found_by"].update(x.get("found_by") for x in fps)
        for h in r.get("hiddenness", {}).values():
            b = h.get("boundary")
            if b and b.get("n_starts"):
                g["boundary_searches"] += 1
                g["boundary_newton_runs"] += b["n_starts"]
                g["boundary_new_equilibria"] += len(b.get("new_fp_ids", []))
    return {k: dict(v, found_by=dict(v["found_by"])) for k, v in sorted(out.items())}


def main():
    res = {"note": "check of the completeness of the equilibrium sets; see the docstring of paper_completeness.py"}
    res["esn_full_protocol_found_by"] = stage_c_found_by()
    try:
        res["esn_saturation"], _ = esn_part()
    except FileNotFoundError as exc:
        res["esn_saturation"] = {"missing": str(exc)}
    res["certified_3d"], _ = models_part()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "completeness.json").write_text(json.dumps(res, indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item()),
                                           encoding="utf-8")
    print(json.dumps({k: (v if k != "esn_saturation" else {kk: vv for kk, vv in v.items() if kk in ("n_networks", "by_N_rho", "reproduction")})
                      for k, v in res.items() if k != "note"}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()

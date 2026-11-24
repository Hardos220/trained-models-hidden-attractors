"""Theory checks for the ESN paper: Theorem 1, Lemma 1, the U >= (K-1)/2 bound, the incompleteness witness,
Proposition 1 (index of model equilibria near true ones) and Proposition 2 (unstable-subspace excitation test).

Inputs (read only), under ROOT = $ESN_DATA (default local/esn-gpu):
  local_results/X/*.json           stage X: hard-tanh nets of stage H with the exact equilibrium set E (32 directions)
  node_results/X256, X2048         stage X recomputed with 256 / 2048 random directions on the nets in
                                   data/x_missed_keys.txt (nets where 32 directions missed a self-excited attractor)
  local_results/I/*.json           stage I: indices of the equilibria of the trained ESN models of stage C
  {gpuq-results,local_results,node_results}/**/C/<regime>/*_esn_*.json
                                   stage C records (nearest_true_eq of each equilibrium; same order as in stage I)
  data/<regime>.json               true equilibria and their eigenvalues

Outputs: paper2_esn/results/numbers/theory.json (all numbers) and theory_points.csv (one row per net / model).
Run: python -X utf8 paper2_esn/gpu/paper_theory.py   (numpy only; a few seconds)
"""
import csv
import json
import time
from collections import Counter, defaultdict
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("ESN_DATA", Path(__file__).resolve().parents[2] / "local" / "esn-gpu"))
OUT = Path(__file__).resolve().parents[1] / "results" / "numbers"
RESULT_DIRS = ("gpuq-results", "local_results", "node_results")   # precedence used by gpu/make_bundles.py
EXACT_TOL = 1e-9        # stage X: a net is "exactly reproduced" if the retrained net keeps every point of E (residual)
TRUE_TOL = 0.01         # model equilibrium is "true-like" (a copy) if dist_true_eq < 0.01, i.e. its output is within 1% of
                        # L = max(box_hi - box_lo) / 2 (half of the largest edge of Q) from a true one (pipeline.fp_info);
                        # same threshold as paper_common.COPY_TOL
REF_BALL = 1000         # random points per (equilibrium, radius) in the reference ball test of stage H
REF_RADII = 6           # radii 1e-6 ... 1e-1 of the reference test
SUB_DELTAS = 2          # radii 1e-6, 1e-4 of the unstable-subspace test


def load_dir(d):
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(d.glob("*.json")) if ".failed" not in f.name]


def frac(a, b):
    return f"{a}/{b} ({100.0 * a / b:.1f}%)" if b else f"{a}/0"


def true_indices(regime):
    """Flow-map indices of the true equilibria: (-1)^{number of real positive eigenvalues of Df}."""
    m = json.loads((ROOT / "data" / f"{regime}.json").read_text(encoding="utf-8"))
    eqs = m["equilibria"] if isinstance(m["equilibria"], list) else []
    return [int((-1) ** sum(1 for re, im in zip(e["eig_re"], e["eig_im"]) if abs(im) < 1e-12 and re > 0)) for e in eqs]


def stage_c_files():
    """key 'regime/model' -> stage C json, first found in RESULT_DIRS order, only records with their .npz (as make_bundles)."""
    out = {}
    for d in RESULT_DIRS:
        for f in sorted((ROOT / d).rglob("C/*/*_esn_*.json")):
            if ".failed" in f.name or not f.with_suffix(".npz").exists():
                continue
            out.setdefault(f.parent.name + "/" + f.stem, f)
    return out


def main():
    t0 = time.time()
    R = {"meta": {"date": time.strftime("%Y-%m-%d %H:%M"), "root": str(ROOT), "exact_residual_tol": EXACT_TOL,
                  "true_like_dist_tol": TRUE_TOL}}
    X = load_dir(ROOT / "local_results" / "X")
    I = load_dir(ROOT / "local_results" / "I")

    # ------------------------------------------------------------------ Theorem 1 on stage X (exact sets)
    for r in X:
        r["exact"] = r["fp_residual_max"] <= EXACT_TOL
    Xe = [r for r in X if r["exact"]]
    res_in = max(r["fp_residual_max"] for r in Xe)
    res_out = min((r["fp_residual_max"] for r in X if not r["exact"]), default=None)
    t1x = {"n_nets_all": len(X), "n_nets_exact": len(Xe),
           "exact_residual_max_inside": res_in, "exact_residual_min_outside": res_out,
           "index_sum_eq1_exact": sum(r["index_sum_E"] == 1 for r in Xe),
           "index_sum_eq1_all": sum(r["index_sum_E"] == 1 for r in X),
           "index_sum_ne1_all": [{"key": r["key"], "index_sum": r["index_sum_E"], "K": r["E_n"],
                                  "residual": r["fp_residual_max"]} for r in X if r["index_sum_E"] != 1],
           "K_odd_exact": sum(r["E_n"] % 2 == 1 for r in Xe),
           "n_points_all": sum(len(r["equilibria"]) for r in X),
           "n_points_exact": sum(len(r["equilibria"]) for r in Xe),
           "K_range_exact": [min(r["E_n"] for r in Xe), max(r["E_n"] for r in Xe)],
           "N_values": sorted({r["N"] for r in X})}
    R["theorem1_X"] = t1x

    # ------------------------------------------------------------------ Theorem 1 on stage I (trained ESN models of stage C)
    I_sum = Counter(r["index_sum"] for r in I)
    ne1 = [r for r in I if r["index_sum"] != 1]
    t1i = {"n_models": len(I), "index_sum_eq1": I_sum.get(1, 0), "index_sum_ne1": len(ne1),
           "index_sum_counts": {str(k): v for k, v in sorted(I_sum.items())},
           "K_even": sum(r["K"] % 2 == 0 for r in I),
           "ne1_iff_K_even": all((r["index_sum"] != 1) == (r["K"] % 2 == 0) for r in I),
           "ne1_models": [{"key": r["key"], "K": r["K"], "index_sum": r["index_sum"]} for r in ne1],
           "n_points": sum(len(r["fixed_points"]) for r in I),
           "residual_max": max(r["residual_max"] for r in I),
           "models_residual_gt_1e-8": sum(r["residual_max"] > 1e-8 for r in I),
           "n_unstable_agrees_with_stage_C": sum(p["n_unstable"] == p["n_unstable_C"] for r in I for p in r["fixed_points"]),
           "models_by_regime": dict(sorted(Counter(r["regime"] for r in I).items())),
           "models_by_N": {str(k): v for k, v in sorted(Counter(r["N"] for r in I).items())}}
    R["theorem1_I"] = t1i

    # ------------------------------------------------------------------ Lemma 1 and the stability of index -1 points
    ptsX = [e for r in X for e in r["equilibria"]]
    ptsXe = [e for r in Xe for e in r["equilibria"]]
    ptsI = [p for r in I for p in r["fixed_points"]]
    lem = lambda P: sum(p["index"] != (-1) ** p["n_real_gt1"] for p in P)       # noqa: E731
    neg_stable = lambda P: sum(p["index"] == -1 and p["n_unstable"] == 0 for p in P)  # noqa: E731
    R["lemma1"] = {"n_points_X_all": len(ptsX), "n_points_X_exact": len(ptsXe), "n_points_I": len(ptsI),
                   "n_points_total": len(ptsX) + len(ptsI), "n_points_total_X_exact_plus_I": len(ptsXe) + len(ptsI),
                   "violations_X_all": lem(ptsX), "violations_I": lem(ptsI),
                   "violations_total": lem(ptsX) + lem(ptsI),
                   "index_minus1_but_stable": neg_stable(ptsX) + neg_stable(ptsI)}

    # ------------------------------------------------------------------ U >= (K-1)/2
    def ubound(rows):
        v = [(K, U) for K, U in rows]
        return {"n": len(v), "violations": sum(U < (K - 1) / 2 for K, U in v),
                "tight": sum(U == (K - 1) / 2 for K, U in v), "mean_U_minus_bound": float(np.mean([U - (K - 1) / 2 for K, U in v]))}
    UI = [(r["K"], sum(p["n_unstable"] > 0 for p in r["fixed_points"])) for r in I]
    R["U_bound"] = {"X_all": ubound([(r["E_n"], r["n_unstable_E"]) for r in X]),
                    "X_exact": ubound([(r["E_n"], r["n_unstable_E"]) for r in Xe]),
                    "I_all": ubound(UI),
                    "I_index_sum1": ubound([ku for ku, r in zip(UI, I) if r["index_sum"] == 1])}

    # ------------------------------------------------------------------ incompleteness witness (Newton sets E_f of stage H)
    def witness(rows):
        inc = [r for r in rows if r["Ef_n"] < r["E_n"]]
        com = [r for r in rows if r["Ef_n"] == r["E_n"]]
        return {"n_nets": len(rows), "n_incomplete": len(inc), "n_complete": len(com),
                "flagged_by_index_sum": sum(r["index_sum_Ef"] != 1 for r in inc),
                "flagged_by_parity": sum(r["Ef_n"] % 2 == 0 for r in inc),
                "missed": sum(r["index_sum_Ef"] == 1 for r in inc),
                "missed_points_in_unflagged": sorted(r["E_n"] - r["Ef_n"] for r in inc if r["index_sum_Ef"] == 1),
                "false_alarms_complete": sum(r["index_sum_Ef"] != 1 for r in com),
                "Ef_superset_of_E": sum(r["Ef_n"] > r["E_n"] for r in rows)}
    R["incompleteness_witness"] = {"X_exact": witness(Xe), "X_all": witness(X)}

    # ------------------------------------------------------------------ Proposition 1 (stage I + stage C nearest_true_eq)
    cfiles = stage_c_files()
    tind = {}
    p1_tot, p1_ok = Counter(), Counter()
    p1_tot_tight, p1_ok_tight = Counter(), Counter()
    cover, bij, n_mod, cov_hist = Counter(), Counter(), Counter(), defaultdict(Counter)
    mism, excluded, bij_index_ok = [], Counter(), Counter()
    order_ok = True
    for r in I:
        reg = r["regime"]
        if reg not in tind:
            tind[reg] = true_indices(reg)
        ti = tind[reg]
        n_mod[reg] += 1
        c = json.loads(cfiles[r["key"]].read_text(encoding="utf-8"))
        cf = c["fixed_points"]
        if len(cf) != len(r["fixed_points"]) or any(q.get("dist_true_eq") != p["dist_true_eq"] for p, q in zip(r["fixed_points"], cf)):
            order_ok = False
        near = defaultdict(list)
        for p, q in zip(r["fixed_points"], cf):
            d = p["dist_true_eq"]
            if d is None or d >= TRUE_TOL:
                continue
            if not ti or "nearest_true_eq" not in q:          # le1: line of equilibria, no isolated true index
                excluded[reg] += 1
                continue
            j = q["nearest_true_eq"]
            near[j].append(p["index"])
            ok = p["index"] == ti[j]
            for k in ("all", reg):
                p1_tot[k] += 1
                p1_ok[k] += ok
                if d < 1e-3:
                    p1_tot_tight[k] += 1
                    p1_ok_tight[k] += ok
            if not ok:
                mism.append({"key": r["key"], "true_eq": j, "index_model": p["index"], "index_true": ti[j],
                             "dist": d, "rho": p["rho"], "n_real_gt1": p["n_real_gt1"]})
        if not ti:
            continue
        cov_hist[reg][len(near)] += 1
        if len(near) == len(ti):
            cover[reg] += 1
            if all(len(v) == 1 for v in near.values()):
                bij[reg] += 1
                bij_index_ok[reg] += all(near[j][0] == ti[j] for j in near)
    regs = sorted(tind)
    R["prop1"] = {
        "true_indices": {g: tind[g] for g in regs},
        "s": {g: (sum(tind[g]) if tind[g] else None) for g in regs},
        "stage_I_stage_C_order_consistent": order_ok,
        "n_true_like_points": p1_tot["all"], "n_index_match": p1_ok["all"],
        "match_fraction": p1_ok["all"] / p1_tot["all"],
        "by_regime": {g: {"n": p1_tot[g], "match": p1_ok[g]} for g in regs if p1_tot[g]},
        "dist_lt_1e-3": {"n": p1_tot_tight["all"], "match": p1_ok_tight["all"]},
        "excluded_line_of_equilibria_points": dict(excluded),
        "mismatch_median_dist": float(np.median([m["dist"] for m in mism])) if mism else None,
        "mismatch_median_rho": float(np.median([m["rho"] for m in mism])) if mism else None,
        "mismatches": mism,
        "n_models": dict(n_mod),
        "n_true_eq_covered_hist": {g: {str(k): v for k, v in sorted(cov_hist[g].items())} for g in regs if cov_hist[g]},
        "reproduce_all_one_to_one": {g: bij[g] for g in regs if tind[g]},
        "reproduce_all_one_to_one_with_true_indices": {g: bij_index_ok[g] for g in regs if tind[g]},
        "reproduce_all_at_least_one": {g: cover[g] for g in regs if tind[g]},
        "chua_I_reproduce_all_three": cover["chua_I"],
        "chua_I_max_covered": max(cov_hist["chua_I"]) if cov_hist["chua_I"] else None}

    # ------------------------------------------------------------------ Proposition 2 on stage X (32 directions)
    def prop2(rows):
        conf = Counter()
        hid_exc = []
        pts_sub, n_eq_delta = 0, 0
        for r in rows:
            for k, cnt in r["subspace_counts"].items():
                pts_sub += sum(cnt.values())
                n_eq_delta += 1
            for a, v in r["attractors"].items():
                if not v.get("ref_ok") or v.get("label_reference") is None:
                    continue
                conf[(v["label_reference"], bool(v["excited_sub_E"]))] += 1
                if v["label_reference"] != "self_excited" and v["excited_sub_E"]:
                    hid_exc.append({"key": r["key"], "attractor": a, "kind": v["kind"], "label_reference": v["label_reference"],
                                    "r_sub_E": v["r_sub_E"]})
        se = conf[("self_excited", True)] + conf[("self_excited", False)]
        mean_sub = pts_sub / n_eq_delta if n_eq_delta else None
        return {"self_excited_reference": se, "self_excited_found_by_subspace": conf[("self_excited", True)],
                "self_excited_missed": conf[("self_excited", False)],
                "hidden_stat_reference": conf[("hidden_stat", True)] + conf[("hidden_stat", False)],
                "hidden_stat_excited_by_subspace": conf[("hidden_stat", True)],
                "excited_r_reference": conf[("excited_r", True)] + conf[("excited_r", False)],
                "excited_r_excited_by_subspace": conf[("excited_r", True)],
                "excited_by_subspace_not_self_excited_by_reference": hid_exc,
                "nets_with_missed_self_excited": sum(any(v.get("ref_ok") and v.get("label_reference") == "self_excited"
                                                         and not v["excited_sub_E"] for v in r["attractors"].values()) for r in rows),
                "subspace_points_per_unstable_eq_and_radius": mean_sub,
                "subspace_radii": SUB_DELTAS,
                "reference_ball_points_per_eq_and_radius": REF_BALL, "reference_radii": REF_RADII,
                "points_ratio_per_radius": REF_BALL / mean_sub if mean_sub else None,
                "points_ratio_ball_one_radius_vs_subspace_all_radii": REF_BALL / (mean_sub * SUB_DELTAS) if mean_sub else None,
                "points_ratio_total": REF_BALL * REF_RADII / (mean_sub * SUB_DELTAS) if mean_sub else None}
    R["prop2_X"] = {"exact": prop2(Xe), "all": prop2(X)}

    # ------------------------------------------------------------------ §4g: number of random directions (32 / 256 / 2048)
    keys = [k.strip() for k in (ROOT / "data" / "x_missed_keys.txt").read_text(encoding="utf-8").split() if k.strip()]
    dirs = {"32": ROOT / "local_results" / "X", "256": ROOT / "node_results" / "X256", "2048": ROOT / "node_results" / "X2048"}
    rows = []
    for k in keys:
        f = k.replace("/", "_") + ".json"
        res = {c: json.loads((d / f).read_text(encoding="utf-8")) for c, d in dirs.items()}
        nu = max([e["n_unstable"] for e in res["32"]["equilibria"]] or [0])
        for a, v in res["32"]["attractors"].items():
            row = {"key": k, "a": a, "kind": v["kind"], "ref": v["label_reference"], "ref_ok": v.get("ref_ok"),
                   "exact": res["32"]["fp_residual_max"] <= EXACT_TOL, "nu": nu}
            for c in dirs:
                va = res[c]["attractors"].get(a, {})
                row[c] = bool(va.get("excited_sub_E"))
                row["r" + c] = va.get("r_sub_E")
            rows.append(row)
    se = [r for r in rows if r["ref"] == "self_excited"]
    missed = [r for r in se if not r["32"]]
    still = [r for r in se if not r["2048"]]
    other = [r for r in rows if r["ref"] != "self_excited"]
    R["directions_4g"] = {
        "n_nets": len(keys), "nets_all_exact": all(r["exact"] for r in rows),
        "self_excited_reference": len(se), "self_excited_reference_ref_ok": sum(bool(r["ref_ok"]) for r in se),
        "found": {c: sum(r[c] for r in se) for c in dirs},
        "missed_at_32": len(missed),
        "missed_at_32_caught": {c: sum(r[c] for r in missed) for c in ("256", "2048")},
        "caught_at_32_lost": {c: sum(1 for r in se if r["32"] and not r[c]) for c in ("256", "2048")},
        "never_caught": len(still), "never_caught_kinds": dict(Counter(r["kind"] for r in still)),
        "never_caught_max_dim_Eu": sorted(r["nu"] for r in still),
        "never_caught_list": [{"key": r["key"], "attractor": r["a"], "kind": r["kind"], "max_dim_Eu": r["nu"]} for r in still],
        "not_self_excited": [{"key": r["key"], "attractor": r["a"], "kind": r["kind"], "label_reference": r["ref"],
                              "excited": {c: r[c] for c in dirs}, "r_sub": {c: r["r" + c] for c in dirs}} for r in other],
        "hidden_stat_excited_any": sum(1 for r in other if r["ref"] == "hidden_stat" and any(r[c] for c in dirs)),
        "hidden_stat_n": sum(1 for r in other if r["ref"] == "hidden_stat")}

    # ------------------------------------------------------------------ outputs
    OUT.mkdir(parents=True, exist_ok=True)
    R["meta"]["time_s"] = round(time.time() - t0, 1)
    (OUT / "theory.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    with open(OUT / "theory_points.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "key", "regime", "N", "K", "U", "index_sum", "complete", "exact", "set",
                    "K_newton", "U_newton", "index_sum_newton"])
        for r in sorted(X, key=lambda r: r["key"]):
            w.writerow(["X", r["key"], "lorenz", r["N"], r["E_n"], r["n_unstable_E"], r["index_sum_E"],
                        int(r["Ef_n"] == r["E_n"]), int(r["exact"]), "", r["Ef_n"], r["n_unstable_Ef"], r["index_sum_Ef"]])
        for (K, U), r in sorted(zip(UI, I), key=lambda t: t[1]["key"]):
            w.writerow(["I", r["key"], r["regime"], r["N"], K, U, r["index_sum"], "", "", r["set"], "", "", ""])
    report(R)


def report(R):
    a, b = R["theorem1_X"], R["theorem1_I"]
    print("=== Theorem 1")
    print(f"  stage X: exact nets (residual <= {EXACT_TOL:g}) {a['n_nets_exact']} of {a['n_nets_all']} "
          f"(gap: max inside {a['exact_residual_max_inside']:.1e}, min outside {a['exact_residual_min_outside']:.1e})")
    print(f"    sum ind = 1: exact {frac(a['index_sum_eq1_exact'], a['n_nets_exact'])}, all {frac(a['index_sum_eq1_all'], a['n_nets_all'])};"
          f" K odd (exact) {a['K_odd_exact']}; K range {a['K_range_exact']}")
    for e in a["index_sum_ne1_all"]:
        print(f"    exception (not exact): {e['key']} sum {e['index_sum']} K {e['K']} residual {e['residual']:.1e}")
    print(f"  stage I: sum ind = 1 in {frac(b['index_sum_eq1'], b['n_models'])}; counts {b['index_sum_counts']}; "
          f"K even {b['K_even']}; (sum != 1) <=> (K even): {b['ne1_iff_K_even']}")
    print(f"    points {b['n_points']}, max residual {b['residual_max']:.1e}, n_unstable = stage C at "
          f"{b['n_unstable_agrees_with_stage_C']} points")
    L = R["lemma1"]
    print(f"=== Lemma 1: violations {L['violations_total']} of {L['n_points_total']} points "
          f"(X {L['n_points_X_all']}, of them in exact nets {L['n_points_X_exact']}; I {L['n_points_I']}); "
          f"index -1 but stable: {L['index_minus1_but_stable']}")
    U = R["U_bound"]
    print("=== U >= (K-1)/2: violations " + ", ".join(f"{k} {v['violations']}/{v['n']} (tight {v['tight']})" for k, v in U.items()))
    for k, w in R["incompleteness_witness"].items():
        print(f"=== incompleteness witness, {k}: incomplete Newton sets {w['n_incomplete']} of {w['n_nets']}; flagged by "
              f"sum != 1: {w['flagged_by_index_sum']}, by parity: {w['flagged_by_parity']}; false alarms on "
              f"{w['n_complete']} complete: {w['false_alarms_complete']}; unflagged miss {w['missed_points_in_unflagged']} points")
    p = R["prop1"]
    print(f"=== Proposition 1: index match {frac(p['n_index_match'], p['n_true_like_points'])} "
          f"(dist < 1e-3: {frac(p['dist_lt_1e-3']['match'], p['dist_lt_1e-3']['n'])}); excluded (line of equilibria) "
          f"{p['excluded_line_of_equilibria_points']}")
    print("    by regime: " + ", ".join(f"{g} {v['match']}/{v['n']}" for g, v in p["by_regime"].items()))
    print(f"    true indices {p['true_indices']}, s {p['s']}")
    print(f"    reproduce all true eq. one-to-one: {p['reproduce_all_one_to_one']} "
          f"(with true indices {p['reproduce_all_one_to_one_with_true_indices']}); at least one point each: "
          f"{p['reproduce_all_at_least_one']}")
    print(f"    Chua I: all three reproduced in {p['chua_I_reproduce_all_three']} of {p['n_models']['chua_I']} models, "
          f"covered-count histogram {p['n_true_eq_covered_hist']['chua_I']}")
    print(f"    mismatches: median dist {p['mismatch_median_dist']:.1e}, median rho {p['mismatch_median_rho']:.2f}")
    for k, q in R["prop2_X"].items():
        print(f"=== Proposition 2, stage X {k}: subspace test found {frac(q['self_excited_found_by_subspace'], q['self_excited_reference'])}"
              f" self-excited (missed in {q['nets_with_missed_self_excited']} nets); hidden_stat excited "
              f"{q['hidden_stat_excited_by_subspace']}/{q['hidden_stat_reference']}; excited_r excited "
              f"{q['excited_r_excited_by_subspace']}/{q['excited_r_reference']}")
        print(f"    points per (unstable eq., radius): subspace {q['subspace_points_per_unstable_eq_and_radius']:.1f} vs ball "
              f"{REF_BALL} -> x{q['points_ratio_per_radius']:.0f} per radius, x{q['points_ratio_total']:.0f} in total, "
              f"x{q['points_ratio_ball_one_radius_vs_subspace_all_radii']:.1f} (one ball radius vs both subspace radii)")
        for e in q["excited_by_subspace_not_self_excited_by_reference"]:
            print(f"    excited only by subspace: {e}")
    d = R["directions_4g"]
    print(f"=== §4g directions: {d['n_nets']} nets (all exact: {d['nets_all_exact']}); self-excited {d['self_excited_reference']}; "
          f"found {d['found']}; missed at 32 {d['missed_at_32']}, caught later {d['missed_at_32_caught']}; "
          f"lost vs 32 {d['caught_at_32_lost']}")
    print(f"    never caught {d['never_caught']} {d['never_caught_kinds']} max dim E^u {d['never_caught_max_dim_Eu']}; "
          f"hidden_stat excited {d['hidden_stat_excited_any']}/{d['hidden_stat_n']}")
    for e in d["not_self_excited"]:
        print(f"    not self-excited: {e['key']} {e['attractor']} {e['kind']} {e['label_reference']} excited {e['excited']} r {e['r_sub']}")
    print(f"written {OUT / 'theory.json'}, {OUT / 'theory_points.csv'}; {R['meta']['time_s']} s")


if __name__ == "__main__":
    main()

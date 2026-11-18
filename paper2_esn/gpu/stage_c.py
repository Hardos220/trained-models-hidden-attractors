"""Этап C: модели по рядам семи режимов. Одна задача обрабатывает список моделей;
результат каждой модели — $GPUQ_OUT/C/<режим>/<модель>.json и .npz; готовые модели пропускаются.

Запуск: python gpu/stage_c.py --regimes se11,chua_II --sets H1,E1 --kinds esn [--limit K] [--quick]
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gpu"))
from hsearch import esn_pilot as P  # noqa: E402
import esngpu as G  # noqa: E402
import pipeline as PL  # noqa: E402
import ngrc_gpu as NG  # noqa: E402

REGIMES = ["se11", "se3", "wangchen2", "wangchen2_a0.03", "le1", "chua_I", "chua_II"]
SETS = ["H1", "H0", "E1", "HE1"]
ESN_GRID = [(rho, eta, seed) for rho in (0.6, 0.9, 1.2) for eta in (1e-7, 1e-4) for seed in (0, 1, 2)]
NGRC_GRID = [1e-3, 1e-1, 10.0]
LAM_C = 0.02


def load_regime(key):
    z = np.load(ROOT / "data" / f"{key}.npz")
    meta = json.load(open(ROOT / "data" / f"{key}.json", encoding="utf-8"))
    l1 = np.load(ROOT / "data" / f"{key}_l1.npz")
    return z, meta, l1


def train_sets(z):
    E = np.split(z["train_E1"], np.cumsum(z["train_E1_len"])[:-1])
    return {"H1": ([z["train_H1"]], 500), "H0": ([z["train_H0"]], 500), "E1": (E, 50),
            "HE1": ([z["train_H1"]] + E, [500] + [50] * len(E))}


def one_step_nrmse(esn, X, w=100, n=2000):
    """Прогрев w отсчётами ряда X (физ.), затем прогноз на шаг на n отсчётах; NRMSE по координатам."""
    U = X[: w + n + 1] / esn["scale"]
    r = np.zeros(esn["n_res"])
    preds, targ = [], []
    for k in range(w + n):
        r = P.act(esn["W"] @ r + esn["W_in"] @ U[k] + esn["b"], esn["activation"])
        if k >= w - 1:
            preds.append(esn["W_out"] @ r)
            targ.append(U[k + 1])
    preds, targ = np.array(preds), np.array(targ)
    return float(np.mean(np.sqrt(np.mean((preds - targ) ** 2, axis=0)) / targ.std(axis=0)))


def analyze_esn(key, z, meta, l1, set_name, rho, eta, seed, quick=False, log=print, N=300):
    t_start = time.time()
    tim = {}
    series, wash = train_sets(z)[set_name]
    esn = P.train_esn_multi(series, n_res=N, spectral_radius=rho, ridge=eta, input_scale=0.5, density=0.05,
                            washout=wash, seed=seed)
    net = G.Net(esn, h=meta["Delta_t"])
    n_tr, n_obs = (4000, 1000) if quick else (20000, 4000)
    n_life = 10000 if quick else 50000
    rng = np.random.default_rng(1000 + seed)
    true_atts = {k[4:]: z[k][:10000] for k in z.files if k.startswith("att_")}
    true_eq = [e["point"] for e in meta["equilibria"]]
    L_scale = float(np.max(np.array(meta["box_hi"]) - np.array(meta["box_lo"]))) / 2
    eq_line = key == "le1"
    rec = {"regime": key, "set": set_name, "model": "esn", "N": N, "rho": rho, "eta": eta, "seed": seed,
           "precision_bulk": PL.PREC_BULK,
           "train_rmse": esn["train_rmse"],
           "nrmse_1step": {k: one_step_nrmse(esn, Y) for k, Y in true_atts.items()}}
    tim["train"] = time.time() - t_start
    # 2. неподвижные точки
    t0 = time.time()
    L = PL.Labeler(net, LAM_C)
    nQ, nT = (1000, 500) if quick else (5000, 2500)
    starts = [rng.uniform(-1, 1, size=(nQ, N)),
              np.clip(esn["R_train"][rng.integers(0, len(esn["R_train"]), nT)] + 0.05 * rng.normal(size=(nT, N)), -1, 1)]
    eqs = list(true_eq)
    if eq_line:
        eqs = [[0.0, 0.0, zz] for zz in np.linspace(-10, -0.5, 20)]
    for e in eqs:
        xs = np.asarray(e)[None] + 0.01 * L_scale * rng.normal(size=(200, len(e)))
        starts.append(net.lift_const(esn, np.vstack([np.asarray(e)[None], xs]), w=100))
    S = np.vstack(starts)
    X, ok = net.newton(S, max_iter=40)
    found_by = {}
    jj = L.add_fps(X[ok]) if ok.any() else []
    for j in jj:
        found_by.setdefault(j, "start")
    tim["fp_newton"] = time.time() - t0
    # 3. обучающий аттрактор
    t0 = time.time()
    atr = PL.training_attractor(net, L, esn["R_train"][-1])
    rec["A_tr"] = {"kind": atr["kind"], "label": atr["label"]}
    # 4. перепись
    groups = {}
    groups["uL2"] = net.lift_const(esn, z["ics_uniform"], w=100)
    for w in (10, 100):
        R = net.lift_drive(esn, l1["l1"], w)
        groups[f"uL1_{w}"] = R
    basins = sorted(k[10:] for k in z.files if k.startswith("ics_basin_"))
    for b in basins:
        groups[f"b_{b}"] = net.lift_const(esn, z[f"ics_basin_{b}"], w=100)
    groups["Q"] = rng.uniform(-1, 1, size=(1000 if quick else 5000, N))
    orb = atr["orbit"]
    groups["kick"] = np.clip(orb[rng.integers(0, len(orb), 1000)] + 0.1 * rng.normal(size=(1000, N)), -1, 1)
    echo = [net.lift_const(esn, z["ics_uniform"][:500], w=100, R0=rng.uniform(-1, 1, size=(500, N))) for _ in range(4)]
    groups["echo"] = np.vstack(echo)
    names, states, valid = [], [], []
    for g, R in groups.items():
        ok_rows = np.isfinite(R).all(1)
        names.append(g)
        states.append(R)
        valid.append(ok_rows)
    allS = np.vstack(states)
    allV = np.concatenate(valid)
    run_idx = np.where(allV)[0]
    out = net.run(allS[run_idx], n_tr=n_tr, n_obs=n_obs, seed=seed + 7, prec=PL.PREC_BULK)
    tim["census_run"] = time.time() - t0
    t0 = time.time()
    lab0 = L.label(out)
    cands = PL.discover(net, L, out, lab0, n_life=n_life, n_obs=n_obs)
    lab = np.asarray(L.label(out), dtype=str)
    full = np.array(["lift_undefined"] * len(allS), dtype=object)
    full[run_idx] = lab
    full = full.astype(str)
    offs = np.cumsum([0] + [len(s) for s in states])
    glab = {g: full[offs[i]:offs[i + 1]] for i, g in enumerate(names)}
    rec["census"] = {g: PL.counts(v) for g, v in glab.items()}
    rec["discovery"] = cands
    tim["labels"] = time.time() - t0
    # матрицы соответствия
    tu = z["labels_uniform"]
    conf = {}
    for g, tl in (("uL2", tu), ("uL1_10", tu[:2000]), ("uL1_100", tu[:2000])):
        m = {}
        for a, b in zip(tl, glab[g]):
            m.setdefault(str(a), {}).setdefault(b, 0)
            m[str(a)][b] += 1
        conf[g] = m
    for b in basins:
        conf[f"b_{b}"] = {b: PL.counts(glab[f"b_{b}"])}
    rec["confusion"] = conf
    e1 = glab["uL2"][:500]
    E = np.stack([glab["echo"][i * 500:(i + 1) * 500] for i in range(4)])
    rec["echo_all_same"] = float(np.mean(np.all(E == e1[None], axis=0)))
    # 6. дополнительные старты: отрезки и дефляция
    t0 = time.time()
    stable = [i for i, x in enumerate(L.fps) if np.max(np.abs(net.stability(x)[0])) < 1]
    seg = []
    reps = [r["rep"] for r in L.refs]
    if reps and stable:
        per = max(1, (500 if quick else 2000) // (len(reps) * len(stable)))
        for rp in reps:
            orb_r = PL.G.N_(net.orbit(rp, 1000, every=5))
            for i in stable:
                lam_ = rng.uniform(0, 1, size=(per, 1))
                seg.append(lam_ * L.fps[i][None] + (1 - lam_) * orb_r[rng.integers(0, len(orb_r), per)])
    if seg:
        X, ok = net.newton(np.vstack(seg), max_iter=40)
        for j in (L.add_fps(X[ok]) if ok.any() else []):
            found_by.setdefault(j, "segment")
    if L.fps:
        D0 = rng.uniform(-1, 1, size=(200 if quick else 1000, N))
        X, ok = net.newton(D0, max_iter=60, deflate=np.asarray(L.fps))
        for j in (L.add_fps(X[ok]) if ok.any() else []):
            found_by.setdefault(j, "deflation")
    tim["fp_extra"] = time.time() - t0
    # 8. окрестности неустойчивых точек
    t0 = time.time()
    unst = [i for i in range(len(L.fps)) if np.max(np.abs(net.stability(L.fps[i])[0])) > 1 + 1e-9]
    nb = PL.radius_test(net, L, unst, n_ball=50 if quick else 200, n_tr=n_tr, n_obs=n_obs)
    tim["radius"] = time.time() - t0
    hits = PL.hits_by_attractor(nb)
    # 9. кандидаты и граница бассейна
    t0 = time.time()
    hid = {}
    census_states = allS[run_idx]
    for k in range(len(L.refs)):
        a = f"a{k}"
        h = hits.get(a, {"r_hat": np.inf, "from": {}})
        cand = not (h["r_hat"] <= 1e-3)
        rec_h = {"r_hat": h["r_hat"] if np.isfinite(h["r_hat"]) else None, "from": h["from"], "candidate": cand}
        cons = None
        if cand:
            bs = PL.boundary_search(net, L, a, census_states, lab, K=200 if quick else 1000,
                                    n_starts=2000 if quick else 20000, n_tr=n_tr, n_obs=n_obs)
            for j in bs["new_fp_ids"]:
                found_by.setdefault(j, "boundary")
            new_unst = [j for j in bs["new_fp_ids"] if np.max(np.abs(net.stability(L.fps[j])[0])) > 1 + 1e-9]
            if new_unst:
                nb2 = PL.radius_test(net, L, new_unst, n_ball=50 if quick else 200, n_tr=n_tr, n_obs=n_obs)
                nb.update(nb2)
                h2 = PL.hits_by_attractor(nb2).get(a)
                if h2:
                    rec_h["r_hat"] = min(rec_h["r_hat"] or np.inf, h2["r_hat"])
                    rec_h["from"].update(h2["from"])
            mm = PL.missing_mass_bounds(list(bs["multiplicities"].values()), bs["n_starts"])
            cons = mm["conservative"]
            rec_h["boundary"] = {k_: v for k_, v in bs.items() if k_ != "boundary_points"}
            rec_h["missing_mass"] = mm
        r_hat = rec_h["r_hat"] if rec_h["r_hat"] is not None else np.inf
        still_cand = cand and not (r_hat <= 1e-3)
        rec_h["label"] = PL.hidden_label(r_hat, still_cand, cons)
        hid[a] = rec_h
    rec["hiddenness"] = hid
    rec["radius_counts"] = nb
    tim["boundary"] = time.time() - t0
    # 10. соответствие
    t0 = time.time()
    att_rec, att_samples = [], []
    for k, r in enumerate(L.refs):
        Xm = net.samples(r["rep"], 10000, burn=10000)
        att_samples.append(Xm[:2000])
        att_rec.append({"name": f"a{k}", "kind": r["kind"], "lam": r["lam"],
                        "metrics": PL.correspondence(Xm, true_atts)})
    rec["attractors"] = att_rec
    rec["true_attractors"] = {k: {"kind": v["kind"], "lam": v["lam"]} for k, v in meta["attractors"].items()}
    rec["fixed_points"] = [dict(PL.fp_info(net, x, true_eq, L_scale, eq_line), found_by=found_by.get(i, "label"))
                           for i, x in enumerate(L.fps)]
    tim["corr"] = time.time() - t0
    rec["time_s"] = {k: round(v, 1) for k, v in tim.items()}
    rec["time_total_s"] = round(time.time() - t_start, 1)
    legend = sorted(set(full.tolist()))
    arrays = {"fps": np.asarray(L.fps) if L.fps else np.zeros((0, N)),
              "reps": np.asarray([r["rep"] for r in L.refs]) if L.refs else np.zeros((0, N)),
              "att_samples": np.asarray(att_samples) if att_samples else np.zeros((0, 2000, 3)),
              "labels": np.array([legend.index(x) for x in full], dtype=np.int16),
              "legend": np.array(legend), "group_offsets": offs, "group_names": np.array(names)}
    return rec, arrays


def analyze_ngrc(key, z, meta, l1, set_name, eta, quick=False):
    t_start = time.time()
    series, wash = train_sets(z)[set_name]
    ng = P.NGRC(series, ridge=eta)
    absmax = np.max(np.abs(np.vstack(series)), axis=0)
    net = NG.NGRCNet(ng, meta["Delta_t"], meta["box_lo"], meta["box_hi"], absmax)
    n_tr, n_obs = (4000, 1000) if quick else (20000, 4000)
    rng = np.random.default_rng(77)
    true_atts = {k[4:]: z[k][:10000] for k in z.files if k.startswith("att_")}
    true_eq = [e["point"] for e in meta["equilibria"]]
    L_scale = float(np.max(np.array(meta["box_hi"]) - np.array(meta["box_lo"]))) / 2
    eq_line = key == "le1"
    d = net.d
    lo, hi = net.box_lo_n, net.box_hi_n
    rec = {"regime": key, "set": set_name, "model": "ngrc", "eta": eta, "train_rmse": ng.train_rmse}
    L = PL.Labeler(net, LAM_C)
    starts = lo + (hi - lo) * rng.random((10000, d))
    eqs = [[0.0, 0.0, zz] for zz in np.linspace(-10, -0.5, 20)] if eq_line else true_eq
    if len(eqs):
        starts = np.vstack([starts, np.asarray(eqs) / net.scale])
    X, ok = net.newton(np.hstack([starts, starts]))
    if ok.any():
        L.add_fps(X[ok])
    last = series[0] / net.scale
    s_last = np.concatenate([last[-1], last[-2]])
    atr = PL.training_attractor(net, L, s_last)
    rec["A_tr"] = {"kind": atr["kind"], "label": atr["label"]}
    x0 = z["ics_uniform"] / net.scale
    x1 = l1["next_uniform"] / net.scale
    groups = {"uL0": np.hstack([x0, x0]), "uL1": np.hstack([x1, x0])}
    basins = sorted(k[10:] for k in z.files if k.startswith("ics_basin_"))
    for b in basins:
        xb = z["ics_basin_" + b] / net.scale
        xb1 = l1["next_" + b] / net.scale
        groups["b_" + b] = np.hstack([xb, xb])
        groups["b1_" + b] = np.hstack([xb1, xb])
    q1 = lo + (hi - lo) * rng.random((5000, d))
    q2 = lo + (hi - lo) * rng.random((5000, d))
    groups["Q"] = np.hstack([q1, q2])
    orb = atr["orbit"]
    groups["kick"] = orb[rng.integers(0, len(orb), 1000)] + 0.1 * rng.normal(size=(1000, 2 * d))
    names = list(groups)
    states = [np.nan_to_num(groups[g], nan=1e6) for g in names]
    allS = np.vstack(states)
    out = net.run(allS, n_tr=n_tr, n_obs=n_obs, seed=5)
    lab0 = L.label(out)
    cands = PL.discover(net, L, out, lab0, n_life=10000 if quick else 50000, n_obs=n_obs)
    lab = np.asarray(L.label(out), dtype=str)
    offs = np.cumsum([0] + [len(x) for x in states])
    glab = {g: lab[offs[i]:offs[i + 1]] for i, g in enumerate(names)}
    rec["census"] = {g: PL.counts(v) for g, v in glab.items()}
    rec["discovery"] = cands
    tu = z["labels_uniform"]
    conf = {}
    for g in ("uL0", "uL1"):
        m = {}
        for a, b in zip(tu, glab[g]):
            m.setdefault(str(a), {}).setdefault(b, 0)
            m[str(a)][b] += 1
        conf[g] = m
    rec["confusion"] = conf
    unst = [i for i in range(len(L.fps)) if np.max(np.abs(net.stability(L.fps[i])[0])) > 1 + 1e-9]
    nb = PL.radius_test(net, L, unst, n_ball=50 if quick else 200, n_tr=n_tr, n_obs=n_obs)
    hits = PL.hits_by_attractor(nb)
    hid = {}
    for k in range(len(L.refs)):
        a = "a" + str(k)
        h = hits.get(a, {"r_hat": np.inf, "from": {}})
        cand = not (h["r_hat"] <= 1e-3)
        rec_h = {"r_hat": h["r_hat"] if np.isfinite(h["r_hat"]) else None, "from": h["from"], "candidate": cand}
        cons = None
        if cand:
            bs = PL.boundary_search(net, L, a, allS, lab, K=200 if quick else 1000, n_starts=2000 if quick else 20000,
                                    n_tr=n_tr, n_obs=n_obs)
            new_unst = [j for j in bs["new_fp_ids"] if np.max(np.abs(net.stability(L.fps[j])[0])) > 1 + 1e-9]
            if new_unst:
                nb2 = PL.radius_test(net, L, new_unst, n_ball=50 if quick else 200, n_tr=n_tr, n_obs=n_obs)
                nb.update(nb2)
                h2 = PL.hits_by_attractor(nb2).get(a)
                if h2:
                    rec_h["r_hat"] = min(rec_h["r_hat"] or np.inf, h2["r_hat"])
            mm = PL.missing_mass_bounds(list(bs["multiplicities"].values()), bs["n_starts"])
            cons = mm["conservative"]
            rec_h["boundary"] = {k_: v for k_, v in bs.items() if k_ != "boundary_points"}
            rec_h["missing_mass"] = mm
        r_hat = rec_h["r_hat"] if rec_h["r_hat"] is not None else np.inf
        rec_h["label"] = PL.hidden_label(r_hat, cand and not (r_hat <= 1e-3), cons)
        hid[a] = rec_h
    rec["hiddenness"] = hid
    rec["radius_counts"] = nb
    att_rec, att_samples = [], []
    for k, r in enumerate(L.refs):
        Xm = net.samples(r["rep"], 10000, burn=10000)
        att_samples.append(Xm[:2000])
        att_rec.append({"name": "a" + str(k), "kind": r["kind"], "lam": r["lam"],
                        "metrics": PL.correspondence(Xm, true_atts)})
    rec["attractors"] = att_rec
    rec["true_attractors"] = {k: {"kind": v["kind"], "lam": v["lam"]} for k, v in meta["attractors"].items()}
    fpl = []
    for x in L.fps:
        w, _ = net.stability(x)
        xo = net.output_of(x)
        f = {"rho": float(np.max(np.abs(w))), "n_unstable": int(np.sum(np.abs(w) > 1 + 1e-9)), "x": xo.tolist()}
        if eq_line:
            f["dist_true_eq"] = float(np.hypot(xo[0], xo[1]) / L_scale)
        elif len(true_eq):
            dd = np.linalg.norm(np.asarray(true_eq) - xo, axis=1)
            f["nearest_true_eq"], f["dist_true_eq"] = int(np.argmin(dd)), float(dd.min() / L_scale)
        fpl.append(f)
    rec["fixed_points"] = fpl
    rec["time_total_s"] = round(time.time() - t_start, 1)
    legend = sorted(set(lab.tolist()))
    arrays = {"fps": np.asarray(L.fps) if L.fps else np.zeros((0, 2 * d)),
              "reps": np.asarray([r["rep"] for r in L.refs]) if L.refs else np.zeros((0, 2 * d)),
              "att_samples": np.asarray(att_samples) if att_samples else np.zeros((0, 2000, d)),
              "labels": np.array([legend.index(x) for x in lab], dtype=np.int16), "legend": np.array(legend),
              "group_offsets": offs, "group_names": np.array(names)}
    return rec, arrays


def default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default=",".join(REGIMES))
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--kinds", default="esn")
    ap.add_argument("--grid", default="all", help="all | индексы через запятую в ESN_GRID")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--N", type=int, default=300)
    ap.add_argument("--rhos", default="")
    ap.add_argument("--etas", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip", default="", help="файл со списком готовых моделей <режим>/<имя>")
    ap.add_argument("--seen-dirs", default="", help="каталоги out других задач через запятую")
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    out_root = Path(os.environ.get("GPUQ_OUT", "out")) / "C"
    grid = ESN_GRID if a.grid == "all" else [ESN_GRID[int(i)] for i in a.grid.split(",")]
    if a.rhos or a.etas or a.seeds:
        rhos = [float(x) for x in a.rhos.split(",")] if a.rhos else [0.6, 0.9, 1.2]
        etas = [float(x) for x in a.etas.split(",")] if a.etas else [1e-7, 1e-4]
        seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else [0, 1, 2]
        grid = [(r_, e_, s_) for r_ in rhos for e_ in etas for s_ in seeds]
    jobs = []
    for key in a.regimes.split(","):
        for s in a.sets.split(","):
            if "esn" in a.kinds.split(","):
                jobs += [(key, s, "esn", g) for g in grid]
            if "ngrc" in a.kinds.split(","):
                jobs += [(key, s, "ngrc", (None, eta, None)) for eta in NGRC_GRID]
    if a.limit:
        jobs = jobs[: a.limit]
    if a.reverse:
        jobs = jobs[::-1]
    seen = [Path(x) for x in a.seen_dirs.split(",") if x]
    print(f"stage C: {len(jobs)} models, quick={a.quick}", flush=True)
    skip = set()
    if a.skip and Path(a.skip).exists():
        skip = set(x.strip() for x in Path(a.skip).read_text(encoding="utf-8").splitlines() if x.strip())
    cache = {}
    t_all = time.time()
    for n, (key, s, kind, g) in enumerate(jobs):
        rho, eta, seed = g
        name = f"{s}_{kind}_rho{rho}_eta{eta:g}_s{seed}" if kind == "esn" else f"{s}_ngrc_eta{eta:g}"
        if kind == "esn" and a.N != 300:
            name = f"{s}_esn_N{a.N}_rho{rho}_eta{eta:g}_s{seed}"
        d = out_root / key
        d.mkdir(parents=True, exist_ok=True)
        if (d / f"{name}.json").exists() or f"{key}/{name}" in skip or \
                any((sd / "C" / key / f"{name}.json").exists() for sd in seen):
            continue
        if key not in cache:
            cache = {key: load_regime(key)}
        z, meta, l1 = cache[key]
        G.report_progress(n, len(jobs), f"{key} {name}")
        try:
            if kind == "esn":
                rec, arr = analyze_esn(key, z, meta, l1, s, rho, eta, seed, quick=a.quick, N=a.N)
            else:
                rec, arr = analyze_ngrc(key, z, meta, l1, s, eta, quick=a.quick)
            rec["gpu_peak_gb"] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 2) if torch.cuda.is_available() else None
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            np.savez_compressed(d / f"{name}.npz", **arr)
            (d / f"{name}.json").write_text(json.dumps(rec, default=default, ensure_ascii=False, indent=1), encoding="utf-8")
            hl = {k: v["label"] for k, v in rec["hiddenness"].items()}
            best = {at["name"]: min((m.get("delta_att", 9e9) for m in at["metrics"].values()), default=None) for at in rec["attractors"]}
            print(f"[{n + 1}/{len(jobs)}] {key} {name}: A_tr {rec['A_tr']['kind']}, FP {len(rec['fixed_points'])}, "
                  f"attractors {len(rec['attractors'])} {hl} best Δatt {best}; {rec['time_total_s']} s "
                  f"{rec.get('time_s', '')} peak {rec['gpu_peak_gb']} GB", flush=True)
        except Exception as exc:
            if G.fatal_cuda(exc):
                print("fatal CUDA error, stopping the process", flush=True)
                traceback.print_exc()
                sys.exit(3)
            print(f"[{n + 1}/{len(jobs)}] {key} {name}: FAILED", flush=True)
            traceback.print_exc()
            (d / f"{name}.failed.txt").write_text(traceback.format_exc(), encoding="utf-8")
    G.report_progress(len(jobs), len(jobs), "done")
    print(f"stage C done in {time.time() - t_all:.0f} s", flush=True)


if __name__ == "__main__":
    main()

"""Этап N: число и устойчивость равновесий обученной ESN в зависимости от размера N (облегчённый протокол).

Для каждой модели: обучение; равновесия (метод Ньютона из куба, из окрестности обучающих состояний, из поднятых
истинных равновесий, затем дефляция); для каждого — ind = sign det(I − J), число неустойчивых направлений и
вещественных собственных значений > 1, расстояние до истинного равновесия; воспроизведение скрытого аттрактора
(синхронизация сети на 100 отсчётах истинной траектории, затем замкнутый цикл); проверка окрестностей всех
неустойчивых равновесий (радиусы 1e-6, 1e-4, 1e-3, вдоль неустойчивых направлений и 50 случайных точек);
доля бассейна воспроизведённого аттрактора по 2000 точкам куба и 1000 поднятым точкам бассейна истинного HA.
Без поиска границы бассейна и оценки недостающей массы: метка «not_excited» — кандидат, а не «hidden_stat».

Запуск: python gpu/stage_n.py --regimes se3 --sets HE1 --Ns 100,200 --rhos 0.9 --etas 1e-7,1e-4 --seeds 0,1,2
Результат: $GPUQ_OUT/N/<режим>/<набор>_N<N>_rho<ρ>_eta<η>_s<seed>.json
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
import stage_c as SC  # noqa: E402

LAM_C = SC.LAM_C
if os.environ.get("RMQ_CPU") and G.DEV.type == "cpu":
    torch.set_num_threads(int(os.environ["RMQ_CPU"]))


def eq_extra(net, x):
    J = net.jac(x)
    w = torch.linalg.eigvals(J)
    w = G.N_(w)
    I = torch.eye(net.N, dtype=G.DT, device=G.DEV)
    sign, logdet = torch.linalg.slogdet(I - J)
    real = np.abs(w.imag) <= 1e-10 * np.maximum(1.0, np.abs(w))
    return {"index": int(round(float(sign))), "logabsdet": float(logdet),
            "n_real_gt1": int(np.sum(real & (w.real > 1 + 1e-9)))}


def analyze(key, z, meta, set_name, N, rho, eta, seed, quick=False, input_scale=0.5, density=0.05):
    t_start = time.time()
    series, wash = SC.train_sets(z)[set_name]
    esn = P.train_esn_multi(series, n_res=N, spectral_radius=rho, ridge=eta, input_scale=input_scale, density=density,
                            washout=wash, seed=seed)
    t_train = time.time() - t_start
    rec, _ = evaluate(key, z, meta, esn, seed, quick=quick)
    rec = {"regime": key, "set": set_name, "N": N, "rho": rho, "eta": eta, "seed": seed,
           "input_scale": input_scale, "density": density, **rec}
    rec["time_s"] = {"train": round(t_train, 1), **rec["time_s"]}
    rec["time_total_s"] = round(time.time() - t_start, 1)
    return rec


def evaluate(key, z, meta, esn, seed, quick=False, eq_only=False):
    """Облегчённый протокол для готовой модели esn. Возвращает (сведения, массив равновесий K×N).
    eq_only — только равновесия и их индексы (без аттрактора, окрестностей и переписи)."""
    tim = {}
    N = esn["n_res"]
    net = G.Net(esn, h=meta["Delta_t"])
    n_tr, n_obs = (4000, 1000) if quick else (20000, 4000)
    rng = np.random.default_rng(3000 + seed)
    true_eq = [e["point"] for e in meta["equilibria"]]
    L_scale = float(np.max(np.array(meta["box_hi"]) - np.array(meta["box_lo"]))) / 2
    eq_line = key == "le1"
    rec = {"train_rmse": esn["train_rmse"]}
    # равновесия
    t0 = time.time()
    L = PL.Labeler(net, LAM_C)
    nQ, nT = (1000, 500) if quick else ((5000, 2500) if N < 1000 else (2000, 1000))   # при N ≥ 1000 меньше стартов
    nD = 200 if quick else (1000 if N < 1000 else 500)
    rec["n_starts"] = {"cube": nQ, "train": nT, "deflation": nD}
    starts = [rng.uniform(-1, 1, size=(nQ, N)),
              np.clip(esn["R_train"][rng.integers(0, len(esn["R_train"]), nT)] + 0.05 * rng.normal(size=(nT, N)), -1, 1)]
    eqs = [[0.0, 0.0, zz] for zz in np.linspace(-10, -0.5, 20)] if eq_line else list(true_eq)
    for e in eqs:
        xs = np.asarray(e)[None] + 0.01 * L_scale * rng.normal(size=(200, len(e)))
        starts.append(net.lift_const(esn, np.vstack([np.asarray(e)[None], xs]), w=100))
    X, ok = net.newton(np.vstack(starts), max_iter=40)
    if ok.any():
        L.add_fps(X[ok])
    if L.fps:
        X, ok = net.newton(rng.uniform(-1, 1, size=(nD, N)), max_iter=60, deflate=np.asarray(L.fps))
        if ok.any():
            L.add_fps(X[ok])
    tim["fp"] = time.time() - t0
    if eq_only:
        fpi = []
        for x in L.fps:
            info = PL.fp_info(net, x, true_eq, L_scale, eq_line)
            info.pop("x", None)
            fpi.append({**info, **eq_extra(net, x)})
        rec.update({"fixed_points": fpi, "K": len(fpi), "U": int(sum(f["n_unstable"] > 0 for f in fpi)),
                    "index_sum": int(sum(f["index"] for f in fpi)), "time_s": {k: round(v, 1) for k, v in tim.items()}})
        return rec, np.asarray(L.fps) if L.fps else np.zeros((0, N))
    # скрытый аттрактор: синхронизация на истинной траектории
    t0 = time.time()
    hid = meta["hidden"]
    match = {}
    for y in hid:
        Y = z[f"att_{y}"]
        ykind = "chaos" if meta["attractors"][y]["kind"] == "chaotic" else "regular"
        s0 = rng.integers(0, len(Y) - 101, 64)
        U = np.stack([Y[s:s + 100] for s in s0])
        R = net.lift_drive(esn, U, 100)
        o = net.run(R, n_tr=n_tr, n_obs=n_obs, seed=seed + 5)
        chaos = o["lam"] > LAM_C
        keep = ~o["fp"] & (chaos if ykind == "chaos" else ~chaos)
        m = {"frac_osc": float(np.mean(~o["fp"])), "frac_kind": float(np.mean(keep))}
        if keep.sum() >= 8:
            ref = PL.make_ref(PL.feats(o, net.scale)[keep], o["lam"][keep], ykind, o["final"][np.where(keep)[0][0]])
            Xm = net.samples(ref["rep"], 10000, burn=5000)
            corr = PL.correspondence(Xm, {y: Y[:10000]})[y]
            m.update(corr)
            m["reproduced"] = bool(corr.get("delta_att", 9) < 0.2 and corr.get("d_H", 9) < 0.2)
            m["attractor"] = f"a{len(L.refs)}"
            m["lam"] = ref["lam"]
            L.refs.append(ref)
        else:
            m["reproduced"] = False
        match[y] = m
    rec["hidden_match"] = match
    tim["match"] = time.time() - t0
    # точки покоя: сведения и индексы
    t0 = time.time()
    fpi = []
    for x in L.fps:
        info = PL.fp_info(net, x, true_eq, L_scale, eq_line)
        info.pop("x", None)
        fpi.append({**info, **eq_extra(net, x)})
    rec["fixed_points"] = fpi
    rec["K"] = len(fpi)
    rec["U"] = int(sum(f["n_unstable"] > 0 for f in fpi))
    rec["index_sum"] = int(sum(f["index"] for f in fpi))
    tim["fp_info"] = time.time() - t0
    # окрестности неустойчивых точек и доли бассейнов
    t0 = time.time()
    unst = [i for i, f in enumerate(fpi) if f["n_unstable"] > 0]
    nb = PL.radius_test(net, L, unst, radii=(1e-6, 1e-4, 1e-3), n_ball=50, n_tr=n_tr, n_obs=n_obs)
    hits = PL.hits_by_attractor(nb)
    rec["radius_counts"] = nb
    for y, m in match.items():
        a = m.get("attractor")
        if not a:
            continue
        h = hits.get(a, {"r_hat": np.inf, "from": {}})
        m["r_hat"] = None if not np.isfinite(h["r_hat"]) else h["r_hat"]
        m["from"] = h["from"]
        m["label"] = "self_excited" if h["r_hat"] <= 1e-6 else ("excited_r" if h["r_hat"] <= 1e-3 else "not_excited")
    groups = {"cube": rng.uniform(-1, 1, size=(500 if quick else 2000, N))}
    for y in hid:
        if f"ics_basin_{y}" in z.files:
            groups[f"basin_{y}"] = net.lift_const(esn, z[f"ics_basin_{y}"][:1000], w=100)
    for g, S in groups.items():
        o = net.run(S, n_tr=n_tr, n_obs=n_obs, seed=seed + 9)
        rec[f"census_{g}"] = PL.counts(np.asarray(L.label(o), dtype=str))
    tim["radius_census"] = time.time() - t0
    rec["time_s"] = {k: round(v, 1) for k, v in tim.items()}
    return rec, np.asarray(L.fps) if L.fps else np.zeros((0, N))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default="se3")
    ap.add_argument("--sets", default="HE1")
    ap.add_argument("--Ns", default="100,200,300")
    ap.add_argument("--rhos", default="0.9")
    ap.add_argument("--etas", default="1e-7,1e-4")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--input-scales", default="0.5", help="масштаб входа W_in (по умолчанию 0.5)")
    ap.add_argument("--densities", default="0.05", help="плотность W (по умолчанию 0.05)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    jobs = [(k, s, int(N), float(r), float(e), int(sd), float(isc), float(dn))
            for k in a.regimes.split(",") for s in a.sets.split(",") for N in a.Ns.split(",") for r in a.rhos.split(",")
            for e in a.etas.split(",") for sd in a.seeds.split(",") for isc in a.input_scales.split(",")
            for dn in a.densities.split(",")]
    if a.reverse:
        jobs = jobs[::-1]
    out_root = Path(os.environ.get("GPUQ_OUT", "out")) / "N"
    print(f"stage N: {len(jobs)} models, device {G.DEV}", flush=True)
    cache = {}
    for n, (key, s, N, rho, eta, seed, isc, dn) in enumerate(jobs):
        name = f"{s}_N{N}_rho{rho}_eta{eta:g}_s{seed}"
        if isc != 0.5:
            name += f"_in{isc:g}"
        if dn != 0.05:
            name += f"_dens{dn:g}"
        d = out_root / key
        d.mkdir(parents=True, exist_ok=True)
        if (d / f"{name}.json").exists():
            continue
        if key not in cache:
            cache = {key: SC.load_regime(key)}
        z, meta, _ = cache[key]
        G.report_progress(n, len(jobs), f"{key} {name}")
        try:
            rec = analyze(key, z, meta, s, N, rho, eta, seed, quick=a.quick, input_scale=isc, density=dn)
            (d / f"{name}.json").write_text(json.dumps(rec, default=float), encoding="utf-8")
            hm = {y: (m.get("reproduced"), m.get("label"), round(m.get("delta_att", 9), 3)) for y, m in rec["hidden_match"].items()}
            print(f"[{n + 1}/{len(jobs)}] {key} {name}: K {rec['K']} U {rec['U']} Σind {rec['index_sum']} {hm}; "
                  f"{rec['time_total_s']} s {rec['time_s']}", flush=True)
        except Exception as exc:
            traceback.print_exc()
            if G.fatal_cuda(exc):
                print("fatal CUDA error, stopping the process", flush=True)
                sys.exit(3)
            (d / f"{name}.failed.txt").write_text(traceback.format_exc(), encoding="utf-8")
    G.report_progress(len(jobs), len(jobs), "done")


if __name__ == "__main__":
    main()

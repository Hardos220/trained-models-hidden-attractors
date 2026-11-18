"""Этап A: перепись эхо-сетей, обученных на системе Лоренца, по плану data/a_census_design.csv.

Запуск: python gpu/stage_a.py --rows 0-63 [--quick] [--skip FILE]
Результат: $GPUQ_OUT/A/<строка>_<b|b0>.json и .npz.
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

LAM_C = 0.05
H = 0.02
LAM_MAX = 0.9056
LORENZ_EQ = np.array([[0.0, 0.0, 0.0], [8.485, 8.485, 27.0], [-8.485, -8.485, 27.0]])


def design(path="data/a_census_design.csv"):
    rows = []
    for line in (ROOT / path).read_text(encoding="utf-8").splitlines()[1:]:
        r = line.split(",")
        rows.append({"row": int(r[0]), "seed": int(r[1]), "N": int(r[2]), "rho": float(r[3]), "eta": float(r[4]),
                     "input_scale": float(r[5]), "n_train": int(r[6])})
    return rows


def survival(net, orb, stable_fps, rng, n=400, n_steps=50000, every=100):
    R = G.T(np.clip(orb[rng.integers(0, len(orb), n)] + 1e-6 * rng.normal(size=(n, net.N)), -1, 1))
    t_hit = np.full(n, np.inf)
    which = np.full(n, -1)
    if not stable_fps:
        return {"fraction_collapsed": 0.0, "n": n, "t_hit": [], "to_fp": {}}
    Pf = G.T(np.asarray(stable_fps))
    tol = 1e-2 * (1 + Pf.norm(dim=1))
    for k in range(0, n_steps, every):
        R = net.advance(R, every)
        d = torch.cdist(R, Pf)
        j = d.argmin(1)
        near = d[torch.arange(n, device=G.DEV), j] < tol[j]
        near = G.N_(near)
        jj = G.N_(j)
        new = near & ~np.isfinite(t_hit)
        t_hit[new] = k + every
        which[new] = jj[new]
    fin = np.isfinite(t_hit)
    grid = np.arange(0, n_steps + 1, 500)
    return {"fraction_collapsed": float(fin.mean()), "n": n, "t_hit": t_hit[fin].tolist(),
            "S": [float(np.mean(t_hit > t)) for t in grid], "t_grid": grid.tolist(),
            "to_fp": PL.counts([f"s{j}" for j in which[which >= 0]]) if fin.any() else {}}


def vpt(esn, net, series, att_rms, thresholds=(0.2, 0.4, 0.8)):
    res = {th: [] for th in thresholds}
    for k in range(20):
        st = 40000 + 200 * k
        r = net.lift_drive(esn, series[st:st + 100][None], 100)
        X = net.samples(r[0], 1000, burn=0)
        # samples() делает шаг перед первым отсчётом: x̂_{101..}; первый прогноз x̂_100 = выход самого r
        x100 = net.output_of(r[0])
        Xh = np.vstack([x100[None], X[:-1]])
        Xt = series[st + 100: st + 1100]
        e = np.linalg.norm(Xh - Xt, axis=1) / att_rms
        for th in thresholds:
            over = np.where(e > th)[0]
            res[th].append((over[0] if len(over) else 1000) * H * LAM_MAX)
    return {str(th): {"mean": float(np.mean(v)), "median": float(np.median(v)), "all": v} for th, v in res.items()}


def cond_lyap(esn, u, d0=1e-8):
    W = G.T(esn["W"]); Win = G.T(esn["W_in"]); b = G.T(esn["b"])
    Un = G.T(u / esn["scale"])
    r = torch.zeros(1, W.shape[0], dtype=G.DT, device=G.DEV)
    for k in range(500):
        r = torch.tanh(r @ W.T + Un[k] @ Win.T + b)
    v = torch.randn_like(r)
    q = r + d0 * v / v.norm()
    s = 0.0
    for k in range(500, len(Un)):
        r = torch.tanh(r @ W.T + Un[k] @ Win.T + b)
        q = torch.tanh(q @ W.T + Un[k] @ Win.T + b)
        dq = q - r
        dist = float(dq.norm())
        s += np.log(max(dist, 1e-300) / d0)
        q = r + dq * (d0 / max(dist, 1e-300))
    return s / ((len(Un) - 500) * H)


def lyap_spectrum(net, r0, k=6, n=10000, burn=2000):
    R = net.advance(G.T(r0).reshape(1, -1), burn)[0]
    Q = torch.linalg.qr(torch.randn(net.N, k, dtype=G.DT, device=G.DEV))[0]
    s = torch.zeros(k, dtype=G.DT, device=G.DEV)
    for _ in range(n):
        z = net.A @ R + net.b
        t = torch.tanh(z)
        Q = (1 - t * t)[:, None] * (net.A @ Q)
        R = t
        Q, Rr = torch.linalg.qr(Q)
        s += torch.log(torch.abs(torch.diagonal(Rr)))
    return (G.N_(s) / (n * H)).tolist()


def analyze(spec, zero_bias, series, l1, ics_phys, att, quick=False):
    t_start = time.time()
    tim = {}
    N = spec["N"]
    u = series[: spec["n_train"]]
    esn = P.train_esn_multi([u], n_res=N, spectral_radius=spec["rho"], ridge=spec["eta"], input_scale=spec["input_scale"],
                            density=0.05, washout=500, seed=spec["seed"], zero_bias=zero_bias)
    net = G.Net(esn, h=H)
    n_tr, n_obs = (4000, 1000) if quick else (20000, 4000)
    rng = np.random.default_rng(7000 + spec["row"] + (1 if zero_bias else 0))
    rec = dict(spec, zero_bias=zero_bias, train_rmse=esn["train_rmse"], precision_bulk=PL.PREC_BULK)
    t0 = time.time()
    L = PL.Labeler(net, LAM_C)
    nQ, nT = (2000, 1000) if quick else (10000, 5000)
    S0 = np.vstack([rng.uniform(-1, 1, size=(nQ, N)),
                    np.clip(esn["R_train"][rng.integers(0, len(esn["R_train"]), nT)] + 0.05 * rng.normal(size=(nT, N)), -1, 1)])
    X, ok = net.newton(S0)
    found_by = {}
    for j in (L.add_fps(X[ok]) if ok.any() else []):
        found_by.setdefault(j, "start")
    tim["fp_newton"] = time.time() - t0
    t0 = time.time()
    atr = PL.training_attractor(net, L, esn["R_train"][-1])
    rec["A_tr"] = {"kind": atr["kind"], "label": atr["label"], "reconstructed_chaotic": atr["kind"] == "chaos"}
    orb = atr["orbit"]
    groups = {"Q": rng.uniform(-1, 1, size=(nQ, N)),
              "kick": np.clip(orb[rng.integers(0, len(orb), 1000)] + 0.1 * rng.normal(size=(1000, N)), -1, 1),
              "pL1_10": net.lift_drive(esn, l1[:, :10], 10),
              "pL2": net.lift_const(esn, ics_phys, w=100)}
    names = list(groups)
    states = [groups[g] for g in names]
    allS = np.vstack(states)
    out = net.run(allS, n_tr=n_tr, n_obs=n_obs, seed=spec["row"], prec=PL.PREC_BULK)
    lab0 = L.label(out)
    cands = PL.discover(net, L, out, lab0, n_life=10000 if quick else 50000, n_obs=n_obs)
    lab = np.asarray(L.label(out), dtype=str)
    offs = np.cumsum([0] + [len(x) for x in states])
    rec["census"] = {g: PL.counts(lab[offs[i]:offs[i + 1]]) for i, g in enumerate(names)}
    rec["discovery"] = cands
    tim["census"] = time.time() - t0
    t0 = time.time()
    stable = [i for i, x in enumerate(L.fps) if np.max(np.abs(net.stability(x)[0])) < 1]
    seg = []
    reps = [r["rep"] for r in L.refs]
    if reps and stable:
        per = max(1, (500 if quick else 2000) // (len(reps) * len(stable)))
        for rp in reps:
            o_r = G.N_(net.orbit(rp, 1000, every=5))
            for i in stable:
                lam_ = rng.uniform(0, 1, size=(per, 1))
                seg.append(lam_ * L.fps[i][None] + (1 - lam_) * o_r[rng.integers(0, len(o_r), per)])
    if seg:
        X, ok = net.newton(np.vstack(seg))
        for j in (L.add_fps(X[ok]) if ok.any() else []):
            found_by.setdefault(j, "segment")
    if L.fps:
        X, ok = net.newton(rng.uniform(-1, 1, size=(200 if quick else 1000, N)), max_iter=60, deflate=np.asarray(L.fps))
        for j in (L.add_fps(X[ok]) if ok.any() else []):
            found_by.setdefault(j, "deflation")
    tim["fp_extra"] = time.time() - t0
    t0 = time.time()
    stable = [i for i, x in enumerate(L.fps) if np.max(np.abs(net.stability(x)[0])) < 1]
    rec["survival"] = survival(net, orb, [L.fps[i] for i in stable], rng, n=100 if quick else 400,
                               n_steps=10000 if quick else 50000)
    tim["survival"] = time.time() - t0
    t0 = time.time()
    unst = [i for i in range(len(L.fps)) if np.max(np.abs(net.stability(L.fps[i])[0])) > 1 + 1e-9]
    nb = PL.radius_test(net, L, unst, n_ball=50 if quick else 200, n_tr=n_tr, n_obs=n_obs)
    hits = PL.hits_by_attractor(nb)
    hid = {}
    for k in range(len(L.refs)):
        a = f"a{k}"
        h = hits.get(a, {"r_hat": np.inf, "from": {}})
        cand = not (h["r_hat"] <= 1e-3)
        rec_h = {"r_hat": h["r_hat"] if np.isfinite(h["r_hat"]) else None, "from": h["from"], "candidate": cand}
        cons = None
        if cand:
            bs = PL.boundary_search(net, L, a, allS, lab, K=200 if quick else 1000, n_starts=2000 if quick else 20000,
                                    n_tr=n_tr, n_obs=n_obs)
            for j in bs["new_fp_ids"]:
                found_by.setdefault(j, "boundary")
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
    tim["hiddenness"] = time.time() - t0
    t0 = time.time()
    att_rec, att_samples = [], []
    for k, r in enumerate(L.refs):
        Xm = net.samples(r["rep"], 10000, burn=10000)
        att_samples.append(Xm[:2000])
        att_rec.append({"name": f"a{k}", "kind": r["kind"], "lam": r["lam"], "metrics": PL.correspondence(Xm, {"lorenz": att[:10000]})})
    rec["attractors"] = att_rec
    rms = float(np.sqrt(np.mean(np.sum(att ** 2, axis=1))))
    rec["vpt"] = vpt(esn, net, series, rms)
    rec["cle"] = cond_lyap(esn, series[40000:45000])
    if atr["kind"] in ("chaos", "regular"):
        rec["lyap_spectrum"] = lyap_spectrum(net, L.refs[0]["rep"], n=2000 if quick else 10000)
    rec["fixed_points"] = [dict(PL.fp_info(net, x, LORENZ_EQ, 25.0), found_by=found_by.get(i, "label"))
                           for i, x in enumerate(L.fps)]
    tim["covariates"] = time.time() - t0
    rec["time_s"] = {k: round(v, 1) for k, v in tim.items()}
    rec["time_total_s"] = round(time.time() - t_start, 1)
    legend = sorted(set(lab.tolist()))
    arrays = {"fps": np.asarray(L.fps) if L.fps else np.zeros((0, N)),
              "reps": np.asarray([r["rep"] for r in L.refs]) if L.refs else np.zeros((0, N)),
              "att_samples": np.asarray(att_samples) if att_samples else np.zeros((0, 2000, 3)),
              "labels": np.array([legend.index(x) for x in lab], dtype=np.int16), "legend": np.array(legend),
              "group_offsets": offs, "group_names": np.array(names)}
    return rec, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="0-127")
    ap.add_argument("--design", default="data/a_census_design.csv")
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip", default="")
    ap.add_argument("--seen-dirs", default="", help="каталоги out других задач через запятую")
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.rows.split("-"))
    D = design(a.design)
    zl = np.load(ROOT / "data" / "lorenz_l1.npz")
    series, l1 = zl["series"], zl["l1"]
    zz = np.load(ROOT / "data" / "lorenz.npz")
    ics_phys, att = zz["ics_phys"], zz["att_lorenz"]
    out_root = Path(os.environ.get("GPUQ_OUT", "out")) / "A"
    out_root.mkdir(parents=True, exist_ok=True)
    skip = set()
    if a.skip and Path(a.skip).exists():
        skip = set(x.strip() for x in Path(a.skip).read_text(encoding="utf-8").splitlines() if x.strip())
    jobs = [(spec, zb) for spec in D[lo:hi + 1] for zb in (False, True)]
    if a.reverse:
        jobs = jobs[::-1]
    seen = [Path(x) for x in a.seen_dirs.split(",") if x]
    print(f"stage A: {len(jobs)} networks", flush=True)
    for n, (spec, zb) in enumerate(jobs):
        name = f"{a.tag}r{spec['row']:03d}_{'b0' if zb else 'b'}"
        if (out_root / f"{name}.json").exists() or name in skip or \
                any((sd / "A" / f"{name}.json").exists() for sd in seen):
            continue
        G.report_progress(n, len(jobs), name)
        try:
            rec, arr = analyze(spec, zb, series, l1, ics_phys, att, quick=a.quick)
            np.savez_compressed(out_root / f"{name}.npz", **arr)
            (out_root / f"{name}.json").write_text(json.dumps(rec, default=float, ensure_ascii=False), encoding="utf-8")
            hl = {k: v["label"] for k, v in rec["hiddenness"].items()}
            print(f"[{n + 1}/{len(jobs)}] {name} N={spec['N']}: A_tr {rec['A_tr']['kind']}, FP {len(rec['fixed_points'])}, "
                  f"att {len(rec['attractors'])} {hl}, collapsed {rec['survival']['fraction_collapsed']:.3f}, "
                  f"VPT0.4 {rec['vpt']['0.4']['mean']:.2f}; {rec['time_total_s']} s {rec['time_s']}", flush=True)
        except Exception as exc:
            if G.fatal_cuda(exc):
                print("fatal CUDA error, stopping the process", flush=True)
                traceback.print_exc()
                sys.exit(3)
            print(f"[{n + 1}/{len(jobs)}] {name}: FAILED", flush=True)
            traceback.print_exc()
            (out_root / f"{name}.failed.txt").write_text(traceback.format_exc(), encoding="utf-8")
    G.report_progress(len(jobs), len(jobs), "done")


if __name__ == "__main__":
    main()

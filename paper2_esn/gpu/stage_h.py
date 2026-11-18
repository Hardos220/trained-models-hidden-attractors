"""Этап H: эхо-сети с hard-tanh, точное множество неподвижных точек перебором 3^N областей, эталонная проверка
окрестностей, протокол 4.6 без знания точного множества и проверка оценок недостающей массы.

Запуск: python gpu/stage_h.py --N 8,10,12 [--rho ...] [--seeds 0,1,2,3,4] [--quick] [--skip FILE]
Результат: $GPUQ_OUT/H/N<N>/<сеть>.json и .npz.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gpu"))
from hsearch import esn_pilot as P  # noqa: E402
import esngpu as G  # noqa: E402
import pipeline as PL  # noqa: E402

LAM_C = 0.05


def map_to_E(X, E, tol=1e-8):
    """Индекс ближайшей точки E (или −1) для каждой строки X."""
    if not len(E) or not len(X):
        return np.full(len(X), -1)
    out = np.empty(len(X), int)
    for i0 in range(0, len(X), 8192):
        F = X[i0:i0 + 8192]
        d = np.sqrt(((F[:, None, :] - E[None]) ** 2).sum(-1))
        j = d.argmin(1)
        ok = d[np.arange(len(F)), j] < tol * (1 + np.linalg.norm(E[j], axis=1))
        out[i0:i0 + 8192] = np.where(ok, j, -1)
    return out


def analyze(u, N, rho, eta, seed, quick=False):
    t_start = time.time()
    tim = {}
    esn = P.train_esn_multi([u], n_res=N, spectral_radius=rho, ridge=eta, input_scale=0.5, density=min(1.0, 10.0 / N),
                            washout=500, seed=seed, activation="hardtanh")
    net = G.Net(esn, h=0.02)
    n_tr, n_obs = (4000, 1000) if quick else (20000, 4000)
    rng = np.random.default_rng(5000 + seed)
    rec = {"N": N, "rho": rho, "eta": eta, "seed": seed, "train_rmse": esn["train_rmse"], "precision_bulk": PL.PREC_BULK}
    # 2. точное множество
    t0 = time.time()
    E, regs, deg = G.hardtanh_enumerate(net.A_np, net.b_np, chunk=max(10000, int(float(os.environ.get("ESN_ENUM_BYTES", "2e8")) / (N * N * 8))))
    tim["enumerate"] = time.time() - t0
    E_info = []
    for x in E:
        w, _ = net.stability(x)
        E_info.append({"rho": float(np.max(np.abs(w))), "n_unstable": int(np.sum(np.abs(w) > 1 + 1e-9)),
                       "n_linear": int(np.sum(np.abs(net.A_np @ x + net.b_np) < 1))})
    rec["E"] = {"n": int(len(E)), "degenerate": int(deg), "info": E_info}
    L = PL.Labeler(net, LAM_C, list(E))
    # 3. обучающий аттрактор и перепись
    t0 = time.time()
    atr = PL.training_attractor(net, L, esn["R_train"][-1])
    rec["A_tr"] = {"kind": atr["kind"], "label": atr["label"]}
    nQ = 2000 if quick else 10000
    Qs = rng.uniform(-1, 1, size=(nQ, N))
    orb = atr["orbit"]
    kicks = np.clip(orb[rng.integers(0, len(orb), 1000)] + 0.1 * rng.normal(size=(1000, N)), -1, 1)
    S = np.vstack([Qs, kicks])
    out = net.run(S, n_tr=n_tr, n_obs=n_obs, seed=seed + 3, prec=PL.PREC_BULK)
    lab0 = L.label(out)
    cands = PL.discover(net, L, out, lab0, n_life=10000 if quick else 50000, n_obs=n_obs)
    lab = np.asarray(L.label(out), dtype=str)
    rec["census"] = {"Q": PL.counts(lab[:nQ]), "kick": PL.counts(lab[nQ:])}
    rec["discovery"] = cands
    rec["fp_not_in_E"] = int(len(L.fps) - len(E))
    tim["census"] = time.time() - t0
    unstE = [i for i in range(len(E)) if E_info[i]["rho"] > 1 + 1e-9]
    # 4. эталонная проверка окрестностей от всех неустойчивых точек E
    t0 = time.time()
    nb_ref = PL.radius_test(net, L, unstE, n_ball=200 if quick else 1000, n_tr=n_tr, n_obs=n_obs, seed=61)
    hits_ref = PL.hits_by_attractor(nb_ref)
    tim["radius_ref"] = time.time() - t0
    # 5. протокол без знания E: E_f методом Ньютона по областям
    t0 = time.time()
    St = np.vstack([rng.uniform(-1, 1, size=(nQ, N)),
                    np.clip(esn["R_train"][rng.integers(0, len(esn["R_train"]), nQ)] + 0.05 * rng.normal(size=(nQ, N)), -1, 1)])
    Xf, okf = net.newton(St, chunk=int(os.environ.get("ESN_NEWTON_CHUNK", "65536")))
    jf = map_to_E(Xf[okf], E)
    Ef_ids = sorted(set(int(j) for j in jf if j >= 0))
    rec["E_f"] = {"ids": Ef_ids, "n_converged": int(okf.sum()), "outside_E": int(np.sum(jf < 0))}
    unstF = [i for i in Ef_ids if E_info[i]["rho"] > 1 + 1e-9]
    nb_f = PL.radius_test(net, L, unstF, n_ball=50 if quick else 200, n_tr=n_tr, n_obs=n_obs, seed=71)
    hits_f = PL.hits_by_attractor(nb_f)
    tim["radius_protocol"] = time.time() - t0
    # 6. для каждого колебательного аттрактора: граница, протокол, калибровка недостающей массы
    t0 = time.time()
    att = {}
    n_cal = 100000 if quick else 1000000
    reps = 10 if quick else 50
    sizes = (1000, 10000, 20000)
    for k in range(len(L.refs)):
        a = f"a{k}"
        hr = hits_ref.get(a, {"r_hat": np.inf, "from": {}})
        hf = hits_f.get(a, {"r_hat": np.inf, "from": {}})
        r_ref = hr["r_hat"]
        r_f = hf["r_hat"]
        cand = not (r_f <= 1e-3)
        # точки у границы (для всех аттракторов) — бисекция
        rng_a = np.random.default_rng(900 + k)
        ia = np.where(lab == a)[0]
        io = np.where(lab != a)[0]
        K = 200 if quick else 1000
        eps = 1e-3
        if len(io) == 0:
            att[a] = {"note": "no other outcomes in census", "r_hat_ref": None if not np.isfinite(r_ref) else r_ref}
            continue
        if len(ia) >= 50:
            Ap = S[rng_a.choice(ia, K)]
        else:
            o_r = G.N_(net.orbit(L.refs[k]["rep"], 2000, every=2))
            Ap = np.clip(o_r[rng_a.integers(0, len(o_r), K)] + eps * rng_a.normal(size=(K, N)) / np.sqrt(N), -1, 1)
        Op = S[rng_a.choice(io, K)]

        def in_basin(X):
            o = net.run(np.clip(X, -1, 1), n_tr=n_tr, n_obs=n_obs, seed=int(rng_a.integers(1 << 30)))
            return np.asarray(L.label(o), dtype=str) == a
        Bp = PL.bisect(in_basin, Ap, Op, eps)

        def nu(n):
            return np.clip(Bp[rng_a.integers(0, K, n)] + eps * rng_a.normal(size=(n, N)) / np.sqrt(N), -1, 1)
        # частоты p̂_j по 10⁶ стартов
        X1, ok1 = net.newton(nu(n_cal), chunk=int(os.environ.get("ESN_NEWTON_CHUNK", "65536")))
        j1 = map_to_E(X1[ok1], E)
        p_hat = np.bincount(j1[j1 >= 0], minlength=len(E)) / n_cal
        new_outside = int(np.sum(j1 < 0))
        # протокол 4.6, п. 4 — один набор из 2·10⁴ стартов
        Xb, okb = net.newton(nu(20000), chunk=int(os.environ.get("ESN_NEWTON_CHUNK", "65536")))
        jb = map_to_E(Xb[okb], E)
        mult = np.bincount(jb[jb >= 0], minlength=len(E))
        mm = PL.missing_mass_bounds(mult[mult > 0], 20000)
        new_ids = sorted(set(int(j) for j in jb if j >= 0) - set(Ef_ids))
        r_f2 = r_f
        if cand and new_ids:
            nb2 = PL.radius_test(net, L, [i for i in new_ids if E_info[i]["rho"] > 1 + 1e-9],
                                 n_ball=50 if quick else 200, n_tr=n_tr, n_obs=n_obs, seed=81)
            h2 = PL.hits_by_attractor(nb2).get(a)
            if h2:
                r_f2 = min(r_f, h2["r_hat"])
        cand2 = cand and not (r_f2 <= 1e-3)
        label = PL.hidden_label(r_f2, cand2, mm["conservative"] if cand else None)
        label_ref = PL.hidden_label(r_ref, not (r_ref <= 1e-3), 0.0)
        # повторы для покрытия границ
        rows = []
        for n_d in sizes:
            for rep in range(reps):
                Xr, okr = net.newton(nu(n_d), chunk=int(os.environ.get("ESN_NEWTON_CHUNK", "65536")))
                jr = map_to_E(Xr[okr], E)
                m_r = np.bincount(jr[jr >= 0], minlength=len(E))
                seen = m_r > 0
                b = PL.missing_mass_bounds(m_r[seen], n_d)
                rows.append({"n": n_d, "rep": rep, "mult": m_r.tolist(), "M_true": float(p_hat[~seen].sum()),
                             "G": b["G"], "conservative": b["conservative"], "refined": b["refined"],
                             "n_converged": int(okr.sum())})
        att[a] = {"kind": L.refs[k]["kind"], "lam": L.refs[k]["lam"],
                  "r_hat_ref": None if not np.isfinite(r_ref) else r_ref, "from_ref": hr["from"],
                  "r_hat_protocol": None if not np.isfinite(r_f2) else r_f2, "candidate": cand,
                  "label_protocol": label, "label_reference": label_ref, "missing_mass": mm,
                  "p_hat": p_hat.tolist(), "outside_E_cal": new_outside, "replicates": rows}
    rec["attractors"] = att
    tim["calibration"] = time.time() - t0
    # частоты находок из куба
    t0 = time.time()
    Xq, okq = net.newton(rng.uniform(-1, 1, size=(n_cal, N)), chunk=int(os.environ.get("ESN_NEWTON_CHUNK", "65536")))
    jq = map_to_E(Xq[okq], E)
    rec["p_cube"] = (np.bincount(jq[jq >= 0], minlength=len(E)) / n_cal).tolist()
    rec["cube_outside_E"] = int(np.sum(jq < 0))
    tim["cube"] = time.time() - t0
    rec["radius_ref"] = nb_ref
    rec["radius_protocol"] = nb_f
    rec["time_s"] = {k: round(v, 1) for k, v in tim.items()}
    rec["time_total_s"] = round(time.time() - t_start, 1)
    arrays = {"E": E, "regions": regs, "reps": np.asarray([r["rep"] for r in L.refs]) if L.refs else np.zeros((0, N))}
    return rec, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", default="8,10,12,14,16,18")
    ap.add_argument("--rho", default="0.6,0.9,1.2,1.5")
    ap.add_argument("--eta", default="1e-7,1e-4")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip", default="")
    ap.add_argument("--seen-dirs", default="", help="каталоги out других задач через запятую")
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    u = np.load(ROOT / "data" / "lorenz_l1.npz")["series"][:20000]
    out_root = Path(os.environ.get("GPUQ_OUT", "out")) / "H"
    skip = set()
    if a.skip and Path(a.skip).exists():
        skip = set(x.strip() for x in Path(a.skip).read_text(encoding="utf-8").splitlines() if x.strip())
    jobs = [(int(N), float(r), float(e), int(s)) for N in a.N.split(",") for r in a.rho.split(",")
            for e in a.eta.split(",") for s in a.seeds.split(",")]
    if a.reverse:
        jobs = jobs[::-1]
    seen = [Path(x) for x in a.seen_dirs.split(",") if x]
    print(f"stage H: {len(jobs)} networks", flush=True)
    for n, (N, rho, eta, seed) in enumerate(jobs):
        name = f"N{N}_rho{rho}_eta{eta:g}_s{seed}"
        d = out_root / f"N{N}"
        d.mkdir(parents=True, exist_ok=True)
        if (d / f"{name}.json").exists() or f"N{N}/{name}" in skip or \
                any((sd / "H" / f"N{N}" / f"{name}.json").exists() for sd in seen):
            continue
        G.report_progress(n, len(jobs), name)
        try:
            rec, arr = analyze(u, N, rho, eta, seed, quick=a.quick)
            np.savez_compressed(d / f"{name}.npz", **arr)
            (d / f"{name}.json").write_text(json.dumps(rec, default=float, ensure_ascii=False), encoding="utf-8")
            sm = {k: (v.get("label_reference"), v.get("label_protocol"), v.get("missing_mass", {}).get("conservative"))
                  for k, v in rec["attractors"].items()}
            print(f"[{n + 1}/{len(jobs)}] {name}: |E| {rec['E']['n']}, |E_f| {len(rec['E_f']['ids'])}, A_tr {rec['A_tr']['kind']}, "
                  f"attractors {sm}; {rec['time_total_s']} s {rec['time_s']}", flush=True)
        except Exception as exc:
            if G.fatal_cuda(exc):
                print("fatal CUDA error, stopping the process", flush=True)
                traceback.print_exc()
                sys.exit(3)
            print(f"[{n + 1}/{len(jobs)}] {name}: FAILED", flush=True)
            traceback.print_exc()
            (d / f"{name}.failed.txt").write_text(traceback.format_exc(), encoding="utf-8")
    G.report_progress(len(jobs), len(jobs), "done")


if __name__ == "__main__":
    main()

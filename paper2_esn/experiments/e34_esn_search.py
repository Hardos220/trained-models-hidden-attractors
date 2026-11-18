"""Этап 3: применение метода поиска к замкнутому резервуару (эхо-сеть N = 300, обучена на Лоренце).

Задача: найти редкий паразитный режим обученной сети в кубе B = [-1, 1]^N (образ tanh).
Эталоны берутся из проверки e33 (неподвижные точки, подтверждённые паразитные аттракторы,
реконструированный аттрактор); целевой режим задаётся именем (например, fp:5), остальные режимы
считаются известными. Метки — тем же классификатором, что в e33, с горизонтом n_tr + n_obs.
Признаки для стратегий — в общей раскладке nfeat(N) (статистики состояний резервуара по раннему
окну и окну наблюдения), поэтому расщепление, активное обучение и портфель работают без изменений.
Общие случайные числа: генератор запуска r — default_rng(1000 + r), как в e20.

Запуск: python e34_esn_search.py <seed> <target> [--budget 3000] [--R 20] [--strategies ...] [--procs 12]
Выход: results/e34_esn_search_s<seed>_<target>.json (с возобновлением по стратегиям).
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e33_esn_verify import Labeler, build, fu, reconstructed  # noqa: E402
from hsearch.identify import LAB_UNKNOWN  # noqa: E402
from hsearch.integrate import F_FP, F_LAM, F_RMAX, feature_slices, nfeat  # noqa: E402
from hsearch.search.harness import Problem, n_delta_from_runs, run_search  # noqa: E402
from hsearch.search.strategies import STRATEGIES  # noqa: E402

N_TR, N_OBS, N_EARLY = 6000, 2000, 500


class ESNProxy:
    """Резервуар как «система» для каркаса поиска: куб [-1, 1]^N, отображение."""
    kind = "map"

    def __init__(self, N, h):
        self.dim = N
        self.box_lo = -np.ones(N)
        self.box_hi = np.ones(N)
        self.n_tr, self.n_obs = N_TR, N_OBS
        self.dt = h
        self.name = "esn"
        self.equilibria = []

    def sample_box(self, n, rng):
        return rng.uniform(self.box_lo, self.box_hi, size=(int(n), self.dim))


def run_feats(m, X0, rng, n_tr=N_TR, n_obs=N_OBS, n_early=N_EARLY, d0=1e-7):
    """Пакетный прогон с отпечатками: признаки в раскладке nfeat(N) и статистики выхода."""
    R = np.array(X0, float, copy=True)
    n, N = R.shape
    es = np.zeros((n, N)); eq = np.zeros((n, N))
    emn = np.full((n, N), np.inf); emx = np.full((n, N), -np.inf)
    for k in range(n_tr):
        R = m.step(R)
        if k < n_early:
            es += R; eq += R * R
            np.minimum(emn, R, out=emn); np.maximum(emx, R, out=emx)
    ne = max(1, min(n_tr, n_early))
    e_mean = es / ne
    e_std = np.sqrt(np.maximum(eq / ne - e_mean ** 2, 0.0))
    v = rng.normal(size=R.shape)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    Q = R + d0 * v
    s1 = np.zeros((n, N)); s2 = np.zeros((n, N))
    smn = np.full((n, N), np.inf); smx = np.full((n, N), -np.inf)
    u1 = np.zeros((n, 3)); u2 = np.zeros((n, 3))
    umn = np.full((n, 3), np.inf); umx = np.full((n, 3), -np.inf)
    logsum = np.zeros(n); drift = np.zeros(n)
    for k in range(n_obs):
        Rn = m.step(R)
        Q = m.step(Q)
        if k >= n_obs - 50:
            drift = np.maximum(drift, np.abs(Rn - R).max(axis=1))
        R = Rn
        dQ = Q - R
        dist = np.linalg.norm(dQ, axis=1)
        dist = np.where(dist > 0, dist, d0)
        logsum += np.log(dist / d0)
        Q = R + dQ * (d0 / dist)[:, None]
        s1 += R; s2 += R * R
        np.minimum(smn, R, out=smn); np.maximum(smx, R, out=smx)
        u = m.output(R)
        u1 += u; u2 += u * u
        np.minimum(umn, u, out=umn); np.maximum(umx, u, out=umx)
    lam = logsum / (n_obs * m.h)
    fp = drift < 1e-9 * (1 + np.abs(R).max(axis=1))
    feats = np.zeros((n, nfeat(N)))
    sl = feature_slices(N)
    mean = s1 / n_obs
    feats[:, F_FP] = fp.astype(float)
    feats[:, F_LAM] = lam
    feats[:, F_RMAX] = np.abs(smx).max(axis=1)
    feats[:, sl["mean"]] = mean
    feats[:, sl["std"]] = np.sqrt(np.maximum(s2 / n_obs - mean ** 2, 0.0))
    feats[:, sl["min"]] = smn
    feats[:, sl["max"]] = smx
    feats[:, sl["emean"]] = e_mean
    feats[:, sl["estd"]] = e_std
    feats[:, sl["emin"]] = emn
    feats[:, sl["emax"]] = emx
    um = u1 / n_obs
    out = {"u_mean": um, "u_std": np.sqrt(np.maximum(u2 / n_obs - um ** 2, 0.0)), "u_min": umn, "u_max": umx,
           "lam": lam, "fp": fp, "final": R}
    return feats, out


def load_reference_set(seed, m, ref, rng, n_res=300):
    """Неподвижные точки и подтверждённые паразитные аттракторы из e33; кластеры перекалибруются
    на окне наблюдения поиска."""
    rec = next(r for r in json.loads((ROOT / "results" / f"e33_esn_verify_N{n_res}.json").read_text(encoding="utf-8"))
               if r["seed"] == seed)
    z = np.load(ROOT / "results" / f"e33_esn_attractors_N{n_res}.npz")
    fps = [z[f"s{seed}_fp{i}_r"] for i in range(len(rec["fixed_points"]))]
    lab = Labeler(m, ref, fps)
    clusters = []
    for a in rec["spurious_attractors"]:
        rep = z[f"s{seed}_{a['name']}_r"]
        X = rep[None, :] + 1e-7 * rng.normal(size=(32, m.N))
        o = m.run(X, n_tr=2000, n_obs=N_OBS, rng=rng)
        F = fu(o)
        mu = F.mean(0)
        clusters.append({"kind": a["kind"], "mu": mu, "sd": F.std(0),
                         "floor": (0.05 if a["kind"] == "chaos" else 0.02) * (1 + np.abs(mu)), "name": a["name"]})
    lab.clusters = clusters
    names = ["lorenz"] + [f"fp:{i}" for i in rec["stable_fp"]] + [c["name"] for c in clusters]
    return rec, lab, names


class ESNProblem(Problem):
    """Оценка пачки начальных состояний резервуара; метки — индексы в списке имён режимов."""

    def setup(self, m, labeler, names):
        self.m, self.labeler, self.names = m, labeler, names
        self.index = {nm: i for i, nm in enumerate(names)}
        return self

    def evaluate(self, ics):
        ics = np.clip(np.ascontiguousarray(ics, dtype=np.float64), -1.0, 1.0)
        feats, out = run_feats(self.m, ics, self.rng)
        lab = self.labeler.label(out)
        labels = np.array([self.index.get(str(s), LAB_UNKNOWN) for s in lab], dtype=int)
        self.n_eval += len(ics)
        return feats, out["final"], labels


G = {}


def init_worker(seed, n_res, target):
    u_all, esn, m = build(seed, n_res)
    m.fast = False
    rng = np.random.default_rng(3400 + seed)
    att, ref, ok = reconstructed(m, esn, rng)
    rec, lab, names = load_reference_set(seed, m, ref, rng, n_res)
    if target not in names:
        raise ValueError(f"цель {target} не найдена среди {names}")
    G.update(m=m, lab=lab, names=names, proxy=ESNProxy(m.N, m.h), target=names.index(target),
             known=[i for i in range(len(names)) if names[i] != target])


def task(args):
    strategy, r, budget = args
    t0 = time.time()
    rng = np.random.default_rng(1000 + r)
    prob = ESNProblem(G["proxy"], G["names"], [G["target"]], G["known"], rng).setup(G["m"], G["lab"], G["names"])
    strat = STRATEGIES[strategy](prob, batch=20, n_init=50, n_pop=50)
    res = run_search(prob, strat, budget)
    disc = [(int(n), int(lb)) for n, lb in res["discoveries"][:10]]
    return strategy, r, res["found_at"], disc, round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("target")
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--R", type=int, default=20)
    ap.add_argument("--strategies", default="uniform,sus,portfolio_no_sus,portfolio")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--n_res", type=int, default=300)
    a = ap.parse_args()
    import multiprocessing as mp
    out_path = ROOT / "results" / f"e34_esn_search_s{a.seed}_{a.target.replace(':', '')}.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    todo = [s for s in a.strategies.split(",") if s not in out]
    tasks = [(s, r, a.budget) for s in todo for r in range(a.R)]
    t0 = time.time()
    results = {s: [None] * a.R for s in todo}
    disc = {s: [None] * a.R for s in todo}
    times = {s: [0.0] * a.R for s in todo}
    if tasks:
        ctx = mp.get_context("spawn")
        with ctx.Pool(a.procs, initializer=init_worker, initargs=(a.seed, a.n_res, a.target)) as pool:
            for s, r, fa, dc, dt in pool.imap_unordered(task, tasks):
                results[s][r] = fa
                disc[s][r] = dc
                times[s][r] = dt
                print(f"{s} r={r}: найдено на {fa} ({dt} с)", flush=True)
    for s in todo:
        q, lo, hi = n_delta_from_runs(results[s])
        out[s] = {"found_at": results[s], "N_0.9": q, "ci95": [lo, hi], "discoveries": disc[s], "time_s": times[s]}
    if "uniform" in out:
        nu = out["uniform"]["N_0.9"]
        for s, v in out.items():
            if s.startswith("_"):
                continue
            v["G_0.9"] = nu / v["N_0.9"] if v["N_0.9"] and np.isfinite(v["N_0.9"]) else 0.0
    out["_meta"] = {"seed": a.seed, "target": a.target, "budget": a.budget, "R": a.R, "n_tr": N_TR, "n_obs": N_OBS,
                    "time_s": round(time.time() - t0, 1)}
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for s, v in out.items():
        if not s.startswith("_"):
            print(f"{s}: N_0.9 = {v['N_0.9']:.0f} [{v['ci95'][0]:.0f}, {v['ci95'][1]:.0f}], G = {v.get('G_0.9', float('nan')):.2f}")


if __name__ == "__main__":
    main()

"""Этап 3: проверка «прочих» режимов замкнутых резервуаров длинными прогонами, кластеризация
паразитных аттракторов, доли бассейнов в кубе и проверка скрытости.

Для каждой реализации (seed) резервуар обучается заново (детерминированно, как в e32), затем:
 1. неподвижные точки (Ньютон из 320 стартов) и их устойчивость;
 2. ансамбль: 1000 равномерных состояний куба [-1, 1]^N и 400 толчков sigma = 0.1 от
    реконструированного аттрактора; переходный процесс 20 000 шагов (400 ед. времени), окно 4000;
 3. «прочие» исходы кластеризуются по отпечаткам выхода (DBSCAN отдельно для хаотических и
    регулярных); каждый кластер проверяется продолжением на 50 000 шагов (время жизни);
 4. скрытость: окрестности всех неустойчивых неподвижных точек (вдоль неустойчивых направлений
    и в шаре радиуса eps, eps в {1e-6, 1e-4, 1e-2}), те же длинные прогоны;
 5. выживание реконструированного аттрактора: 400 состояний на нём (толчок 1e-6), 50 000 шагов,
    момент первого попадания в окрестность устойчивой неподвижной точки;
 6. для рисунков: отрезки траекторий выхода на каждом режиме и пример «обрушения».
Выход: results/e33_esn_verify_N<N>.json, results/e33_esn_attractors_N<N>.npz. С возобновлением.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from hsearch.reservoir_analysis import ClosedLoopESN, classify_runs  # noqa: E402
from hsearch.systems.reservoir import lorenz_data, train_esn  # noqa: E402

N_TR, N_OBS, N_LIFE = 20000, 4000, 50000
EPS_NB = (1e-6, 1e-4, 1e-2)


def fu(out):
    return np.column_stack([out["u_mean"], out["u_std"], out["u_min"], out["u_max"]])


def build(seed, n_res):
    u_all = lorenz_data(25000)
    esn = train_esn(u_all[:20000], n_res=n_res, seed=seed)
    return u_all, esn, ClosedLoopESN(esn, fast=True)


def reconstructed(m, esn, rng):
    R = esn["R_train"][-1][None, :].copy()
    att = []
    for k in range(3000):
        R = m.step(R)
        if k > 500 and k % 10 == 0:
            att.append(R[0].copy())
    att = np.array(att)
    oref = m.run(att[rng.integers(0, len(att), size=48)], n_tr=1000, n_obs=2000, rng=rng)
    keep = (~oref["fp"]) & (oref["lam"] > 0.3)
    F = fu(oref)
    ok = bool(keep.sum() >= 5)
    ref = (F[keep].mean(0), F[keep].std(0)) if ok else (F.mean(0), F.std(0))
    return att, ref, ok


def fp_index(final, fps, tol=1e-3):
    if not fps:
        return np.full(len(final), -1)
    P = np.array(fps)
    d = np.linalg.norm(P[None, :, :] - final[:, None, :], axis=2)
    j = d.argmin(axis=1)
    ok = d[np.arange(len(final)), j] < tol * (1 + np.linalg.norm(P[j], axis=1))
    return np.where(ok, j, -1)


class Labeler:
    """Метки: lorenz / fp:<k> / c<k> (подтверждаемые кластеры прочих режимов) / other_*."""

    def __init__(self, m, ref, fps):
        self.m, self.ref, self.fps = m, ref, list(fps)
        self.clusters = []

    def match_cluster(self, F, lam):
        best, bz = None, np.inf
        for k, c in enumerate(self.clusters):
            if (lam > 0.05) != (c["kind"] == "chaos"):
                continue
            z = np.max(np.abs(F - c["mu"]) / (c["sd"] + c["floor"]))
            if z < bz:
                best, bz = k, z
        return best if bz < 6.0 else None

    def label(self, out):
        lab, _ = classify_runs(out, self.ref)
        lab = lab.astype(object)
        F = fu(out)
        fpi = fp_index(out["final"], self.fps)
        for i in range(len(lab)):
            if lab[i] == "fp":
                if fpi[i] < 0:
                    fpi[i] = fp_index(out["final"][i:i + 1], self.fps)[0]      # список мог пополниться
                if fpi[i] < 0:
                    new = self.m.fixed_points(out["final"][i:i + 1])
                    if new:
                        j = fp_index(new[0][None, :], self.fps, tol=1e-6)[0]
                        if j < 0:
                            self.fps.append(new[0])
                            j = len(self.fps) - 1
                        fpi[i] = j
                lab[i] = f"fp:{fpi[i]}" if fpi[i] >= 0 else "fp:new"
            elif lab[i] in ("other_chaos", "other_periodic") and self.clusters:
                c = self.match_cluster(F[i], out["lam"][i])
                if c is not None:
                    lab[i] = f"c{c}"
        return lab


def counts(lab):
    v, c = np.unique(np.asarray(lab, dtype=str), return_counts=True)
    return {str(a): int(b) for a, b in zip(v, c)}


def discover_clusters(out, lab):
    """DBSCAN по (u_mean, u_std) отдельно для хаотических и регулярных прочих режимов."""
    F = fu(out)
    idx_all = np.where(np.isin(np.asarray(lab, dtype=str), ["other_chaos", "other_periodic"]))[0]
    clusters = []
    for kind, mask in (("chaos", out["lam"][idx_all] > 0.05), ("regular", out["lam"][idx_all] <= 0.05)):
        idx = idx_all[mask]
        if len(idx) == 0:
            continue
        X = F[idx][:, :6]
        eps = 0.5 if kind == "chaos" else 0.1
        cl = DBSCAN(eps=eps, min_samples=1).fit(X).labels_
        for c in np.unique(cl):
            mem = idx[cl == c]
            mu = F[mem].mean(0)
            sd = F[mem].std(0) if len(mem) > 1 else np.zeros_like(mu)
            floor = (0.05 if kind == "chaos" else 0.02) * (1 + np.abs(mu))
            clusters.append({"kind": kind, "members": mem, "mu": mu, "sd": sd, "floor": floor,
                             "lam": float(np.median(out["lam"][mem]))})
    return clusters


def period_of(m, r, n=4000, tol=1e-7):
    """Период орбиты отображения (если найден) или доминирующая частота выхода (инвариантная кривая)."""
    R = r[None, :].copy()
    for _ in range(2000):
        R = m.step(R)
    X = np.empty((n, m.N))
    for k in range(n):
        R = m.step(R)
        X[k] = R[0]
    d = np.linalg.norm(X[1:] - X[0], axis=1)
    hits = np.where(d < tol * (1 + np.linalg.norm(X[0])))[0]
    u = m.output(X)
    if len(hits):
        return {"map_period": int(hits[0] + 1), "u": u}
    x = u[:, 0] - u[:, 0].mean()
    sp = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), d=m.h)
    k = int(np.argmax(sp[1:]) + 1)
    return {"map_period": None, "T_dominant": float(1.0 / f[k]) if f[k] > 0 else None, "u": u}


def traj_u(m, r, n=4000, burn=1000):
    R = r[None, :].copy()
    for _ in range(burn):
        R = m.step(R)
    U = np.empty((n, 3))
    for k in range(n):
        R = m.step(R)
        U[k] = m.output(R)[0]
    return U


def survival(m, att, stable_fps, rng, n=400, n_steps=N_LIFE, every=100, n_rec=24):
    """Траектории с реконструированного аттрактора: момент первого попадания в окрестность
    устойчивой неподвижной точки (inf — не попали за n_steps); запись выхода n_rec траекторий."""
    R = att[rng.integers(0, len(att), size=n)] + 1e-6 * rng.normal(size=(n, m.N))
    t_hit = np.full(n, np.inf)
    which = np.full(n, -1)
    P = np.array(stable_fps) if stable_fps else None
    rec = np.empty((n_steps // 2, n_rec, 3), np.float32)
    for k in range(n_steps):
        R = m.step(R)
        if k % 2 == 0:
            rec[k // 2] = m.output(R[:n_rec])
        if P is not None and k % every == 0:
            d = np.linalg.norm(R[:, None, :] - P[None, :, :], axis=2)
            j = d.argmin(axis=1)
            near = d[np.arange(n), j] < 1e-2 * (1 + np.linalg.norm(P[j], axis=1))
            new = near & ~np.isfinite(t_hit)
            t_hit[new] = k
            which[new] = j[new]
    return t_hit, which, rec, R


def study(seed, n_res=300):
    rng = np.random.default_rng(3300 + seed)
    t0 = time.time()
    u_all, esn, m = build(seed, n_res)
    att, ref, recon_ok = reconstructed(m, esn, rng)
    rec = {"seed": seed, "n_res": n_res, "train_rmse": esn["train_rmse"], "reconstructed_chaotic": recon_ok}
    arrays = {}
    # 1. неподвижные точки
    starts = np.vstack([att[rng.integers(0, len(att), size=120)], rng.uniform(-1, 1, size=(120, n_res)),
                        0.3 * rng.uniform(-1, 1, size=(80, n_res))])
    fps = m.fixed_points(starts)
    L = Labeler(m, ref, fps)
    # 2. ансамбль
    X_cube = rng.uniform(-1, 1, size=(1000, n_res))
    X_kick = np.clip(att[rng.integers(0, len(att), size=400)] + 0.1 * rng.normal(size=(400, n_res)), -1, 1)
    out = m.run(np.vstack([X_cube, X_kick]), n_tr=N_TR, n_obs=N_OBS, rng=rng)
    lab0 = L.label(out)
    # 3. кластеры прочих режимов и время жизни
    cl = discover_clusters(out, lab0)
    confirmed = []
    if cl:
        # время жизни: все кандидаты одним пакетом
        mem_all = [c["members"][:20] for c in cl]
        o2 = m.run(out["final"][np.concatenate(mem_all)], n_tr=N_LIFE, n_obs=N_OBS, rng=rng)
        pos = 0
        for c, mem in zip(cl, mem_all):
            sl = slice(pos, pos + len(mem))
            pos += len(mem)
            sub = {k: v[sl] for k, v in o2.items()}
            tmp = Labeler(m, ref, L.fps)
            tmp.clusters = [c]
            l2 = np.asarray(tmp.label(sub), dtype=str)
            L.fps = tmp.fps
            c["persist"] = float(np.mean(l2 == "c0"))
            c["after_life"] = counts(l2)
            if c["persist"] > 0:
                c["rep"] = sub["final"][np.where(l2 == "c0")[0][0]]
                confirmed.append(c)
    L.clusters = confirmed
    lab = L.label(out)
    lab_c, lab_k = np.asarray(lab[:1000], dtype=str), np.asarray(lab[1000:], dtype=str)
    rec["cube"] = counts(lab_c)
    rec["kick_0.1"] = counts(lab_k)
    rec["candidates"] = [{"kind": c["kind"], "n_members": int(len(c["members"])), "lam": c["lam"], "persist": c["persist"],
                          "after_life": c["after_life"]} for c in cl]
    # 4. скрытость: окрестности всех неустойчивых неподвижных точек одним пакетом
    eig = {}
    X0, tags = [], []
    for i, r in enumerate(list(L.fps)):
        w, dirs = m.unstable_directions(r)
        eig[i] = w
        if np.max(np.abs(w)) < 1:
            continue
        for eps in EPS_NB:
            pts = []
            for dvec in dirs[:10]:
                pts += [r + eps * dvec, r - eps * dvec]
            pts += list(r + eps * rng.normal(size=(40, n_res)) / np.sqrt(n_res))
            X0 += pts
            tags += [(i, eps)] * len(pts)
    hid, excited = {}, {}
    if X0:
        o3 = m.run(np.clip(np.array(X0), -1, 1), n_tr=N_TR, n_obs=N_OBS, rng=rng)
        l3 = np.asarray(L.label(o3), dtype=str)
        for (i, eps) in sorted(set(tags)):
            msk = np.array([t == (i, eps) for t in tags])
            cnt = counts(l3[msk])
            hid[f"fp{i}_eps{eps:g}"] = cnt
            for k_ in cnt:
                excited.setdefault(k_, set()).add(i)
    rec["fp_neighbourhoods"] = hid
    rec["excited_from"] = {k: sorted(int(x) for x in v) for k, v in excited.items()}
    # неподвижные точки (включая найденные по ходу) и устойчивость
    fp_info = []
    for i, r in enumerate(L.fps):
        w = eig[i] if i in eig else m.stability(r)[0]
        fp_info.append({"rho": float(np.max(np.abs(w))), "n_unstable": int(np.sum(np.abs(w) > 1 + 1e-9)),
                        "u": m.output(r[None, :])[0].round(4).tolist(), "norm": float(np.linalg.norm(r))})
        arrays[f"s{seed}_fp{i}_eig"] = w.astype(np.complex64)
        arrays[f"s{seed}_fp{i}_r"] = r
    rec["fixed_points"] = fp_info
    stable = [i for i, d in enumerate(fp_info) if d["rho"] < 1]
    rec["stable_fp"] = stable
    # сводка подтверждённых паразитных режимов
    att_list = []
    for k, c in enumerate(confirmed):
        name = f"c{k}"
        per = period_of(m, c["rep"])
        arrays[f"s{seed}_{name}_u"] = per.pop("u").astype(np.float32)
        arrays[f"s{seed}_{name}_r"] = c["rep"]
        n_cube = rec["cube"].get(name, 0)
        att_list.append({"name": name, "kind": c["kind"], "lam": c["lam"], "u_mean": c["mu"][:3].round(4).tolist(),
                         "u_std": c["mu"][3:6].round(4).tolist(), "p_cube": n_cube / 1000.0,
                         "n_kick": rec["kick_0.1"].get(name, 0), "persist": c["persist"],
                         "self_excited": name in rec["excited_from"],
                         "excited_from": rec["excited_from"].get(name, []), **per})
    rec["spurious_attractors"] = att_list
    rec["stable_fp_basins"] = {f"fp:{i}": rec["cube"].get(f"fp:{i}", 0) / 1000.0 for i in stable}
    # 5. выживание реконструированного аттрактора
    t_hit, which, rtraj, Rend = survival(m, att, [L.fps[i] for i in stable], rng)
    grid = np.arange(0, N_LIFE + 1, 500)
    rec["survival"] = {"t_steps": grid.tolist(), "S": [float(np.mean(t_hit > t)) for t in grid],
                       "n": int(len(t_hit)), "fraction_collapsed": float(np.mean(np.isfinite(t_hit))),
                       "to_fp": counts([f"fp:{stable[j]}" for j in which[which >= 0]]) if stable else {}}
    o_end = m.run(Rend, n_tr=0, n_obs=N_OBS, rng=rng)
    rec["survival"]["end_labels"] = counts(L.label(o_end))
    fin = np.where(np.isfinite(t_hit[:rtraj.shape[1]]))[0]
    if len(fin):
        j = fin[np.argmax(t_hit[fin])] if np.any(t_hit[fin] > 3000) else fin[0]
        arrays[f"s{seed}_collapse_u"] = rtraj[: int(min(t_hit[j] // 2 + 500, len(rtraj))), j]
        rec["collapse_example_t"] = float(t_hit[j])
    arrays[f"s{seed}_lorenz_u"] = traj_u(m, att[0]).astype(np.float32)
    rec["time_s"] = round(time.time() - t0, 1)
    return rec, arrays


def main(seeds=range(8), n_res=300):
    pj = ROOT / "results" / f"e33_esn_verify_N{n_res}.json"
    pz = ROOT / "results" / f"e33_esn_attractors_N{n_res}.npz"
    res = json.loads(pj.read_text(encoding="utf-8")) if pj.exists() else []
    arrays = dict(np.load(pz)) if pz.exists() else {}
    done = {r["seed"] for r in res}
    for seed in seeds:
        if seed in done:
            continue
        rec, arr = study(seed, n_res)
        res.append(rec)
        arrays.update(arr)
        sa = [(a["name"], a["kind"], round(a["p_cube"], 3), "самовозб." if a["self_excited"] else "СКРЫТЫЙ?")
              for a in rec["spurious_attractors"]]
        print(f"seed {seed}: куб {rec['cube']}; толчок 0.1 {rec['kick_0.1']}; FP {len(rec['fixed_points'])}, устойчивых "
              f"{len(rec['stable_fp'])}; паразитные {sa}; выживание: обрушилось {rec['survival']['fraction_collapsed']:.3f} "
              f"за {N_LIFE} шагов ({rec['time_s']} с)", flush=True)
        pj.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        np.savez_compressed(pz, **arrays)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else range(8)
    main(seeds=seeds, n_res=n)

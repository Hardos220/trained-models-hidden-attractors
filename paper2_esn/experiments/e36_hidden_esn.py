"""Этап 3: строгая проверка кандидата в скрытый паразитный аттрактор эхо-сети (N = 100, реализация 0).

Кандидат c1 из e33 (регулярный режим вне области данных, ~3 % куба, не достигнут из окрестностей
неустойчивых неподвижных точек). Проверки:
 1. полнота множества неподвижных точек: Ньютон из 3000 стартов (куб, окрестности орбиты c1,
    отрезки между c1 и устойчивой точкой, окрестность аттрактора);
 2. окрестности всех неустойчивых неподвижных точек: eps в {1e-6, 1e-4, 1e-2, 1e-1}, вдоль
    неустойчивых направлений и 200 точек в шаре на каждый радиус; горизонт 2·10⁴ + 4·10³;
 3. мера скрытости h_eps(c1) и для сравнения доли других аттракторов;
 4. время жизни c1: 20 копий, 2·10⁵ шагов;
 5. сечение бассейнов плоскостью через устойчивую точку, точку c1 и точку аттрактора (для рисунка).
Выход: results/e36_hidden_esn.json, results/e36_hidden_esn.npz
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e33_esn_verify import Labeler, build, counts, fp_index, fu, reconstructed  # noqa: E402

N_TR, N_OBS = 20000, 4000


def main(seed=0, n_res=100, cand="c1"):
    t0 = time.time()
    rng = np.random.default_rng(3600 + seed)
    rec = next(r for r in json.loads((ROOT / "results" / f"e33_esn_verify_N{n_res}.json").read_text(encoding="utf-8"))
               if r["seed"] == seed)
    z = np.load(ROOT / "results" / f"e33_esn_attractors_N{n_res}.npz")
    u_all, esn, m = build(seed, n_res)
    att, ref, ok = reconstructed(m, esn, rng)
    fps = [z[f"s{seed}_fp{i}_r"] for i in range(len(rec["fixed_points"]))]
    L = Labeler(m, ref, fps)
    # кластеры паразитных режимов: перекалибровка по представителям
    for a in rec["spurious_attractors"]:
        rep = z[f"s{seed}_{a['name']}_r"]
        o = m.run(rep[None, :] + 1e-7 * rng.normal(size=(32, m.N)), n_tr=2000, n_obs=N_OBS, rng=rng)
        F = fu(o)
        mu = F.mean(0)
        L.clusters.append({"kind": a["kind"], "mu": mu, "sd": F.std(0),
                           "floor": (0.05 if a["kind"] == "chaos" else 0.02) * (1 + np.abs(mu)), "name": a["name"]})
    names = {f"c{k}": c["name"] for k, c in enumerate(L.clusters)}
    rep = z[f"s{seed}_{cand}_r"]
    # орбита кандидата
    R = rep[None, :].copy()
    orbit = []
    for k in range(3000):
        R = m.step(R)
        if k % 5 == 0:
            orbit.append(R[0].copy())
    orbit = np.array(orbit)
    stable = [i for i, fp in enumerate(rec["fixed_points"]) if fp["rho"] < 1]
    out = {"seed": seed, "n_res": n_res, "candidate": cand, "n_fp_e33": len(fps)}
    # 1. полнота неподвижных точек
    starts = [rng.uniform(-1, 1, size=(1000, m.N)),
              orbit[rng.integers(0, len(orbit), 500)] + 0.05 * rng.normal(size=(500, m.N)),
              orbit[rng.integers(0, len(orbit), 500)] + 0.2 * rng.normal(size=(500, m.N)),
              att[rng.integers(0, len(att), 500)] + 0.05 * rng.normal(size=(500, m.N))]
    for i in stable:
        lam = rng.uniform(0, 1, size=(500, 1))
        starts.append(lam * fps[i][None, :] + (1 - lam) * orbit[rng.integers(0, len(orbit), 500)])
    S = np.clip(np.vstack(starts), -1, 1)
    new = m.fixed_points(S)
    added = 0
    for x in new:
        if fp_index(x[None, :], L.fps, tol=1e-6)[0] < 0:
            L.fps.append(x)
            added += 1
    out["n_starts"] = int(len(S))
    out["n_fp_total"] = len(L.fps)
    out["n_fp_new"] = added
    info = []
    for i, r in enumerate(L.fps):
        w, dirs = m.unstable_directions(r)
        info.append({"rho": float(np.max(np.abs(w))), "n_unstable": int(np.sum(np.abs(w) > 1 + 1e-9)),
                     "u": m.output(r[None, :])[0].round(4).tolist()})
    out["fixed_points"] = info
    print(f"неподвижных точек: было {len(fps)}, стало {len(L.fps)} (новых {added}) из {len(S)} стартов "
          f"({time.time() - t0:.0f} с)", flush=True)
    # 2–3. окрестности всех неустойчивых точек
    X0, tags = [], []
    for i, r in enumerate(L.fps):
        if info[i]["rho"] < 1:
            continue
        w, dirs = m.unstable_directions(r)
        for eps in (1e-6, 1e-4, 1e-2, 1e-1):
            pts = []
            for dvec in dirs[:20]:
                pts += [r + eps * dvec, r - eps * dvec]
            pts += list(r + eps * rng.normal(size=(200, m.N)) / np.sqrt(m.N))
            X0 += pts
            tags += [(i, eps)] * len(pts)
    o3 = m.run(np.clip(np.array(X0), -1, 1), n_tr=N_TR, n_obs=N_OBS, rng=rng)
    lab = np.array([names.get(str(x), str(x)) for x in L.label(o3)])
    tags = np.array(tags, dtype=object)
    nb = {}
    h = {}
    for (i, eps) in sorted(set(map(tuple, tags))):
        msk = np.array([tuple(t) == (i, eps) for t in tags])
        cnt = counts(lab[msk])
        nb[f"fp{i}_eps{eps:g}"] = cnt
        h.setdefault(f"{eps:g}", {"n": 0})
        h[f"{eps:g}"]["n"] += int(msk.sum())
        for k_, v in cnt.items():
            h[f"{eps:g}"][k_] = h[f"{eps:g}"].get(k_, 0) + v
    out["fp_neighbourhoods"] = nb
    out["h_eps"] = h
    hits = int(np.sum(lab == cand))
    out["candidate_hits_from_unstable_fp"] = hits
    out["n_neighbourhood_points"] = int(len(lab))
    print(f"окрестности неустойчивых точек: {len(lab)} траекторий, попаданий в {cand}: {hits} "
          f"({time.time() - t0:.0f} с)", flush=True)
    # окрестности устойчивых точек большими радиусами (для сравнения)
    st_nb = {}
    for i in stable:
        for eps in (0.1, 0.3, 1.0):
            Xs = np.clip(L.fps[i][None, :] + eps * rng.normal(size=(200, m.N)) / np.sqrt(m.N), -1, 1)
            o = m.run(Xs, n_tr=N_TR, n_obs=N_OBS, rng=rng)
            st_nb[f"fp{i}_eps{eps:g}"] = counts([names.get(str(x), str(x)) for x in L.label(o)])
    out["stable_fp_neighbourhoods"] = st_nb
    # 4. время жизни
    Xc = np.clip(orbit[rng.integers(0, len(orbit), 20)] + 1e-6 * rng.normal(size=(20, m.N)), -1, 1)
    o4 = m.run(Xc, n_tr=200000, n_obs=N_OBS, rng=rng)
    out["lifetime_200k"] = counts([names.get(str(x), str(x)) for x in L.label(o4)])
    out["candidate_lam"] = float(np.median(o4["lam"]))
    print(f"время жизни 2·10⁵ шагов: {out['lifetime_200k']}, λ = {out['candidate_lam']:.4f} ({time.time() - t0:.0f} с)", flush=True)
    # 5. сечение бассейнов: плоскость через устойчивую точку, точку c1 и точку аттрактора
    P0 = L.fps[stable[0]] if stable else att[0]
    P1 = orbit[0]
    P2 = att[len(att) // 2]
    e1 = P1 - P0
    e2 = P2 - P0
    e2 = e2 - e1 * (e1 @ e2) / (e1 @ e1)
    na, nb_ = 140, 140
    A, Bv = np.meshgrid(np.linspace(-0.6, 1.8, na), np.linspace(-0.6, 1.8, nb_))
    G = P0[None, :] + A.ravel()[:, None] * e1[None, :] + Bv.ravel()[:, None] * e2[None, :]
    G = np.clip(G, -1, 1)
    o5 = m.run(G, n_tr=N_TR, n_obs=N_OBS, rng=rng)
    lab5 = np.array([names.get(str(x), str(x)) for x in L.label(o5)])
    uniq = sorted(set(lab5))
    img = np.array([uniq.index(x) for x in lab5]).reshape(nb_, na)
    out["slice_labels"] = uniq
    out["slice_counts"] = counts(lab5)
    print(f"сечение: {out['slice_counts']} ({time.time() - t0:.0f} с)", flush=True)
    # траектории выхода для рисунка
    arrays = {"slice": img, "slice_a": A, "slice_b": Bv, "orbit_u": m.output(orbit),
              "att_u": m.output(att), "fp_u": np.array([m.output(r[None, :])[0] for r in L.fps]),
              "fp_rho": np.array([d["rho"] for d in info]), "scale": esn["scale"],
              "P_u": np.array([m.output(P0[None, :])[0], m.output(P1[None, :])[0], m.output(P2[None, :])[0]])}
    for k, c in enumerate(L.clusters):
        Rr = z[f"s{seed}_{c['name']}_r"][None, :].copy()
        U = []
        for j in range(4000):
            Rr = m.step(Rr)
            U.append(m.output(Rr)[0])
        arrays[f"{c['name']}_u"] = np.array(U)
    out["time_s"] = round(time.time() - t0, 1)
    (ROOT / "results" / "e36_hidden_esn.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(ROOT / "results" / "e36_hidden_esn.npz", **arrays)
    print("готово", out["time_s"], "с")


if __name__ == "__main__":
    main()

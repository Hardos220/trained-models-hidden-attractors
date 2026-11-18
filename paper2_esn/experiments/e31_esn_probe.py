"""Проверка модуля анализа резервуара: обучение ESN на Лоренце, эталон реконструированного
аттрактора, классификация 400 начальных состояний вблизи обучающих данных (шум 0.05) и в кубе,
неподвижные точки методом Ньютона и их устойчивость. Выход: results/e31_esn_probe.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from hsearch.reservoir_analysis import ClosedLoopESN, classify_runs, lorenz_reference  # noqa: E402
from hsearch.systems.reservoir import lorenz_data, train_esn  # noqa: E402


def main(n_res=300, seed=0):
    rng = np.random.default_rng(31)
    t0 = time.time()
    u = lorenz_data(20000)
    esn = train_esn(u, n_res=n_res, seed=seed)
    m = ClosedLoopESN(esn)
    print(f"обучение: RMSE {esn['train_rmse']:.2e} ({time.time() - t0:.1f} с)")
    t0 = time.time()
    ref, out_ref = lorenz_reference(m, rng)
    print(f"эталон: λ копий {np.round(np.median(out_ref['lam']), 3)} (Лоренц ≈ 0.906), fp у копий {out_ref['fp'].sum()} "
          f"({time.time() - t0:.1f} с)")
    res = {"n_res": n_res, "seed": seed, "train_rmse": esn["train_rmse"], "lam_ref_median": float(np.median(out_ref["lam"]))}
    for region, n in (("near_train_0.05", 400), ("cube", 400)):
        t0 = time.time()
        if region.startswith("near"):
            R = esn["R_train"]
            X0 = np.clip(R[rng.integers(0, len(R), size=n)] + 0.05 * rng.normal(size=(n, n_res)), -1, 1)
        else:
            X0 = rng.uniform(-1, 1, size=(n, n_res))
        out = m.run(X0, rng=rng)
        lab, z = classify_runs(out, ref)
        vals, cnt = np.unique(lab, return_counts=True)
        res[region] = dict(zip(vals.tolist(), cnt.tolist()))
        fps = out["final"][out["fp"]]
        res[region + "_n_distinct_fp"] = int(len({tuple(np.round(f, 4)) for f in fps}))
        print(f"{region}: {res[region]}; различных неподвижных точек среди сошедшихся: {res[region + '_n_distinct_fp']} "
              f"({time.time() - t0:.1f} с)")
    t0 = time.time()
    starts = np.vstack([esn["R_train"][rng.integers(0, len(esn["R_train"]), size=60)], rng.uniform(-1, 1, size=(60, n_res))])
    fps = m.fixed_points(starts)
    info = []
    for r in fps:
        w, dirs = m.unstable_directions(r)
        info.append({"norm": float(np.linalg.norm(r)), "rho": float(np.max(np.abs(w))), "n_unstable": int(np.sum(np.abs(w) > 1)),
                     "u": m.output(r[None, :])[0].round(3).tolist()})
    info.sort(key=lambda d: d["rho"])
    res["fixed_points"] = info
    print(f"неподвижных точек: {len(fps)} ({time.time() - t0:.1f} с)")
    for d in info:
        print(f"   |r|={d['norm']:.2f} спектр. радиус {d['rho']:.3f}, неуст. направлений {d['n_unstable']}, выход u={d['u']}")
    (ROOT / "results" / "e31_esn_probe.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

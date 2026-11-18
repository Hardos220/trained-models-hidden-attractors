"""Входы для GPU-расчётов пилота эхо-сетей, которым нужен numba (на сервере его нет):

- data/<режим>_l1.npz: l1 — (2000, 100, d) первые 100 отсчётов истинной траектории из ics_uniform[:2000]
  (u_0 = x_0, шаг выборки Δt, NaN после ухода |x| > esc_r); next_uniform — x(Δt) для всех ics_uniform;
  next_<имя> — x(Δt) для ics_basin_<имя>;
- data/lorenz_l1.npz: series — lorenz_data(45000) (h = 0.02); l1 — (2000, 100, 3) из ics_phys (RK4 dt = 0.002,
  отсчёт каждые 10 шагов, без переходного участка).
Запуск: python paper2_esn/experiments/e72_esn_gpu_inputs.py <каталог data>
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e70_esn_pilot_data as D  # noqa: E402
from hsearch.integrate import make_flow_kernels  # noqa: E402
from hsearch.systems.flows3d import lorenz_rhs  # noqa: E402
from hsearch.systems.reservoir import lorenz_data  # noqa: E402


def samples(s, X0, n, every):
    out = np.empty((len(X0), n, s.dim))
    for i, x0 in enumerate(X0):
        tr = s.trajectory(np.asarray(x0, float), s.params, s.dt, (n - 1) * every, s.esc_r)
        out[i] = tr[::every][:n]
    return out


def main():
    data = Path(sys.argv[1])
    for key in D.REGIMES:
        s = D.make_system(key)
        meta = json.load(open(data / f"{key}.json", encoding="utf-8"))
        every = meta["sample_every"]
        z = np.load(data / f"{key}.npz")
        arr = {"l1": samples(s, z["ics_uniform"][:2000], 100, every),
               "next_uniform": samples(s, z["ics_uniform"], 2, every)[:, 1]}
        for k in z.files:
            if k.startswith("ics_basin_"):
                arr["next_" + k[10:]] = samples(s, z[k], 2, every)[:, 1]
        np.savez_compressed(data / f"{key}_l1.npz", **arr)
        und = np.isnan(arr["l1"]).any(axis=(1, 2))
        print(f"{key}: L1(100) undefined for {und.mean():.3f} of 2000; L1(10) undefined for "
              f"{np.isnan(arr['l1'][:, :10]).any(axis=(1, 2)).mean():.3f}", flush=True)
    zl = np.load(data / "lorenz.npz")
    _, traj = make_flow_kernels(lorenz_rhs, 3)
    p = np.array([10.0, 28.0, 8.0 / 3.0])
    l1 = np.empty((2000, 100, 3))
    for i, x0 in enumerate(zl["ics_phys"]):
        tr = traj(np.asarray(x0, float), p, 0.002, 99 * 10, 1e6)
        l1[i] = tr[::10][:100]
    np.savez_compressed(data / "lorenz_l1.npz", series=lorenz_data(45000), l1=l1)
    print("lorenz done", flush=True)


if __name__ == "__main__":
    main()

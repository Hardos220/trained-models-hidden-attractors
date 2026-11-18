"""Data for the example panel of the schematic figure: a one-layer perceptron trained on Chua's circuit (regime I, set HE1,
width 32) whose reproduced hidden attractor is excited from a spurious saddle through a branch of its unstable manifold.

Trains the 20 seeds of the group as in stage M, then for each seed: equilibria and their stability, reproduction of the
hidden attractors, and both branches of the one-dimensional unstable manifolds of the unstable equilibria inside the
region (iterated from e +- 1e-6 v_u). Seeds where a branch of a spurious saddle lands on a reproduced attractor are saved.

    python gpu/fig_example.py [--regime chua_I --set HE1 --cls mlp --width 32 --steps 20000]
Result: $GPUQ_OUT/FIG/example_<regime>_<cls>_<set>_H<width>.npz and .json
"""
import argparse
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gpu"))
import pipeline as PL  # noqa: E402
import stage_c as SC  # noqa: E402
import stage_m as SM  # noqa: E402

if os.environ.get("RMQ_CPU") and SM.DEV.type == "cpu":
    torch.set_num_threads(int(os.environ["RMQ_CPU"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="chua_I")
    ap.add_argument("--set", default="HE1")
    ap.add_argument("--cls", default="mlp")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20000)
    a = ap.parse_args()
    out = Path(os.environ.get("GPUQ_OUT", "out")) / "FIG"
    out.mkdir(parents=True, exist_ok=True)
    z, meta, _ = SC.load_regime(a.regime)
    series, _ = SC.train_sets(z)[a.set]
    allx = np.vstack([np.asarray(s, float) for s in series])
    mu, sig = allx.mean(0), allx.std(0)
    Win = SM.windows([(np.asarray(s, float) - mu) / sig for s in series])
    ws, mse = SM.train_batch(a.cls, a.width, a.seeds, Win,
                             seed=1000 * a.width + zlib.crc32(f"{a.regime}|{a.set}|{a.cls}".encode()) % 1000, steps=a.steps)
    lo_p, hi_p = np.array(meta["box_lo"]), np.array(meta["box_hi"])
    true_eq = [e["point"] for e in meta["equilibria"]]
    L_scale = float(np.max(hi_p - lo_p)) / 2
    rng = np.random.default_rng(3)
    arrays, summary = {}, []
    for s, w in enumerate(ws):
        m = SM.Map(a.cls, w)
        bnd = m.out_bound()
        starts = [-bnd + 2 * bnd * rng.random((12000, 3))] + [((np.asarray(e) - mu) / sig)[None] + 0.01 * rng.normal(size=(50, 3)) for e in true_eq]
        E = SM.equilibria(m, np.vstack(starts))
        info = SM.eq_info(m, E, sig, mu, true_eq, L_scale, False)
        # model attractors from states on each true hidden attractor
        atts = {}
        for y in meta["hidden"]:
            Y = (z[f"att_{y}"] - mu) / sig
            o = m.run(Y[rng.integers(0, len(Y), 32)], 20000, 1000)
            keep = ~o["fp"] & ~o["esc"]
            if keep.sum() >= 4:
                S = m.orbit(o["final"][np.where(keep)[0][0]], 8000)
                corr = PL.correspondence(S * sig + mu, {y: z[f"att_{y}"][:10000]})[y]
                if corr.get("delta_att", 9) < 0.2 and corr.get("d_H", 9) < 0.2:
                    atts[y] = S
        branches = []
        for i, (x, f) in enumerate(zip(E, info)):
            xp = np.array(f["x"])
            if f["n_unstable"] != 1 or not np.all((xp >= lo_p) & (xp <= hi_p)):
                continue
            Jm = m.J(torch.as_tensor(x[None], dtype=SM.DT, device=SM.DEV))[0].cpu().numpy()
            wv, V = np.linalg.eig(Jm)
            k = int(np.argmax(np.abs(wv)))
            v = np.real(V[:, k])
            v /= np.linalg.norm(v)
            for sgn in (1, -1):
                U = torch.as_tensor((x + sgn * 1e-6 * v)[None], dtype=SM.DT, device=SM.DEV)
                tr = []
                for _ in range(3000):
                    U = m.F(U)
                    tr.append(U[0].cpu().numpy().copy())
                tr = np.array(tr)
                end = None
                for y, S in atts.items():
                    if np.min(np.linalg.norm(S[::4] - tr[-1], axis=1)) < 0.05:
                        end = y
                branches.append({"eq": i, "sign": sgn, "spurious": f.get("dist_true_eq", 9) >= 0.01, "ends_on": end,
                                 "lam": float(np.real(wv[k])), "traj": tr})
        hit = [b for b in branches if b["spurious"] and b["ends_on"]]
        summary.append({"seed": s, "mse": float(mse[s]), "K": len(E), "U": int(sum(f["n_unstable"] > 0 for f in info)),
                        "reproduced": sorted(atts), "branches": [{k: v for k, v in b.items() if k != "traj"} for b in branches],
                        "spurious_branch_to_attractor": len(hit)})
        print(summary[-1], flush=True)
        if hit:
            arrays[f"s{s}_E"] = E * sig + mu
            arrays[f"s{s}_E_unstable"] = np.array([f["n_unstable"] for f in info])
            arrays[f"s{s}_E_spurious"] = np.array([f.get("dist_true_eq", 9) >= 0.01 for f in info])
            for y, S in atts.items():
                arrays[f"s{s}_att_{y}"] = S * sig + mu
            for j, b in enumerate(branches):
                arrays[f"s{s}_branch{j}_eq{b['eq']}_{'p' if b['sign'] > 0 else 'm'}_{b['ends_on'] or 'none'}"] = b["traj"] * sig + mu
    for y in meta["hidden"]:
        arrays[f"true_att_{y}"] = z[f"att_{y}"][:20000]
    arrays["true_eq"] = np.array(true_eq)
    name = f"example_{a.regime}_{a.cls}_{a.set}_H{a.width}"
    np.savez_compressed(out / f"{name}.npz", **arrays)
    (out / f"{name}.json").write_text(json.dumps(summary, default=float, indent=1), encoding="utf-8")
    print("seeds with a spurious branch onto a reproduced attractor:", [x["seed"] for x in summary if x["spurious_branch_to_attractor"]])


if __name__ == "__main__":
    main()

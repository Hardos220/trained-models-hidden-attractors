"""Stage MW: retrain stage M models with the stage M code and seeds, SAVE the weights, and repeat the stage M equilibrium
search on them (input of the certified enumeration krawczyk_m.py; check of completeness).

Training is stage_m.train_batch with the same arguments as stage M (20 seeds in one batch, 20000 steps, group seed
1000 H + crc32("<regime>|<set>|<class>") mod 1000), so on the same GPU and software the models coincide with those of
stage M (compare train_mse with the stage M records). Equilibria: stage_m.equilibria from the stage M starts
(rng = default_rng(11): 12000 uniform points of the image box of F for mlp and mlp2, of the box c +- (hi - lo) around
the region Q for res; 50 points near each true equilibrium), then stage_m.eq_info.

    python gpu/stage_mw.py --regimes se3 --sets H1,HE1 --specs mlp:64,mlp:256,mlp2:64,mlp2:256,res:64
Result: $GPUQ_OUT/MW/<regime>/<class>_<set>_H<width>_s<seed>.json (stage M equilibrium fields, u coordinates) and .npz
(weights W1, b1, W2, b2[, W3, b3], mu, sig, Newton equilibria E in u coordinates, start box).
"""
import argparse
import json
import os
import sys
import time
import traceback
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gpu"))
import esngpu as G  # noqa: E402
import stage_c as SC  # noqa: E402
import stage_m as SM  # noqa: E402

DEV, DT = SM.DEV, SM.DT


def newton_search(key, meta, cls, m, mu, sig, quick=False):
    """The equilibrium part of stage_m.analyze_model (same random numbers)."""
    rng = np.random.default_rng(11)
    true_eq = [e["point"] for e in meta["equilibria"]]
    L_scale = float(np.max(np.array(meta["box_hi"]) - np.array(meta["box_lo"]))) / 2
    eq_line = key == "le1"
    lo = (np.array(meta["box_lo"]) - mu) / sig
    hi = (np.array(meta["box_hi"]) - mu) / sig
    if cls == "res":
        c, r = (lo + hi) / 2, (hi - lo)
        box_lo, box_hi = c - r, c + r
    else:
        bnd = m.out_bound()
        box_lo, box_hi = -bnd, bnd
    nS = 3000 if quick else 12000
    starts = [box_lo + (box_hi - box_lo) * rng.random((nS, 3))]
    eqs = [[0.0, 0.0, zz] for zz in np.linspace(-10, -0.5, 20)] if eq_line else true_eq
    for e in eqs:
        starts.append((np.asarray(e)[None] - mu) / sig + 0.01 * rng.normal(size=(50, 3)))
    E = SM.equilibria(m, np.vstack(starts))
    info = SM.eq_info(m, E, sig, mu, true_eq, L_scale, eq_line)
    for f, u in zip(info, E):
        f["u"] = u.tolist()
    return E, info, box_lo, box_hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default="se3")
    ap.add_argument("--sets", default="H1,HE1")
    ap.add_argument("--specs", default="mlp:64,mlp:256,mlp2:64,mlp2:256,res:64")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out_root = Path(os.environ.get("GPUQ_OUT", "out")) / "MW"
    specs = [(s.split(":")[0], int(s.split(":")[1])) for s in a.specs.split(",")]
    groups = [(k, s, c, H) for k in a.regimes.split(",") for s in a.sets.split(",") for c, H in specs]
    print(f"stage MW: {len(groups)} groups x {a.seeds} seeds, device {DEV}", flush=True)
    cache = {}
    stop = os.environ.get("RMQ_STOP_FILE")
    for gi, (key, set_name, cls, H) in enumerate(groups):
        if stop and os.path.exists(stop):
            print("stop file: exiting before the next group (finished groups are kept)", flush=True)
            sys.exit(5)
        rp = os.environ.get("RMQ_PROGRESS_FILE")
        if rp:
            try:
                Path(rp).write_text(f"{gi}/{len(groups)} {key} {cls} {set_name} H{H}", encoding="utf-8")
            except OSError:
                pass
        d = out_root / key
        d.mkdir(parents=True, exist_ok=True)
        names = [f"{cls}_{set_name}_H{H}_s{s}" for s in range(a.seeds)]
        if all((d / f"{nm}.json").exists() and (d / f"{nm}.npz").exists() for nm in names):
            continue
        if key not in cache:
            cache = {key: SC.load_regime(key)}
        z, meta, _ = cache[key]
        G.report_progress(gi, len(groups), f"{key} {cls} {set_name} H{H}")
        try:
            series, _ = SC.train_sets(z)[set_name]
            allx = np.vstack([np.asarray(s, float) for s in series])
            mu, sig = allx.mean(0), allx.std(0)
            Win = SM.windows([(np.asarray(s, float) - mu) / sig for s in series])
            t0 = time.time()
            ws, mse = SM.train_batch(cls, H, a.seeds, Win,
                                     seed=1000 * H + zlib.crc32(f"{key}|{set_name}|{cls}".encode()) % 1000,
                                     steps=1000 if a.quick else a.steps)
            t_train = time.time() - t0
            for s, (w, e) in enumerate(zip(ws, mse)):
                nm = names[s]
                t1 = time.time()
                m = SM.Map(cls, w)
                E, info, blo, bhi = newton_search(key, meta, cls, m, mu, sig, quick=a.quick)
                arr = {k: v.detach().cpu().numpy() for k, v in w.items()}
                np.savez_compressed(d / f"{nm}.npz", mu=mu, sig=sig, E=E, start_lo=blo, start_hi=bhi, **arr)
                rec = {"regime": key, "set": set_name, "class": cls, "H": H, "seed": s, "train_mse": float(e),
                       "device": DEV.type, "batch_seeds": a.seeds, "steps": 1000 if a.quick else a.steps,
                       "K": len(E), "U": int(sum(f["n_unstable"] > 0 for f in info)),
                       "index_sum": int(sum(f["index"] for f in info)), "fixed_points": info,
                       "train_time_group_s": round(t_train, 1), "time_newton_s": round(time.time() - t1, 1)}
                (d / f"{nm}.json").write_text(json.dumps(rec, default=float), encoding="utf-8")
            print(f"[{gi + 1}/{len(groups)}] {key} {cls} {set_name} H{H}: mse {np.median(mse):.2e}, train {t_train:.0f} s",
                  flush=True)
        except Exception as exc:
            traceback.print_exc()
            if G.fatal_cuda(exc):
                print("fatal CUDA error, stopping the process", flush=True)
                sys.exit(3)
    G.report_progress(len(groups), len(groups), "done")


if __name__ == "__main__":
    main()

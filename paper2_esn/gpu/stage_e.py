"""Stage E: controls for training without the hidden attractor (set E1) with model classes that contain the true dynamics.

Each realization uses a random subset of 15 of the 20 E1 transients. Two model classes:
  ngrc - next-generation reservoir computer with two delays and all quadratic monomials (as in stage C), ridge eta;
         analyzed with the stage C protocol (stage_c.analyze_ngrc);
  poly - sparse quadratic map u_{n+1} = u_n + Theta(u_n) Xi fitted by sequentially thresholded least squares (a discrete
         SINDy); for SE3, SE11, LE1 and Wang-Chen the Euler step of the true vector field is in this class. Analyzed with
         the stage M protocol (equilibria, reproduction, excitation labels, census).

    python gpu/stage_e.py --regimes se3 --kinds poly --reps 10 [--thresholds 0.003,0.01,0.03] [--etas 1e-3,1e-1,10]
Result: $GPUQ_OUT/E/<regime>/poly_thr<t>_r<r>.json, ngrc_eta<eta>_r<r>.json
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
import esngpu as G  # noqa: E402
import stage_c as SC  # noqa: E402
import stage_m as SM  # noqa: E402

DEV, DT = SM.DEV, SM.DT
if os.environ.get("RMQ_CPU") and DEV.type == "cpu":
    torch.set_num_threads(int(os.environ["RMQ_CPU"]))
QUAD = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


class SubsetZ:
    """The regime archive with train_E1 replaced by a subset of its transients and, if window is given, train_H1 by the
    contiguous window [start, start + length) of the trajectory on the hidden attractor."""

    def __init__(self, z, keep, window=None):
        self.z = z
        E = np.split(z["train_E1"], np.cumsum(z["train_E1_len"])[:-1])
        sub = [E[i] for i in keep]
        self.over = {"train_E1": np.vstack(sub), "train_E1_len": np.array([len(s) for s in sub])}
        if window is not None:
            a, n = window
            self.over["train_H1"] = np.asarray(z["train_H1"])[a:a + n]
        self.files = list(z.files)

    def __getitem__(self, k):
        return self.over[k] if k in self.over else self.z[k]


def theta(U):
    """Quadratic library [1, u, u_a u_b]; U: (n, 3) array -> (n, 10)."""
    return np.hstack([np.ones((len(U), 1)), U] + [U[:, [a]] * U[:, [b]] for a, b in QUAD])


def stlsq(Th, Y, thr, ridge=1e-8, iters=10):
    n = Th.shape[1]
    Xi = np.linalg.solve(Th.T @ Th + ridge * np.eye(n), Th.T @ Y)
    for _ in range(iters):
        small = np.abs(Xi) < thr
        Xi[small] = 0.0
        for j in range(Y.shape[1]):
            big = ~small[:, j]
            if big.any():
                A = Th[:, big]
                Xi[big, j] = np.linalg.solve(A.T @ A + ridge * np.eye(big.sum()), A.T @ Y[:, j])
    return Xi


class PolyMap(SM.Map):
    """u -> u + Theta(u) Xi in normalized coordinates, with the run/orbit machinery of stage M."""

    def __init__(self, Xi):
        super().__init__("res", None)
        self.Xi = torch.as_tensor(Xi, dtype=DT, device=DEV)

    def F(self, U):
        cols = [torch.ones_like(U[:, :1]), U] + [U[:, [a]] * U[:, [b]] for a, b in QUAD]
        return U + torch.cat(cols, dim=1) @ self.Xi

    def J(self, U):
        n = len(U)
        Jm = torch.eye(3, dtype=DT, device=DEV)[None].repeat(n, 1, 1) + self.Xi[1:4].T[None]
        for k, (a, b) in enumerate(QUAD):
            c = self.Xi[4 + k]                                           # (3,) coefficients for the 3 outputs
            Jm[:, :, a] += c[None, :] * U[:, [b]]
            Jm[:, :, b] += c[None, :] * U[:, [a]]
        return Jm


def run_poly(key, z, meta, sz, set_name, keep, thr, quick):
    series, _ = SC.train_sets(sz)[set_name]
    allx = np.vstack(series)
    mu, sig = allx.mean(0), allx.std(0)
    X, Y = [], []
    for s in series:
        u = (np.asarray(s, float) - mu) / sig
        X.append(u[:-1])
        Y.append(u[1:] - u[:-1])
    X, Y = np.vstack(X), np.vstack(Y)
    Xi = stlsq(theta(X), Y, thr)
    rmse = float(np.sqrt(np.mean((theta(X) @ Xi - Y) ** 2)))
    rec = SM.analyze_model(key, z, meta, "res", None, mu, sig, quick=quick, model=PolyMap(Xi))
    rec.update({"kind": "poly", "thr": thr, "n_terms": int(np.count_nonzero(Xi)), "train_rmse": rmse, "Xi": Xi.tolist(),
                "keep": [int(i) for i in keep]})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default="se3")
    ap.add_argument("--kinds", default="poly,ngrc")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--thresholds", default="0.003,0.01,0.03")
    ap.add_argument("--etas", default="1e-3,1e-1,10")
    ap.add_argument("--sets", default="E1", help="E1, H1, HE1: H1 uses a random window of 75%% of the trajectory on the "
                                                  "hidden attractor, HE1 that window plus 15 of the 20 transients")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out_root = Path(os.environ.get("GPUQ_OUT", "out")) / "E"
    jobs = []
    for key in a.regimes.split(","):
        for s in a.sets.split(","):
            for r in range(a.reps):
                if "poly" in a.kinds:
                    jobs += [(key, s, "poly", float(t), r) for t in a.thresholds.split(",")]
                if "ngrc" in a.kinds:
                    jobs += [(key, s, "ngrc", float(e), r) for e in a.etas.split(",")]
    print(f"stage E: {len(jobs)} models, device {DEV}", flush=True)
    cache = {}
    for n, (key, set_name, kind, par, r) in enumerate(jobs):
        d = out_root / key
        d.mkdir(parents=True, exist_ok=True)
        nm = f"poly_thr{par:g}_r{r}" if kind == "poly" else f"ngrc_eta{par:g}_r{r}"
        if set_name != "E1":                       # E1 names are kept as before for the finished runs
            nm = f"{set_name}_{nm}"
        if (d / f"{nm}.json").exists():
            continue
        if key not in cache:
            cache = {key: SC.load_regime(key)}
        z, meta, l1 = cache[key]
        keep = np.sort(np.random.default_rng(1000 + r).choice(len(z["train_E1_len"]), 15, replace=False))
        window = None
        if set_name in ("H1", "HE1"):
            nH = len(z["train_H1"])
            L = int(0.75 * nH)
            window = (int(np.random.default_rng(2000 + r).integers(0, nH - L + 1)), L)
        sz = SubsetZ(z, keep, window)
        G.report_progress(n, len(jobs), f"{key} {nm}")
        t0 = time.time()
        try:
            if kind == "poly":
                rec = run_poly(key, z, meta, sz, set_name, keep, par, a.quick)
            else:
                rec, arr = SC.analyze_ngrc(key, sz, meta, l1, set_name, par, quick=a.quick)
                rec.update({"kind": "ngrc", "keep": [int(i) for i in keep]})
                np.savez_compressed(d / f"{nm}.npz", **arr)
            rec.update({"regime": key, "set": set_name, "rep": r, "window": window, "time_total_s": round(time.time() - t0, 1)})
            (d / f"{nm}.json").write_text(json.dumps(rec, default=SC.default), encoding="utf-8")
            hm = rec.get("hidden_match")
            info = ({y: (m.get("reproduced"), m.get("label")) for y, m in hm.items()} if hm
                    else {k: v.get("label") for k, v in rec.get("hiddenness", {}).items()})
            print(f"[{n + 1}/{len(jobs)}] {key} {nm}: {info}; {time.time() - t0:.0f} s", flush=True)
        except Exception as exc:
            if G.fatal_cuda(exc):
                raise
            (d / f"{nm}.failed.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[{n + 1}/{len(jobs)}] {key} {nm}: failed: {exc}", flush=True)
    G.report_progress(len(jobs), len(jobs), "done")


if __name__ == "__main__":
    main()

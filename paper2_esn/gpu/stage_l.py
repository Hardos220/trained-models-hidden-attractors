"""Этап L: контроль на системе Ли–Спротта.

1. Непрерывные сети Röhm et al. (RohmESN), пачкой по M сетей (bmm), по 100 затравок на случай.
2. Дискретные ESN (16 на случай) с перепиcью, неподвижными точками и протоколом скрытости.
Запуск: python gpu/stage_l.py [--part rohm|esn|all] [--seeds 100] [--quick]
Результат: $GPUQ_OUT/L/rohm_<случай>.json, $GPUQ_OUT/L/esn_<случай>_<модель>.json/.npz
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

CASES = {"li_sprott_2_0.8": dict(G=0.3, b=1.0, eta=1e-3, re_max=0.95, names=("LC_a", "LC_b", "T")),
         "li_sprott_6_0.1": dict(G=0.01, b=3.0, eta=1e-5, re_max=0.99, names=("SA_a", "SA_b", "T"))}
LAM_C = 0.02
SKIP = set()


class RohmBatch:
    """M непрерывных сетей одновременно; матрицы — из esn_pilot.RohmESN (тот же порядок случайных чисел)."""

    def __init__(self, seeds, m, N=300, rho_s=0.1, G_=0.3, b=1.0, re_max=0.95, theta=2.5, dt=0.1):
        nets = [P.RohmESN(m=m, N=N, rho_s=rho_s, G=G_, b=b, re_max=re_max, theta=theta, dt=dt, seed=s) for s in seeds]
        self.W = G.T(np.stack([n.W for n in nets]))                 # (M, N, N)
        self.Win = G.T(np.stack([n.W_in for n in nets])) * G_       # G·W_in
        self.B = G.T(np.stack([n.B for n in nets]))[:, :, None]     # (M, N, 1)
        self.M, self.N, self.m = len(seeds), N, m
        self.n_sub, self.dt = int(round(theta / dt)), dt
        self.Wout = None

    def _rhs(self, X, Gu):
        return -X + torch.tanh(torch.bmm(self.W, X) + Gu + self.B)

    def advance(self, X, u):
        """X (M, N, k), u (M, m, k) — постоянный вход на интервале θ."""
        Gu = torch.bmm(self.Win, u)
        h = self.dt
        for _ in range(self.n_sub):
            k1 = self._rhs(X, Gu)
            k2 = self._rhs(X + 0.5 * h * k1, Gu)
            k3 = self._rhs(X + 0.5 * h * k2, Gu)
            k4 = self._rhs(X + h * k3, Gu)
            X = X + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return X

    def initial_state(self, k=1, t_free=300.0):
        X = torch.zeros(self.M, self.N, k, dtype=G.DT, device=G.DEV)
        u = torch.zeros(self.M, self.m, k, dtype=G.DT, device=G.DEV)
        n_int = int(round(t_free / (self.dt * self.n_sub)))
        for _ in range(n_int):
            X = self.advance(X, u)
        return X

    def train(self, U, eta, washout=1000):
        Ut = G.T(U)
        X = self.initial_state(1)
        StS = torch.zeros(self.M, self.N + 1, self.N + 1, dtype=G.DT, device=G.DEV)
        StY = torch.zeros(self.M, self.N + 1, self.m, dtype=G.DT, device=G.DEV)
        one = torch.ones(self.M, 1, 1, dtype=G.DT, device=G.DEV)
        sq = torch.zeros(self.M, dtype=G.DT, device=G.DEV)
        for k in range(len(U) - 1):
            X = self.advance(X, Ut[k][None, :, None].expand(self.M, -1, -1))
            if k >= washout:
                s = torch.cat([X, one], dim=1)                      # (M, N+1, 1)
                StS += torch.bmm(s, s.transpose(1, 2))
                StY += s * Ut[k + 1][None, None, :]
        I = torch.eye(self.N + 1, dtype=G.DT, device=G.DEV)[None]
        self.Wout = torch.linalg.solve(StS + eta * I, StY)          # (M, N+1, m)
        return self

    def autonomous(self, warm, n_steps):
        """warm (K, w, m): K ряда прогрева, общих для всех сетей; выход (M, K, n_steps, m)."""
        K, w, _ = warm.shape
        Wt = G.T(warm)
        X = self.initial_state(1).expand(-1, -1, K).contiguous()
        for k in range(w):
            X = self.advance(X, Wt[:, k].T[None].expand(self.M, -1, -1))
        out = torch.empty(self.M, K, n_steps, self.m, dtype=G.DT, device=G.DEV)
        one = torch.ones(self.M, 1, K, dtype=G.DT, device=G.DEV)
        for k in range(n_steps):
            y = torch.bmm(self.Wout.transpose(1, 2), torch.cat([X, one], dim=1))   # (M, m, K)
            out[:, :, k] = y.transpose(1, 2)
            y = torch.where(torch.isfinite(y) & (y.abs() < 1e3), y, torch.zeros_like(y))
            X = self.advance(X, y)
        return G.N_(out)


def run_rohm(out_dir, n_seeds, quick):
    for case, par in CASES.items():
        if (out_dir / f"rohm_{case}.json").exists():
            continue
        z = np.load(ROOT / "data" / f"{case}.npz")
        U = z["train_noisy"]
        names = par["names"]
        warm = np.stack([z[f"transient_{nm}"][:1000] for nm in names])
        truth = {nm: z[f"transient_{nm}"][1000:] for nm in names}
        n_steps = 2000 if quick else 10000
        res = []
        Mb = 25
        for s0 in range(0, n_seeds, Mb):
            t0 = time.time()
            seeds = list(range(s0, min(n_seeds, s0 + Mb)))
            rb = RohmBatch(seeds, m=4, G_=par["G"], b=par["b"], re_max=par["re_max"]).train(U if not quick else U[:3000], par["eta"])
            Y = rb.autonomous(warm, n_steps)
            for i, sd in enumerate(seeds):
                rec = {"seed": sd, "lifts": {}}
                for j, nm in enumerate(names):
                    Xm = Y[i, j]
                    div = bool((~np.isfinite(Xm)).any() or np.abs(Xm).max() > 100)
                    met = {"diverged": div}
                    if not div:
                        for tn, Tt in truth.items():
                            met[tn] = {"delta_att": PL.delta_att(Xm, Tt), "d_H": PL.hausdorff(Xm[::4], Tt[::4]) / PL.diam(Tt),
                                       "SW1": PL.sliced_w1(Xm, Tt) / float(np.mean(Tt.std(0)))}
                    rec["lifts"][nm] = met
                res.append(rec)
            print(f"rohm {case}: seeds {seeds[0]}–{seeds[-1]} in {time.time() - t0:.0f} s", flush=True)
            G.report_progress(s0 + len(seeds), n_seeds, f"rohm {case}")
        (out_dir / f"rohm_{case}.json").write_text(json.dumps(res, default=float), encoding="utf-8")
        ok = [r for r in res if all(not r["lifts"][nm]["diverged"] for nm in names)]
        own = {nm: [r["lifts"][nm][nm]["delta_att"] for r in ok] for nm in names}
        print(f"rohm {case}: {len(ok)}/{len(res)} without divergence; median Δatt to own attractor "
              f"{ {k: round(float(np.median(v)), 3) if v else None for k, v in own.items()} }; "
              f"share Δ<0.1: { {k: round(float(np.mean(np.array(v) < 0.1)), 2) if v else None for k, v in own.items()} }", flush=True)


def analyze_esn_ls(case, z, rho, eta, seed, quick):
    t_start = time.time()
    names = CASES[case]["names"]
    meta = json.load(open(ROOT / "data" / f"{case}.json", encoding="utf-8"))
    h = meta["Delta_t"]
    U = z["train_noisy"]
    esn = P.train_esn_multi([U], n_res=300, spectral_radius=rho, ridge=eta, input_scale=0.5, density=0.05, washout=500, seed=seed)
    net = G.Net(esn, h=h)
    n_tr, n_obs = (4000, 1000) if quick else (20000, 4000)
    rng = np.random.default_rng(300 + seed)
    truth = {nm: z[f"transient_{nm}"][1000:] for nm in names}
    rec = {"case": case, "rho": rho, "eta": eta, "seed": seed, "train_rmse": esn["train_rmse"], "lifts": {}}
    warm = np.stack([z[f"transient_{nm}"][:1000] for nm in names])
    R = net.lift_drive(esn, warm, 1000)
    for j, nm in enumerate(names):
        Xm = net.samples(R[j], 10000)
        met = {"diverged": bool(not np.all(np.isfinite(Xm)))}
        if not met["diverged"]:
            met.update(PL.correspondence(Xm, truth))
        rec["lifts"][nm] = met
    L = PL.Labeler(net, LAM_C)
    nQ = 2000 if quick else 10000
    X, ok = net.newton(np.vstack([rng.uniform(-1, 1, size=(nQ // 2, 300)),
                                  np.clip(esn["R_train"][rng.integers(0, len(esn["R_train"]), nQ // 2)] + 0.05 * rng.normal(size=(nQ // 2, 300)), -1, 1)]))
    if ok.any():
        L.add_fps(X[ok])
    atr = PL.training_attractor(net, L, esn["R_train"][-1])
    rec["A_tr"] = {"kind": atr["kind"], "label": atr["label"]}
    orb = atr["orbit"]
    groups = {"Q": rng.uniform(-1, 1, size=(nQ, 300)),
              "kick": np.clip(orb[rng.integers(0, len(orb), 1000)] + 0.1 * rng.normal(size=(1000, 300)), -1, 1),
              "lift": R}
    gn = list(groups)
    allS = np.vstack([groups[g] for g in gn])
    out = net.run(allS, n_tr=n_tr, n_obs=n_obs, seed=seed)
    cands = PL.discover(net, L, out, L.label(out), n_life=10000 if quick else 50000, n_obs=n_obs)
    lab = np.asarray(L.label(out), dtype=str)
    offs = np.cumsum([0] + [len(groups[g]) for g in gn])
    rec["census"] = {g: PL.counts(lab[offs[i]:offs[i + 1]]) for i, g in enumerate(gn)}
    rec["lift_labels"] = dict(zip(names, lab[offs[2]:offs[3]].tolist()))
    rec["discovery"] = cands
    unst = [i for i in range(len(L.fps)) if np.max(np.abs(net.stability(L.fps[i])[0])) > 1 + 1e-9]
    nb = PL.radius_test(net, L, unst, n_ball=50 if quick else 200, n_tr=n_tr, n_obs=n_obs)
    hits = PL.hits_by_attractor(nb)
    hid = {}
    for k in range(len(L.refs)):
        a = f"a{k}"
        hh = hits.get(a, {"r_hat": np.inf, "from": {}})
        cand = not (hh["r_hat"] <= 1e-3)
        rec_h = {"r_hat": hh["r_hat"] if np.isfinite(hh["r_hat"]) else None, "from": hh["from"], "candidate": cand}
        cons = None
        if cand and L.fps:
            bs = PL.boundary_search(net, L, a, allS, lab, K=200 if quick else 1000, n_starts=2000 if quick else 20000,
                                    n_tr=n_tr, n_obs=n_obs)
            new_unst = [j for j in bs["new_fp_ids"] if np.max(np.abs(net.stability(L.fps[j])[0])) > 1 + 1e-9]
            if new_unst:
                h2 = PL.hits_by_attractor(PL.radius_test(net, L, new_unst, n_ball=50 if quick else 200, n_tr=n_tr, n_obs=n_obs)).get(a)
                if h2:
                    rec_h["r_hat"] = min(rec_h["r_hat"] or np.inf, h2["r_hat"])
            mm = PL.missing_mass_bounds(list(bs["multiplicities"].values()), bs["n_starts"])
            cons = mm["conservative"]
            rec_h["missing_mass"] = mm
        r_hat = rec_h["r_hat"] if rec_h["r_hat"] is not None else np.inf
        rec_h["label"] = PL.hidden_label(r_hat, cand and not (r_hat <= 1e-3), cons) if L.fps else "no_fixed_points"
        hid[a] = rec_h
    rec["hiddenness"] = hid
    att = []
    for k, r in enumerate(L.refs):
        Xm = net.samples(r["rep"], 10000, burn=10000)
        att.append({"name": f"a{k}", "kind": r["kind"], "lam": r["lam"], "metrics": PL.correspondence(Xm, truth)})
    rec["attractors"] = att
    rec["fixed_points"] = [PL.fp_info(net, x) for x in L.fps]
    rec["time_total_s"] = round(time.time() - t_start, 1)
    arrays = {"fps": np.asarray(L.fps) if L.fps else np.zeros((0, 300)),
              "reps": np.asarray([r["rep"] for r in L.refs]) if L.refs else np.zeros((0, 300))}
    return rec, arrays


def run_esn(out_dir, quick):
    for case in CASES:
        z = np.load(ROOT / "data" / f"{case}.npz")
        for rho in (0.9, 1.2):
            for eta in (1e-7, 1e-4):
                for seed in range(4):
                    name = f"esn_{case}_rho{rho}_eta{eta:g}_s{seed}"
                    if (out_dir / f"{name}.json").exists() or name in SKIP:
                        continue
                    try:
                        rec, arr = analyze_esn_ls(case, z, rho, eta, seed, quick)
                        np.savez_compressed(out_dir / f"{name}.npz", **arr)
                        (out_dir / f"{name}.json").write_text(json.dumps(rec, default=float), encoding="utf-8")
                        own = {nm: (round(v[nm]["delta_att"], 3) if not v.get("diverged") else "div") for nm, v in rec["lifts"].items()}
                        print(f"{name}: lifts Δatt {own} → {rec['lift_labels']}; FP {len(rec['fixed_points'])}, "
                              f"att {len(rec['attractors'])} { {k: v['label'] for k, v in rec['hiddenness'].items()} }; {rec['time_total_s']} s",
                              flush=True)
                    except Exception:
                        print(f"{name}: FAILED", flush=True)
                        traceback.print_exc()
                        (out_dir / f"{name}.failed.txt").write_text(traceback.format_exc(), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip", default="")
    a = ap.parse_args()
    if a.skip and Path(a.skip).exists():
        SKIP.update(Path(a.skip).read_text(encoding="utf-8").split())
    out_dir = Path(os.environ.get("GPUQ_OUT", "out")) / "L"
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.part in ("rohm", "all"):
        run_rohm(out_dir, a.seeds, a.quick)
    if a.part in ("esn", "all"):
        run_esn(out_dir, a.quick)


if __name__ == "__main__":
    main()

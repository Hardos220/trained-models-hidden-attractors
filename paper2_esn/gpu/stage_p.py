"""Stage P: intervention on U at fixed data. Perceptrons (mlp, mlp2) are trained as in stage M, with and without a
penalty on the unstable equilibria inside the region Q of the true system; the analysis is that of stage M.

During the rollout phase of training (after the first 30% of the steps), every `every` steps the equilibria of each model
in the batch are located by Newton's method from fixed starts (uniform points of Q and training states); for the found
equilibria inside Q the penalty  lam * mean_starts relu(rho(DF(u*)) - (1 - margin))^2  is added to the loss, with the
equilibria held fixed between updates. It pushes the spectral radius of unstable equilibria below one: equilibria of
index +1 can become stable, and those of index -1 can only disappear by folding with a partner. lam = 0 runs the same
code path without the penalty, so the two arms are paired (same seeds, same minibatches).

    python gpu/stage_p.py --regimes se3 --sets HE1 --classes mlp2 --widths 256 --lams 0,1 --seeds 20 [--quick]
Result: $GPUQ_OUT/P/<regime>/<class>_<set>_H<width>_lam<lam>_s<seed>.json (stage M fields + lam, penalty log)
"""
import argparse
import json
import math
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

DEV = SM.DEV
if os.environ.get("RMQ_CPU") and DEV.type == "cpu":
    torch.set_num_threads(int(os.environ["RMQ_CPU"]))


def jac_batch(cls, p, U):
    """Jacobians of the batched map at U: (B, S, 3) -> (B, S, 3, 3).
    Products are taken as (B, 3S, H) @ (B, H, .) so that the weight matrices are not copied for every point."""
    B, S, _ = U.shape
    z1 = torch.baddbmm(p["b1"][:, None, :], U, p["W1"].transpose(1, 2))
    d1 = 1 - torch.tanh(z1) ** 2                                              # (B, S, H)
    H = d1.shape[-1]
    if cls == "mlp2":
        h1 = torch.tanh(z1)
        d2 = 1 - torch.tanh(torch.baddbmm(p["b2"][:, None, :], h1, p["W2"].transpose(1, 2))) ** 2
        A = (p["W3"][:, None] * d2[:, :, None, :]).reshape(B, 3 * S, -1)       # (B, 3S, H2)
        M = torch.bmm(A, p["W2"]).reshape(B, S, 3, H)                           # (B, S, 3, H)
    else:
        M = p["W2"][:, None].expand(B, S, 3, H)                                 # (B, S, 3, H), a view
    return torch.bmm((M * d1[:, :, None, :]).reshape(B, 3 * S, H), p["W1"]).reshape(B, S, 3, 3)


def distinct_equilibria(U, ok, tol=1e-4, kmax=64):
    """Distinct converged equilibria of each model. U (B, S, 3), ok (B, S) -> points (B, K, 3), weights (B, K) equal to
    the share of the S starts that converged to the point, and a mask (B, K) of real entries."""
    B, S, _ = U.shape
    Uc, okc = U.detach().cpu().numpy(), ok.cpu().numpy()
    pts, wts = [], []
    for b in range(B):
        X = Uc[b][okc[b]]
        if len(X) == 0:
            pts.append(np.zeros((0, 3), np.float32)), wts.append(np.zeros(0, np.float32))
            continue
        _, first, cnt = np.unique(np.round(X / tol).astype(np.int64), axis=0, return_index=True, return_counts=True)
        order = np.argsort(-cnt, kind="stable")[:kmax]
        pts.append(X[first[order]]), wts.append(cnt[order] / S)
    K = max(1, max(len(w) for w in wts))
    P, W, Mk = np.zeros((B, K, 3), np.float32), np.zeros((B, K), np.float32), np.zeros((B, K), bool)
    for b in range(B):
        k = len(wts[b])
        P[b, :k], W[b, :k], Mk[b, :k] = pts[b], wts[b], True
    return (torch.tensor(P, device=U.device), torch.tensor(W, device=U.device), torch.tensor(Mk, device=U.device))


def radius_at(cls, p, P, mask):
    """Spectral radii of DF at the real entries of P (B, K, 3); eigenvalues of the few 3x3 matrices on the CPU (the
    autograd graph passes through the transfer). Returns radii (n,) and the model index of each entry."""
    J = jac_batch(cls, p, P)[mask]
    bidx = mask.nonzero()[:, 0]
    if J.shape[0] == 0:
        return J.new_zeros(0), bidx
    return torch.linalg.eigvals(J.cpu()).abs().amax(-1).to(J.device), bidx


def find_fixed(cls, p, starts, lo, hi, iters=30):
    """Newton from starts (B, S, 3) without gradients; returns points (B, S, 3) and a mask of converged points in Q."""
    with torch.no_grad():
        U = starts.clone()
        I = torch.eye(3, device=U.device, dtype=U.dtype)
        for _ in range(iters):
            g = U - SM.fwd_batch(cls, p, U)
            M = I - jac_batch(cls, p, U)
            sol, info = torch.linalg.solve_ex(M, g[..., None])
            step = sol[..., 0]
            good = (info == 0) & torch.isfinite(step).all(-1)
            U = torch.where(good[..., None], U - step.clamp(-1.0, 1.0), U)
        g = (U - SM.fwd_batch(cls, p, U)).norm(dim=-1)
        inside = ((U >= lo) & (U <= hi)).all(-1)
        ok = (g < 1e-5) & inside & torch.isfinite(U).all(-1)
    return U, ok


def train_batch_pen(cls, H, B, Win, seed, steps, lam, starts, lo, hi, every=50, margin=0.02, wd=1e-6, batch=None):
    """train_batch of stage M plus the penalty; returns weights, one-step MSE and a log of the penalty."""
    if batch is None:
        batch = 4096 if (cls != "mlp2" or H <= 256) else (2048 if H <= 512 else 1024)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(int(seed))
    p = SM.init_params(cls, B, H, gen)
    opt = torch.optim.Adam(p.values(), lr=3e-3)
    Wt = torch.tensor(Win, dtype=torch.float32, device=DEV)
    Xi, Xo = Wt[:, 0], Wt[:, 1]
    n = len(Wt)
    S0 = torch.tensor(starts, dtype=torch.float32, device=DEV)[None].expand(B, -1, -1).contiguous()
    lo_t = torch.tensor(lo, dtype=torch.float32, device=DEV)
    hi_t = torch.tensor(hi, dtype=torch.float32, device=DEV)
    Pk = None
    log = []
    t_pen = int(0.3 * steps)
    for t in range(steps):
        lr = 3e-5 + 0.5 * (3e-3 - 3e-5) * (1 + math.cos(math.pi * t / steps))
        for gp in opt.param_groups:
            gp["lr"] = lr
        idx = torch.randint(0, n, (B, min(batch, n)), generator=gen, device=DEV)
        Wb = Wt[idx]
        if t < t_pen:
            loss_b = ((SM.fwd_batch(cls, p, Wb[:, :, 0]) - Wb[:, :, 1]) ** 2).mean(dim=(1, 2))
        else:
            U = Wb[:, :, 0]
            loss_b = 0.0
            for j in range(1, Wb.shape[2]):
                U = SM.fwd_batch(cls, p, U)
                loss_b = loss_b + ((U - Wb[:, :, j]) ** 2).mean(dim=(1, 2))
            loss_b = loss_b / (Wb.shape[2] - 1)
        reg = sum((v ** 2).sum(dim=tuple(range(1, v.dim()))) for k, v in p.items() if k.startswith("W"))
        loss = loss_b + wd * reg
        if t >= t_pen and (t - t_pen) % every == 0:
            Ustar, okstar = find_fixed(cls, p, S0, lo_t, hi_t)
            Pk, Wk, Mk = distinct_equilibria(Ustar, okstar)
            with torch.no_grad():
                rho, bidx = radius_at(cls, p, Pk, Mk)
                share = torch.zeros(B, device=DEV).index_add(0, bidx, Wk[Mk] * (rho > 1).float())
                anyu = torch.zeros(B, device=DEV).index_add(0, bidx, (rho > 1).float()) > 0
                log.append([t, float(share.mean()), float(anyu.float().mean())])
        if lam > 0 and Pk is not None:
            # mean over the starts of relu(rho - (1 - margin))^2 at the converged equilibria in Q, i.e. the sum over the
            # distinct equilibria weighted by the share of starts; gradient with respect to the weights at fixed points
            rho, bidx = radius_at(cls, p, Pk, Mk)
            pen = torch.zeros(B, device=DEV).index_add(0, bidx, Wk[Mk] * torch.relu(rho - (1 - margin)) ** 2)
            loss = loss + lam * pen
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        opt.step()
    with torch.no_grad():
        full = []
        for i0 in range(0, n, 8192):
            full.append(((SM.fwd_batch(cls, p, Xi[i0:i0 + 8192][None].expand(B, -1, -1)) - Xo[i0:i0 + 8192]) ** 2).mean(dim=2))
        mse = torch.cat(full, dim=1).mean(dim=1)
    return [{k: v[b].detach().double() for k, v in p.items()} for b in range(B)], mse.double().cpu().numpy(), log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default="se3")
    ap.add_argument("--sets", default="HE1")
    ap.add_argument("--classes", default="mlp2")
    ap.add_argument("--widths", default="256")
    ap.add_argument("--lams", default="0,1")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--n-starts", type=int, default=768)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--train-only", action="store_true", help="skip the stage M analysis (tuning: report U_in only)")
    a = ap.parse_args()
    out_root = Path(os.environ.get("GPUQ_OUT", "out")) / "P"
    groups = [(k, s, c, int(H), float(lm)) for k in a.regimes.split(",") for s in a.sets.split(",")
              for c in a.classes.split(",") for H in a.widths.split(",") for lm in a.lams.split(",")]
    print(f"stage P: {len(groups)} groups x {a.seeds} seeds, device {DEV}", flush=True)
    cache = {}
    for gi, (key, set_name, cls, H, lam) in enumerate(groups):
        d = out_root / key
        d.mkdir(parents=True, exist_ok=True)
        names = [f"{cls}_{set_name}_H{H}_lam{lam:g}_s{s}" for s in range(a.seeds)]
        if all((d / f"{nm}.json").exists() for nm in names):
            continue
        if key not in cache:
            cache = {key: SC.load_regime(key)}
        z, meta, _ = cache[key]
        G.report_progress(gi, len(groups), f"{key} {cls} {set_name} H{H} lam{lam:g}")
        try:
            series, _ = SC.train_sets(z)[set_name]
            allx = np.vstack([np.asarray(s, float) for s in series])
            mu, sig = allx.mean(0), allx.std(0)
            Win = SM.windows([(np.asarray(s, float) - mu) / sig for s in series])
            lo = (np.array(meta["box_lo"]) - mu) / sig
            hi = (np.array(meta["box_hi"]) - mu) / sig
            rng = np.random.default_rng(zlib.crc32(f"P|{key}|{set_name}".encode()))
            n_box = int(0.75 * a.n_starts)
            starts = np.vstack([lo + (hi - lo) * rng.random((n_box, 3)),
                                Win[rng.integers(0, len(Win), a.n_starts - n_box), 0]])
            t0 = time.time()
            ws, mse, log = train_batch_pen(cls, H, a.seeds, Win,
                                           seed=1000 * H + zlib.crc32(f"{key}|{set_name}|{cls}".encode()) % 1000,
                                           steps=1000 if a.quick else a.steps, lam=lam, starts=starts, lo=lo, hi=hi)
            t_train = time.time() - t0
            for s, (w, e) in enumerate(zip(ws, mse)):
                nm = names[s]
                if (d / f"{nm}.json").exists():
                    continue
                try:
                    if a.train_only:
                        m = SM.Map(cls, w)
                        E = SM.equilibria(m, np.vstack([starts, (np.asarray(meta['box_lo']) - mu) / sig * 0]))
                        info = SM.eq_info(m, E, sig, mu, [], 1.0, False)
                        lo_p, hi_p = np.array(meta["box_lo"]), np.array(meta["box_hi"])
                        Un = sum(1 for x in info if x["n_unstable"] > 0 and np.all(np.array(x["x"]) >= lo_p)
                                 and np.all(np.array(x["x"]) <= hi_p))
                        rec = {"K": len(E), "U": int(sum(x["n_unstable"] > 0 for x in info)), "U_in": Un,
                               "index_sum": int(sum(x["index"] for x in info))}
                    else:
                        rec = SM.analyze_model(key, z, meta, cls, w, mu, sig, quick=a.quick)
                except Exception as exc:
                    if G.fatal_cuda(exc):
                        raise
                    (d / f"{nm}.failed.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    continue
                rec.update({"regime": key, "set": set_name, "class": cls, "H": H, "seed": s, "lam": lam,
                            "train_mse": float(e), "train_time_group_s": round(t_train, 1), "penalty_log": log})
                (d / f"{nm}.json").write_text(json.dumps(rec, default=float), encoding="utf-8")
            reps = [json.loads((d / f"{nm}.json").read_text(encoding="utf-8")) for nm in names if (d / f"{nm}.json").exists()]
            msg = f"[{gi + 1}/{len(groups)}] {key} {cls} {set_name} H{H} lam {lam:g}: mse {np.median(mse):.1e}, " \
                  f"U {np.mean([r['U'] for r in reps]):.1f}"
            if a.train_only:
                msg += f", U_in {np.mean([r['U_in'] for r in reps]):.2f}"
            else:
                hm = [next(iter(r["hidden_match"].values())) for r in reps]
                msg += (f", repr {sum(h.get('reproduced', False) for h in hm)}/{len(reps)}, not excited "
                        f"{sum(h.get('label') == 'not_excited' for h in hm)}")
            print(msg + f"; train {t_train:.0f} s; last log {log[-1] if log else None}", flush=True)
        except Exception as exc:
            traceback.print_exc()
            if G.fatal_cuda(exc):
                print("fatal CUDA error, stopping the process", flush=True)
                sys.exit(3)
    G.report_progress(len(groups), len(groups), "done")


if __name__ == "__main__":
    main()

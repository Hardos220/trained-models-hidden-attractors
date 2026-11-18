"""Этап M: полносвязные суррогаты в физическом пространстве (3D) — обобщение закона за пределы эхо-сетей.

Классы отображения u_{k+1} = F(u_k) (u = (x − μ)/σ по обучающим рядам):
  mlp  — F(u) = W₂ tanh(W₁u + b₁) + b₂           (выход ограничен: теорема о степени, Σ ind = 1);
  mlp2 — F(u) = W₃ tanh(W₂ tanh(W₁u + b₁) + b₂) + b₃ (тоже ограничен);
  res  — F(u) = u + W₂ tanh(W₁u + b₁) + b₂        (не ограничен: теорема неприменима — контраст).
Обучение — пачкой по seed (одна ширина, один класс) на GPU, Adam по минипакетам, одношаговая MSE.
Для каждой модели: равновесия методом Ньютона в 3D (старты — по всей области значений F для mlp/mlp2 и по
расширенному ящику для res), индексы и Σ ind; воспроизведение скрытого аттрактора (старты на истинной траектории),
Δatt, d_H; проверка окрестностей неустойчивых равновесий (вдоль неустойчивых направлений и 64 точки шара, радиусы
1e-6, 1e-4, 1e-3); перепись из 10⁴ равномерных точек истинной системы с матрицей «истинный исход → исход модели»
(в том числе для траекторий, уходящих в истинной системе на бесконечность).

Запуск: python gpu/stage_m.py --regimes se3 --sets H1,HE1,E1 --classes mlp,mlp2,res --widths 8,16 --seeds 20 [--quick]
Результат: $GPUQ_OUT/M/<режим>/<класс>_<набор>_H<ширина>_s<seed>.json
"""
import argparse
import json
import math
import zlib
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gpu"))
import esngpu as G  # noqa: E402  (устройство, проверка CUDA внутри gpuq, report_progress)
import pipeline as PL  # noqa: E402
import stage_c as SC  # noqa: E402

DEV = G.DEV
DT = torch.float64
LAM_C = SC.LAM_C
RADII = (1e-6, 1e-4, 1e-3)
if os.environ.get("RMQ_CPU") and DEV.type == "cpu":
    torch.set_num_threads(int(os.environ["RMQ_CPU"]))


# ---------------------------------------------------------------- обучение пачкой
K_ROLL = 4


def windows(series, K=K_ROLL):
    """Окна (n, K+1, 3) подряд идущих отсчётов внутри каждого ряда."""
    out = []
    for s in series:
        s = np.asarray(s, float)
        if len(s) > K:
            out.append(np.stack([s[j:len(s) - K + j] for j in range(K + 1)], axis=1))
    return np.concatenate(out)


def init_params(cls, B, H, gen):
    def rn(*shape, sc=1.0):
        return (torch.randn(*shape, generator=gen, device=DEV) * sc).float()
    p = {"W1": rn(B, H, 3, sc=1.0), "b1": (torch.rand(B, H, generator=gen, device=DEV) * 2 - 1).float()}
    if cls == "mlp2":
        p.update({"W2": rn(B, H, H, sc=1.0 / math.sqrt(H)), "b2": (torch.rand(B, H, generator=gen, device=DEV) - 0.5).float(),
                  "W3": rn(B, 3, H, sc=0.5 / math.sqrt(H)), "b3": torch.zeros(B, 3, device=DEV)})
    else:
        p.update({"W2": rn(B, 3, H, sc=0.5 / math.sqrt(H)), "b2": torch.zeros(B, 3, device=DEV)})
    for v in p.values():
        v.requires_grad_(True)
    return p


def fwd_batch(cls, p, U):
    """U: (B, n, 3) → (B, n, 3)."""
    h = torch.tanh(torch.baddbmm(p["b1"][:, None, :], U, p["W1"].transpose(1, 2)))
    if cls == "mlp2":
        h = torch.tanh(torch.baddbmm(p["b2"][:, None, :], h, p["W2"].transpose(1, 2)))
        out = torch.baddbmm(p["b3"][:, None, :], h, p["W3"].transpose(1, 2))
    else:
        out = torch.baddbmm(p["b2"][:, None, :], h, p["W2"].transpose(1, 2))
    return U + out if cls == "res" else out


def train_batch(cls, H, B, Win, seed, steps, wd=1e-6, batch=None):
    """Win: окна (n, K+1, 3). Первые 30% шагов — одношаговая ошибка, затем — ошибка прогноза на 1…K шагов."""
    if batch is None:                                                 # память: у mlp2 слой H×H
        batch = 4096 if (cls != "mlp2" or H <= 256) else (2048 if H <= 512 else 1024)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(int(seed))
    p = init_params(cls, B, H, gen)
    opt = torch.optim.Adam(p.values(), lr=3e-3)
    Wt = torch.tensor(Win, dtype=torch.float32, device=DEV)
    Xi, Xo = Wt[:, 0], Wt[:, 1]
    n = len(Wt)
    for t in range(steps):
        lr = 3e-5 + 0.5 * (3e-3 - 3e-5) * (1 + math.cos(math.pi * t / steps))
        for gp in opt.param_groups:
            gp["lr"] = lr
        idx = torch.randint(0, n, (B, min(batch, n)), generator=gen, device=DEV)
        Wb = Wt[idx]                                                      # (B, b, K+1, 3)
        if t < 0.3 * steps:
            loss_b = ((fwd_batch(cls, p, Wb[:, :, 0]) - Wb[:, :, 1]) ** 2).mean(dim=(1, 2))
        else:
            U = Wb[:, :, 0]
            loss_b = 0.0
            for j in range(1, Wb.shape[2]):
                U = fwd_batch(cls, p, U)
                loss_b = loss_b + ((U - Wb[:, :, j]) ** 2).mean(dim=(1, 2))
            loss_b = loss_b / (Wb.shape[2] - 1)
        reg = sum((v ** 2).sum(dim=tuple(range(1, v.dim()))) for k, v in p.items() if k.startswith("W"))
        loss = (loss_b + wd * reg).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        full = []
        for i0 in range(0, n, 8192):
            full.append(((fwd_batch(cls, p, Xi[i0:i0 + 8192][None].expand(B, -1, -1)) - Xo[i0:i0 + 8192]) ** 2).mean(dim=2))
        mse = torch.cat(full, dim=1).mean(dim=1)
    return [{k: v[b].detach().double() for k, v in p.items()} for b in range(B)], mse.double().cpu().numpy()


# ---------------------------------------------------------------- одна модель
class Map:
    def __init__(self, cls, w):
        self.cls, self.w = cls, w

    def F(self, U):
        w = self.w
        h = torch.tanh(U @ w["W1"].T + w["b1"])
        if self.cls == "mlp2":
            h = torch.tanh(h @ w["W2"].T + w["b2"])
            out = h @ w["W3"].T + w["b3"]
        else:
            out = h @ w["W2"].T + w["b2"]
        return U + out if self.cls == "res" else out

    def J(self, U):
        """Якобианы (n, 3, 3)."""
        w = self.w
        z1 = U @ w["W1"].T + w["b1"]
        d1 = 1 - torch.tanh(z1) ** 2                                      # (n, H)
        if self.cls == "mlp2":
            h1 = torch.tanh(z1)
            d2 = 1 - torch.tanh(h1 @ w["W2"].T + w["b2"]) ** 2
            M = (w["W3"][None] * d2[:, None, :]) @ w["W2"][None]          # (n, 3, H)
            Jm = (M * d1[:, None, :]) @ w["W1"][None]
        else:
            Jm = (w["W2"][None] * d1[:, None, :]) @ w["W1"][None]
        if self.cls == "res":
            Jm = Jm + torch.eye(3, dtype=DT, device=DEV)[None]
        return Jm

    def out_bound(self):
        w = self.w
        Wl, bl = (w["W3"], w["b3"]) if self.cls == "mlp2" else (w["W2"], w["b2"])
        return (Wl.abs().sum(1) + bl.abs()).cpu().numpy()                 # |F_i(u)| ≤ это при всех u

    def run(self, U0, n_tr, n_obs, lam=False):
        U = torch.as_tensor(U0, dtype=DT, device=DEV).clone()
        esc = torch.zeros(len(U), dtype=torch.bool, device=DEV)
        for k in range(n_tr):
            U = self.F(U)
            if (k + 1) % 200 == 0:
                bad = ~torch.isfinite(U).all(1) | (U.abs().amax(1) > 1e3)
                esc |= bad
                U[bad] = 0.0
        drift = torch.zeros(len(U), dtype=DT, device=DEV)
        ls = torch.zeros(len(U), dtype=DT, device=DEV)
        v = torch.randn(U.shape, dtype=DT, device=DEV)
        v /= v.norm(dim=1, keepdim=True)
        for k in range(n_obs):
            Un = self.F(U)
            if k >= n_obs - 50:
                drift = torch.maximum(drift, (Un - U).abs().amax(1))
            if lam:
                v = (self.J(U) @ v[:, :, None])[:, :, 0]
                nv = v.norm(dim=1)
                ls += torch.log(torch.clamp(nv, min=1e-300))
                v = v / torch.clamp(nv, min=1e-300)[:, None]
            U = Un
        bad = ~torch.isfinite(U).all(1) | (U.abs().amax(1) > 1e3)
        esc |= bad
        fp = (drift < 1e-9 * (1 + U.abs().amax(1))) & ~esc
        return {"final": U.cpu().numpy(), "fp": fp.cpu().numpy(), "esc": esc.cpu().numpy(),
                "lam": (ls / n_obs).cpu().numpy() if lam else None}

    def orbit(self, u0, n, burn=2000):
        U = torch.as_tensor(np.asarray(u0, float)[None], dtype=DT, device=DEV)
        for _ in range(burn):
            U = self.F(U)
        S = torch.empty(n, 3, dtype=DT, device=DEV)
        for k in range(n):
            U = self.F(U)
            S[k] = U[0]
        return S.cpu().numpy()


def equilibria(m, starts, iters=60):
    U = torch.as_tensor(starts, dtype=DT, device=DEV).clone()
    I = torch.eye(3, dtype=DT, device=DEV)[None]
    ok = torch.zeros(len(U), dtype=torch.bool, device=DEV)
    for _ in range(iters):
        g = U - m.F(U)
        ok = g.norm(dim=1) < 1e-12 * (1 + U.norm(dim=1))
        M = I - m.J(U)
        sol, info = torch.linalg.solve_ex(M, g[:, :, None])
        step = sol[:, :, 0]
        good = (info == 0) & torch.isfinite(step).all(1)
        U = torch.where((good & ~ok)[:, None], U - step, U)
    g = (U - m.F(U)).norm(dim=1)
    # relative tolerance; points beyond 1e6 are Newton runaways of unbounded maps (tanh saturated), not equilibria
    ok = (g < 1e-10 * (1 + U.norm(dim=1))) & torch.isfinite(U).all(1) & (U.norm(dim=1) < 1e6)
    P = U[ok].cpu().numpy()
    E = []
    for x in P:
        if all(np.linalg.norm(x - y) > 1e-7 * (1 + np.linalg.norm(y)) for y in E):
            E.append(x)
    return np.array(E).reshape(-1, 3)


def eq_info(m, E, sig, mu, true_eq, L_scale, eq_line):
    out = []
    if not len(E):
        return out
    Js = m.J(torch.as_tensor(E, dtype=DT, device=DEV)).cpu().numpy()
    for x, Jm in zip(E, Js):
        w, V = np.linalg.eig(Jm)
        real = np.abs(w.imag) <= 1e-10 * np.maximum(1, np.abs(w))
        sign = np.sign(np.linalg.det(np.eye(3) - Jm))
        xp = x * sig + mu
        rec = {"x": xp.tolist(), "rho": float(np.max(np.abs(w))), "n_unstable": int(np.sum(np.abs(w) > 1 + 1e-9)),
               "n_real_gt1": int(np.sum(real & (w.real > 1 + 1e-9))), "index": int(sign)}
        if eq_line:
            rec["dist_true_eq"] = float(np.hypot(xp[0], xp[1]) / L_scale)
        elif true_eq:
            dd = np.linalg.norm(np.asarray(true_eq) - xp, axis=1)
            rec["nearest_true_eq"], rec["dist_true_eq"] = int(np.argmin(dd)), float(dd.min() / L_scale)
        out.append(rec)
    return out


def labeler(E, atts):
    """atts: список (имя, отсчёты орбиты (n,3), порог). Возвращает функцию меток для результата run."""
    trees = [(nm, cKDTree(S), thr) for nm, S, thr in atts]
    Et = cKDTree(E) if len(E) else None

    def lab(o):
        L = np.empty(len(o["final"]), dtype=object)
        L[o["esc"]] = "esc"
        f = o["final"]
        idx = np.where(o["fp"])[0]
        if len(idx):
            if Et is not None:
                d, j = Et.query(f[idx])
                L[idx] = [f"f{jj}" if dd < 1e-5 * (1 + np.linalg.norm(f[i])) else "fx" for i, dd, jj in zip(idx, d, j)]
            else:
                L[idx] = "fx"
        osc = np.where(~o["fp"] & ~o["esc"])[0]
        if len(osc):
            best = np.full(len(osc), "osc", dtype=object)
            bd = np.full(len(osc), np.inf)
            for nm, tr, thr in trees:
                d, _ = tr.query(f[osc])
                b = (d < thr) & (d < bd)
                best[b] = nm
                bd[b] = d[b]
            L[osc] = best
        return L.astype(str)
    return lab


def analyze_model(key, z, meta, cls, w, mu, sig, quick=False, model=None):
    t0 = time.time()
    m = model if model is not None else Map(cls, w)   # model: any object with F, J, run, orbit (stage E)
    n_tr, n_obs = (4000, 500) if quick else (20000, 1000)
    rng = np.random.default_rng(11)
    true_eq = [e["point"] for e in meta["equilibria"]]
    L_scale = float(np.max(np.array(meta["box_hi"]) - np.array(meta["box_lo"]))) / 2
    eq_line = key == "le1"
    lo = (np.array(meta["box_lo"]) - mu) / sig
    hi = (np.array(meta["box_hi"]) - mu) / sig
    # равновесия
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
    E = equilibria(m, np.vstack(starts))
    info = eq_info(m, E, sig, mu, true_eq, L_scale, eq_line)
    rec = {"K": len(E), "U": int(sum(f["n_unstable"] > 0 for f in info)), "index_sum": int(sum(f["index"] for f in info)),
           "fixed_points": info}
    # скрытые аттракторы: старты на истинной траектории
    atts, match = [], {}
    for y in meta["hidden"]:
        Y = (z[f"att_{y}"] - mu) / sig
        ykind = "chaos" if meta["attractors"][y]["kind"] == "chaotic" else "regular"
        o = m.run(Y[rng.integers(0, len(Y), 64)], n_tr, n_obs, lam=True)
        lamt = o["lam"] / meta["Delta_t"]
        keep = ~o["fp"] & ~o["esc"] & ((lamt > LAM_C) if ykind == "chaos" else (lamt <= LAM_C))
        mm = {"frac_osc": float(np.mean(~o["fp"] & ~o["esc"])), "frac_kind": float(np.mean(keep))}
        if keep.sum() >= 8:
            S = m.orbit(o["final"][np.where(keep)[0][0]], 10000 if not quick else 3000)
            Xp = S * sig + mu
            corr = PL.correspondence(Xp, {y: z[f"att_{y}"][:10000]})[y]
            mm.update(corr)
            mm["reproduced"] = bool(corr.get("delta_att", 9) < 0.2 and corr.get("d_H", 9) < 0.2)
            mm["lam"] = float(np.median(lamt[keep]))
            nm = f"a{len(atts)}"
            mm["attractor"] = nm
            atts.append((nm, S, 0.02 * float(np.ptp(S, axis=0).max())))
        else:
            mm["reproduced"] = False
        match[y] = mm
    rec["hidden_match"] = match
    lab = labeler(E, atts)
    # окрестности неустойчивых равновесий
    X0, tags = [], []
    for i, (x, f) in enumerate(zip(E, info)):
        if f["n_unstable"] == 0:
            continue
        Jm = m.J(torch.as_tensor(x[None], dtype=DT, device=DEV))[0].cpu().numpy()
        wv, V = np.linalg.eig(Jm)
        dirs = []
        for k in np.where(np.abs(wv) > 1 + 1e-9)[0]:
            for comp in (V[:, k].real, V[:, k].imag):
                if np.linalg.norm(comp) > 1e-12:
                    dirs.append(comp / np.linalg.norm(comp))
        for eps in RADII:
            pts = [x + eps * s * v for v in dirs for s in (1, -1)]
            B = rng.normal(size=(64, 3))
            pts += list(x + eps * B / np.linalg.norm(B, axis=1, keepdims=True) * rng.random((64, 1)) ** (1 / 3))
            X0 += pts
            tags += [(i, eps)] * len(pts)
    nb = {}
    if X0:
        o = m.run(np.array(X0), n_tr, n_obs)
        L = lab(o)
        for (i, eps), l in zip(tags, L):
            nb.setdefault(f"{i}|{eps:g}", {}).setdefault(l, 0)
            nb[f"{i}|{eps:g}"][l] += 1
    rec["radius_counts"] = nb
    for y, mm in match.items():
        a = mm.get("attractor")
        if not a:
            continue
        r_hat = min((float(k.split("|")[1]) for k, c in nb.items() if c.get(a, 0) > 0), default=np.inf)
        mm["r_hat"] = None if not np.isfinite(r_hat) else r_hat
        mm["label"] = "self_excited" if r_hat <= 1e-6 else ("excited_r" if r_hat <= 1e-3 else "not_excited")
    # перепись: истинный исход → исход модели
    ics = (z["ics_uniform"] - mu) / sig
    tl = np.asarray(z["labels_uniform"]).astype(str)
    o = m.run(ics[:3000] if quick else ics, n_tr, n_obs)
    L = lab(o)
    conf = {}
    for a_, b_ in zip(tl[:len(L)], L):
        conf.setdefault(a_, {}).setdefault(b_, 0)
        conf[a_][b_] += 1
    rec["census_confusion"] = conf
    rec["census_counts"] = PL.counts(L)
    rec["time_s"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default="se3")
    ap.add_argument("--sets", default="H1,HE1,E1")
    ap.add_argument("--classes", default="mlp,mlp2,res")
    ap.add_argument("--widths", default="8,16,32,64,128,256,512,1024")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out_root = Path(os.environ.get("GPUQ_OUT", "out")) / "M"
    groups = [(k, s, c, int(H)) for k in a.regimes.split(",") for s in a.sets.split(",") for c in a.classes.split(",")
              for H in a.widths.split(",")]
    print(f"stage M: {len(groups)} groups × {a.seeds} seeds, device {DEV}", flush=True)
    cache = {}
    for gi, (key, set_name, cls, H) in enumerate(groups):
        d = out_root / key
        d.mkdir(parents=True, exist_ok=True)
        names = [f"{cls}_{set_name}_H{H}_s{s}" for s in range(a.seeds)]
        if all((d / f"{nm}.json").exists() for nm in names):
            continue
        if key not in cache:
            cache = {key: SC.load_regime(key)}
        z, meta, _ = cache[key]
        G.report_progress(gi, len(groups), f"{key} {cls} {set_name} H{H}")
        try:
            series, _ = SC.train_sets(z)[set_name]
            allx = np.vstack([np.asarray(s, float) for s in series])
            mu, sig = allx.mean(0), allx.std(0)
            Win = windows([(np.asarray(s, float) - mu) / sig for s in series])
            t0 = time.time()
            ws, mse = train_batch(cls, H, a.seeds, Win, seed=1000 * H + zlib.crc32(f"{key}|{set_name}|{cls}".encode()) % 1000,
                                  steps=1000 if a.quick else a.steps)
            t_train = time.time() - t0
            for s, (w, e) in enumerate(zip(ws, mse)):
                nm = names[s]
                if (d / f"{nm}.json").exists():
                    continue
                try:
                    rec = analyze_model(key, z, meta, cls, w, mu, sig, quick=a.quick)
                except Exception as exc:
                    if G.fatal_cuda(exc):
                        raise
                    (d / f"{nm}.failed.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    continue
                rec.update({"regime": key, "set": set_name, "class": cls, "H": H, "seed": s, "train_mse": float(e),
                            "train_time_group_s": round(t_train, 1)})
                (d / f"{nm}.json").write_text(json.dumps(rec, default=float), encoding="utf-8")
            reps = [json.loads((d / f"{nm}.json").read_text(encoding="utf-8")) for nm in names if (d / f"{nm}.json").exists()]
            hm = [next(iter(r["hidden_match"].values())) for r in reps]
            print(f"[{gi + 1}/{len(groups)}] {key} {cls} {set_name} H{H}: mse {np.median(mse):.1e}, U сред "
                  f"{np.mean([r['U'] for r in reps]):.1f}, Σ≠1 {sum(r['index_sum'] != 1 for r in reps)}, воспр "
                  f"{sum(h.get('reproduced', False) for h in hm)}/{len(reps)}, не возб. "
                  f"{sum(h.get('label') == 'not_excited' for h in hm)}; обучение {t_train:.0f} с, "
                  f"анализ {np.mean([r['time_s'] for r in reps]):.1f} с/модель", flush=True)
        except Exception as exc:
            traceback.print_exc()
            if G.fatal_cuda(exc):
                print("fatal CUDA error, stopping the process", flush=True)
                sys.exit(3)
    G.report_progress(len(groups), len(groups), "done")


if __name__ == "__main__":
    main()

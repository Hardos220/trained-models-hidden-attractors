"""NG-RC на GPU с интерфейсом esngpu.Net: состояние S = (x_n, x_{n-1}) в координатах x/scale, D = 2d."""
import math

import numpy as np
import torch

import esngpu as G
from esngpu import DEV, DT, N_, T


class NGRCNet:
    def __init__(self, ngrc, h, box_lo, box_hi, data_absmax):
        self.Wm = T(ngrc.W)                       # (d, F)
        self.scale = np.asarray(ngrc.scale, float)
        self.d = ngrc.d
        self.N = 2 * self.d
        self.h = float(h)
        self.kind = "ngrc"
        lo, hi = np.asarray(box_lo) / self.scale, np.asarray(box_hi) / self.scale
        self.box_lo_n, self.box_hi_n = lo, hi
        self.L_M = float(np.max(hi - lo)) / 2
        self.bound = T(10.0 * np.asarray(data_absmax) / self.scale)
        iu = np.triu_indices(2 * self.d)
        self.iu0 = torch.as_tensor(iu[0], device=DEV)
        self.iu1 = torch.as_tensor(iu[1], device=DEV)
        self.sc = T(self.scale)

    def phi(self, Xn, Xm):
        O = torch.cat([Xn, Xm], dim=1)
        Q = O[:, self.iu0] * O[:, self.iu1]
        return torch.cat([torch.ones(len(O), 1, dtype=DT, device=DEV), O, Q], dim=1)

    def step(self, S):
        d = self.d
        Xn, Xm = S[:, :d], S[:, d:]
        Xnew = Xn + self.phi(Xn, Xm) @ self.Wm.T
        return torch.cat([Xnew, Xn], dim=1)

    def output_of(self, s):
        return np.asarray(s, float)[: self.d] * self.scale

    def clip_state(self, X):
        return np.nan_to_num(np.asarray(X, float))

    def _escaped(self, S):
        X = S[:, : self.d]
        return ~torch.isfinite(X).all(1) | (X.abs() > self.bound).any(1)

    def run(self, S0, n_tr, n_obs, d0=1e-7, seed=0, fp_tol=1e-9, chunk=65536, v=None, prec="fp64"):
        S0 = T(S0) if not torch.is_tensor(S0) else S0
        outs = [self._run(S0[i:i + chunk], n_tr, n_obs, d0, seed + i, fp_tol) for i in range(0, S0.shape[0], chunk)]
        return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}

    def _run(self, S, n_tr, n_obs, d0, seed, fp_tol):
        n, d = S.shape[0], self.d
        S = S.clone()
        esc = torch.zeros(n, dtype=torch.bool, device=DEV)
        for _ in range(n_tr):
            Sn = self.step(S)
            e = self._escaped(Sn)
            esc |= e
            S = torch.where(esc[:, None], S, Sn)
        g = torch.Generator(device=DEV)
        g.manual_seed(int(seed))
        v = torch.randn(S.shape, generator=g, dtype=DT, device=DEV)
        v /= v.norm(dim=1, keepdim=True)
        Q = S + d0 * v
        s1 = torch.zeros(n, d, dtype=DT, device=DEV); s2 = torch.zeros_like(s1); sa = torch.zeros_like(s1)
        xmin = torch.full((n, d), math.inf, dtype=DT, device=DEV); xmax = torch.full((n, d), -math.inf, dtype=DT, device=DEV)
        logsum = torch.zeros(n, dtype=DT, device=DEV); drift = torch.zeros(n, dtype=DT, device=DEV)
        for k in range(n_obs):
            Sn = self.step(S)
            Qn = self.step(Q)
            e = self._escaped(Sn)
            esc |= e
            Sn = torch.where(esc[:, None], S, Sn)
            Qn = torch.where(esc[:, None], Sn + d0 * v, Qn)
            if k >= n_obs - 50:
                drift = torch.maximum(drift, (Sn - S).abs().amax(1))
            S = Sn
            dQ = Qn - S
            dist = dQ.norm(dim=1)
            dist = torch.where((dist > 0) & torch.isfinite(dist), dist, torch.full_like(dist, d0))
            logsum += torch.log(dist / d0)
            Q = S + dQ * (d0 / dist)[:, None]
            Q = torch.where(torch.isfinite(Q), Q, S + d0 * v)
            x = S[:, :d] * self.sc
            s1 += x; s2 += x * x; sa += x.abs()
            xmin = torch.minimum(xmin, x); xmax = torch.maximum(xmax, x)
        mean = s1 / n_obs
        std = torch.sqrt(torch.clamp(s2 / n_obs - mean ** 2, min=0.0))
        fp = (drift < fp_tol * (1 + S.abs().amax(1))) & ~esc
        return {"x_mean": N_(mean), "x_std": N_(std), "x_min": N_(xmin), "x_max": N_(xmax), "x_absmean": N_(sa / n_obs),
                "lam": N_(logsum / (n_obs * self.h)), "fp": N_(fp), "final": N_(S), "esc": N_(esc)}

    def samples(self, s0, n, burn=0):
        S = T(np.asarray(s0, float)).reshape(1, -1)
        for _ in range(burn):
            S = self.step(S)
        X = []
        for _ in range(n):
            S = self.step(S)
            X.append(S[0, : self.d] * self.sc)
        return N_(torch.stack(X))

    def orbit(self, s0, n, every=1, burn=0):
        S = (s0 if torch.is_tensor(s0) else T(s0)).reshape(1, -1)
        for _ in range(burn):
            S = self.step(S)
        out = []
        for k in range(n):
            S = self.step(S)
            if k % every == 0:
                out.append(S[0].clone())
        return torch.stack(out)

    def g_eq(self, X):
        return self.phi(X, X) @ self.Wm.T

    def newton(self, S0, max_iter=60, tol=1e-12, chunk=65536, deflate=None, **kw):
        """Равновесия (x*, x*): W Φ(x*, x*) = 0 по первой половине стартов; якобиан конечными разностями."""
        S0 = np.asarray(S0, float)
        X = T(S0[:, : self.d]).clone()
        n, d = X.shape
        ok = torch.zeros(n, dtype=torch.bool, device=DEV)
        bad = torch.zeros(n, dtype=torch.bool, device=DEV)
        hstep = 1e-7
        for _ in range(max_iter):
            gx = self.g_eq(X)
            ok |= (gx.norm(dim=1) < tol) & ~bad
            act = ~ok & ~bad
            if not act.any():
                break
            ia = act.nonzero().squeeze(1)
            Xa = X[ia]
            ga = gx[ia]
            J = torch.empty(len(ia), d, d, dtype=DT, device=DEV)
            for j in range(d):
                E = Xa.clone()
                E[:, j] += hstep
                J[:, :, j] = (self.g_eq(E) - ga) / hstep
            sol, info = torch.linalg.solve_ex(J, ga[:, :, None])
            Xn = Xa - sol[:, :, 0]
            fin = torch.isfinite(Xn).all(1) & (info == 0) & (Xn.abs() < 1e6).all(1)
            X[ia] = torch.where(fin[:, None], Xn, Xa)
            bad[ia[~fin]] = True
        Xn = N_(X)
        return np.hstack([Xn, Xn]), N_(ok)

    def jac(self, s):
        s = np.asarray(s, float)
        D = self.N
        base = self.step(T(s)[None])[0]
        J = torch.empty(D, D, dtype=DT, device=DEV)
        hstep = 1e-7
        for j in range(D):
            e = s.copy()
            e[j] += hstep
            J[:, j] = (self.step(T(e)[None])[0] - base) / hstep
        return J

    def stability(self, s):
        w, V = torch.linalg.eig(self.jac(s))
        return N_(w), N_(V)

    def unstable_directions(self, s, tol=1e-9, max_dirs=20):
        w, V = self.stability(s)
        dirs = []
        for k in np.argsort(-np.abs(w)):
            if np.abs(w[k]) <= 1 + tol:
                break
            for comp in (np.real(V[:, k]), np.imag(V[:, k])):
                nrm = np.linalg.norm(comp)
                if nrm > 1e-12:
                    dirs.append(comp / nrm)
        return w, dirs[:max_dirs]

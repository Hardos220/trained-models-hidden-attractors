"""GPU-ядро пилота эхо-сетей (torch, float64): замкнутый цикл, отпечатки, метод Ньютона, hard-tanh.

Смысл всех величин — как в src/hsearch/esn_pilot.py (CPU-эталон); здесь только пакетные реализации.
"""
import json
import math
import os
import time

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DT = torch.float64
if os.environ.get("GPUQ_JOB_ID") and "ESN_CUDA_RETRY" not in os.environ:
    time.sleep(float(np.random.default_rng().uniform(0, 20)))     # разнести старты задач, начатых в одну секунду
DEV = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
if os.environ.get("GPUQ_JOB_ID") and DEV.type != "cuda" and int(os.environ.get("ESN_CUDA_RETRY", "0")) < 5:
    # MPS может отказать клиенту при одновременном старте; отказ кэшируется в процессе, поэтому перезапуск процесса
    import sys as _sys
    k = int(os.environ.get("ESN_CUDA_RETRY", "0")) + 1
    print(f"[esngpu] CUDA unavailable, restarting the process (attempt {k} of 5)", flush=True)
    os.environ["ESN_CUDA_RETRY"] = str(k)
    time.sleep(30 * k)
    os.execv(_sys.executable, [_sys.executable] + _sys.argv)
if os.environ.get("GPUQ_JOB_ID") and DEV.type != "cuda" and os.environ.get("ESN_ALLOW_CPU") != "1":
    # внутри задачи gpuq видеокарта обязана быть: без неё расчёт молча ушёл бы на процессор (так было при отказе MPS)
    import sys as _sys
    print(f"FATAL: CUDA is not available in gpuq job {os.environ.get('GPUQ_JOB_ID')} "
          f"(torch {torch.__version__}); refusing to run on CPU", flush=True)
    _sys.exit(4)
print(f"[esngpu] device {DEV}" + (f" {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""), flush=True)
USE_GRAPHS = os.environ.get("ESN_GRAPHS", "0") == "1"


class Pacer:
    """Ровный темп при перегреве: раз в минуту смотрит throttled_s в RMQ_CONDITIONS_FILE; если регулятор
    температуры урезал задачи — уменьшает долю занятости видеокарты (паузы до 50 мс между блоками)."""

    def __init__(self):
        self.path = os.environ.get("RMQ_CONDITIONS_FILE")
        self.enabled = bool(self.path) and DEV.type == "cuda"
        self.share = 1.0
        self.last_check = time.monotonic()
        self.cool_since = time.monotonic()
        self.last_thr = self._throttled()
        self.t0 = time.perf_counter()

    def _throttled(self):
        if not self.path:
            return None
        try:
            with open(self.path, encoding="utf-8") as f:
                return float(json.load(f).get("throttled_s", 0.0))
        except (OSError, ValueError, TypeError):
            return None

    def start(self):
        self.t0 = time.perf_counter()

    def tick(self):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        busy = time.perf_counter() - self.t0
        if self.share < 1.0:
            time.sleep(min(0.05, busy * (1.0 / self.share - 1.0)))
        now = time.monotonic()
        if now - self.last_check >= 60:
            self.last_check = now
            thr = self._throttled()
            if thr is not None and self.last_thr is not None:
                if thr - self.last_thr > 5:
                    self.share, self.cool_since = max(0.4, self.share - 0.1), now
                elif now - self.cool_since >= 300:
                    self.share = min(1.0, self.share + 0.1)
            self.last_thr = thr
        self.t0 = time.perf_counter()


PACER = None


def pacer():
    global PACER
    if PACER is None:
        PACER = Pacer()
    return PACER


def T(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=DT, device=DEV)


def N_(t):
    return t.detach().cpu().numpy()


class Net:
    """Замкнутый цикл r_{k+1} = φ(A r_k + b), A = s·W + W_in W_out; выход x = scale·W_out r."""

    def __init__(self, esn, h, s=1.0):
        W_out = np.asarray(esn["W_out"], float)
        A = s * np.asarray(esn["W"], float) + np.asarray(esn["W_in"], float) @ W_out
        self.A_np = A
        self.b_np = np.asarray(esn["b"], float)
        self.AT = T(A.T).contiguous()
        self.A = T(A).contiguous()
        self.b = T(self.b_np)
        self.b_esn = self.b
        self.scale = np.asarray(esn["scale"], float)
        self.WoT = T((W_out * self.scale[:, None]).T).contiguous()     # (N, d): физический выход
        self.WoTn = T(W_out.T).contiguous()                            # нормированный выход x/scale
        self.kind = esn.get("activation", "tanh")
        self.N = A.shape[0]
        self.d = W_out.shape[0]
        self.h = float(h)

    L_M = 1.0

    def output_of(self, r):
        return (np.asarray(r, float) @ self.WoT.cpu().numpy())

    def clip_state(self, X):
        return np.clip(np.asarray(X, float), -1.0, 1.0)

    def lift_const(self, esn, X0, w=100, R0=None):
        """L2: w шагов r = φ(W r + W_in x0/scale + b) от R0 (по умолчанию 0); возвращает numpy."""
        W = T(esn["W"]); Win = T(esn["W_in"])
        X0 = np.atleast_2d(np.asarray(X0, float))
        drive = T(X0 / self.scale) @ Win.T + self.b_esn
        R = torch.zeros(len(X0), self.N, dtype=DT, device=DEV) if R0 is None else T(R0)
        for _ in range(w):
            R = self.act_(torch.addmm(drive, R, W.T))
        return N_(R)

    def lift_drive(self, esn, U, w):
        """L1: w входов истинной траектории U (n, >=w, d) от r = 0; строки с NaN дают NaN."""
        W = T(esn["W"]); Win = T(esn["W_in"])
        U = np.asarray(U, float)
        Un = T(np.nan_to_num(U[:, :w] / self.scale, nan=0.0))
        R = torch.zeros(len(U), self.N, dtype=DT, device=DEV)
        for k in range(w):
            R = self.act_(torch.addmm(self.b_esn, R, W.T) + Un[:, k] @ Win.T)
        R = N_(R)
        R[np.isnan(U[:, :w]).any(axis=(1, 2))] = np.nan
        return R

    def act_(self, Z):
        if self.kind == "tanh":
            return Z.tanh_()
        return Z.clamp_(-1.0, 1.0)

    def step_into(self, R, out):
        torch.addmm(self.b, R, self.AT, out=out)
        return self.act_(out)

    def advance(self, R, n):
        """n итераций; возвращает новый тензор (R не меняется)."""
        a = R.clone()
        b = torch.empty_like(a)
        for _ in range(n):
            self.step_into(a, b)
            a, b = b, a
        return a

    def _fp32(self):
        if not hasattr(self, "_AT32"):
            self._AT32 = self.AT.float()
            self._b32 = self.b.float()
            self._WoT32 = self.WoT.float()
        return self._AT32, self._b32, self._WoT32

    def run(self, R0, n_tr, n_obs, d0=1e-7, seed=0, fp_tol=1e-9, chunk=16384, v=None, prec="fp64"):
        if prec == "fp32":
            R0 = R0 if torch.is_tensor(R0) else T(R0)
            outs = [self._run_fp32(R0[i:i + chunk], n_tr, n_obs, seed + i) for i in range(0, R0.shape[0], chunk)]
            return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
        return self._run_dispatch(R0, n_tr, n_obs, d0, seed, fp_tol, chunk, v)

    def _run_fp32(self, R0, n_tr, n_obs, seed, d0=1e-4, fp_tol=1e-5):
        """То же, что _run_eager, в float32: отклонение для показателя Ляпунова 1e-4, порог сходимости 1e-5."""
        AT, b, WoT = self._fp32()
        F = torch.float32
        n, d = R0.shape[0], self.d
        pc = pacer()
        pc.start()
        R = R0.to(F).clone()
        Rn = torch.empty_like(R)

        def step(src, dst):
            torch.addmm(b, src, AT, out=dst)
            return dst.tanh_() if self.kind == "tanh" else dst.clamp_(-1.0, 1.0)
        for k in range(n_tr):
            step(R, Rn)
            R, Rn = Rn, R
            if (k + 1) % 50 == 0:
                pc.tick()
        g = torch.Generator(device=DEV)
        g.manual_seed(int(seed))
        v = torch.randn(R.shape, generator=g, dtype=F, device=DEV)
        v /= v.norm(dim=1, keepdim=True)
        Q = R + d0 * v
        Qn = torch.empty_like(R)
        s1 = torch.zeros(n, d, dtype=DT, device=DEV); s2 = torch.zeros_like(s1); sa = torch.zeros_like(s1)
        xmin = torch.full((n, d), math.inf, dtype=DT, device=DEV); xmax = torch.full((n, d), -math.inf, dtype=DT, device=DEV)
        logsum = torch.zeros(n, dtype=DT, device=DEV)
        drift = torch.zeros(n, dtype=F, device=DEV)
        for k in range(n_obs):
            step(R, Rn)
            step(Q, Qn)
            if k >= n_obs - 50:
                drift = torch.maximum(drift, (Rn - R).abs().amax(dim=1))
            R, Rn = Rn, R
            Q, Qn = Qn, Q
            dQ = Q - R
            dist = dQ.norm(dim=1)
            dist = torch.where(dist > 0, dist, torch.full_like(dist, d0))
            logsum += torch.log(dist / d0).double()
            Q = R + dQ * (d0 / dist)[:, None]
            x = (R @ WoT).double()
            s1 += x; s2 += x * x; sa += x.abs()
            torch.minimum(xmin, x, out=xmin); torch.maximum(xmax, x, out=xmax)
            if (k + 1) % 50 == 0:
                pc.tick()
        mean = s1 / n_obs
        std = torch.sqrt(torch.clamp(s2 / n_obs - mean ** 2, min=0.0))
        fp = drift < fp_tol * (1 + R.abs().amax(dim=1))
        return {"x_mean": N_(mean), "x_std": N_(std), "x_min": N_(xmin), "x_max": N_(xmax), "x_absmean": N_(sa / n_obs),
                "lam": N_(logsum / (n_obs * self.h)), "fp": N_(fp), "final": N_(R.double())}

    def _run_dispatch(self, R0, n_tr, n_obs, d0=1e-7, seed=0, fp_tol=1e-9, chunk=16384, v=None):
        """Отпечатки как esn_pilot.ClosedLoop.run (физический выход), по пачкам chunk состояний.
        v — направления начальных отклонений для показателя Ляпунова (numpy n×N); по умолчанию случайные."""
        R0 = R0 if torch.is_tensor(R0) else T(R0)
        outs = []
        for i in range(0, R0.shape[0], chunk):
            vi = None if v is None else T(v[i:i + chunk])
            outs.append(self._run(R0[i:i + chunk], n_tr, n_obs, d0, seed + i, fp_tol, vi))
        return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}

    def _run(self, R0, n_tr, n_obs, d0, seed, fp_tol, v=None):
        if USE_GRAPHS and DEV.type == "cuda" and n_obs % 50 == 0 and n_obs >= 100 and R0.shape[0] >= 64:
            return self._run_graph(R0, n_tr, n_obs, d0, seed, fp_tol, v)
        return self._run_eager(R0, n_tr, n_obs, d0, seed, fp_tol, v)

    def _capture(self, body):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        return g

    def _run_graph(self, R0, n_tr, n_obs, d0, seed, fp_tol, v=None):
        n, N, d = R0.shape[0], self.N, self.d
        BT, BO = 100, 50
        Ra = torch.empty_like(R0); Rb = torch.empty_like(R0)

        def tr_body():
            for k in range(BT // 2):
                self.step_into(Ra, Rb)
                self.step_into(Rb, Ra)
        Ra.copy_(R0)
        g_tr = self._capture(tr_body) if n_tr >= BT else None
        Ra.copy_(R0)
        for _ in range(n_tr // BT):
            g_tr.replay()
        for _ in range(n_tr % BT):
            self.step_into(Ra, Rb)
            Ra, Rb = Rb, Ra
        R = Ra
        if v is None:
            gen = torch.Generator(device=DEV)
            gen.manual_seed(int(seed))
            v = torch.randn(R.shape, generator=gen, dtype=DT, device=DEV)
        else:
            v = v.clone()
        v /= v.norm(dim=1, keepdim=True)
        R0s = R.clone()
        Qs = R0s + d0 * v
        R1, R2 = torch.empty_like(R), torch.empty_like(R)
        Q1, Q2 = torch.empty_like(R), torch.empty_like(R)
        s1 = torch.zeros(n, d, dtype=DT, device=DEV); s2 = torch.zeros_like(s1); sa = torch.zeros_like(s1)
        xmin = torch.full((n, d), math.inf, dtype=DT, device=DEV); xmax = torch.full((n, d), -math.inf, dtype=DT, device=DEV)
        logsum = torch.zeros(n, dtype=DT, device=DEV)
        drift = torch.zeros(n, dtype=DT, device=DEV)
        x = torch.empty(n, d, dtype=DT, device=DEV)
        d0t = torch.full((n,), d0, dtype=DT, device=DEV)

        def obs_body(track):
            pairs = ((R1, R2, Q1, Q2), (R2, R1, Q2, Q1))
            for k in range(BO):
                Rs, Rd, Qs_, Qd = pairs[k % 2]
                self.step_into(Rs, Rd)
                self.step_into(Qs_, Qd)
                if track:
                    torch.maximum(drift, (Rd - Rs).abs().amax(dim=1), out=drift)
                dQ = Qd - Rd
                dist = dQ.norm(dim=1)
                dist = torch.where(dist > 0, dist, d0t)
                logsum.add_(torch.log(dist / d0))
                torch.addcmul(Rd, dQ, (d0 / dist)[:, None], out=Qd)
                torch.mm(Rd, self.WoT, out=x)
                s1.add_(x); s2.addcmul_(x, x); sa.add_(x.abs())
                torch.minimum(xmin, x, out=xmin); torch.maximum(xmax, x, out=xmax)

        def reset():
            R1.copy_(R0s); Q1.copy_(Qs)
            s1.zero_(); s2.zero_(); sa.zero_(); logsum.zero_(); drift.zero_()
            xmin.fill_(math.inf); xmax.fill_(-math.inf)
        reset()
        g_obs = self._capture(lambda: obs_body(False))
        reset()
        g_last = self._capture(lambda: obs_body(True))
        reset()
        for _ in range(n_obs // BO - 1):
            g_obs.replay()
        g_last.replay()
        R = R1
        mean = s1 / n_obs
        std = torch.sqrt(torch.clamp(s2 / n_obs - mean ** 2, min=0.0))
        fp = drift < fp_tol * (1 + R.abs().amax(dim=1))
        out = {"x_mean": N_(mean), "x_std": N_(std), "x_min": N_(xmin), "x_max": N_(xmax), "x_absmean": N_(sa / n_obs),
               "lam": N_(logsum / (n_obs * self.h)), "fp": N_(fp), "final": N_(R)}
        del g_tr, g_obs, g_last
        return out

    def _run_eager(self, R0, n_tr, n_obs, d0, seed, fp_tol, v=None):
        n = R0.shape[0]
        pc = pacer()
        pc.start()
        R = R0.clone()
        Rb = torch.empty_like(R)
        for k in range(n_tr):
            self.step_into(R, Rb)
            R, Rb = Rb, R
            if (k + 1) % 50 == 0:
                pc.tick()
        if v is None:
            g = torch.Generator(device=DEV)
            g.manual_seed(int(seed))
            v = torch.randn(R.shape, generator=g, dtype=DT, device=DEV)
        else:
            v = v.clone()
        v /= v.norm(dim=1, keepdim=True)
        Q = R + d0 * v
        Rn, Qn = torch.empty_like(R), torch.empty_like(R)
        d = self.d
        s1 = torch.zeros(n, d, dtype=DT, device=DEV); s2 = torch.zeros_like(s1); sa = torch.zeros_like(s1)
        xmin = torch.full((n, d), math.inf, dtype=DT, device=DEV); xmax = torch.full((n, d), -math.inf, dtype=DT, device=DEV)
        logsum = torch.zeros(n, dtype=DT, device=DEV)
        drift = torch.zeros(n, dtype=DT, device=DEV)
        x = torch.empty(n, d, dtype=DT, device=DEV)
        for k in range(n_obs):
            self.step_into(R, Rn)
            self.step_into(Q, Qn)
            if k >= n_obs - 50:
                drift = torch.maximum(drift, (Rn - R).abs().amax(dim=1))
            R, Rn = Rn, R
            Q, Qn = Qn, Q
            dQ = Q - R
            dist = dQ.norm(dim=1)
            dist = torch.where(dist > 0, dist, torch.full_like(dist, d0))
            logsum += torch.log(dist / d0)
            torch.addcmul(R, dQ, (d0 / dist)[:, None], out=Q)
            torch.mm(R, self.WoT, out=x)
            s1 += x; s2 += x * x; sa += x.abs()
            torch.minimum(xmin, x, out=xmin); torch.maximum(xmax, x, out=xmax)
            if (k + 1) % 50 == 0:
                pc.tick()
        mean = s1 / n_obs
        std = torch.sqrt(torch.clamp(s2 / n_obs - mean ** 2, min=0.0))
        fp = drift < fp_tol * (1 + R.abs().amax(dim=1))
        return {"x_mean": N_(mean), "x_std": N_(std), "x_min": N_(xmin), "x_max": N_(xmax), "x_absmean": N_(sa / n_obs),
                "lam": N_(logsum / (n_obs * self.h)), "fp": N_(fp), "final": N_(R)}

    def samples(self, r0, n, burn=0):
        R = (r0 if torch.is_tensor(r0) else T(r0)).reshape(1, -1)
        R = self.advance(R, burn)
        X = torch.empty(n, self.d, dtype=DT, device=DEV)
        Rn = torch.empty_like(R)
        for k in range(n):
            self.step_into(R, Rn)
            R, Rn = Rn, R
            X[k] = (R @ self.WoT)[0]
        return N_(X)

    def orbit(self, r0, n, every=1, burn=0):
        """Состояния орбиты (n//every, N) после burn шагов."""
        R = (r0 if torch.is_tensor(r0) else T(r0)).reshape(1, -1)
        R = self.advance(R, burn)
        S = []
        Rn = torch.empty_like(R)
        for k in range(n):
            self.step_into(R, Rn)
            R, Rn = Rn, R
            if k % every == 0:
                S.append(R[0].clone())
        return torch.stack(S)

    # ------------------------------------------------------------------ неподвижные точки
    def jac(self, r):
        r = T(r) if not torch.is_tensor(r) else r
        z = self.A @ r + self.b
        if self.kind == "tanh":
            dphi = 1 - torch.tanh(z) ** 2
        else:
            dphi = (z.abs() < 1).to(DT)
        return dphi[:, None] * self.A

    def stability(self, r):
        w, V = torch.linalg.eig(self.jac(r))
        return N_(w), N_(V)

    def unstable_directions(self, r, tol=1e-9, max_dirs=20):
        w, V = self.stability(r)
        dirs = []
        for k in np.argsort(-np.abs(w)):
            if np.abs(w[k]) <= 1 + tol:
                break
            for comp in (np.real(V[:, k]), np.imag(V[:, k])):
                nrm = np.linalg.norm(comp)
                if nrm > 1e-12:
                    dirs.append(comp / nrm)
        return w, dirs[:max_dirs]

    def newton(self, X0, max_iter=40, tol=1e-12, chunk=512, deflate=None, power=2.0, shift=1.0):
        """Пакетный метод Ньютона (tanh) или метод Ньютона по областям (hard-tanh). Возвращает (X, ok) numpy.
        deflate — список известных точек (numpy, k×N) для дефляции Farrell et al. (только tanh)."""
        X0 = T(X0)
        Xs, oks = [], []
        K = T(np.asarray(deflate)) if deflate is not None and len(deflate) else None
        chunk = max(16, min(chunk, int(3e8 / (self.N * self.N * 8))))      # ~0.3 ГБ на пачку якобианов
        for i in range(0, X0.shape[0], chunk):
            if self.kind == "tanh":
                X, ok = self._newton_tanh(X0[i:i + chunk].clone(), max_iter, tol, K, power, shift)
            else:
                X, ok = self._newton_pw(X0[i:i + chunk].clone(), max_iter)
            Xs.append(X); oks.append(ok)
        return np.concatenate(Xs), np.concatenate(oks)

    def _newton_tanh(self, X, max_iter, tol, K, power, shift):
        n, N = X.shape
        I = torch.eye(N, dtype=DT, device=DEV)
        ok = torch.zeros(n, dtype=torch.bool, device=DEV)
        bad = torch.zeros(n, dtype=torch.bool, device=DEV)
        for _ in range(max_iter):
            t = torch.tanh(torch.addmm(self.b, X, self.AT))
            g = X - t
            ok |= (g.norm(dim=1) < tol) & ~bad
            act = ~ok & ~bad
            if not act.any():
                break
            ia = act.nonzero().squeeze(1)
            J = torch.mul((1 - t[ia] ** 2)[:, :, None], self.A[None])
            J.neg_()
            J.diagonal(dim1=1, dim2=2).add_(1.0)
            sol, info = batched_solve(J, g[ia][:, :, None])
            del J
            dx = -sol[:, :, 0]
            if K is not None:
                diff = X[ia][:, None, :] - K[None]                       # (m, k, N)
                nr = diff.norm(dim=2)                                     # (m, k)
                mk = nr ** (-power) + shift
                grad = ((-power * nr ** (-power - 2) / mk)[:, :, None] * diff).sum(1)   # ∇log M
                denom = 1.0 - (grad * dx).sum(1)
                dx = torch.where(denom.abs()[:, None] > 1e-14, dx / denom[:, None], dx)
            Xn = X[ia] + dx
            fin = torch.isfinite(Xn).all(1) & (info == 0)
            X[ia] = torch.where(fin[:, None], Xn, X[ia])
            bad[ia[~fin]] = True
        return N_(X), N_(ok)

    def _newton_pw(self, X, max_iter):
        n, N = X.shape
        I = torch.eye(N, dtype=DT, device=DEV)
        prev = None
        ok = torch.zeros(n, dtype=torch.bool, device=DEV)
        done = torch.zeros(n, dtype=torch.bool, device=DEV)
        for _ in range(max_iter):
            z = torch.addmm(self.b, X, self.AT)
            sig = torch.where(z >= 1, 1.0, torch.where(z <= -1, -1.0, 0.0)).to(DT)
            if prev is not None:
                same = (sig == prev).all(1) & ~done
                res = (torch.clamp(z, -1, 1) - X).norm(dim=1) < 1e-10 * (1 + X.norm(dim=1))
                ok |= same & res
                done |= same
            if done.all():
                break
            prev = sig
            D = (sig == 0).to(DT)                                         # линейные нейроны
            M = torch.mul(D[:, :, None], self.A[None])
            M.neg_()
            M.diagonal(dim1=1, dim2=2).add_(1.0)
            rhs = D * self.b[None] + (1 - D) * sig
            sol, info = torch.linalg.solve_ex(M, rhs[:, :, None])
            del M
            Xn = sol[:, :, 0]
            upd = ~done & (info == 0) & torch.isfinite(Xn).all(1)
            X = torch.where(upd[:, None], Xn, X)
        return N_(X), N_(ok)


def batched_solve(M, rhs):
    """Пачка систем M x = rhs → (решения, info). На CPU — numpy (LAPACK): torch на CPU бросает исключения и пишет
    ошибки oneMKL (DLASWP) на вырожденных матрицах и на некоторых размерах пачки, из-за чего метод Ньютона при
    N = 150 бросал большую часть стартов. На GPU — torch.linalg.solve_ex."""
    if M.device.type == "cpu":
        Mn, rn = M.numpy(), rhs.numpy()
        try:
            return torch.from_numpy(np.linalg.solve(Mn, rn)), torch.zeros(M.shape[0], dtype=torch.int32)
        except np.linalg.LinAlgError:
            sol = np.zeros_like(rn)
            info = np.zeros(M.shape[0], np.int32)
            for k in range(M.shape[0]):
                try:
                    sol[k] = np.linalg.solve(Mn[k], rn[k])
                except np.linalg.LinAlgError:
                    info[k] = 1
            return torch.from_numpy(sol), torch.from_numpy(info)
    try:
        return torch.linalg.solve_ex(M, rhs)
    except RuntimeError:
        return solve_each(M, rhs)


def solve_each(M, rhs):
    """Решение пачки систем по одной: вырожденные получают info = 1 (как solve_ex)."""
    sol = torch.zeros_like(rhs)
    info = torch.zeros(M.shape[0], dtype=torch.int32, device=M.device)
    for k in range(M.shape[0]):
        try:
            sol[k] = torch.linalg.solve(M[k], rhs[k])
        except RuntimeError:
            info[k] = 1
    return sol, info


def hardtanh_enumerate(A, b, chunk=200_000, tol=1e-12):
    """Все неподвижные точки r = clip(A r + b) перебором 3^N областей на GPU (полная система N×N в каждой области).
    Возвращает (точки numpy k×N, области k×N, число вырожденных систем)."""
    A_ = T(A); b_ = T(b)
    N = A.shape[0]
    total = 3 ** N
    I = torch.eye(N, dtype=DT, device=DEV)
    pw = torch.tensor([3 ** i for i in range(N)], dtype=torch.int64, device=DEV)
    pts, regs, degenerate = [], [], 0
    for s0 in range(0, total, chunk):
        idx = torch.arange(s0, min(total, s0 + chunk), dtype=torch.int64, device=DEV)
        digits = (idx[:, None] // pw[None]) % 3                           # 0, 1, 2 -> -1, 0, +1
        sig = (digits - 1).to(DT)
        D = (sig == 0).to(DT)
        M = I[None] - D[:, :, None] * A_[None]
        rhs = D * b_[None] + (1 - D) * sig
        sol, info = torch.linalg.solve_ex(M, rhs[:, :, None])
        x = sol[:, :, 0]
        degenerate += int((info != 0).sum())
        z = x @ A_.T + b_
        lin_ok = ((D * x).abs() <= 1 + tol).all(1)
        sat_ok = (((1 - D) * sig * z) >= (1 - D) * (1 - tol)).all(1)
        res_ok = (torch.clamp(z, -1, 1) - x).norm(dim=1) <= 1e-9 * (1 + x.norm(dim=1))
        good = (info == 0) & lin_ok & sat_ok & res_ok & torch.isfinite(x).all(1)
        if good.any():
            pts.append(N_(x[good])); regs.append(N_(sig[good]).astype(np.int8))
    if not pts:
        return np.zeros((0, N)), np.zeros((0, N), np.int8), degenerate
    P = np.concatenate(pts); S = np.concatenate(regs)
    # слияние точек на стыках областей
    keep, U = [], []
    for i, x in enumerate(P):
        if all(np.linalg.norm(x - y) > 1e-9 * (1 + np.linalg.norm(y)) for y in U):
            U.append(x); keep.append(i)
    return P[keep], S[keep], degenerate


def fatal_cuda(exc):
    """Ошибка CUDA портит контекст процесса: дальнейшие модели в этом процессе считать нельзя."""
    msg = f"{type(exc).__name__}: {exc}"
    return "CUDA error" in msg or "CUBLAS_STATUS" in msg or "AcceleratorError" in msg or "illegal memory" in msg


def report_progress(done, total, note=""):
    p = os.environ.get("GPUQ_OUT")
    if p:
        try:
            with open(os.path.join(p, "progress.txt"), "w", encoding="utf-8") as f:
                f.write(f"{done}/{total} {note}\n")
        except OSError:
            pass

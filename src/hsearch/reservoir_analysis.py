"""Анализ замкнутого резервуарного вычислителя r_{k+1} = tanh(A r_k + b) (этап 3).

Пакетное моделирование на BLAS (R @ A.T для n траекторий сразу), отпечатки в пространстве
выхода u = W_out r (3 координаты Лоренца, нормированные), показатель Ляпунова методом двух
траекторий, неподвижные точки методом Ньютона, их устойчивость и неустойчивые многообразия.
"""
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _tanh_bias_inplace(Z, b):
    n, N = Z.shape
    for i in prange(n):
        for j in range(N):
            Z[i, j] = np.tanh(Z[i, j] + b[j])
    return Z


class ClosedLoopESN:
    def __init__(self, esn, h=0.02, fast=False):
        self.A = np.ascontiguousarray(esn["A"])
        self.AT = np.ascontiguousarray(self.A.T)
        self.b = np.ascontiguousarray(esn["b"], dtype=np.float64)
        self.W_out = np.asarray(esn["W_out"])
        self.N = self.A.shape[0]
        self.h = float(h)
        self.R_train = esn.get("R_train")
        self.fast = bool(fast)     # параллельный tanh (numba) для больших пакетов

    def step(self, R):
        if self.fast and R.ndim == 2 and R.shape[0] >= 64:
            return _tanh_bias_inplace(np.ascontiguousarray(R @ self.AT), self.b)
        return np.tanh(R @ self.AT + self.b)

    def output(self, R):
        return R @ self.W_out.T

    def run(self, R0, n_tr=4000, n_obs=2000, d0=1e-7, noise=0.0, rng=None, n_early=500, fp_tol=1e-9):
        """Возвращает словарь отпечатков (n траекторий):
        u_mean/u_std/u_min/u_max (n, 3) — по окну наблюдения; lam (n,) — на единицу времени;
        fp (n,) — сошлась к неподвижной точке; e_mean/e_std (n, 3) — по первым n_early шагам;
        final (n, N) — конечное состояние; drift (n,) — max |r_{k+1} - r_k| на последних 50 шагах."""
        rng = np.random.default_rng(0) if rng is None else rng
        R = np.array(R0, float, copy=True)
        n = len(R)
        es = np.zeros((n, 3)); eq = np.zeros((n, 3))
        for k in range(n_tr):
            R = self.step(R)
            if noise:
                R += noise * rng.normal(size=R.shape)
            if k < n_early:
                u = self.output(R)
                es += u; eq += u * u
        ne = min(n_tr, n_early)
        e_mean = es / max(ne, 1)
        e_std = np.sqrt(np.maximum(eq / max(ne, 1) - e_mean ** 2, 0.0))
        v = rng.normal(size=R.shape)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        Q = R + d0 * v
        s1 = np.zeros((n, 3)); s2 = np.zeros((n, 3))
        umin = np.full((n, 3), np.inf); umax = np.full((n, 3), -np.inf)
        logsum = np.zeros(n)
        drift = np.zeros(n)
        for k in range(n_obs):
            Rn = self.step(R)
            Q = self.step(Q)
            if k >= n_obs - 50:
                drift = np.maximum(drift, np.abs(Rn - R).max(axis=1))
            R = Rn
            if noise:
                R += noise * rng.normal(size=R.shape)
            dQ = Q - R
            dist = np.linalg.norm(dQ, axis=1)
            dist = np.where(dist > 0, dist, d0)
            logsum += np.log(dist / d0)
            Q = R + dQ * (d0 / dist)[:, None]
            u = self.output(R)
            s1 += u; s2 += u * u
            umin = np.minimum(umin, u); umax = np.maximum(umax, u)
        u_mean = s1 / n_obs
        u_std = np.sqrt(np.maximum(s2 / n_obs - u_mean ** 2, 0.0))
        lam = logsum / (n_obs * self.h)
        return {"u_mean": u_mean, "u_std": u_std, "u_min": umin, "u_max": umax, "lam": lam,
                "fp": drift < fp_tol * (1 + np.abs(R).max(axis=1)), "drift": drift,
                "e_mean": e_mean, "e_std": e_std, "final": R}

    # --- неподвижные точки
    def fixed_points(self, starts, max_iter=60, tol=1e-12, dedupe_tol=1e-6):
        found = []
        I = np.eye(self.N)
        for r in np.asarray(starts, float):
            x = r.copy()
            ok = False
            for _ in range(max_iter):
                t = np.tanh(self.A @ x + self.b)
                g = x - t
                if np.linalg.norm(g) < tol:
                    ok = True
                    break
                J = I - (1.0 - t * t)[:, None] * self.A
                try:
                    dx = np.linalg.solve(J, -g)
                except np.linalg.LinAlgError:
                    break
                x = x + dx
                if not np.all(np.isfinite(x)):
                    break
            if ok and all(np.linalg.norm(x - y) > dedupe_tol * (1 + np.linalg.norm(y)) for y in found):
                found.append(x)
        return found

    def jacobian(self, r):
        t = np.tanh(self.A @ r + self.b)
        return (1.0 - t * t)[:, None] * self.A

    def stability(self, r):
        w, V = np.linalg.eig(self.jacobian(r))
        return w, V

    def unstable_directions(self, r, tol=1e-9):
        w, V = self.stability(r)
        dirs = []
        for k in np.where(np.abs(w) > 1 + tol)[0]:
            v = V[:, k]
            for comp in (np.real(v), np.imag(v)):
                nrm = np.linalg.norm(comp)
                if nrm > 1e-12:
                    dirs.append(comp / nrm)
        return w, dirs


def classify_runs(out, lorenz_ref, lam_chaos=0.05, tol=6.0, floor_rel=0.05):
    """Метки: 'lorenz' — совпадение с эталоном реконструированного аттрактора по (λ, статистикам u);
    'fp' — неподвижная точка; 'other_chaos' / 'other_periodic' — прочие режимы."""
    F = np.column_stack([out["u_mean"], out["u_std"], out["u_min"], out["u_max"]])
    mu, sd = lorenz_ref
    z = np.max(np.abs(F - mu) / (sd + floor_rel * (1 + np.abs(mu))), axis=1)
    lab = np.where(out["fp"], "fp", np.where(z < tol, "lorenz", np.where(out["lam"] > lam_chaos, "other_chaos", "other_periodic")))
    return lab, z


def lorenz_reference(esn_model, rng, n_copies=48, n_tr=2000, n_obs=2000):
    """Эталон реконструированного аттрактора: копии с точек обучающей траектории."""
    R = esn_model.R_train
    picks = R[rng.integers(len(R) // 2, len(R), size=n_copies)]
    out = esn_model.run(picks, n_tr=n_tr, n_obs=n_obs, rng=rng)
    F = np.column_stack([out["u_mean"], out["u_std"], out["u_min"], out["u_max"]])
    keep = ~out["fp"]
    return (F[keep].mean(axis=0), F[keep].std(axis=0)), out

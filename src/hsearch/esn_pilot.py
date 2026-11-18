"""Эталонные CPU-реализации для пилота эхо-сетей (задание на расчёты на A100).

Функции задают точный смысл всех величин задания; GPU-реализация сверяется с ними тестами. Скорость не важна.

Модели:
- train_esn_multi — эхо-сеть с обучением по нескольким рядам (при одном ряде совпадает с systems.reservoir.train_esn),
  активация tanh или hard-tanh, вариант b = 0 с теми же W и W_in;
- ClosedLoop — замкнутая сеть r_{k+1} = φ(A r_k + b), A = s·W + W_in W_out, отпечатки траекторий в пространстве выхода;
- lift_drive, lift_const — подъём физического начального условия в резервуар (L1, L2);
- NGRC — NG-RC с двумя задержками и квадратичными мономами, x_{n+1} = x_n + W Φ(x_n, x_{n-1});
- RohmESN — непрерывная эхо-сеть Röhm, Gauthier, Fischer 2021 (Chaos 31, 103127), приложения A и C.
Неподвижные точки: newton_tanh (пакетный метод Ньютона), deflated_newton (дефляция Farrell et al. 2015),
hardtanh_fixed_points_bruteforce (полный перебор 3^N областей), piecewise_newton (метод Ньютона для hard-tanh).
Метрики: delta_att (Röhm et al., формулы 8–10), hausdorff, sliced_w1. Проверка скрытости: radius_points,
bisect_boundary, missing_mass_bounds.
"""
import itertools

import numpy as np
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------------------------------------------
# Активации
# ----------------------------------------------------------------------------------------------------------

def act(z, kind):
    if kind == "tanh":
        return np.tanh(z)
    if kind == "hardtanh":
        return np.clip(z, -1.0, 1.0)
    raise ValueError(kind)


def dact_from_out(t, z, kind):
    """Производная активации; t = φ(z)."""
    if kind == "tanh":
        return 1.0 - t * t
    return (np.abs(z) < 1.0).astype(float)


# ----------------------------------------------------------------------------------------------------------
# Эхо-сеть: обучение по нескольким рядам
# ----------------------------------------------------------------------------------------------------------

def train_esn_multi(series, n_res=300, spectral_radius=1.1, input_scale=0.5, density=0.05, ridge=1e-6, washout=500,
                    seed=0, activation="tanh", zero_bias=False, scale=None):
    """series — список рядов (T_j, m) в физических координатах. Порядок случайных чисел тот же, что у
    systems.reservoir.train_esn: W (нормальные × маска плотности), W_in, b. При zero_bias = True b вычисляется
    и обнуляется, так что W и W_in совпадают с вариантом со случайным b.
    Нормировка: u = x / scale, scale — стандартное отклонение по всем рядам вместе (если не задано).
    Каждый ряд прогоняется от r = 0; первые washout состояний ряда в регрессию не входят (washout — число или список
    по рядам); пары (r_k, u_{k+1}) берутся внутри каждого ряда."""
    rng = np.random.default_rng(seed)
    series = [np.asarray(s, float) for s in series]
    m = series[0].shape[1]
    W = rng.normal(size=(n_res, n_res)) * (rng.random((n_res, n_res)) < density)
    ev = np.max(np.abs(np.linalg.eigvals(W)))
    W *= spectral_radius / ev
    W_in = rng.uniform(-input_scale, input_scale, size=(n_res, m))
    b = rng.uniform(-0.2, 0.2, size=n_res)
    if zero_bias:
        b = np.zeros(n_res)
    if scale is None:
        scale = np.vstack(series).std(axis=0)
    Xs, Ys, Rs = [], [], []
    washes = list(washout) if np.ndim(washout) else [int(washout)] * len(series)
    for s, washout in zip(series, washes):
        un = s / scale
        R = np.zeros((len(un), n_res))
        r = np.zeros(n_res)
        for k in range(len(un)):
            r = act(W @ r + W_in @ un[k] + b, activation)
            R[k] = r
        Xs.append(R[washout:-1])
        Ys.append(un[washout + 1:])
        Rs.append(R[washout:])
    X = np.vstack(Xs)
    Y = np.vstack(Ys)
    W_out = np.linalg.solve(X.T @ X + ridge * np.eye(n_res), X.T @ Y).T
    train_err = np.sqrt(np.mean((X @ W_out.T - Y) ** 2))
    return {"W": W, "W_in": W_in, "W_out": W_out, "b": b, "A": W + W_in @ W_out, "scale": np.asarray(scale, float),
            "train_rmse": float(train_err), "R_train": np.vstack(Rs), "n_res": n_res, "activation": activation}


def lift_drive(esn, U_phys, w):
    """L1: прогрев истинным рядом. r = 0, затем w входов u_0 … u_{w-1} (физические отсчёты истинной траектории из
    начального условия, u_0 — само начальное условие). Возвращает состояние, с которого запускается замкнутый цикл.
    U_phys: (n, w, m) или (w, m)."""
    U = np.asarray(U_phys, float)
    single = U.ndim == 2
    if single:
        U = U[None]
    R = np.zeros((U.shape[0], esn["n_res"]))
    for k in range(w):
        R = act(R @ esn["W"].T + (U[:, k] / esn["scale"]) @ esn["W_in"].T + esn["b"], esn["activation"])
    return R[0] if single else R


def lift_const(esn, x0_phys, w=100, R0=None):
    """L2: постоянный вход. r = R0 (по умолчанию 0), затем w шагов r = φ(W r + W_in x0/scale + b)."""
    X0 = np.atleast_2d(np.asarray(x0_phys, float))
    R = np.zeros((len(X0), esn["n_res"])) if R0 is None else np.array(R0, float, copy=True)
    drive = (X0 / esn["scale"]) @ esn["W_in"].T + esn["b"]
    for _ in range(w):
        R = act(R @ esn["W"].T + drive, esn["activation"])
    return R


class ClosedLoop:
    """Замкнутая сеть r_{k+1} = φ(A r_k + b), A = s·W + W_in W_out (s — множитель W после обучения, по умолчанию 1).
    h — шаг выборки обучающего ряда (единица времени на итерацию). Выход в физических координатах: x = scale·W_out r."""

    def __init__(self, esn, h, s=1.0):
        self.W_out = np.asarray(esn["W_out"])
        self.A = s * np.asarray(esn["W"]) + np.asarray(esn["W_in"]) @ self.W_out
        self.b = np.asarray(esn["b"], float)
        self.scale = np.asarray(esn["scale"], float)
        self.kind = esn.get("activation", "tanh")
        self.N = self.A.shape[0]
        self.d = self.W_out.shape[0]
        self.h = float(h)

    def step(self, R):
        return act(R @ self.A.T + self.b, self.kind)

    def output(self, R):
        return (R @ self.W_out.T) * self.scale

    def run(self, R0, n_tr, n_obs, d0=1e-7, rng=None, fp_tol=1e-9):
        """Отпечатки (как ClosedLoopESN.run из reservoir_analysis): по окну n_obs — x_mean, x_std, x_min, x_max
        (физические координаты выхода), x_absmean; lam — наибольший показатель Ляпунова на единицу времени методом
        двух траекторий (начальное отклонение d0, перенормировка каждый шаг); fp — сошлась к неподвижной точке
        (max|r_{k+1} − r_k| на последних 50 шагах < fp_tol·(1 + max|r|)); final — конечное состояние."""
        rng = np.random.default_rng(0) if rng is None else rng
        R = np.array(R0, float, copy=True)
        n = len(R)
        for _ in range(n_tr):
            R = self.step(R)
        v = rng.normal(size=R.shape)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        Q = R + d0 * v
        s1 = np.zeros((n, self.d)); s2 = np.zeros((n, self.d)); sa = np.zeros((n, self.d))
        xmin = np.full((n, self.d), np.inf); xmax = np.full((n, self.d), -np.inf)
        logsum = np.zeros(n); drift = np.zeros(n)
        for k in range(n_obs):
            Rn = self.step(R)
            Q = self.step(Q)
            if k >= n_obs - 50:
                drift = np.maximum(drift, np.abs(Rn - R).max(axis=1))
            R = Rn
            dQ = Q - R
            dist = np.linalg.norm(dQ, axis=1)
            dist = np.where(dist > 0, dist, d0)
            logsum += np.log(dist / d0)
            Q = R + dQ * (d0 / dist)[:, None]
            x = self.output(R)
            s1 += x; s2 += x * x; sa += np.abs(x)
            xmin = np.minimum(xmin, x); xmax = np.maximum(xmax, x)
        mean = s1 / n_obs
        return {"x_mean": mean, "x_std": np.sqrt(np.maximum(s2 / n_obs - mean ** 2, 0.0)), "x_min": xmin,
                "x_max": xmax, "x_absmean": sa / n_obs, "lam": logsum / (n_obs * self.h),
                "fp": drift < fp_tol * (1 + np.abs(R).max(axis=1)), "final": R}

    def samples(self, r0, n, burn=0):
        """n отсчётов выхода (физические координаты) с траектории из r0 после burn шагов."""
        R = np.array(r0, float)[None, :]
        for _ in range(burn):
            R = self.step(R)
        X = np.empty((n, self.d))
        for k in range(n):
            R = self.step(R)
            X[k] = self.output(R)[0]
        return X

    def jacobian(self, r):
        z = self.A @ r + self.b
        t = act(z, self.kind)
        return dact_from_out(t, z, self.kind)[:, None] * self.A

    def unstable_directions(self, r, tol=1e-9, max_dirs=20):
        w, V = np.linalg.eig(self.jacobian(r))
        dirs = []
        for k in np.argsort(-np.abs(w)):
            if np.abs(w[k]) <= 1 + tol:
                break
            for comp in (np.real(V[:, k]), np.imag(V[:, k])):
                nrm = np.linalg.norm(comp)
                if nrm > 1e-12:
                    dirs.append(comp / nrm)
        return w, dirs[:max_dirs]


def newton_tanh(model, starts, max_iter=60, tol=1e-12):
    """Метод Ньютона для r = φ(A r + b) (φ = tanh) из каждого старта независимо. Возвращает (точки, сошлись)."""
    I = np.eye(model.N)
    out = np.array(starts, float, copy=True)
    ok = np.zeros(len(out), bool)
    for i in range(len(out)):
        x = out[i]
        for _ in range(max_iter):
            t = np.tanh(model.A @ x + model.b)
            g = x - t
            if np.linalg.norm(g) < tol:
                ok[i] = True
                break
            J = I - (1.0 - t * t)[:, None] * model.A
            try:
                x = x - np.linalg.solve(J, g)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(x)):
                break
        out[i] = x
    return out, ok


def deflated_newton(model, starts, known, power=2.0, shift=1.0, max_iter=100, tol=1e-12):
    """Дефляция (Farrell, Birkisson, Funke 2015): метод Ньютона для G(x) = M(x)·F(x), F = x − tanh(Ax + b),
    M(x) = Π_k (1/|x − x_k|^power + shift) по уже известным решениям known. Старты независимы друг от друга
    (список known фиксирован на время пакета). Возвращает (точки, сошлись по |F| < tol)."""
    I = np.eye(model.N)
    K = [np.asarray(k, float) for k in known]
    out = np.array(starts, float, copy=True)
    ok = np.zeros(len(out), bool)
    for i in range(len(out)):
        x = out[i]
        for _ in range(max_iter):
            t = np.tanh(model.A @ x + model.b)
            F = x - t
            if np.linalg.norm(F) < tol:
                ok[i] = True
                break
            J = I - (1.0 - t * t)[:, None] * model.A
            # шаг Ньютона для G = M F: dx = −(M J + F ∇M^T)^{-1} M F; через формулу Шермана–Моррисона
            # эквивалентно dx = τ·dx_N, dx_N = −J^{-1}F, τ = 1 / (1 − (∇M·dx_N)/M)
            try:
                dxN = -np.linalg.solve(J, F)
            except np.linalg.LinAlgError:
                break
            M, gradlogM = 1.0, np.zeros(model.N)
            for xk in K:
                diff = x - xk
                nr = np.linalg.norm(diff)
                if nr == 0:
                    M = np.inf
                    break
                mk = nr ** (-power) + shift
                M *= mk
                gradlogM += (-power * nr ** (-power - 2) * diff) / mk
            if not np.isfinite(M):
                break
            denom = 1.0 - gradlogM @ dxN
            x = x + (dxN / denom if abs(denom) > 1e-14 else dxN)
            if not np.all(np.isfinite(x)):
                break
        out[i] = x
    return out, ok


def dedupe(points, tol=1e-8):
    """Уникальные точки: совпадение, если |x − y| < tol·(1 + |y|). Возвращает (уникальные, индекс каждой точки)."""
    uniq, idx = [], np.empty(len(points), int)
    for i, x in enumerate(points):
        for j, y in enumerate(uniq):
            if np.linalg.norm(x - y) < tol * (1 + np.linalg.norm(y)):
                idx[i] = j
                break
        else:
            uniq.append(np.array(x))
            idx[i] = len(uniq) - 1
    return uniq, idx


# ----------------------------------------------------------------------------------------------------------
# hard-tanh: точный перебор и метод Ньютона по областям
# ----------------------------------------------------------------------------------------------------------

def hardtanh_fixed_points_bruteforce(A, b, tol=1e-12):
    """Все неподвижные точки r = clip(A r + b, −1, 1) полным перебором 3^N областей.
    Область σ ∈ {−1, 0, +1}^N: σ_i = ±1 — нейрон насыщен (r_i = σ_i, σ_i (A r + b)_i ≥ 1), σ_i = 0 — линейный
    (r_i = (A r + b)_i, |r_i| ≤ 1). Для линейного множества S решается (I − A_SS) r_S = A_{S,S̄} σ_S̄ + b_S.
    Возвращает (список точек, список областей, число вырожденных систем). Точки на стыке областей сливаются."""
    N = len(b)
    pts, regs, degenerate = [], [], 0
    for sig in itertools.product((-1, 0, 1), repeat=N):
        sig = np.array(sig)
        S = np.where(sig == 0)[0]
        Sb = np.where(sig != 0)[0]
        r = sig.astype(float)
        if len(S):
            M = np.eye(len(S)) - A[np.ix_(S, S)]
            rhs = A[np.ix_(S, Sb)] @ sig[Sb] + b[S]
            if abs(np.linalg.det(M)) < 1e-14:
                degenerate += 1
                continue
            r[S] = np.linalg.solve(M, rhs)
            if np.any(np.abs(r[S]) > 1 + tol):
                continue
        z = A @ r + b
        if len(Sb) and np.any(sig[Sb] * z[Sb] < 1 - tol):
            continue
        if np.linalg.norm(np.clip(z, -1, 1) - r) > 1e-9 * (1 + np.linalg.norm(r)):
            continue
        pts.append(r)
        regs.append(sig)
    uniq, idx = dedupe(pts, tol=1e-9)
    regs_u = [regs[int(np.where(idx == j)[0][0])] for j in range(len(uniq))]
    return uniq, regs_u, degenerate


def piecewise_newton(A, b, x0, max_iter=50):
    """Метод Ньютона для hard-tanh: по текущей точке определяется область (σ_i = sign(z_i) при |z_i| ≥ 1, иначе 0),
    в ней решается аффинное уравнение неподвижной точки; повтор, пока область не перестанет меняться.
    Возвращает (точка, сошлась)."""
    x = np.asarray(x0, float).copy()
    N = len(b)
    prev = None
    for _ in range(max_iter):
        z = A @ x + b
        sig = np.where(z >= 1, 1, np.where(z <= -1, -1, 0))
        if prev is not None and np.array_equal(sig, prev):
            ok = np.linalg.norm(np.clip(A @ x + b, -1, 1) - x) < 1e-10 * (1 + np.linalg.norm(x))
            return x, bool(ok)
        prev = sig
        S = np.where(sig == 0)[0]
        Sb = np.where(sig != 0)[0]
        r = sig.astype(float)
        if len(S):
            M = np.eye(len(S)) - A[np.ix_(S, S)]
            try:
                r[S] = np.linalg.solve(M, A[np.ix_(S, Sb)] @ sig[Sb] + b[S])
            except np.linalg.LinAlgError:
                return x, False
        x = r
    return x, False


# ----------------------------------------------------------------------------------------------------------
# NG-RC
# ----------------------------------------------------------------------------------------------------------

class NGRC:
    """NG-RC (Gauthier et al. 2021) с k = 2 задержками и шагом задержки 1: линейная часть O = [x_n, x_{n−1}] (2d),
    нелинейная — все мономы O_i O_j, i ≤ j, постоянный член 1. Модель x_{n+1} = x_n + W Φ(x_n, x_{n−1}).
    Нормировка x/scale (scale — стандартное отклонение по обучающим рядам). Ридж по всем признакам."""

    def __init__(self, series, ridge=1e-6, scale=None):
        series = [np.asarray(s, float) for s in series]
        self.d = series[0].shape[1]
        self.scale = np.vstack(series).std(axis=0) if scale is None else np.asarray(scale, float)
        F, Y = [], []
        for s in series:
            u = s / self.scale
            F.append(self.features(u[1:-1], u[:-2]))
            Y.append(u[2:] - u[1:-1])
        F = np.vstack(F)
        Y = np.vstack(Y)
        self.W = np.linalg.solve(F.T @ F + ridge * np.eye(F.shape[1]), F.T @ Y).T
        self.train_rmse = float(np.sqrt(np.mean((F @ self.W.T - Y) ** 2)))
        self.iu = np.triu_indices(2 * self.d)

    def features(self, xn, xm):
        O = np.hstack([xn, xm])
        iu = np.triu_indices(O.shape[1])
        Q = (O[:, :, None] * O[:, None, :])[:, iu[0], iu[1]]
        return np.hstack([np.ones((len(O), 1)), O, Q])

    def step(self, Xn, Xm):
        """Один шаг в нормированных координатах; состояния (n, d)."""
        return Xn + self.features(Xn, Xm) @ self.W.T, Xn

    def equilibria(self, starts, max_iter=60, tol=1e-12, h=1e-7):
        """Равновесия x* (нормированные): W Φ(x*, x*) = 0, метод Ньютона с конечно-разностным якобианом."""
        out, ok = [], []
        for x in np.asarray(starts, float):
            x = x.copy()
            conv = False
            for _ in range(max_iter):
                g = self.features(x[None], x[None])[0] @ self.W.T
                if np.linalg.norm(g) < tol:
                    conv = True
                    break
                J = np.empty((self.d, self.d))
                for j in range(self.d):
                    e = x.copy(); e[j] += h
                    J[:, j] = (self.features(e[None], e[None])[0] @ self.W.T - g) / h
                try:
                    x = x - np.linalg.solve(J, g)
                except np.linalg.LinAlgError:
                    break
                if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e6:
                    break
            out.append(x)
            ok.append(conv)
        return np.array(out), np.array(ok)

    def delay_jacobian(self, x, h=1e-7):
        """Якобиан отображения (x_n, x_{n−1}) -> (x_{n+1}, x_n) в равновесии (x, x), размер 2d × 2d."""
        d = self.d
        J = np.zeros((2 * d, 2 * d))
        base = self.step(x[None], x[None])[0][0]
        for j in range(2 * d):
            a, c = x.copy(), x.copy()
            if j < d:
                a[j] += h
            else:
                c[j - d] += h
            J[:d, j] = (self.step(a[None], c[None])[0][0] - base) / h
        J[d:, :d] = np.eye(d)
        return J


# ----------------------------------------------------------------------------------------------------------
# Непрерывная эхо-сеть Röhm et al. 2021
# ----------------------------------------------------------------------------------------------------------

class RohmESN:
    """dX/dt = −X + tanh(W X + G W_in u(t) + B), u(t) кусочно-постоянный с интервалом θ; RK4 с шагом 0.1.
    W: разреженная доля rho_s, значения U(−1, 1), затем W = re_max · W / Re(Λ)_max; W_in: N × m, в каждой строке один
    ненулевой элемент U(−1, 1) в случайном столбце; B ~ U(−b, b). Порядок случайных чисел: маска W, значения W,
    столбцы W_in, значения W_in, B."""

    def __init__(self, m, N=300, rho_s=0.1, G=0.3, b=1.0, re_max=0.95, theta=2.5, dt=0.1, seed=0):
        rng = np.random.default_rng(seed)
        mask = rng.random((N, N)) < rho_s
        W = rng.uniform(-1, 1, size=(N, N)) * mask
        lam = np.max(np.real(np.linalg.eigvals(W)))
        self.W = re_max * W / lam
        cols = rng.integers(0, m, size=N)
        vals = rng.uniform(-1, 1, size=N)
        self.W_in = np.zeros((N, m))
        self.W_in[np.arange(N), cols] = vals
        self.B = rng.uniform(-b, b, size=N)
        self.G, self.theta, self.dt, self.N, self.m = G, theta, dt, N, m
        self.n_sub = int(round(theta / dt))
        self.W_out = None

    def _rhs(self, X, u):
        return -X + np.tanh(X @ self.W.T + self.G * (u @ self.W_in.T) + self.B)

    def advance(self, X, u):
        """Интервал θ с постоянным входом u (n, m); X (n, N)."""
        h = self.dt
        for _ in range(self.n_sub):
            k1 = self._rhs(X, u)
            k2 = self._rhs(X + 0.5 * h * k1, u)
            k3 = self._rhs(X + 0.5 * h * k2, u)
            k4 = self._rhs(X + h * k3, u)
            X = X + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return X

    def initial_state(self, n=1, t_free=300.0):
        """Состояние после t_free единиц времени без входа (u = 0) из X = 0."""
        X = np.zeros((n, self.N))
        u = np.zeros((n, self.m))
        h = self.dt
        for _ in range(int(round(t_free / h))):
            k1 = self._rhs(X, u)
            k2 = self._rhs(X + 0.5 * h * k1, u)
            k3 = self._rhs(X + 0.5 * h * k2, u)
            k4 = self._rhs(X + h * k3, u)
            X = X + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return X

    def train(self, u, eta=1e-3, washout=1000):
        """u (K, m): отсчёты 0 … K−1; состояния после входов washout … K−2, цели u_{k+1}. Ридж с η·I_{N+1}."""
        X = self.initial_state(1)
        S = []
        for k in range(len(u) - 1):
            X = self.advance(X, u[k][None])
            if k >= washout:
                S.append(np.append(X[0], 1.0))
        S = np.array(S)
        Y = u[washout + 1:]
        self.W_out = np.linalg.solve(S.T @ S + eta * np.eye(self.N + 1), S.T @ Y)
        self.train_rmse = float(np.sqrt(np.mean((S @ self.W_out - Y) ** 2)))
        return self

    def autonomous(self, u_warm, n_steps):
        """Прогрев истинными отсчётами u_warm (n, w, m) от состояния initial_state, затем n_steps шагов с
        собственным выходом. Возвращает (n, n_steps, m)."""
        U = np.asarray(u_warm, float)
        n, w, _ = U.shape
        X = np.repeat(self.initial_state(1), n, axis=0)
        for k in range(w):
            X = self.advance(X, U[:, k])
        out = np.empty((n, n_steps, self.m))
        for k in range(n_steps):
            y = np.hstack([X, np.ones((n, 1))]) @ self.W_out
            out[:, k] = y
            X = self.advance(X, y)
        return out


# ----------------------------------------------------------------------------------------------------------
# Метрики соответствия аттракторов
# ----------------------------------------------------------------------------------------------------------

def delta_att(model_x, true_x):
    """Ошибка аттрактора Röhm et al. 2021, формулы (8)–(10): Δ = sqrt(Σ_i Δ_i² + Δ_|i|²),
    Δ_i = (⟨x_i⟩ − ⟨x̃_i⟩)/⟨|x̃_i|⟩, Δ_|i| = (⟨|x_i|⟩ − ⟨|x̃_i|⟩)/⟨|x̃_i|⟩; x̃ — истинный аттрактор."""
    m, t = np.asarray(model_x), np.asarray(true_x)
    ta = np.abs(t).mean(0)
    d1 = (m.mean(0) - t.mean(0)) / ta
    d2 = (np.abs(m).mean(0) - ta) / ta
    return float(np.sqrt(np.sum(d1 ** 2 + d2 ** 2)))


def hausdorff(Xa, Xb):
    """Симметричное расстояние Хаусдорфа между облаками точек."""
    da = cKDTree(Xb).query(Xa)[0].max()
    db = cKDTree(Xa).query(Xb)[0].max()
    return float(max(da, db))


def sliced_w1(Xa, Xb, n_dir=64, seed=0):
    """Нарезанное расстояние Вассерштейна W1: среднее по n_dir случайным единичным направлениям (затравка seed)
    одномерных W1 между проекциями; облака приводятся к одному размеру интерполяцией квантилей."""
    rng = np.random.default_rng(seed)
    d = Xa.shape[1]
    V = rng.normal(size=(n_dir, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    q = np.linspace(0, 1, 1001)
    tot = 0.0
    for v in V:
        a = np.quantile(Xa @ v, q)
        b = np.quantile(Xb @ v, q)
        tot += np.mean(np.abs(a - b))
    return float(tot / n_dir)


# ----------------------------------------------------------------------------------------------------------
# Скрытость: окрестности равновесий, граница бассейна, недостающая масса
# ----------------------------------------------------------------------------------------------------------

def radius_points(r, dirs, eps, n_ball, rng):
    """Точки проверки окрестности равновесия r радиуса eps (как в paper2_esn/experiments/e36_hidden_esn.py):
    r ± eps·v для каждого неустойчивого направления v и n_ball точек r + eps·ξ/sqrt(N), ξ ~ N(0, I)."""
    N = len(r)
    pts = []
    for v in dirs:
        pts += [r + eps * v, r - eps * v]
    pts += list(r + eps * rng.normal(size=(n_ball, N)) / np.sqrt(N))
    return np.array(pts)


def bisect_boundary(in_basin, A_pts, O_pts, eps):
    """Точки у границы бассейна: для каждой пары (a, o) — a в бассейне, o нет — бисекция отрезка, пока его длина
    не станет меньше eps. in_basin(X) -> bool для пачки точек. Возвращает середины последних отрезков."""
    a = np.array(A_pts, float, copy=True)
    o = np.array(O_pts, float, copy=True)
    active = np.linalg.norm(a - o, axis=1) >= eps
    while active.any():
        mid = 0.5 * (a[active] + o[active])
        inside = in_basin(mid)
        ia = np.where(active)[0]
        a[ia[inside]] = mid[inside]
        o[ia[~inside]] = mid[~inside]
        active = np.linalg.norm(a - o, axis=1) >= eps
    return 0.5 * (a + o)


def missing_mass_bounds(counts, n, alpha=0.05):
    """counts — кратности находок различных равновесий за n независимых стартов (несошедшиеся старты в counts не
    входят, но входят в n). G = f1/n — оценка Гуда–Тьюринга недостающей массы.
    conservative: G + (1 + √2)·sqrt(ln(2/α)/n) — граница на уровне 1 − α без предположений о распределении
    (верхний хвост M: McAllester, Ortiz 2003, теорема 16; нижний хвост G: неравенство Мак-Диармида).
    refined: G + 1/n + 2·sqrt(f2·t)/n + t/n, t = ln(1/α) — подстановочная граница по теореме 3.9
    Ben-Hamou, Boucheron, Ohannessian 2017 (нестрогая)."""
    c = np.asarray(counts, int)
    f1 = int(np.sum(c == 1))
    f2 = int(np.sum(c == 2))
    G = f1 / n
    cons = G + (1 + np.sqrt(2)) * np.sqrt(np.log(2 / alpha) / n)
    t = np.log(1 / alpha)
    ref = G + 1.0 / n + 2.0 * np.sqrt(f2 * t) / n + t / n
    return {"n": int(n), "distinct": int(len(c)), "f1": f1, "f2": f2, "G": G,
            "conservative": float(min(1.0, cons)), "refined": float(min(1.0, ref))}


def clopper_pearson_upper(k, n, alpha=0.05):
    """Верхняя граница доли на уровне 1 − α (одностороння); при k = 0 равна 1 − α^{1/n}."""
    from scipy.stats import beta
    if k >= n:
        return 1.0
    return float(beta.ppf(1 - alpha, k + 1, n - k))

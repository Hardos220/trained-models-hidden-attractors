"""Конвейер анализа одной модели: неподвижные точки, обучающий аттрактор, перепись, разметка и кластеры,
протокол проверки скрытости, метрики соответствия.

Модель — объект с интерфейсом esngpu.Net (run, newton, jac, unstable_directions, samples, orbit, N, d, h, scale,
kind) или ngrc_gpu.NGRCNet. Разметка повторяет paper2_esn/experiments/e33_esn_verify.py.
"""
import time

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

import esngpu as G

RADII = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
PREC_BULK = __import__("os").environ.get("ESN_PREC_BULK", "fp64")   # fp32 — перепись и разметка в float32


def feats(out, scale):
    return np.column_stack([out["x_mean"] / scale, out["x_std"] / scale, out["x_min"] / scale, out["x_max"] / scale])


def dedupe_against(points, known, tol=1e-8):
    """Индексы points в known (с пополнением known). Возвращает список индексов."""
    idx = []
    for x in points:
        j = -1
        if known:
            K = np.asarray(known)
            dd = np.linalg.norm(K - x, axis=1)
            k = int(np.argmin(dd))
            if dd[k] < tol * (1 + np.linalg.norm(K[k])):
                j = k
        if j < 0:
            known.append(np.array(x))
            j = len(known) - 1
        idx.append(j)
    return idx


class Labeler:
    """Метки: 'a<k>' — колебательный аттрактор модели k; 'f<k>' — неподвижная точка k; 'oc'/'or' — несопоставленный
    хаотический/регулярный режим; 'fx' — сошлась, но точку не удалось уточнить; 'esc' — уход (NG-RC)."""

    def __init__(self, net, lam_c, fps=None):
        self.net, self.lam_c = net, lam_c
        self.fps = [np.asarray(x) for x in (fps or [])]
        self.refs = []            # dict(kind, mu, sd, floor, rep, lam)

    def fp_index(self, finals, tol=1e-3):
        out = np.full(len(finals), -1)
        if not self.fps or not len(finals):
            return out
        K = np.asarray(self.fps)
        step = max(16, min(4096, int(2e8 / (8 * K.shape[0] * K.shape[1]))))   # массив step×k×N не больше ~0.2 ГБ
        for i0 in range(0, len(finals), step):
            F = finals[i0:i0 + step]
            d = np.sqrt(np.maximum(((F[:, None, :] - K[None]) ** 2).sum(-1), 0))
            j = d.argmin(1)
            ok = d[np.arange(len(F)), j] < tol * (1 + np.linalg.norm(K[j], axis=1))
            out[i0:i0 + step] = np.where(ok, j, -1)
        return out

    def add_fps(self, X):
        return dedupe_against(list(X), self.fps, tol=1e-6)

    def label(self, out, F=None):
        n = len(out["lam"])
        lab = np.empty(n, dtype=object)
        esc = out.get("esc", np.zeros(n, bool))
        fp = out["fp"] & ~esc
        lab[esc] = "esc"
        idx = np.where(fp)[0]
        if len(idx):
            j = self.fp_index(out["final"][idx])
            miss = idx[j < 0]
            if len(miss):
                X, ok = self.net.newton(out["final"][miss])
                if ok.any():
                    jj = self.add_fps(X[ok])
                    j_m = np.full(len(miss), -1)
                    j_m[ok] = jj
                    j[j < 0] = j_m
            for i, jj in zip(idx, j):
                lab[i] = f"f{jj}" if jj >= 0 else "fx"
        osc = np.where(~fp & ~esc)[0]
        if len(osc):
            F = feats(out, self.net.scale)[osc] if F is None else F[osc]
            lam = out["lam"][osc]
            best = np.full(len(osc), -1)
            bz = np.full(len(osc), np.inf)
            for k, r in enumerate(self.refs):
                z = np.max(np.abs(F - r["mu"]) / (r["sd"] + r["floor"]), axis=1)
                kind_ok = (lam > self.lam_c) if r["kind"] == "chaos" else (lam <= self.lam_c)
                z[~kind_ok] = np.inf
                better = z < bz
                best[better] = k
                bz[better] = z[better]
            for t, i in enumerate(osc):
                if bz[t] < 6.0:
                    lab[i] = f"a{best[t]}"
                else:
                    lab[i] = "oc" if lam[t] > self.lam_c else "or"
        return lab


def counts(lab):
    v, c = np.unique(np.asarray(lab, dtype=str), return_counts=True)
    return {str(a): int(b) for a, b in zip(v, c)}


def make_ref(F, lam, kind, rep):
    mu = F.mean(0)
    sd = F.std(0) if len(F) > 1 else np.zeros_like(mu)
    floor = (0.05 if kind == "chaos" else 0.02) * (1 + np.abs(mu))
    return {"kind": kind, "mu": mu, "sd": sd, "floor": floor, "rep": np.asarray(rep), "lam": float(np.median(lam))}


def training_attractor(net, L, r_last, n_tr=1000, n_obs=2000, seed=11):
    """A_tr: орбита из последнего обучающего состояния (3000 шагов, состояния через 10 после 500), 48 копий.
    Если большинство копий сходится к точке — A_tr точка (метка f<k>), иначе колебательный эталон refs[0]."""
    orb = G.N_(net.orbit(r_last, 2500, every=10, burn=500))
    rng = np.random.default_rng(seed)
    picks = orb[rng.integers(0, len(orb), 48)]
    o = net.run(picks, n_tr=n_tr, n_obs=n_obs, seed=seed, prec=PREC_BULK)
    esc = o.get("esc", np.zeros(48, bool))
    if np.mean(o["fp"] | esc) >= 0.5:
        X, ok = net.newton(o["final"][o["fp"] & ~esc])
        if ok.any():
            j = L.add_fps(X[ok])
            v, c = np.unique(j, return_counts=True)
            return {"kind": "fp", "label": f"f{int(v[np.argmax(c)])}", "orbit": orb}
        return {"kind": "fx", "label": "fx", "orbit": orb}
    chaos = o["lam"] > L.lam_c
    kind = "chaos" if np.mean(chaos[~o["fp"]]) >= 0.5 else "regular"
    keep = ~o["fp"] & ~esc & (chaos if kind == "chaos" else ~chaos)
    F = feats(o, net.scale)[keep]
    ref = make_ref(F, o["lam"][keep], kind, orb[len(orb) // 2])
    L.refs.insert(0, ref)
    return {"kind": kind, "label": "a0", "orbit": orb}


def discover(net, L, out, lab, n_life=50000, n_obs=4000, seed=21):
    """Кластеры несопоставленных режимов (DBSCAN по среднему и ст. откл.), подтверждение продолжением до 20
    членов каждого кластера на n_life итераций. Возвращает список кандидатов (для отчёта)."""
    F = feats(out, net.scale)
    d = net.d
    lab = np.asarray(lab, dtype=str)
    cands = []
    for kind, code, eps in (("chaos", "oc", 0.5), ("regular", "or", 0.1)):
        idx = np.where(lab == code)[0]
        if not len(idx):
            continue
        if len(idx) > 2000:          # DBSCAN хранит соседства всех точек: память растёт как n²
            idx = np.sort(np.random.default_rng(seed).choice(idx, 2000, replace=False))
        cl = DBSCAN(eps=eps, min_samples=1).fit(F[idx][:, :2 * d]).labels_
        for c in np.unique(cl):
            mem = idx[cl == c]
            ref = make_ref(F[mem], out["lam"][mem], kind, out["final"][mem[0]])
            sub = mem[:20]
            o2 = net.run(out["final"][sub], n_tr=n_life, n_obs=n_obs, seed=seed + int(c), prec=PREC_BULK)
            tmp = Labeler(net, L.lam_c, L.fps)
            tmp.refs = [ref]
            l2 = np.asarray(tmp.label(o2), dtype=str)
            L.fps = tmp.fps
            persist = float(np.mean(l2 == "a0"))
            rec = {"kind": kind, "n_members": int(len(mem)), "lam": ref["lam"], "persist": persist, "after_life": counts(l2)}
            if persist > 0:
                k0 = int(np.where(l2 == "a0")[0][0])
                F2 = feats(o2, net.scale)[l2 == "a0"]
                ref2 = make_ref(np.vstack([F[mem], F2]), np.concatenate([out["lam"][mem], o2["lam"][l2 == "a0"]]), kind,
                                o2["final"][k0])
                L.refs.append(ref2)
                rec["confirmed_as"] = f"a{len(L.refs) - 1}"
            cands.append(rec)
    return cands


def fp_info(net, x, true_eq=None, L_scale=1.0, eq_line=False):
    w, _ = net.stability(x)
    rho = float(np.max(np.abs(w)))
    xo = net.output_of(x)
    rec = {"rho": rho, "n_unstable": int(np.sum(np.abs(w) > 1 + 1e-9)), "x": xo.tolist(),
           "mean_abs_r": float(np.mean(np.abs(x))), "share_sat": float(np.mean(np.abs(x) > 0.9))}
    if eq_line:
        rec["dist_true_eq"] = float(np.hypot(xo[0], xo[1]) / L_scale)
    elif true_eq is not None and len(true_eq):
        dd = np.linalg.norm(np.asarray(true_eq) - xo, axis=1)
        rec["nearest_true_eq"] = int(np.argmin(dd))
        rec["dist_true_eq"] = float(dd.min() / L_scale)
    return rec


def radius_test(net, L, fp_ids, radii=RADII, n_ball=200, n_tr=20000, n_obs=4000, seed=31, max_dirs=20):
    """Окрестности неустойчивых точек: числа исходов по (точка, радиус). L_M = net.L_M."""
    rng = np.random.default_rng(seed)
    X0, tags, n_pts = [], [], {}
    for i in fp_ids:
        r = L.fps[i]
        w, dirs = net.unstable_directions(r, max_dirs=max_dirs)
        if np.max(np.abs(w)) <= 1 + 1e-9:
            continue
        for eps in radii:
            pts = []
            for v in dirs:
                pts += [r + eps * net.L_M * v, r - eps * net.L_M * v]
            pts += list(r + eps * net.L_M * rng.normal(size=(n_ball, len(r))) / np.sqrt(len(r)))
            X0 += pts
            tags += [(i, eps)] * len(pts)
            n_pts[(i, eps)] = len(pts)
    res = {}
    if not X0:
        return res
    X0 = net.clip_state(np.array(X0))
    eps_arr = np.array([eps for _, eps in tags])
    lab = np.empty(len(X0), dtype=object)
    small = eps_arr <= 1e-4                         # малые радиусы — только float64
    for mask, prec in ((small, "fp64"), (~small, PREC_BULK)):
        if mask.any():
            o = net.run(X0[mask], n_tr=n_tr, n_obs=n_obs, seed=seed, prec=prec)
            lab[mask] = L.label(o)
    lab = lab.astype(str)
    tags = np.array([f"{i}|{eps:g}" for i, eps in tags])
    for key in np.unique(tags):
        res[key] = counts(lab[tags == key])
    return res


def hits_by_attractor(nb):
    """r̂ по каждому аттрактору: min радиуса с попаданием по всем точкам; и есть ли попадания при r ≤ 1e-3."""
    out = {}
    for key, cnt in nb.items():
        i, eps = key.split("|")
        eps = float(eps)
        for lab, c in cnt.items():
            if not lab.startswith("a") or c == 0:
                continue
            rec = out.setdefault(lab, {"r_hat": np.inf, "from": {}})
            rec["r_hat"] = min(rec["r_hat"], eps)
            rec["from"][i] = min(rec["from"].get(i, np.inf), eps)
    return out


def boundary_search(net, L, a_label, census_states, census_labels, K=1000, eps_rel=1e-3, n_starts=20000,
                    n_tr=20000, n_obs=4000, seed=41):
    """Точки у границы бассейна аттрактора a_label (бисекция) и n_starts стартов метода Ньютона из ν_∂."""
    rng = np.random.default_rng(seed)
    eps = eps_rel * net.L_M
    lab = np.asarray(census_labels, dtype=str)
    ia = np.where(lab == a_label)[0]
    io = np.where(lab != a_label)[0]
    D = census_states.shape[1]
    if len(io) == 0:                 # вся перепись пришла в A: точек другого исхода для бисекции нет
        return {"boundary_points": np.zeros((0, D)), "n_starts": 0, "n_converged": 0, "multiplicities": {},
                "new_fp_ids": [], "bisect_s": 0.0, "note": "no other outcomes in census"}
    if len(ia) >= 50:
        A_pts = census_states[rng.choice(ia, K)]
    else:
        k = int(a_label[1:])
        orb = G.N_(net.orbit(L.refs[k]["rep"], 2000, every=2))
        A_pts = net.clip_state(orb[rng.integers(0, len(orb), K)] + eps * rng.normal(size=(K, D)) / np.sqrt(D))
    O_pts = census_states[rng.choice(io, K)]

    def in_basin(X):
        o = net.run(net.clip_state(X), n_tr=n_tr, n_obs=n_obs, seed=int(rng.integers(1 << 30)), prec=PREC_BULK)
        return np.asarray(L.label(o), dtype=str) == a_label

    t0 = time.time()
    Bp = bisect(in_basin, A_pts, O_pts, eps)
    tb = time.time() - t0
    starts = net.clip_state(Bp[rng.integers(0, K, n_starts)] + eps * rng.normal(size=(n_starts, D)) / np.sqrt(D))
    X, ok = net.newton(starts)
    n_known_before = len(L.fps)
    j = L.add_fps(X[ok]) if ok.any() else []
    v, c = np.unique(j, return_counts=True) if len(j) else (np.array([], int), np.array([], int))
    return {"boundary_points": Bp, "n_starts": int(n_starts), "n_converged": int(ok.sum()),
            "multiplicities": {int(a): int(b) for a, b in zip(v, c)},
            "new_fp_ids": list(range(n_known_before, len(L.fps))), "bisect_s": round(tb, 1)}


def bisect(in_basin, A_pts, O_pts, eps):
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


def missing_mass_bounds(counts_list, n, alpha=0.025):
    if n == 0:
        return {"n": 0, "distinct": 0, "f1": 0, "f2": 0, "G": None, "conservative": None, "refined": None}
    c = np.asarray(counts_list, int)
    f1, f2 = int(np.sum(c == 1)), int(np.sum(c == 2))
    G_ = f1 / n
    cons = G_ + (1 + np.sqrt(2)) * np.sqrt(np.log(2 / alpha) / n)
    t = np.log(1 / alpha)
    ref = G_ + 1.0 / n + 2.0 * np.sqrt(f2 * t) / n + t / n
    return {"n": int(n), "distinct": int(len(c)), "f1": f1, "f2": f2, "G": G_,
            "conservative": float(min(1.0, cons)), "refined": float(min(1.0, ref))}


def hidden_label(r_hat, cand, cons):
    if r_hat <= 1e-6:
        return "self_excited"
    if r_hat <= 1e-3:
        return "excited_r"
    if cand and cons is not None:
        return "hidden_stat" if cons <= 0.05 else "undetermined"
    return "undetermined"


def diam(X, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    S = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    from scipy.spatial.distance import pdist
    return float(pdist(S).max())


def hausdorff(Xa, Xb):
    return float(max(cKDTree(Xb).query(Xa)[0].max(), cKDTree(Xa).query(Xb)[0].max()))


def sliced_w1(Xa, Xb, n_dir=64, seed=0):
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(n_dir, Xa.shape[1]))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    q = np.linspace(0, 1, 1001)
    return float(np.mean([np.mean(np.abs(np.quantile(Xa @ v, q) - np.quantile(Xb @ v, q))) for v in V]))


def delta_att(m, t):
    ta = np.abs(t).mean(0)
    d1 = (m.mean(0) - t.mean(0)) / ta
    d2 = (np.abs(m).mean(0) - ta) / ta
    return float(np.sqrt(np.sum(d1 ** 2 + d2 ** 2)))


def correspondence(Xm, true_atts):
    """Метрики аттрактора модели (отсчёты выхода) относительно каждого истинного аттрактора."""
    out = {}
    if not np.all(np.isfinite(Xm)):
        return {k: {"finite": False} for k in true_atts}
    for name, Y in true_atts.items():
        out[name] = {"delta_att": delta_att(Xm, Y), "d_H": hausdorff(Xm[::2], Y[::2]) / diam(Y),
                     "SW1": sliced_w1(Xm, Y) / float(np.mean(Y.std(0)))}
    return out

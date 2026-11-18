"""Входные данные пилота эхо-сетей на A100: размеченные начальные условия, эталонные аттракторы и обучающие ряды.

Для каждого режима (система при заданных параметрах):
 1. пул равномерных начальных условий области B с метками исхода (build_reference_set, как e02);
 2. ics_uniform — первые 10⁴ точек пула с метками; ics_basin_<имя> — до 2000 точек пула из бассейна каждого
    аттрактора;
 3. att_<имя> — 2·10⁴ отсчётов с шагом выборки Δt на каждом колебательном аттракторе (истинная система, без шума);
 4. обучающие ряды (отсчёты с шагом Δt, физические координаты):
    H1 — одна траектория на скрытом аттракторе с динамическим шумом, 500 + 10⁴ отсчётов;
    H0 — то же без шума;
    E1 — 20 траекторий из бассейна видимого аттрактора (равновесие, линия равновесий или цикл LC), по 50 + 500
         отсчётов, с шумом;
    динамический шум: после каждого шага RK4 x += sqrt(dt)·σ_i·ξ, σ_i = 0.003·std_i(скрытого аттрактора);
    проверка «без перескоков»: состояния ряда каждые 1000 отсчётов размечаются без шума и должны дать тот же
    аттрактор.
Контроль Ли–Спротта (Röhm et al. 2021, приложение B): Эйлер–Маруяма h = 10⁻³, x += h·f(x) + 0.2·h·ξ, выборка
каждые 300 шагов при (a, b) = (2, 0.8) и каждые 200 шагов при (6, 0.1), 11 000 отсчётов; ряды без шума для прогрева
и эталона.
Лоренц: 2000 физических начальных условий для переписи A0.

Выход: <out>/<режим>.npz и <out>/<режим>.json, <out>/li_sprott_*.npz/json, <out>/lorenz.npz/json, <out>/index.json.
Запуск: python paper2_esn/experiments/e70_esn_pilot_data.py <out> [режимы через запятую]
"""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from hsearch import systems  # noqa: E402
from hsearch.identify import LAB_ESCAPE, LAB_UNKNOWN, build_reference_set, match  # noqa: E402
from hsearch.integrate import run_batch  # noqa: E402
from hsearch.systems.highdim import li_sprott_4d_rhs  # noqa: E402
from hsearch.systems.reservoir import lorenz_data  # noqa: E402

# режим: (система, параметры или None, шаг выборки в шагах dt, размер пула, скрытые, видимый, известные IC)
REGIMES = {
    "se11": ("se11", None, 15, 400_000, ["HA"], "FP", None),
    "se3": ("se3", None, 20, 100_000, ["HA"], "FP", None),
    "wangchen2": ("wangchen2", None, 18, 100_000, ["HA"], "FP", None),
    "wangchen2_a0.03": ("wangchen2", [0.03], 18, 100_000, ["HA"], "E0",
                        {"HA": [-0.35343, 0.24664, 0.00207]}),
    "le1": ("le1", None, 16, 1_000_000, ["HA"], "Eset", None),
    "chua_I": ("chua_I", None, 28, 100_000, ["HA_a", "HA_b"], "F0", None),
    "chua_II": ("chua_II", None, 20, 300_000, ["HA_a", "HA_b"], "LC", None),
}
# переходный участок и окно наблюдения для разметки, если они отличаются от реестра (у Ванга–Чена при a = 0.03
# притяжение к циклу медленное: при штатных 30 000 + 10 000 шагах показатель Ляпунова по окну ещё ≈ 0.03)
TIMING = {"wangchen2_a0.03": (600_000, 40_000)}
N_UNIFORM, N_BASIN, N_ATT = 10_000, 2000, 20_000
H_WASH, H_LEN = 500, 10_000
E_N, E_WASH, E_LEN = 20, 50, 500
NOISE_LEVELS = (0.003, 0.001, 0.0003)     # σ_i / std_i скрытого аттрактора, по убыванию
NOISE_SEEDS = 30                           # затравок на уровень шума
CHECK_H, CHECK_E = 250, 100                # шаг проверки «без перескоков», отсчётов


def make_system(key):
    name, params, every, n_pool, hidden, visible, known = REGIMES[key]
    s = systems.get(name)
    if params is not None:
        p = np.array(params, float)
        eq = [np.array([0.25, 0.0625, -p[0] / 0.0625])] if name == "wangchen2" else s.equilibria
        s = replace(s, params=p, name=key, known=dict(known), equilibria=eq)
    if key in TIMING:
        s = replace(s, n_tr=TIMING[key][0], n_obs=TIMING[key][1])
    return s


def make_noisy(rhs, d):
    @njit(cache=False)
    def run(x0, p, dt, n_samples, every, sig_sqdt, xi):
        # xi: (n_samples*every, d) стандартные нормальные; sig_sqdt = σ·sqrt(dt) (d,)
        x = x0.copy()
        out = np.empty((n_samples, d))
        k1 = np.empty(d); k2 = np.empty(d); k3 = np.empty(d); k4 = np.empty(d); tmp = np.empty(d)
        step = 0
        for s in range(n_samples):
            for _ in range(every):
                rhs(x, p, k1)
                for i in range(d):
                    tmp[i] = x[i] + 0.5 * dt * k1[i]
                rhs(tmp, p, k2)
                for i in range(d):
                    tmp[i] = x[i] + 0.5 * dt * k2[i]
                rhs(tmp, p, k3)
                for i in range(d):
                    tmp[i] = x[i] + dt * k3[i]
                rhs(tmp, p, k4)
                for i in range(d):
                    x[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) + sig_sqdt[i] * xi[step, i]
                step += 1
            for i in range(d):
                out[s, i] = x[i]
        return out
    return run


def sampled(s, runner, x0, n_samples, every, sig, rng):
    """Отсчёты с шагом every·dt; динамический шум σ (d,), sig = 0 — без шума."""
    sig = np.asarray(sig, float) * np.ones(s.dim)
    chunks, x, left = [], np.asarray(x0, float).copy(), n_samples
    while left > 0:
        m = min(left, 2000)
        xi = rng.standard_normal((m * every, s.dim)) if np.any(sig > 0) else np.zeros((m * every, s.dim))
        seg = runner(x, s.params, s.dt, m, every, sig * np.sqrt(s.dt), xi)
        chunks.append(seg)
        x = seg[-1].copy()
        left -= m
    return np.vstack(chunks)


def label_states(s, refs, X):
    feats, finals = run_batch(s, np.ascontiguousarray(X))
    lab, _ = match(s, feats, finals, refs)
    return lab


def name_of(lab, refs):
    if lab == LAB_ESCAPE:
        return "escape"
    if lab == LAB_UNKNOWN:
        return "unknown"
    return refs[lab].name


def regime(key, out_dir, seed):
    t0 = time.time()
    name, params, every, n_pool, hidden, visible, known = REGIMES[key]
    s = make_system(key)
    rng = np.random.default_rng(seed)
    refs, ics, feats, finals, labels, hist = build_reference_set(s, rng, n_sample=n_pool, verbose=False)
    lab_names = np.array([name_of(int(l), refs) for l in labels])
    # обнаруженные эталоны new*: продолжение 16 копий в 10 раз дольше; если большинство приходит в другой режим,
    # эталон считается переходным процессом и его метки заменяются меткой этого режима
    merged = {}
    for k, r in enumerate(refs):
        if not r.name.startswith("new") or r.example is None:
            continue
        X = r.example[None, :] + 1e-6 * rng.normal(size=(16, s.dim))
        f2, fin2 = run_batch(s, X, n_tr=10 * s.n_tr, n_obs=s.n_obs)
        l2, _ = match(s, f2, fin2, refs)
        v, c = np.unique([name_of(int(l), refs) for l in l2], return_counts=True)
        top = v[np.argmax(c)]
        if top != r.name:
            merged[r.name] = str(top)
            lab_names[lab_names == r.name] = top
    refs = [r for r in refs if r.name not in merged]
    names = [r.name for r in refs]
    frac = {str(nm): float(np.mean(lab_names == nm)) for nm in sorted(set(lab_names))}
    print(f"[{key}] пул {n_pool}: {frac} ({time.time() - t0:.0f} с)", flush=True)
    arrays = {"ics_uniform": ics[:N_UNIFORM], "labels_uniform": lab_names[:N_UNIFORM].astype("U16")}
    for nm in sorted(set(lab_names) - {"escape", "unknown"}):
        idx = np.where(lab_names == nm)[0][:N_BASIN]
        arrays[f"ics_basin_{nm}"] = ics[idx]
    runner = make_noisy(s.rhs, s.dim)
    meta_att = {}
    for r in refs:
        if r.kind not in ("periodic", "chaotic") or r.example is None:
            continue
        X = sampled(s, runner, r.example, N_ATT, every, 0.0, rng)
        arrays[f"att_{r.name}"] = X
        meta_att[r.name] = {"kind": r.kind, "lam": float(r.lam), "p_pool": frac.get(r.name, 0.0),
                            "mean": X.mean(0).tolist(), "std": X.std(0).tolist(), "abs_mean": np.abs(X).mean(0).tolist()}
    fps = [r for r in refs if r.kind == "fp" and r.point is not None]
    eq_info = []
    for r in fps:
        e = r.point
        h = 1e-6
        J = np.zeros((s.dim, s.dim))
        f0 = np.zeros(s.dim)
        s.rhs(e, s.params, f0)
        for j in range(s.dim):
            ep = e.copy(); ep[j] += h
            f1 = np.zeros(s.dim)
            s.rhs(ep, s.params, f1)
            J[:, j] = (f1 - f0) / h
        w = np.linalg.eigvals(J)
        eq_info.append({"name": r.name, "point": e.tolist(), "eig_re": np.real(w).tolist(), "eig_im": np.imag(w).tolist(),
                        "stable": bool(np.all(np.real(w) < 0)), "p_pool": frac.get(r.name, 0.0)})
    # обучающие ряды
    hid = hidden[0]
    att = arrays[f"att_{hid}"]
    x_on = att[-1]
    train = {}

    def stays(X, target, step):
        lab = label_states(s, refs, X[::step])
        return all(name_of(int(l), refs) == target for l in lab)

    # H1: первый уровень шума и первая затравка, при которых все состояния ряда через CHECK_H отсчётов
    # (продолженные без шума) приходят на скрытый аттрактор
    sig, chosen = None, None
    for lev in NOISE_LEVELS:
        for j in range(NOISE_SEEDS):
            r_ = np.random.default_rng([seed, 1, int(lev * 1e6), j])
            X = sampled(s, runner, x_on, H_WASH + H_LEN, every, lev * att.std(0), r_)
            if stays(X, hid, CHECK_H):
                sig, chosen = lev * att.std(0), {"level": lev, "seed_index": j}
                train["H1"] = X
                break
        if sig is not None:
            break
    if sig is None:
        raise RuntimeError(f"{key}: no noisy H1 series without hopping")
    train["H0"] = sampled(s, runner, x_on, H_WASH + H_LEN, every, 0.0, rng)
    assert stays(train["H0"], hid, CHECK_H)
    # E1: каждая траектория из бассейна видимого аттрактора — с тем же σ; затравка подбирается так, чтобы все
    # состояния через CHECK_E отсчётов приходили на видимый аттрактор; начальные условия берутся по порядку
    basin_v = arrays[f"ics_basin_{visible}"]
    E, e_info, j_ic = [], [], 0
    while len(E) < E_N and j_ic < len(basin_v):
        for j in range(NOISE_SEEDS):
            r_ = np.random.default_rng([seed, 2, j_ic, j])
            X = sampled(s, runner, basin_v[j_ic], E_WASH + E_LEN, every, sig, r_)
            if np.all(np.isfinite(X)) and stays(X, visible, CHECK_E):
                E.append(X)
                e_info.append({"ic_index": j_ic, "seed_index": j})
                break
        j_ic += 1
    if len(E) < E_N:
        raise RuntimeError(f"{key}: only {len(E)} E1 series")
    train["E1"] = np.vstack(E)
    arrays["train_E1_len"] = np.full(E_N, E_WASH + E_LEN)
    for k, v in train.items():
        arrays[f"train_{k}"] = v
    checks = {"H1": chosen, "E1": e_info}
    ok_h = ok_e = True
    meta = {"regime": key, "system": name, "params": s.params.tolist(), "dim": s.dim, "dt": s.dt,
            "sample_every": every, "Delta_t": every * s.dt, "box_lo": s.box_lo.tolist(), "box_hi": s.box_hi.tolist(),
            "esc_r": s.esc_r, "n_tr": s.n_tr, "n_obs": s.n_obs, "n_pool": n_pool, "seed": seed,
            "pool_fractions": frac, "refs": names, "hidden": hidden, "visible": visible,
            "attractors": meta_att, "equilibria": eq_info,
            "equilibrium_set": "x = y = 0, any z (stable for z < 0)" if key == "le1" else None,
            "noise_sigma": sig.tolist(), "noise_level_rel": chosen["level"],
            "noise_rule": "after each RK4 step x += sqrt(dt)*sigma*xi; sigma = level*std(hidden attractor)",
            "merged_refs": merged,
            "no_hopping_rule": f"states every {CHECK_H} (H) / {CHECK_E} (E) samples, continued without noise, "
                               "reach the same attractor",
            "train": {"H1": {"washout": H_WASH, "len": H_LEN, "noise": True},
                      "H0": {"washout": H_WASH, "len": H_LEN, "noise": False},
                      "E1": {"n_traj": E_N, "washout": E_WASH, "len": E_LEN, "noise": True, "basin_of": visible}},
            "no_hopping_checks": checks, "no_hopping_ok": bool(ok_h and ok_e),
            "time_s": round(time.time() - t0, 1)}
    np.savez_compressed(out_dir / f"{key}.npz", **arrays)
    (out_dir / f"{key}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{key}] готово: перескоков нет = {meta['no_hopping_ok']} {checks} ({meta['time_s']} с)", flush=True)
    return meta


@njit(cache=False)
def _euler_ls(x0, a, b, h, n_samples, every, noise, xi):
    x = x0.copy()
    out = np.empty((n_samples, 4))
    f = np.empty(4)
    k = 0
    for s in range(n_samples):
        for _ in range(every):
            f[0] = x[1] - x[0]
            f[1] = -x[0] * x[2] + x[3]
            f[2] = x[0] * x[1] - a
            f[3] = -b * x[1]
            for i in range(4):
                x[i] += h * f[i] + noise * h * xi[k, i]
            k += 1
        for i in range(4):
            out[s, i] = x[i]
    return out


def li_sprott(out_dir, seed):
    """Ряды для контроля Röhm et al. 2021 (Chaos 31, 103127), приложение B."""
    cases = {"li_sprott_2_0.8": (2.0, 0.8, 300, [5, 1, 1, 1],
                                 {"LC_a": [5, 1, 1, 1], "LC_b": [-5, -1, 1, -1], "T": [4, 1, -1, 1]}),
             "li_sprott_6_0.1": (6.0, 0.1, 200, [0, -4, 0, 5],
                                 {"SA_a": [0, -4, 0, 5], "SA_b": [0, 4, 0, -5], "T": [1, -1, 1, -1]})}
    metas = {}
    for key, (a, b, every, x_train, starts) in cases.items():
        n = 11_000
        arrays = {}
        # шумный ряд без перескоков между аттракторами (у Röhm et al. — визуальная проверка): окна по 200 отсчётов
        # должны лежать на обучающем аттракторе; затравки перебираются по порядку, берётся первая подходящая
        tries = []
        for seed_try in range(seed, seed + 200):
            rng = np.random.default_rng(seed_try)
            chunks, x = [], np.array(x_train, float)
            for c in range(0, n, 1000):
                xi = rng.standard_normal((1000 * every, 4))
                seg = _euler_ls(x, a, b, 1e-3, 1000, every, 0.2, xi)
                chunks.append(seg)
                x = seg[-1].copy()
            X = np.vstack(chunks)
            wm = np.array([X[i:i + 200, 3].mean() for i in range(0, n, 200)])
            wz = np.array([X[i:i + 200, 2].std() for i in range(0, n, 200)])
            ok = bool(np.all(wm > 0.05) and np.all(wz > 1.6)) if key.startswith("li_sprott_2") else bool(np.all(wm > 3.0))
            tries.append({"seed": seed_try, "ok": ok, "min_window_u_mean": float(wm.min())})
            if ok:
                break
        arrays["train_noisy"] = X
        zero = np.zeros((1, 4))
        for nm, x0 in starts.items():
            Xs = []
            x = np.array(x0, float)
            for c in range(0, n, 1000):
                seg = _euler_ls(x, a, b, 1e-3, 1000, every, 0.0, np.zeros((1000 * every, 4)))
                Xs.append(seg)
                x = seg[-1].copy()
            X = np.vstack(Xs)
            arrays[f"transient_{nm}"] = X           # первые 1000 — прогрев, остальные — эталон аттрактора
        del zero
        np.savez_compressed(out_dir / f"{key}.npz", **arrays)
        meta = {"case": key, "a": a, "b": b, "scheme": "Euler–Maruyama h = 1e-3: x += h*f(x) + noise*h*xi",
                "noise_train": 0.2, "sample_every_steps": every, "Delta_t": every * 1e-3, "n_samples": n,
                "train_start": x_train, "transient_starts": starts, "train_seed": tries[-1]["seed"],
                "train_seed_ok": tries[-1]["ok"], "train_seed_tries": len(tries),
                "no_hopping_rule": "every 200-sample window: mean u > 0.05 and std z > 1.6 (case 2, 0.8); mean u > 3 (case 6, 0.1)",
                "attractor_stats": {nm: {"mean": arrays[f"transient_{nm}"][1000:].mean(0).tolist(),
                                         "abs_mean": np.abs(arrays[f"transient_{nm}"][1000:]).mean(0).tolist()}
                                    for nm in starts},
                "source": "Röhm, Gauthier, Fischer 2021, Chaos 31, 103127, appendices A–C"}
        (out_dir / f"{key}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        metas[key] = meta
        print(f"[{key}] {meta['attractor_stats']}", flush=True)
    return metas


def lorenz(out_dir, seed):
    rng = np.random.default_rng(seed)
    lo, hi = np.array([-20.0, -25.0, 0.0]), np.array([20.0, 25.0, 50.0])
    ics = lo + (hi - lo) * rng.random((2000, 3))
    att = lorenz_data(45000)[25000:]          # продолжение того же ряда, что у обучения e33 (h = 0.02)
    np.savez_compressed(out_dir / "lorenz.npz", ics_phys=ics, att_lorenz=att)
    meta = {"box_lo": lo.tolist(), "box_hi": hi.tolist(), "n_ics": 2000, "h": 0.02,
            "att_from": "hsearch.systems.reservoir.lorenz_data(45000)[25000:]",
            "train_from": "lorenz_data(25000)[:20000] as in paper2_esn/experiments/e33_esn_verify.build",
            "lyapunov_max": 0.9056}
    (out_dir / "lorenz.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[lorenz] готово", flush=True)
    return meta


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sys.argv[2].split(",") if len(sys.argv) > 2 else list(REGIMES) + ["li_sprott", "lorenz"]
    index_path = out_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    for i, key in enumerate(keys):
        if key == "li_sprott":
            index.update({k: {"no_hopping_ok": None} for k in li_sprott(out_dir, 7300)})
        elif key == "lorenz":
            lorenz(out_dir, 7200)
            index["lorenz"] = {}
        else:
            m = regime(key, out_dir, 7000 + list(REGIMES).index(key))
            index[key] = {"no_hopping_ok": m["no_hopping_ok"], "pool_fractions": m["pool_fractions"]}
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Этап 3: скрытая (паразитная) динамика замкнутых резервуаров, обученных на аттракторе Лоренца.

Для каждой реализации (seed): обучение (h = 0.02, 20 000 шагов), время достоверного прогноза
(VPT, в ляпуновских временах, порог 0.4), «климат» (λ и статистики выхода), неподвижные точки
(Ньютон) и их устойчивость; исходы замкнутой динамики:
  * толчки от состояний реконструированного аттрактора: r + σ ξ, σ ∈ {1e-3 … 0.3};
  * равномерные состояния куба [-1, 1]^N;
  * прогрев: случайное состояние, затем T_w шагов вынужденной динамики истинным рядом, затем
    замыкание (T_w ∈ {5, 20, 50, 200});
  * скрытость: окрестности неустойчивых неподвижных точек вдоль неустойчивых направлений
    (ε ∈ {1e-4, 1e-2}) и в шаре радиуса ε.
Метки исходов: lorenz / fp:<номер неподвижной точки> / other_chaos / other_periodic.
Выход: results/e32_esn_study.json (+ примеры конечных состояний в results/e32_esn_states.npz)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from hsearch.reservoir_analysis import ClosedLoopESN, classify_runs  # noqa: E402
from hsearch.systems.reservoir import lorenz_data, train_esn  # noqa: E402

LYAP_LORENZ = 0.906


def vpt(m, esn, u_true_norm, start_state, n_max=3000, thr=0.4):
    R = start_state[None, :].copy()
    errs = []
    denom = np.sqrt(np.mean(np.sum(u_true_norm ** 2, axis=1)))
    for j in range(min(n_max, len(u_true_norm))):
        u_hat = m.output(R)[0]
        e = np.linalg.norm(u_hat - u_true_norm[j]) / denom
        errs.append(e)
        if e > thr:
            return j * m.h * LYAP_LORENZ, errs
        R = m.step(R)
    return n_max * m.h * LYAP_LORENZ, errs


def fp_label(final, fps, tol=1e-3):
    if not fps:
        return None
    d = np.linalg.norm(np.array(fps)[None, :, :] - final[:, None, :], axis=2)
    j = d.argmin(axis=1)
    return np.where(d[np.arange(len(final)), j] < tol * (1 + np.linalg.norm(np.array(fps)[j], axis=1)), j, -1)


def label_runs(m, out, ref, fps):
    lab, z = classify_runs(out, ref)
    lab = lab.astype(object)
    fpl = fp_label(out["final"], fps)
    for i in range(len(lab)):
        if lab[i] == "fp":
            lab[i] = f"fp:{fpl[i]}" if fpl is not None and fpl[i] >= 0 else "fp:new"
    return lab


def counts(lab):
    v, c = np.unique(lab, return_counts=True)
    return {str(a): int(b) for a, b in zip(v, c)}


def study(seed, n_res=300, n_per=200, rng=None):
    rng = np.random.default_rng(1000 + seed) if rng is None else rng
    t0 = time.time()
    u_all = lorenz_data(25000)
    T_train = 20000
    esn = train_esn(u_all[:T_train], n_res=n_res, seed=seed)
    m = ClosedLoopESN(esn)
    un_future = u_all[T_train:] / esn["scale"]
    v, errs = vpt(m, esn, un_future, esn["R_train"][-1])
    rec = {"seed": seed, "n_res": n_res, "train_rmse": esn["train_rmse"], "vpt_lyap": v}
    # реконструированный аттрактор: длинный замкнутый прогон с последнего обучающего состояния
    out0 = m.run(esn["R_train"][-1][None, :], n_tr=2000, n_obs=6000, rng=rng)
    rec["climate"] = {"lam": float(out0["lam"][0]), "u_mean": out0["u_mean"][0].tolist(), "u_std": out0["u_std"][0].tolist(),
                      "fp": bool(out0["fp"][0])}
    # точки реконструированного аттрактора как базы для толчков и эталона
    R = esn["R_train"][-1][None, :].copy()
    att = []
    for k in range(3000):
        R = m.step(R)
        if k > 500 and k % 10 == 0:
            att.append(R[0].copy())
    att = np.array(att)
    reconstructed_ok = not out0["fp"][0] and out0["lam"][0] > 0.3
    rec["reconstructed_chaotic"] = bool(reconstructed_ok)
    picks = att[rng.integers(0, len(att), size=48)]
    oref = m.run(picks, n_tr=1000, n_obs=2000, rng=rng)
    keep = (~oref["fp"]) & (oref["lam"] > 0.3)
    F = np.column_stack([oref["u_mean"], oref["u_std"], oref["u_min"], oref["u_max"]])
    ref = (F[keep].mean(axis=0), F[keep].std(axis=0)) if keep.sum() >= 5 else (F.mean(axis=0), F.std(axis=0))
    # неподвижные точки
    starts = np.vstack([att[rng.integers(0, len(att), size=80)], rng.uniform(-1, 1, size=(80, n_res)),
                        0.3 * rng.uniform(-1, 1, size=(40, n_res))])
    fps = m.fixed_points(starts)
    fp_info = []
    for r in fps:
        w, dirs = m.unstable_directions(r)
        fp_info.append({"rho": float(np.max(np.abs(w))), "n_unstable": int(np.sum(np.abs(w) > 1 + 1e-9)),
                        "u": m.output(r[None, :])[0].round(3).tolist(), "norm": float(np.linalg.norm(r))})
    rec["fixed_points"] = fp_info
    rec["n_stable_fp"] = int(sum(1 for d in fp_info if d["rho"] < 1))
    outcomes = {}
    examples = {}
    # толчки от аттрактора
    for sig in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1):
        X0 = np.clip(att[rng.integers(0, len(att), size=n_per)] + sig * rng.normal(size=(n_per, n_res)), -1, 1)
        out = m.run(X0, rng=rng)
        lab = label_runs(m, out, ref, fps)
        outcomes[f"kick_{sig:g}"] = counts(lab)
        for name in set(lab):
            if not str(name).startswith("lorenz") and name not in examples:
                examples[name] = out["final"][list(lab).index(name)]
    # куб
    X0 = rng.uniform(-1, 1, size=(n_per, n_res))
    out = m.run(X0, rng=rng)
    lab = label_runs(m, out, ref, fps)
    outcomes["cube"] = counts(lab)
    for name in set(lab):
        if not str(name).startswith("lorenz") and name not in examples:
            examples[name] = out["final"][list(lab).index(name)]
    # прогрев
    u_tr = u_all[:T_train] / esn["scale"]
    for Tw in (5, 20, 50, 200):
        X = rng.uniform(-1, 1, size=(n_per, n_res))
        starts_t = rng.integers(1000, T_train - Tw - 1, size=n_per)
        for k in range(Tw):
            uk = u_tr[starts_t + k]
            X = np.tanh(X @ esn["W"].T + uk @ esn["W_in"].T + esn["b"])
        out = m.run(X, rng=rng)
        lab = label_runs(m, out, ref, fps)
        outcomes[f"warmup_{Tw}"] = counts(lab)
    # скрытость: окрестности неустойчивых неподвижных точек
    hid = {}
    for i, r in enumerate(fps):
        w, dirs = m.unstable_directions(r)
        if not dirs:
            continue
        for eps in (1e-4, 1e-2):
            X0 = []
            for dvec in dirs[:10]:
                X0 += [r + eps * dvec, r - eps * dvec]
            nb = 40
            X0 += list(r + eps * rng.normal(size=(nb, n_res)) / np.sqrt(n_res))
            X0 = np.clip(np.array(X0), -1, 1)
            out = m.run(X0, rng=rng)
            lab = label_runs(m, out, ref, fps)
            hid[f"fp{i}_eps{eps:g}"] = counts(lab)
    rec["outcomes"] = outcomes
    rec["fp_neighbourhoods"] = hid
    rec["time_s"] = round(time.time() - t0, 1)
    return rec, examples, m, esn


def main(seeds=range(8), n_res=300):
    res = []
    all_examples = {}
    for seed in seeds:
        rec, ex, m, esn = study(seed, n_res=n_res)
        res.append(rec)
        for k, v in ex.items():
            all_examples[f"s{seed}_{k}"] = v
        oc = rec["outcomes"]
        print(f"seed {seed}: RMSE {rec['train_rmse']:.1e}, VPT {rec['vpt_lyap']:.2f} λt, λ_climate {rec['climate']['lam']:.3f}, "
              f"FP: {len(rec['fixed_points'])} (устойчивых {rec['n_stable_fp']}); толчок 0.01 {oc['kick_0.01']}, 0.1 {oc['kick_0.1']}, "
              f"куб {oc['cube']}, прогрев 20 {oc['warmup_20']} ({rec['time_s']} с)", flush=True)
        (ROOT / "results" / f"e32_esn_study_N{n_res}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        np.savez_compressed(ROOT / "results" / f"e32_esn_states_N{n_res}.npz", **all_examples)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    main(n_res=n)

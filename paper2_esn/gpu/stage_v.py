"""Этап V: вмешательство — меняет ли изменение числа паразитных равновесий скрытость (причинная проверка закона).

Резервуар (W, W_in, b) и нормировка (scale по ряду H1) одинаковы во всех вариантах; меняется только то, по чему
обучается выход W_out:
  H1      — только траектория на скрытом аттракторе (база);
  eqc     — H1 с линейным ограничением W_out r*_j = u*_j: r*_j — неподвижная точка разомкнутого резервуара при
            постоянном входе u*_j = x*_j/scale, поэтому r*_j — точное равновесие замкнутой сети на месте истинного
            (для le1 — 5 точек линии равновесий);
  E5, E10, E20 — H1 + 5, 10, 20 из 20 переходов к равновесию (E20 = набор HE1): доза данных вне аттрактора;
  L       — H1 + начальные участки (≥ 60 отсчётов, до ухода) истинных траекторий из 2000 случайных точек;
  E20+L   — оба дополнения.
Каждый вариант оценивается облегчённым протоколом этапа N (gpu/stage_n.evaluate).
Пробный вариант «вырезания» паразитных равновесий по месту (дополнение точками вокруг них) отброшен: в пробе U
выросло 4 → 8, ошибка обучения — в 37 раз (невырожденное равновесие не исчезает от малой поправки, а сдвигается).

Запуск: python gpu/stage_v.py --regimes se3 --Ns 300 --etas 1e-7,1e-4 --seeds 0,1,2 [--quick]
Результат: $GPUQ_OUT/V/<режим>/N<N>_rho<ρ>_eta<η>_s<seed>.json
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gpu"))
from hsearch import esn_pilot as P  # noqa: E402
import esngpu as G  # noqa: E402
import stage_c as SC  # noqa: E402
import stage_n as SN  # noqa: E402


def eq_constraints(key, meta, esn):
    pts = [[0.0, 0.0, zz] for zz in np.linspace(-10, -0.5, 5)] if key == "le1" else [e["point"] for e in meta["equilibria"]]
    U = np.asarray(pts, float) / esn["scale"]
    R = np.zeros((len(U), esn["n_res"]))
    drive = U @ esn["W_in"].T + esn["b"]
    for _ in range(500):
        R = P.act(R @ esn["W"].T + drive, esn["activation"])
    return R, U


def constrained(esn, series, wash, eta, C, D):
    """Повторное обучение выхода при тех же W, W_in, b, scale с ограничением C W_outᵀ = D."""
    Xs, Ys = [], []
    for s, w in zip(series, wash):
        un = np.asarray(s, float) / esn["scale"]
        R = np.zeros((len(un), esn["n_res"]))
        r = np.zeros(esn["n_res"])
        for k in range(len(un)):
            r = P.act(esn["W"] @ r + esn["W_in"] @ un[k] + esn["b"], esn["activation"])
            R[k] = r
        Xs.append(R[w:-1]); Ys.append(un[w + 1:])
    X, Y = np.vstack(Xs), np.vstack(Ys)
    G_ = X.T @ X + eta * np.eye(esn["n_res"])
    Wt = np.linalg.solve(G_, X.T @ Y)
    MC = np.linalg.solve(G_, C.T)
    Wt = Wt + MC @ np.linalg.solve(C @ MC, D - C @ Wt)
    e = dict(esn)
    e["W_out"] = Wt.T
    e["A"] = esn["W"] + esn["W_in"] @ Wt.T
    e["train_rmse"] = float(np.sqrt(np.mean((X @ Wt - Y) ** 2)))
    return e


def arms_data(z, l1):
    H = [z["train_H1"]]
    E = np.split(z["train_E1"], np.cumsum(z["train_E1_len"])[:-1])
    Lt = []
    for tr in l1["l1"]:
        bad = ~np.isfinite(tr).all(1)
        n = int(np.argmax(bad)) if bad.any() else len(tr)
        if n >= 60:
            Lt.append(tr[:n])
    return {"H1": (H, [500]), "E5": (H + E[:5], [500] + [50] * 5), "E10": (H + E[:10], [500] + [50] * 10),
            "E20": (H + E, [500] + [50] * len(E)), "L": (H + Lt, [500] + [50] * len(Lt)),
            "E20+L": (H + E + Lt, [500] + [50] * (len(E) + len(Lt)))}, len(Lt)


def analyze(key, z, meta, l1, N, rho, eta, seed, quick=False):
    t0 = time.time()
    data, nL = arms_data(z, l1)
    scale = z["train_H1"].std(axis=0)
    out = {"regime": key, "N": N, "rho": rho, "eta": eta, "seed": seed, "n_L_segments": nL, "arms": {}}
    base = None
    for arm in ("H1", "eqc", "E5", "E10", "E20", "L", "E20+L"):
        ta = time.time()
        series, wash = data["H1" if arm == "eqc" else arm]
        esn = P.train_esn_multi(series, n_res=N, spectral_radius=rho, ridge=eta, input_scale=0.5, density=0.05,
                                washout=wash, seed=seed, scale=scale)
        if arm == "H1":
            base = esn
        if arm == "eqc":
            C, D = eq_constraints(key, meta, base)
            esn = constrained(base, series, wash, eta, C, D)
        rec, _ = SN.evaluate(key, z, meta, esn, seed, quick=quick)
        if arm == "eqc":
            rec["constraint_fp_residual"] = float(np.abs(np.tanh(C @ esn["A"].T + esn["b"]) - C).max())
        rec["n_train_series"] = len(series)
        rec["time_s_arm"] = round(time.time() - ta, 1)
        out["arms"][arm] = rec
        y, m = next(iter(rec["hidden_match"].items()))
        print(f"    {arm:6s} K {rec['K']} U {rec['U']} Σind {rec['index_sum']} воспр {m.get('reproduced')} "
              f"Δatt {m.get('delta_att', 9):.3f} {m.get('label')} rmse {esn['train_rmse']:.2e} {rec['time_s_arm']} s", flush=True)
    out["time_total_s"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default="se3")
    ap.add_argument("--Ns", default="300")
    ap.add_argument("--rhos", default="0.9")
    ap.add_argument("--etas", default="1e-7,1e-4")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    jobs = [(k, int(N), float(r), float(e), int(sd)) for k in a.regimes.split(",") for N in a.Ns.split(",")
            for r in a.rhos.split(",") for e in a.etas.split(",") for sd in a.seeds.split(",")]
    out_root = Path(os.environ.get("GPUQ_OUT", "out")) / "V"
    print(f"stage V: {len(jobs)} models, device {G.DEV}", flush=True)
    cache = {}
    for n, (key, N, rho, eta, seed) in enumerate(jobs):
        name = f"N{N}_rho{rho}_eta{eta:g}_s{seed}"
        d = out_root / key
        d.mkdir(parents=True, exist_ok=True)
        if (d / f"{name}.json").exists():
            continue
        if key not in cache:
            cache = {key: SC.load_regime(key)}
        z, meta, l1 = cache[key]
        G.report_progress(n, len(jobs), f"{key} {name}")
        print(f"[{n + 1}/{len(jobs)}] {key} {name}", flush=True)
        try:
            rec = analyze(key, z, meta, l1, N, rho, eta, seed, quick=a.quick)
            (d / f"{name}.json").write_text(json.dumps(rec, default=float), encoding="utf-8")
        except Exception as exc:
            traceback.print_exc()
            if G.fatal_cuda(exc):
                print("fatal CUDA error, stopping the process", flush=True)
                sys.exit(3)
            (d / f"{name}.failed.txt").write_text(traceback.format_exc(), encoding="utf-8")
    G.report_progress(len(jobs), len(jobs), "done")


if __name__ == "__main__":
    main()

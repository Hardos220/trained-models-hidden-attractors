"""Этап I: индексы равновесий обученных моделей ESN этапа C (проверка теоремы о степени).

Для каждой модели: переобучение (те же параметры и seed), уточнение каждой сохранённой точки методом Ньютона,
якобиан J = diag(1 − tanh²) A, ind = sign det(I − J), число вещественных собственных значений > 1,
число неустойчивых; Σ ind по всем точкам и отдельно по «истинным» (dist_true_eq < 0.01) и паразитным.

Входы: data/i_index.json, data/i_bundle.npz (gpu/make_bundles.py), data/<режим>.npz/.json.
Запуск: python gpu/stage_i.py [--part i/n]
Результат: $GPUQ_OUT/I/<режим>_<модель>.json
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gpu"))
from hsearch import esn_pilot as P  # noqa: E402
import esngpu as G  # noqa: E402
import stage_c as SC  # noqa: E402

if os.environ.get("RMQ_CPU"):
    torch.set_num_threads(int(os.environ["RMQ_CPU"]))


def analyze(item, fps, cache):
    t0 = time.time()
    key = item["regime"]
    if key not in cache:
        cache.clear()
        cache[key] = SC.load_regime(key)
    z, meta, _ = cache[key]
    series, wash = SC.train_sets(z)[item["set"]]
    esn = P.train_esn_multi(series, n_res=item["N"], spectral_radius=item["rho"], ridge=item["eta"], input_scale=0.5,
                            density=0.05, washout=wash, seed=item["seed"])
    net = G.Net(esn, h=meta["Delta_t"])
    rows = []
    A, b = net.A_np, net.b_np
    I = np.eye(len(A))
    for j, x0 in enumerate(np.asarray(fps)):
        x = x0.copy()
        for _ in range(8):                           # уточнение методом Ньютона (по одной точке, numpy)
            t = np.tanh(A @ x + b)
            g = x - t
            if np.linalg.norm(g) < 1e-13:
                break
            try:
                x = x - np.linalg.solve(I - (1 - t ** 2)[:, None] * A, g)
            except np.linalg.LinAlgError:
                break
        if not np.all(np.isfinite(x)) or np.linalg.norm(np.tanh(A @ x + b) - x) > 1e-10:
            x = x0                                   # не сошлось — остаётся сохранённая точка (видно по residual)
        if True:
            J = (1 - np.tanh(A @ x + b) ** 2)[:, None] * A
            w = np.linalg.eigvals(J)
            real = np.abs(w.imag) <= 1e-10 * np.maximum(1.0, np.abs(w))
            sign, logdet = np.linalg.slogdet(np.eye(len(J)) - J)
            res = float(np.linalg.norm(np.tanh(net.A_np @ x + net.b_np) - x))
            info = item["fp"][j] if j < len(item["fp"]) else {}
            rows.append({"index": int(sign), "logabsdet": float(logdet), "rho": float(np.max(np.abs(w))),
                         "n_unstable": int(np.sum(np.abs(w) > 1 + 1e-9)),
                         "n_real_gt1": int(np.sum(real & (w.real > 1 + 1e-9))),
                         "residual": res, "shift": float(np.linalg.norm(x - x0)),
                         "n_unstable_C": info.get("n_unstable"), "dist_true_eq": info.get("dist_true_eq"),
                         "found_by": info.get("found_by")})
    true_like = [r for r in rows if (r["dist_true_eq"] if r["dist_true_eq"] is not None else 9) < 0.01]
    spur = [r for r in rows if r not in true_like]
    return {"key": item["key"], "regime": key, "set": item["set"], "N": item["N"], "rho": item["rho"], "eta": item["eta"],
            "seed": item["seed"], "K": len(rows), "index_sum": int(sum(r["index"] for r in rows)),
            "index_sum_true": int(sum(r["index"] for r in true_like)), "index_sum_spur": int(sum(r["index"] for r in spur)),
            "n_true_like": len(true_like), "residual_max": float(max((r["residual"] for r in rows), default=0.0)),
            "fixed_points": rows, "time_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    pi, pn = (int(x) for x in a.part.split("/"))
    index = json.loads((ROOT / "data" / "i_index.json").read_text(encoding="utf-8"))
    index = sorted(index, key=lambda r: (r["regime"], r["N"]))[pi::pn]       # по режимам — меньше перезагрузок данных
    if a.limit:
        index = index[: a.limit]
    bundle = np.load(ROOT / "data" / "i_bundle.npz")
    out = Path(os.environ.get("GPUQ_OUT", "out")) / "I"
    out.mkdir(parents=True, exist_ok=True)
    print(f"stage I: {len(index)} models, device {G.DEV}", flush=True)
    cache = {}
    for n, item in enumerate(index):
        name = item["key"].replace("/", "_")
        if (out / f"{name}.json").exists():
            continue
        G.report_progress(n, len(index), name)
        try:
            rec = analyze(item, bundle[f"{item['i']}_fps"], cache)
            (out / f"{name}.json").write_text(json.dumps(rec, default=float), encoding="utf-8")
            print(f"[{n + 1}/{len(index)}] {name}: K {rec['K']} Σind {rec['index_sum']} (истинные {rec['index_sum_true']}, "
                  f"паразитные {rec['index_sum_spur']}), res {rec['residual_max']:.1e}; {rec['time_s']} s", flush=True)
        except Exception:
            traceback.print_exc()
            (out / f"{name}.failed.txt").write_text(traceback.format_exc(), encoding="utf-8")
    G.report_progress(len(index), len(index), "done")


if __name__ == "__main__":
    main()

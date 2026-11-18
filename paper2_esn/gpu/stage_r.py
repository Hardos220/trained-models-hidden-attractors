"""Этап R: контроль «до обучения» — число равновесий замкнутого резервуара с обученным выходом и со случайным выходом
той же нормы Фробениуса (те же W, W_in, b). Отвечает на вопрос: создаёт ли паразитные равновесия обучение или они
свойственны классу моделей.

Запуск: python gpu/stage_r.py --regimes se3,se11,chua_I --sets H1 --N 300 --etas 1e-4 --seeds 0,1,2,3 --draws 2
Результат: $GPUQ_OUT/R/<режим>/<набор>_N<N>_eta<η>_s<seed>.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gpu"))
from hsearch import esn_pilot as P  # noqa: E402
import esngpu as G  # noqa: E402
import stage_c as SC  # noqa: E402
import stage_n as SN  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default="se3")
    ap.add_argument("--sets", default="H1")
    ap.add_argument("--N", type=int, default=300)
    ap.add_argument("--etas", default="1e-4")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--draws", type=int, default=2)
    a = ap.parse_args()
    out_root = Path(os.environ.get("GPUQ_OUT", "out")) / "R"
    for key in a.regimes.split(","):
        z, meta, _ = SC.load_regime(key)
        for s in a.sets.split(","):
            series, wash = SC.train_sets(z)[s]
            for eta in (float(e) for e in a.etas.split(",")):
                for seed in (int(x) for x in a.seeds.split(",")):
                    d = out_root / key
                    d.mkdir(parents=True, exist_ok=True)
                    name = f"{s}_N{a.N}_eta{eta:g}_s{seed}"
                    if (d / f"{name}.json").exists():
                        continue
                    esn = P.train_esn_multi(series, n_res=a.N, spectral_radius=0.9, ridge=eta, input_scale=0.5,
                                            density=0.05, washout=wash, seed=seed)
                    rec = {"regime": key, "set": s, "N": a.N, "eta": eta, "seed": seed, "variants": {}}
                    r0, _ = SN.evaluate(key, z, meta, esn, seed, eq_only=True)
                    rec["variants"]["trained"] = {k: r0[k] for k in ("K", "U", "index_sum")}
                    nrm = np.linalg.norm(esn["W_out"])
                    rng = np.random.default_rng(500 + seed)
                    for dr in range(a.draws):
                        Wr = rng.normal(size=esn["W_out"].shape)
                        Wr *= nrm / np.linalg.norm(Wr)
                        e = dict(esn)
                        e["W_out"] = Wr
                        e["A"] = esn["W"] + esn["W_in"] @ Wr
                        rr, _ = SN.evaluate(key, z, meta, e, seed, eq_only=True)
                        rec["variants"][f"random{dr}"] = {k: rr[k] for k in ("K", "U", "index_sum")}
                    e = dict(esn)
                    e["W_out"] = np.zeros_like(esn["W_out"])
                    e["A"] = esn["W"].copy()
                    rr, _ = SN.evaluate(key, z, meta, e, seed, eq_only=True)
                    rec["variants"]["zero"] = {k: rr[k] for k in ("K", "U", "index_sum")}
                    rec["W_out_norm"] = float(nrm)
                    (d / f"{name}.json").write_text(json.dumps(rec), encoding="utf-8")
                    print(key, name, {k: (v["K"], v["U"], v["index_sum"]) for k, v in rec["variants"].items()}, flush=True)


if __name__ == "__main__":
    main()

"""Сверка переписи в float32 с float64: одна сеть на режим, 3000 состояний (2000 поднятых L2 + 1000 из куба),
полная длина траекторий. Метки FP32 ставятся тем же разметчиком, что построен по прогону FP64."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gpu"))
from hsearch import esn_pilot as P  # noqa: E402
import esngpu as G  # noqa: E402
import pipeline as PL  # noqa: E402
import stage_c as SC  # noqa: E402

res = {}
cases = [(k, 0.02) for k in SC.REGIMES] + [("lorenz", 0.05)]
for key, lam_c in cases:
    rng = np.random.default_rng(5)
    if key == "lorenz":
        u = np.load(ROOT / "data" / "lorenz_l1.npz")["series"][:20000]
        esn = P.train_esn_multi([u], n_res=300, seed=0)
        h = 0.02
        X0 = np.load(ROOT / "data" / "lorenz.npz")["ics_phys"][:2000]
    else:
        z, meta, l1 = SC.load_regime(key)
        series, wash = SC.train_sets(z)["H1"]
        esn = P.train_esn_multi(series, n_res=300, spectral_radius=0.9, ridge=1e-4, washout=wash, seed=0)
        h = meta["Delta_t"]
        X0 = z["ics_uniform"][:2000]
    net = G.Net(esn, h=h)
    L = PL.Labeler(net, lam_c)
    X, ok = net.newton(rng.uniform(-1, 1, size=(2000, 300)))
    if ok.any():
        L.add_fps(X[ok])
    PL.training_attractor(net, L, esn["R_train"][-1])
    S = np.vstack([net.lift_const(esn, X0, w=100), rng.uniform(-1, 1, size=(1000, 300))])
    torch.cuda.synchronize(); t0 = time.time()
    o64 = net.run(S, 20000, 4000, seed=1)
    torch.cuda.synchronize(); t64 = time.time() - t0
    lab0 = L.label(o64)
    PL.discover(net, L, o64, lab0, n_life=20000, n_obs=4000)
    l64 = np.asarray(L.label(o64), dtype=str)
    t0 = time.time()
    o32 = net.run(S, 20000, 4000, seed=1, prec="fp32")
    torch.cuda.synchronize(); t32 = time.time() - t0
    l32 = np.asarray(L.label(o32), dtype=str)
    agree = float(np.mean(l64 == l32))
    lam_class = float(np.mean((o64["lam"] > lam_c) == (o32["lam"] > lam_c)))
    dis = {}
    for a, b in zip(l64[l64 != l32], l32[l64 != l32]):
        dis[f"{a}->{b}"] = dis.get(f"{a}->{b}", 0) + 1
    res[key] = {"agree": agree, "lam_class_agree": lam_class, "t64_s": round(t64, 1), "t32_s": round(t32, 1),
                "speedup": round(t64 / t32, 1), "labels64": PL.counts(l64), "disagreements": dis}
    print(key, json.dumps(res[key]), flush=True)
out = Path(__file__).resolve().parents[1] / "local_results"
out.mkdir(exist_ok=True)
(out / "check_fp32.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

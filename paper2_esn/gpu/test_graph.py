"""Сверка прогона с CUDA-графами и без них; замер скорости."""
import sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "gpu"))
from hsearch import esn_pilot as P
import esngpu as G
z = np.load(ROOT / "data" / "se11.npz")
e = P.train_esn_multi([z["train_H1"]], n_res=300, spectral_radius=0.9, ridge=1e-7, washout=500, seed=0)
net = G.Net(e, h=0.15)
rng = np.random.default_rng(1)
R0 = rng.uniform(-1, 1, size=(4096, 300)); v = rng.normal(size=R0.shape)
for nt, no in ((20, 100), (1000, 500)):
    G.USE_GRAPHS = False; a = net.run(R0, nt, no, v=v)
    G.USE_GRAPHS = True; b = net.run(R0, nt, no, v=v)
    print(nt, no, {k: float(np.max(np.abs(a[k].astype(float) - b[k].astype(float)))) for k in a}, flush=True)
for B in (4096, 16384):
    R = rng.uniform(-1, 1, size=(B, 300))
    for gflag in (False, True):
        G.USE_GRAPHS = gflag
        torch.cuda.synchronize(); t0 = time.time()
        net.run(R, 2000, 1000, chunk=B)
        torch.cuda.synchronize()
        print("B", B, "graphs", gflag, round(time.time() - t0, 2), "s for 2000+1000", flush=True)

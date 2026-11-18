"""A100: CUDA-графы против обычного цикла; метод Ньютона при разных размерах пачки и числе итераций."""
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
for B in (4096, 16384):
    R = rng.uniform(-1, 1, size=(B, 300))
    for gflag in (False, True):
        G.USE_GRAPHS = gflag
        torch.cuda.synchronize(); t0 = time.time()
        net.run(R, 4000, 1000, chunk=B)
        torch.cuda.synchronize()
        print("B", B, "graphs", gflag, round(time.time() - t0, 2), "s for 4000+1000", flush=True)
S = rng.uniform(-1, 1, size=(4096, 300))
for chunk in (512, 1024, 2048):
    for it in (25, 40):
        torch.cuda.synchronize(); t0 = time.time()
        X, ok = net.newton(S, max_iter=it, chunk=chunk)
        torch.cuda.synchronize()
        print("newton chunk", chunk, "iter", it, round(time.time() - t0, 2), "s, ok", int(ok.sum()), flush=True)
print("peak GB", round(torch.cuda.max_memory_reserved() / 2**30, 2))

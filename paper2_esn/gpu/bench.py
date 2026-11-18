"""Сверка GPU-ядра с CPU-эталоном esn_pilot и замер скорости на A100."""
import json
import os
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

out_dir = Path(os.environ.get("GPUQ_OUT", "out"))
out_dir.mkdir(parents=True, exist_ok=True)
res = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "torch": torch.__version__}
print(res, flush=True)

z = np.load(ROOT / "data" / "se11.npz")
meta = json.load(open(ROOT / "data" / "se11.json", encoding="utf-8"))
t0 = time.time()
e = P.train_esn_multi([z["train_H1"]], n_res=300, spectral_radius=0.9, ridge=1e-7, washout=500, seed=0)
res["train_s"] = round(time.time() - t0, 1)
cpu = P.ClosedLoop(e, h=meta["Delta_t"])
gpu = G.Net(e, h=meta["Delta_t"])
rng = np.random.default_rng(1)
R0 = rng.uniform(-1, 1, size=(32, 300))
R0[:16] = e["R_train"][rng.integers(0, len(e["R_train"]), 16)]
v = np.random.default_rng(5).normal(size=R0.shape)
oc = cpu.run(R0, n_tr=2000, n_obs=500, rng=np.random.default_rng(5))
og = gpu.run(R0, n_tr=2000, n_obs=500, v=v)
res["fp_equal"] = bool(np.array_equal(oc["fp"], og["fp"]))
res["max_dmean"] = float(np.max(np.abs(oc["x_mean"] - og["x_mean"])))
res["max_dlam"] = float(np.max(np.abs(oc["lam"] - og["lam"])))
res["max_dfinal"] = float(np.max(np.abs(oc["final"] - og["final"])))
oc2 = cpu.run(R0, n_tr=20, n_obs=30, rng=np.random.default_rng(5))
og2 = gpu.run(R0, n_tr=20, n_obs=30, v=v)
res["short_max_dfinal"] = float(np.max(np.abs(oc2["final"] - og2["final"])))
res["short_max_dmean"] = float(np.max(np.abs(oc2["x_mean"] - og2["x_mean"])))
print("run vs CPU:", {k: res[k] for k in ("fp_equal", "max_dmean", "max_dlam", "max_dfinal", "short_max_dfinal", "short_max_dmean")}, flush=True)

# Ньютон
S = rng.uniform(-1, 1, size=(60, 300))
Xc, okc = P.newton_tanh(cpu, S)
Xg, okg = gpu.newton(S)
both = okc & okg
res["newton_ok_cpu"], res["newton_ok_gpu"] = int(okc.sum()), int(okg.sum())
res["newton_same_start_frac"] = float(np.mean(np.max(np.abs(Xc[both] - Xg[both]), axis=1) < 1e-9)) if both.any() else None
Uc, _ = P.dedupe([x for x, k in zip(Xc, okc) if k]); Ug, _ = P.dedupe([x for x, k in zip(Xg, okg) if k])
res["newton_sets"] = [len(Uc), len(Ug), int(sum(min(np.linalg.norm(x - y) for y in Ug) < 1e-9 for x in Uc))]
print("newton:", res["newton_ok_cpu"], res["newton_ok_gpu"], res["newton_same_start_frac"], res["newton_sets"], flush=True)

# скорость прогона
torch.cuda.synchronize()
speed = {}
for B in (4096, 16384):
    R = G.T(rng.uniform(-1, 1, size=(B, 300)))
    torch.cuda.synchronize(); t0 = time.time()
    gpu.advance(R, 1000)
    torch.cuda.synchronize(); ttr = (time.time() - t0) / 1000
    t0 = time.time()
    gpu.run(R, n_tr=0, n_obs=300, chunk=B)
    torch.cuda.synchronize(); tob = (time.time() - t0) / 300
    fl = 2 * B * 300 * 300
    speed[B] = {"tr_step_ms": round(ttr * 1e3, 3), "obs_step_ms": round(tob * 1e3, 3), "tr_TFLOPS": round(fl / ttr / 1e12, 2)}
    print("B", B, speed[B], flush=True)
res["speed_N300"] = speed
# Ньютон: время на 2048 стартов
S = rng.uniform(-1, 1, size=(2048, 300))
torch.cuda.synchronize(); t0 = time.time()
Xg, okg = gpu.newton(S)
torch.cuda.synchronize()
res["newton_2048_s"] = round(time.time() - t0, 2)
res["newton_2048_ok"] = int(okg.sum())
print("newton 2048:", res["newton_2048_s"], "s, ok", res["newton_2048_ok"], flush=True)

# hard-tanh: перебор против CPU (N = 8) и время N = 12, 14, 16
u = z["train_H1"]
for N in (8, 12, 14, 16):
    eh = P.train_esn_multi([u[:5000]], n_res=N, seed=3, density=1.0, activation="hardtanh", washout=200, spectral_radius=1.2)
    mh = G.Net(eh, h=0.15)
    torch.cuda.synchronize(); t0 = time.time()
    pts, regs, deg = G.hardtanh_enumerate(mh.A_np, mh.b_np)
    torch.cuda.synchronize(); tg = time.time() - t0
    rec = {"n_fp": int(len(pts)), "degenerate": deg, "gpu_s": round(tg, 2)}
    if N <= 8:
        pc, _, _ = P.hardtanh_fixed_points_bruteforce(mh.A_np, mh.b_np)
        rec["cpu_n_fp"] = len(pc)
        rec["match"] = bool(len(pc) == len(pts) and all(min(np.linalg.norm(p - q) for q in pts) < 1e-10 for p in pc))
    Xn, okn = mh.newton(np.random.default_rng(2).uniform(-1, 1, size=(4000, N)))
    U, _ = P.dedupe([x for x, k in zip(Xn, okn) if k], tol=1e-9)
    rec["newton_distinct"] = len(U)
    rec["newton_outside_enum"] = int(sum(min(np.linalg.norm(x - y) for y in pts) > 1e-8 for x in U)) if len(pts) else len(U)
    res[f"hardtanh_N{N}"] = rec
    print("hard-tanh", N, rec, flush=True)
res["peak_mem_gb"] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)
json.dump(res, open(out_dir / "bench.json", "w"), indent=1)
print("done", res["peak_mem_gb"], "GB", flush=True)

"""Stage S (saturation): completeness of the equilibrium sets of trained echo state networks (check).

For every network of a design list (data/sat_design.json, written by paper2_esn/gpu/make_sat_design.py) the network is
rebuilt exactly as in stage N / stage C (same training call, same seed), and the equilibrium search of stage N is repeated
with the original number of Newton starts times 1, 2, 4 (and 8):

  block 0  = the stage N search itself, with the same random numbers (rng = default_rng(3000 + seed)): nQ uniform points
             of the cube, nT perturbed training states, the lifted true equilibria with 200 noisy copies each (Newton,
             40 iterations), then nD uniform points with deflation against everything found (60 iterations);
             nQ, nT, nD = 5000, 2500, 1000 for N < 1000 and 2000, 1000, 500 otherwise;
  block b  = the same composition of starts from an independent stream default_rng([seed, 7919, b]), deflation against
             everything found so far (b = 1 .. mult - 1).
"x k starts" = union over blocks 0 .. k-1. After block 0 the hidden attractor is matched exactly as in stage N
(synchronization on 100 samples of the true trajectory, same rng), which also reproduces the stored stage N label.
If the network also belongs to the full-protocol ensemble (stage C, same training call), its stored equilibria
(data/sat_cfps.npz) are refined by Newton on the rebuilt network and merged, flagged in_C.

For every equilibrium: physical image x = scale W_out r*, whether it lies in the region Q (box_lo..box_hi), stability,
index, the first block that found it, and how many plain (non-deflated) and deflated Newton runs converged to it.
Excitation test (stage N radii 1e-6, 1e-4, 1e-3; 50 ball points plus the unstable directions) on ALL unstable
equilibria, so that the label of the hidden attractor can be recomputed for every start budget.

Checkpoint after block 0 + match and after every block ($GPUQ_OUT/S/<regime>/<name>.ckpt.pkl); a finished network
($GPUQ_OUT/S/<regime>/<name>.json) is skipped.

    python gpu/stage_sat.py --design data/sat_design.json --groups n2000a [--quick]
"""
import argparse
import json
import os
import pickle
import sys
import time
import traceback
from collections import Counter
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
import stage_n as SN  # noqa: E402

LAM_C = SC.LAM_C
SALT = 7919
RADII = (1e-6, 1e-4, 1e-3)
if os.environ.get("RMQ_CPU") and G.DEV.type == "cpu":
    torch.set_num_threads(int(os.environ["RMQ_CPU"]))


def n_starts(N, quick=False):
    if quick:
        return 200, 100, 50
    nQ, nT = (5000, 2500) if N < 1000 else (2000, 1000)
    return nQ, nT, (1000 if N < 1000 else 500)


def model_name(e):
    return f"{e['set']}_N{e['N']}_rho{e['rho']}_eta{e['eta']:g}_s{e['seed']}"


class Tracker:
    """Per equilibrium: first block, plain hits, deflated hits."""

    def __init__(self):
        self.first, self.plain, self.defl = {}, Counter(), Counter()

    def add(self, L, X, b, kind):
        if not len(X):
            return 0
        n0 = len(L.fps)
        for j in L.add_fps(X):
            self.first.setdefault(j, b)
            (self.plain if kind == "plain" else self.defl)[j] += 1
        return len(L.fps) - n0


def starts_of_block(net, esn, rng, N, eqs, L_scale, nQ, nT):
    """Same expressions and order of random draws as stage_n.evaluate."""
    starts = [rng.uniform(-1, 1, size=(nQ, N)),
              np.clip(esn["R_train"][rng.integers(0, len(esn["R_train"]), nT)] + 0.05 * rng.normal(size=(nT, N)), -1, 1)]
    for e in eqs:
        xs = np.asarray(e)[None] + 0.01 * L_scale * rng.normal(size=(200, len(e)))
        starts.append(net.lift_const(esn, np.vstack([np.asarray(e)[None], xs]), w=100))
    return np.vstack(starts)


def run_block(net, esn, L, T, rng, b, N, eqs, L_scale, nQ, nT, nD):
    t0 = time.time()
    S = starts_of_block(net, esn, rng, N, eqs, L_scale, nQ, nT)
    X, ok = net.newton(S, max_iter=40)
    n_ok_p = int(ok.sum())
    new_p = T.add(L, X[ok], b, "plain") if ok.any() else 0
    new_d, n_ok_d = 0, 0
    if L.fps:      # stage N draws the deflation starts only when something was found
        X, ok = net.newton(rng.uniform(-1, 1, size=(nD, N)), max_iter=60, deflate=np.asarray(L.fps))
        n_ok_d = int(ok.sum())
        new_d = T.add(L, X[ok], b, "defl") if ok.any() else 0
    return {"block": b, "n_plain": len(S), "n_plain_ok": n_ok_p, "n_defl": nD, "n_defl_ok": n_ok_d,
            "new_plain": new_p, "new_defl": new_d, "K_after": len(L.fps), "time_s": round(time.time() - t0, 1)}


def hidden_match(key, z, meta, net, esn, L, rng, seed, n_tr, n_obs):
    """Copy of the hidden-attractor part of stage_n.evaluate (same random numbers when rng continues block 0)."""
    match = {}
    for y in meta["hidden"]:
        Y = z[f"att_{y}"]
        ykind = "chaos" if meta["attractors"][y]["kind"] == "chaotic" else "regular"
        s0 = rng.integers(0, len(Y) - 101, 64)
        U = np.stack([Y[s:s + 100] for s in s0])
        R = net.lift_drive(esn, U, 100)
        o = net.run(R, n_tr=n_tr, n_obs=n_obs, seed=seed + 5)
        chaos = o["lam"] > LAM_C
        keep = ~o["fp"] & (chaos if ykind == "chaos" else ~chaos)
        m = {"frac_osc": float(np.mean(~o["fp"])), "frac_kind": float(np.mean(keep))}
        if keep.sum() >= 8:
            ref = PL.make_ref(PL.feats(o, net.scale)[keep], o["lam"][keep], ykind, o["final"][np.where(keep)[0][0]])
            Xm = net.samples(ref["rep"], 10000, burn=5000)
            corr = PL.correspondence(Xm, {y: Y[:10000]})[y]
            m.update(corr)
            m["reproduced"] = bool(corr.get("delta_att", 9) < 0.2 and corr.get("d_H", 9) < 0.2)
            m["attractor"] = f"a{len(L.refs)}"
            m["lam"] = ref["lam"]
            L.refs.append(ref)
        else:
            m["reproduced"] = False
        match[y] = m
    return match


def save_ckpt(path, st):
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(st, f)
    os.replace(tmp, path)


def analyze(e, z, meta, cfps, out_dir, quick=False):
    key, set_name, N, rho, eta, seed = e["regime"], e["set"], int(e["N"]), float(e["rho"]), float(e["eta"]), int(e["seed"])
    mult = int(e.get("mult", 8))
    name = model_name(e)
    ck = out_dir / f"{name}.ckpt.pkl"
    t_start = time.time()
    series, wash = SC.train_sets(z)[set_name]
    esn = P.train_esn_multi(series, n_res=N, spectral_radius=rho, ridge=eta, input_scale=0.5, density=0.05,
                            washout=wash, seed=seed)
    t_train = time.time() - t_start
    net = G.Net(esn, h=meta["Delta_t"])
    n_tr, n_obs = (4000, 1000) if quick else (20000, 4000)
    true_eq = [x["point"] for x in meta["equilibria"]]
    L_scale = float(np.max(np.array(meta["box_hi"]) - np.array(meta["box_lo"]))) / 2
    eq_line = key == "le1"
    eqs = [[0.0, 0.0, zz] for zz in np.linspace(-10, -0.5, 20)] if eq_line else list(true_eq)
    nQ, nT, nD = n_starts(N, quick)
    L = PL.Labeler(net, LAM_C)
    st = None
    if ck.exists():
        try:
            with open(ck, "rb") as f:
                st = pickle.load(f)
            print(f"  resume {name} after block {st['blocks_done'] - 1}", flush=True)
        except Exception:
            st = None
    if st is None:
        T = Tracker()
        rng = np.random.default_rng(3000 + seed)
        blk = [run_block(net, esn, L, T, rng, 0, N, eqs, L_scale, nQ, nT, nD)]
        K0 = len(L.fps)
        t0 = time.time()
        match = hidden_match(key, z, meta, net, esn, L, rng, seed, n_tr, n_obs)
        t_match = time.time() - t0
        st = {"fps": [np.asarray(x) for x in L.fps], "refs": L.refs, "first": T.first, "plain": T.plain, "defl": T.defl,
              "blocks": blk, "blocks_done": 1, "K0": K0, "match": match, "t_match": t_match}
        save_ckpt(ck, st)
    L.fps = [np.asarray(x) for x in st["fps"]]
    L.refs = st["refs"]
    T = Tracker()
    T.first, T.plain, T.defl = st["first"], st["plain"], st["defl"]
    for b in range(st["blocks_done"], mult):
        rng_b = np.random.default_rng([seed, SALT, b])
        st["blocks"].append(run_block(net, esn, L, T, rng_b, b, N, eqs, L_scale, nQ, nT, nD))
        st.update(fps=[np.asarray(x) for x in L.fps], first=T.first, plain=T.plain, defl=T.defl, blocks_done=b + 1)
        save_ckpt(ck, st)
        print(f"    block {b}: K {len(L.fps)} ({st['blocks'][-1]['time_s']} s)", flush=True)
    n_search = len(L.fps)
    # stored full-protocol equilibria (stage C) of the same network
    in_C, c_info = set(), None
    if cfps is not None and len(cfps):
        Xc, okc = net.newton(np.asarray(cfps, float), max_iter=20)
        res0 = [float(np.linalg.norm(np.tanh(net.A_np @ x + net.b_np) - x)) for x in np.asarray(cfps, float)]
        n0 = len(L.fps)
        ids = L.add_fps([Xc[i] if okc[i] else np.asarray(cfps[i], float) for i in range(len(cfps))])
        in_C = set(ids)
        c_info = {"n_stored": int(len(cfps)), "refined_ok": int(okc.sum()), "max_residual_stored": float(max(res0)),
                  "n_not_in_search": int(len(L.fps) - n0)}
    # equilibria: information
    t0 = time.time()
    lo, hi = np.array(meta["box_lo"]), np.array(meta["box_hi"])
    fpi = []
    for i, x in enumerate(L.fps):
        info = PL.fp_info(net, x, true_eq, L_scale, eq_line)
        xp = np.asarray(info["x"])
        info.update(SN.eq_extra(net, x))
        info.update({"inside": bool(np.all(xp >= lo) and np.all(xp <= hi)), "first_block": T.first.get(i),
                     "hits_plain": int(T.plain.get(i, 0)), "hits_defl": int(T.defl.get(i, 0)), "in_C": i in in_C,
                     "residual": float(np.linalg.norm(np.tanh(net.A_np @ x + net.b_np) - x))})
        fpi.append(info)
    t_info = time.time() - t0

    def summary(sel):
        f = [fpi[i] for i in sel]
        return {"K": len(f), "U": int(sum(x["n_unstable"] > 0 for x in f)),
                "U_in": int(sum(x["n_unstable"] > 0 and x["inside"] for x in f)),
                "K_in": int(sum(x["inside"] for x in f)), "index_sum": int(sum(x["index"] for x in f))}
    cum = {}
    for k in sorted({1, 2, 4, 8, mult}):
        if k <= mult:
            cum[f"x{k}"] = summary([i for i in range(n_search) if T.first.get(i, 99) < k])
    if in_C:
        cum["C"] = summary(sorted(in_C))
        cum["C_and_x1"] = summary(sorted(in_C | {i for i in range(n_search) if T.first.get(i, 99) < 1}))
        cum["C_and_all"] = summary(list(range(len(fpi))))
    # excitation test on all unstable equilibria
    t0 = time.time()
    n_before = len(L.fps)
    unst = [i for i, f in enumerate(fpi) if f["n_unstable"] > 0]
    nb = PL.radius_test(net, L, unst, radii=RADII, n_ball=50, n_tr=n_tr, n_obs=n_obs)
    hits = PL.hits_by_attractor(nb)
    exc = {}
    for y, m in st["match"].items():
        a = m.get("attractor")
        if a:
            fr = hits.get(a, {"from": {}})["from"]
            exc[y] = {str(i): float(r) for i, r in fr.items()}
            for i in range(len(fpi)):
                fpi[i].setdefault("r_exc", {})[y] = fr.get(str(i))
    t_radius = time.time() - t0
    rec = {"regime": key, "set": set_name, "N": N, "rho": rho, "eta": eta, "seed": seed, "src": e.get("src"),
           "mult": mult, "train_rmse": esn["train_rmse"], "n_starts_block": {"cube": nQ, "train": nT, "lifted": 201 * len(eqs),
           "deflation": nD}, "blocks": st["blocks"], "K0": st["K0"], "cum": cum, "C_stored": c_info,
           "hidden_match": st["match"], "excitation_from": exc, "fixed_points": fpi,
           "n_found_by_label_in_radius_test": len(L.fps) - n_before, "radius_counts": nb,
           "time_s": {"train": round(t_train, 1), "blocks": round(sum(x["time_s"] for x in st["blocks"]), 1),
                      "match": round(st["t_match"], 1), "info": round(t_info, 1), "radius": round(t_radius, 1)},
           "time_total_s": round(time.time() - t_start, 1)}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="data/sat_design.json")
    ap.add_argument("--groups", default="", help="comma-separated design groups (default: all)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mult", type=int, default=0, help="override the number of blocks (tests)")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    design = json.loads((ROOT / a.design).read_text(encoding="utf-8"))
    nets = design["networks"]
    if a.groups:
        gs = set(a.groups.split(","))
        nets = [e for e in nets if e["group"] in gs]
    if a.limit:
        nets = nets[: a.limit]
    cf_path = ROOT / "data" / "sat_cfps.npz"
    cf = np.load(cf_path) if cf_path.exists() else None
    out_root = Path(os.environ.get("GPUQ_OUT", "out")) / "S"
    print(f"stage S: {len(nets)} networks, device {G.DEV}", flush=True)
    cache = {}
    for n, e in enumerate(nets):
        if a.mult:
            e = dict(e, mult=a.mult)
        key = e["regime"]
        name = model_name(e)
        d = out_root / key
        d.mkdir(parents=True, exist_ok=True)
        if (d / f"{name}.json").exists():
            continue
        if key not in cache:
            cache = {key: SC.load_regime(key)}
        z, meta, _ = cache[key]
        G.report_progress(n, len(nets), f"{key} {name}")
        ckey = f"{key}/{name}"
        cfps = cf[ckey] if cf is not None and ckey in cf.files else None
        try:
            rec = analyze(e, z, meta, cfps, d, quick=a.quick)
            rec["group"] = e.get("group")
            (d / f"{name}.json").write_text(json.dumps(rec, default=SC.default), encoding="utf-8")
            ck = d / f"{name}.ckpt.pkl"
            if ck.exists():
                ck.unlink()
            cum = {k: (v["K"], v["U"], v["U_in"], v["index_sum"]) for k, v in rec["cum"].items()}
            hm = {y: (m.get("reproduced"), round(m.get("delta_att", 9), 3)) for y, m in rec["hidden_match"].items()}
            print(f"[{n + 1}/{len(nets)}] {key} {name}: (K,U,U_in,isum) {cum} {hm}; {rec['time_total_s']} s "
                  f"{rec['time_s']}", flush=True)
        except Exception as exc:
            traceback.print_exc()
            if G.fatal_cuda(exc):
                print("fatal CUDA error, stopping the process", flush=True)
                sys.exit(3)
            (d / f"{name}.failed.txt").write_text(traceback.format_exc(), encoding="utf-8")
    G.report_progress(len(nets), len(nets), "done")


if __name__ == "__main__":
    main()

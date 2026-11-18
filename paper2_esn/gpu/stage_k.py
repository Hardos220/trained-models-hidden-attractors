"""Этап K: множитель s матрицы W после обучения для моделей C, восстановивших скрытый аттрактор.

Отбор делается локально по результатам C (gpu/select_k.py) и передаётся файлом data/k_selection.json.
Запуск: python gpu/stage_k.py [--quick]
Результат: $GPUQ_OUT/K/<режим>_<модель>.json
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
import pipeline as PL  # noqa: E402
import stage_c as SC  # noqa: E402

S_GRID = np.round(np.arange(0.80, 1.2001, 0.02), 2)
FINE_GRID = np.round(1.0 + np.array([-0.01, -0.007, -0.004, -0.002, -0.001, -0.0005, 0.0, 0.0005, 0.001, 0.002, 0.004,
                                     0.007, 0.01]), 4)


def scaled(esn, s):
    e = dict(esn)
    e["W"] = s * np.asarray(esn["W"])
    return e


def analyze(sel, quick=False, grid_all=None):
    key, set_name = sel["regime"], sel["set"]
    z, meta, l1 = SC.load_regime(key)
    series, wash = SC.train_sets(z)[set_name]
    esn = P.train_esn_multi(series, n_res=300, spectral_radius=sel["rho"], ridge=sel["eta"], input_scale=0.5,
                            density=0.05, washout=wash, seed=sel["seed"])
    Y = z[f"att_{sel['target']}"][:10000]
    n_tr, n_obs = (4000, 1000) if quick else (20000, 4000)
    rng = np.random.default_rng(4000 + sel["seed"])
    # X при s = 1: состояние-представитель из результата C
    rep0 = np.asarray(sel["rep"], float)
    rows = {}
    for direction in (+1, -1):
        rep = rep0.copy()
        fps_prev = [np.asarray(x) for x in sel.get("fps", [])]
        prevX = None
        lost = False
        grid = [s for s in (S_GRID if grid_all is None else grid_all) if (s >= 1.0 if direction > 0 else s <= 1.0)]
        grid = sorted(grid, reverse=direction < 0)
        for s in grid:
            if direction < 0 and abs(s - 1.0) < 1e-9 and 1.0 in rows:
                continue
            t0 = time.time()
            es = scaled(esn, s)
            net = G.Net(es, h=meta["Delta_t"])
            rec = {"s": float(s)}
            # неподвижные точки: продолжение и новые старты
            St = [np.vstack(fps_prev)] if fps_prev else []
            St += [rng.uniform(-1, 1, size=(1000, 300)),
                   np.clip(esn["R_train"][rng.integers(0, len(esn["R_train"]), 1000)] + 0.05 * rng.normal(size=(1000, 300)), -1, 1)]
            X, ok = net.newton(np.vstack(St))
            L = PL.Labeler(net, SC.LAM_C)
            if ok.any():
                L.add_fps(X[ok])
            fps_prev = list(L.fps)
            rec["n_fp"] = len(L.fps)
            rec["n_stable"] = int(sum(np.max(np.abs(net.stability(x)[0])) < 1 for x in L.fps))
            if not lost:
                C0 = np.clip(rep[None] + 1e-6 * rng.normal(size=(64, 300)), -1, 1)
                o = net.run(C0, n_tr=n_tr, n_obs=n_obs, seed=int(s * 1000))
                if np.mean(o["fp"]) >= 0.5:
                    lost, rec["lost"] = True, "fixed_point"
                else:
                    chaos = o["lam"] > SC.LAM_C
                    kind = "chaos" if np.mean(chaos[~o["fp"]]) >= 0.5 else "regular"
                    keep = ~o["fp"] & (chaos if kind == "chaos" else ~chaos)
                    ref = PL.make_ref(PL.feats(o, net.scale)[keep], o["lam"][keep], kind, o["final"][np.where(keep)[0][0]])
                    Xm = net.samples(ref["rep"], 10000, burn=10000)
                    if prevX is not None and np.all(np.isfinite(Xm)) and PL.delta_att(Xm, prevX) > 0.5:
                        lost, rec["lost"] = True, "jump"
                    else:
                        L.refs = [ref]
                        rep = ref["rep"]
                        prevX = Xm
                        rec.update({"kind": kind, "lam": ref["lam"], "metrics": PL.correspondence(Xm, {"Y": Y})["Y"]})
                        unst = [i for i in range(len(L.fps)) if np.max(np.abs(net.stability(L.fps[i])[0])) > 1 + 1e-9]
                        nb = PL.radius_test(net, L, unst, n_ball=50 if quick else 200, n_tr=n_tr, n_obs=n_obs)
                        h = PL.hits_by_attractor(nb).get("a0", {"r_hat": np.inf})
                        rec["r_hat"] = None if not np.isfinite(h["r_hat"]) else h["r_hat"]
                        Rl = net.lift_const(es, z["ics_uniform"][:2000], w=100)
                        Rq = rng.uniform(-1, 1, size=(2000, 300))
                        o2 = net.run(np.vstack([Rl, Rq]), n_tr=n_tr, n_obs=n_obs, seed=7)
                        l2 = np.asarray(L.label(o2), dtype=str)
                        rec["p_lift"] = float(np.mean(l2[:2000] == "a0"))
                        rec["p_cube"] = float(np.mean(l2[2000:] == "a0"))
            rec["nrmse_1step_Y"] = SC.one_step_nrmse(es, Y)
            # обучающий аттрактор A_tr(s)
            Ltmp = PL.Labeler(net, SC.LAM_C, L.fps)
            at = PL.training_attractor(net, Ltmp, esn["R_train"][-1])
            rec["A_tr_kind"] = at["kind"]
            if at["kind"] in ("chaos", "regular"):
                Xa = net.samples(Ltmp.refs[0]["rep"], 10000, burn=5000)
                atts = {k[4:]: z[k][:10000] for k in z.files if k.startswith("att_")}
                rec["A_tr_delta_att"] = {k: v["delta_att"] for k, v in PL.correspondence(Xa, atts).items()}
            rec["time_s"] = round(time.time() - t0, 1)
            rows[float(s)] = rec
            print(f"  s={s:.2f}: {rec.get('kind', rec.get('lost', '-'))} Δatt {rec.get('metrics', {}).get('delta_att')} "
                  f"r̂ {rec.get('r_hat')} p_lift {rec.get('p_lift')} FP {rec['n_fp']}/{rec['n_stable']} ({rec['time_s']} s)", flush=True)
    return {"selection": {k: v for k, v in sel.items() if k not in ("rep", "fps")}, "rows": [rows[k] for k in sorted(rows)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--sel", default="data/k_selection.json")
    ap.add_argument("--grid", default="coarse", help="coarse (0.80–1.20 через 0.02) | fine (1 ± 0.0005…0.01)")
    ap.add_argument("--part", default="0/1", help="i/n — обработать каждую n-ю модель начиная с i")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    sels = json.loads((ROOT / a.sel).read_text(encoding="utf-8"))
    pi, pn = (int(x) for x in a.part.split("/"))
    sels = sels[pi::pn]
    grid_all = FINE_GRID if a.grid == "fine" else None
    out_dir = Path(os.environ.get("GPUQ_OUT", "out")) / "K"
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, sel in enumerate(sels):
        name = f"{sel['regime']}_{sel['name']}{a.tag}"
        if a.only and a.only not in name:
            continue
        skip = set(Path(a.skip).read_text(encoding="utf-8").split()) if a.skip and Path(a.skip).exists() else set()
        if (out_dir / f"{name}.json").exists() or name in skip:
            continue
        G.report_progress(n, len(sels), name)
        print(f"[{n + 1}/{len(sels)}] {name}", flush=True)
        try:
            res = analyze(sel, quick=a.quick, grid_all=grid_all)
            (out_dir / f"{name}.json").write_text(json.dumps(res, default=float), encoding="utf-8")
        except Exception as exc:
            traceback.print_exc()
            if G.fatal_cuda(exc):
                print("fatal CUDA error, stopping the process", flush=True)
                sys.exit(3)
            (out_dir / f"{name}.failed.txt").write_text(traceback.format_exc(), encoding="utf-8")
    G.report_progress(len(sels), len(sels), "done")


if __name__ == "__main__":
    main()

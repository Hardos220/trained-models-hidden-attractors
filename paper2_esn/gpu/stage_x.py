"""Этап X: точный эталон для теории (сети hard-tanh этапа H, точное множество равновесий E).

Для каждой сети: индексы всех точных равновесий ind(e) = sign det(I − D A) (D — линейные нейроны области),
проверка Σ ind = 1 по E и по найденному методом Ньютона E_f (свидетельство неполноты), число вещественных
собственных значений > 1; тест по неустойчивому подпространству: из каждого неустойчивого равновесия —
точки e ± δ q_j по базису неустойчивого подпространства и 32 случайных направления в нём, δ = 1e-6, 1e-4;
куда приходят траектории. Сравнение с эталонной меткой и меткой протокола.

Входы: data/x_index.json, data/x_bundle.npz (gpu/make_bundles.py), data/lorenz_l1.npz.
Запуск: python gpu/stage_x.py [--part i/n] [--quick]
Результат: $GPUQ_OUT/X/<N>_<сеть>.json
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
import pipeline as PL  # noqa: E402

LAM_C = 0.05
DELTAS = (1e-6, 1e-4)
if os.environ.get("RMQ_CPU"):
    torch.set_num_threads(int(os.environ["RMQ_CPU"]))


def eig_info(J):
    w = np.linalg.eigvals(J)
    real = np.abs(w.imag) <= 1e-10 * np.maximum(1.0, np.abs(w))
    sign, _ = np.linalg.slogdet(np.eye(len(J)) - J)
    return w, {"rho": float(np.max(np.abs(w))), "n_unstable": int(np.sum(np.abs(w) > 1 + 1e-9)),
               "n_real_gt1": int(np.sum(real & (w.real > 1 + 1e-9))), "index": int(sign)}


def unstable_basis(J):
    w, V = np.linalg.eig(J)
    cols = []
    for k in np.where(np.abs(w) > 1 + 1e-9)[0]:
        cols.append(V[:, k].real)
        if abs(w[k].imag) > 1e-10:
            cols.append(V[:, k].imag)
    if not cols:
        return np.zeros((len(J), 0))
    Q, R = np.linalg.qr(np.column_stack(cols))
    keep = np.abs(np.diag(R)) > 1e-10
    return Q[:, keep]


def build_refs(net, reps, kinds, rng):
    refs = []
    for k, rep in enumerate(reps):
        orb = G.N_(net.orbit(rep, 2500, every=10, burn=500))
        picks = orb[rng.integers(0, len(orb), 48)]
        o = net.run(picks, n_tr=1000, n_obs=2000, seed=11 + k)
        if not (~o["fp"]).any():                       # от представителя все копии сошлись к точке
            refs.append(None)
            continue
        kind = kinds.get(f"a{k}") or ("chaos" if np.median(o["lam"][~o["fp"]]) > LAM_C else "regular")
        if kind is None:
            refs.append(None)
            continue
        chaos = o["lam"] > LAM_C
        keep = ~o["fp"] & (chaos if kind == "chaos" else ~chaos)
        if not keep.any():
            refs.append(None)
            continue
        refs.append(PL.make_ref(PL.feats(o, net.scale)[keep], o["lam"][keep], kind, orb[len(orb) // 2]))
    return refs


def analyze(u, item, E, regs, reps, quick=False, n_rand=None):
    t0 = time.time()
    N, rho, eta, seed = item["N"], item["rho"], item["eta"], item["seed"]
    esn = P.train_esn_multi([u], n_res=N, spectral_radius=rho, ridge=eta, input_scale=0.5, density=min(1.0, 10.0 / N),
                            washout=500, seed=seed, activation="hardtanh")
    net = G.Net(esn, h=0.02)
    A, b = net.A_np, net.b_np
    res_fp = float(max((np.linalg.norm(np.clip(A @ x + b, -1, 1) - x) for x in E), default=0.0))
    eq = []
    for x in E:
        D = (np.abs(A @ x + b) < 1).astype(float)
        J = D[:, None] * A
        _, inf = eig_info(J)
        eq.append({**inf, "n_linear": int(D.sum())})
    Ef = set(item["E_f"])
    rec = {"key": item["key"], "N": N, "rho": rho, "eta": eta, "seed": seed, "fp_residual_max": res_fp,
           "E_n": len(E), "Ef_n": len(Ef), "equilibria": eq,
           "index_sum_E": int(sum(e["index"] for e in eq)),
           "index_sum_Ef": int(sum(eq[i]["index"] for i in Ef)),
           "n_unstable_E": int(sum(e["n_unstable"] > 0 for e in eq)),
           "n_unstable_Ef": int(sum(eq[i]["n_unstable"] > 0 for i in Ef))}
    # тест по неустойчивому подпространству
    rng = np.random.default_rng(7000 + seed)
    kinds = {a: v.get("kind") for a, v in item["attractors"].items()}
    refs = build_refs(net, reps, kinds, rng)
    L = PL.Labeler(net, LAM_C, list(E))
    L.refs = [r if r is not None else {"kind": "chaos", "mu": np.full(4 * net.d, 1e9), "sd": np.zeros(4 * net.d),
                                       "floor": np.ones(4 * net.d), "rep": reps[k], "lam": 0.0}
              for k, r in enumerate(refs)]
    X0, tags = [], []
    n_rand = n_rand or (8 if quick else 32)
    for i, x in enumerate(E):
        if eq[i]["n_unstable"] == 0:
            continue
        D = (np.abs(A @ x + b) < 1).astype(float)
        Q = unstable_basis(D[:, None] * A)
        if Q.shape[1] == 0:
            continue
        dirs = [Q[:, j] for j in range(Q.shape[1])] + [-Q[:, j] for j in range(Q.shape[1])]
        C = Q @ rng.normal(size=(Q.shape[1], n_rand))
        dirs += list((C / np.linalg.norm(C, axis=0)).T)
        for dl in DELTAS:
            for v in dirs:
                X0.append(x + dl * v)
                tags.append((i, dl))
    sub = {}
    if X0:
        n_tr, n_obs = (4000, 1000) if quick else (20000, 4000)
        o = net.run(net.clip_state(np.array(X0)), n_tr=n_tr, n_obs=n_obs, seed=seed + 99)
        lab = np.asarray(L.label(o), dtype=str)
        for (i, dl), l in zip(tags, lab):
            sub.setdefault(f"{i}|{dl:g}", {}).setdefault(l, 0)
            sub[f"{i}|{dl:g}"][l] += 1
    rec["subspace_counts"] = sub
    att = {}
    for a, v in item["attractors"].items():
        hitE = [k for k, c in sub.items() if c.get(a, 0) > 0]
        hitF = [k for k in hitE if int(k.split("|")[0]) in Ef]
        att[a] = {**v, "ref_ok": refs[int(a[1:])] is not None if int(a[1:]) < len(refs) else False,
                  "excited_sub_E": bool(hitE), "excited_sub_Ef": bool(hitF),
                  "r_sub_E": min((float(k.split("|")[1]) for k in hitE), default=None),
                  "from_sub_E": sorted({int(k.split("|")[0]) for k in hitE})}
    rec["attractors"] = att
    rec["time_s"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rand-dirs", type=int, default=0, help="число случайных направлений в неустойчивом подпространстве (по умолчанию 32)")
    ap.add_argument("--keys", default="", help="файл со списком ключей сетей (N<N>/<имя>), только они")
    ap.add_argument("--tag", default="", help="суффикс каталога результатов: X<tag>")
    a = ap.parse_args()
    pi, pn = (int(x) for x in a.part.split("/"))
    index = json.loads((ROOT / "data" / "x_index.json").read_text(encoding="utf-8"))[pi::pn]
    if a.keys:
        keep = set(Path(a.keys).read_text(encoding="utf-8").split())
        index = [it for it in index if it["key"] in keep]
    if a.limit:
        index = index[: a.limit]
    bundle = np.load(ROOT / "data" / "x_bundle.npz")
    u = np.load(ROOT / "data" / "lorenz_l1.npz")["series"][:20000]
    out = Path(os.environ.get("GPUQ_OUT", "out")) / f"X{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"stage X: {len(index)} networks, device {G.DEV}", flush=True)
    for n, item in enumerate(index):
        name = item["key"].replace("/", "_")
        if (out / f"{name}.json").exists():
            continue
        G.report_progress(n, len(index), name)
        try:
            i = item["i"]
            rec = analyze(u, item, bundle[f"{i}_E"], bundle[f"{i}_regions"], bundle[f"{i}_reps"], quick=a.quick, n_rand=a.rand_dirs or None)
            (out / f"{name}.json").write_text(json.dumps(rec, default=float), encoding="utf-8")
            sm = {k: (v["label_reference"], v["label_protocol"], v["excited_sub_E"], v["excited_sub_Ef"])
                  for k, v in rec["attractors"].items()}
            print(f"[{n + 1}/{len(index)}] {name}: |E| {rec['E_n']} Σind {rec['index_sum_E']}, |E_f| {rec['Ef_n']} "
                  f"Σind_f {rec['index_sum_Ef']}, res {rec['fp_residual_max']:.1e}; {sm}; {rec['time_s']} s", flush=True)
        except Exception:
            traceback.print_exc()
            (out / f"{name}.failed.txt").write_text(traceback.format_exc(), encoding="utf-8")
    G.report_progress(len(index), len(index), "done")


if __name__ == "__main__":
    main()

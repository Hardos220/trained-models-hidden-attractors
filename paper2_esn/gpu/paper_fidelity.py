"""Fidelity of the trained models beyond hiddenness (check).

Hiddenness is measured relative to the model's own equilibria; this script asks whether a model that keeps a reproduced
hidden attractor hidden is also faithful to the true system, with measures that do not depend on the model's equilibria.
Everything is computed from the stored records; no model is re-run.

Sections of paper2_esn/chaos/numbers/fidelity.json:
  true_systems        per regime: census fractions, p_B of Table I, attractor diameter, rotation axis, rotation core,
                      winding numbers and distances of the true equilibria.
  1_basin_fidelity    (i) basin agreement on Q, recall/precision of the hidden basin, model p_B vs true p_B, by stage,
                      regime, class/ensemble, training set and survival status; stages M (perceptrons, residual nets),
                      E (sparse quadratic maps; NG-RC), C (echo state networks, full protocol, per-point census), P
                      (penalty, lam = 0, 0.1, 1, paired by seed), V and N (echo state networks; only the true hidden basin
                      was censused there, so only its recall).
  1i_pB_table         model p_B of the reproduced hidden attractor by regime and class (median, IQR) vs the true p_B.
  1ii_visible_state   copy of the true visible state (stable copy of the visible equilibrium) and survival split by it.
  1iii_saddle_foci    Chua I and II: copies of the true saddle-foci, their stability type and index, and whether they
                      excite the reproduced attractor; decomposition of the "copies excite in 41%" figure.
  2_penalty_accounting  stage P: equilibria inside and outside Q by index and stability per lam, paired changes,
                      location matching of the index -1 equilibria of lam = 0, excitation from outside Q.
  3_prop1_eqc         stage V arm eqc (W_out r* = x* imposed): indices of the imposed copies, total index, number and
                      stability of the other equilibria (Proposition 1).
  4_mechanism         unstable equilibria of models with a reproduced attractor: distance to the true attractor,
                      winding of the attractor around them, unstable spectrum, matched null (probability that a census
                      point at the same distance lies in the model basin), distance to the rotation core.
  5_ball_mixing       how the small balls around exciting equilibria split between the attractor and other outcomes.

Definitions (also in fidelity.json["definitions"]):
  copy: model equilibrium with dist_true_eq < COPY_TOL = 0.01 (1% of the half-width L of Q) from a true equilibrium.
  correspondence of outcomes (true census label g of a uniform point of Q -> model outcome label l counted correct):
    g = hidden attractor y  <-> l = model attractor matched to y (hidden_match[y].attractor, only if reproduced;
                                 stage C/E-NG-RC: best_match_for), nothing if y is not reproduced;
    g = true stable equilibrium <-> l = f<i> with model equilibrium i a stable copy of it (n_unstable 0, dist < 0.01);
    g = escape  <-> l = esc (models whose trajectories leave |u| > 1e3 in normalized units; bounded models never do);
    g = LC (Chua II, the visible cycle) <-> any oscillatory model outcome other than the reproduced hidden attractors
                                 (osc / or / oc / unmatched a<k>): an approximation, no cycle matching was stored;
    g = Eset (LE1, line of equilibria) <-> stable copy of the line (distance to x = y = 0 below 0.01 L, z < 0);
    g = unknown and model label lift_undefined are excluded from all denominators.
  agree_all = share of census points with a correct outcome; agree_bounded = the same over points whose true outcome is
    bounded (escape excluded): bounded models cannot escape, so agree_all is dominated by the escape share of Q.
  recall = share of the true hidden basin (census points labeled y) that the model sends to its copy of y;
  precision = share of the model basin of that copy that lies in the true basin of y.
  pB_model = share of all census points that the model sends to its copy of y; pB_true = share labeled y (Table I uses
    the 1e5-4e5 point pool; the census has 1e4 points); pB_ratio = pB_model / pB_true; *_bounded: same over the
    points with bounded true outcome; escape_share_of_model_basin = share of the model basin made of points that escape
    in the true system.
  vis_ok (visible state reproduced): the model has a stable copy (n_unstable 0, dist < 0.01) of the true visible
    equilibrium; Chua II (visible = cycle): recall of the true cycle basin by oscillatory model outcomes >= 0.5; LE1:
    a stable copy of the stable half-line.
  hidden: light protocol label not_excited (stages M, P, E-poly, V, N) or full protocol label hidden_stat (stage C,
    NG-RC; undetermined counted as missing).
  excites: an unstable model equilibrium excites the reproduced attractor if a point of its balls at radius <= 1e-3
    reaches the attractor (radius_counts), as in paper_classes.py.
  eq type: (n_unstable, n_real_gt1) of the map Jacobian; index = (-1)^n_real_gt1. In 3-D maps n_unstable 2 with
    n_real_gt1 0 is a complex pair outside the unit circle (unstable focus, 2-D unstable manifold, index +1) unless both
    eigenvalues are real below -1 (rare). The true saddle-foci of Chua's circuit have type (1, 1) and index -1.
  attractor geometry (true attractor samples att_<y> of data/<regime>.npz, a sampled trajectory of 20000 points):
    D = diameter (largest distance in a 2000-point subsample, as pipeline.diam); dist_att = distance to the nearest
    sample / D; rotation axis n = direction of the mean angular momentum sum (x - mu) x dx; winding w(p) = number of turns
    of the trajectory projected on the plane orthogonal to n around the projection of p, normalized by the largest
    winding over a 41 x 41 grid of the attractor's projection (so w = 1 for points enclosed by every turn, 0 for points
    outside all turns); w_pc12: the same in the plane of the two leading principal components; encircled: |w| >= 0.5;
    rotation core = mean of the grid points with w >= 0.9; axis_dist = distance of the projection of p to the core / D.
  matched null: probability that a census point of Q at the same distance from the true attractor (bins of dist_att:
    0-0.01, 0.01-0.03, 0.03-0.1, 0.1-0.3, 0.3-1, >1 diameters) is sent by the model to the reproduced attractor.
    Stage C: directly from the per-point census labels stored in the .npz files. Stages M and P store only the confusion
    counts true label -> model label, so the null is stratified: sum_g P(true label g | bin) * P(model -> A | g), which
    assumes that, within a true outcome, the model outcome does not depend on the distance; its accuracy is checked on
    stage C, where both versions can be computed.

Pooled statistics use the six main regimes of paper_common.MAIN_REGIMES; LE1 is reported per regime and under
"le1_separate" where pooled; the survival split by vis_ok pools the five regimes whose visible state is an equilibrium
(Chua II, visible cycle, reported separately).

    python -X utf8 paper2_esn/gpu/paper_fidelity.py          (via rmq: ~5 GB of stage M JSON; workers = $RMQ_CPU)
"""
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_common import ALL_REGIMES, COPY_TOL, MAIN_REGIMES, POOL_NOTE, SEPARATE  # noqa: E402
import paper_data_effects as PD  # noqa: E402   (collect, best_match_for, van_elteren, clean)
import paper_esn_law as PE  # noqa: E402        (load_dedup, logit_fit, mantel_haenszel, SUFFIX_PAT)

ROOT = Path(os.environ.get("ESN_DATA", Path(__file__).resolve().parents[2] / "local" / "esn-gpu"))
OUT = HERE.parent / "chaos" / "numbers"
FIG = HERE.parent / "chaos" / "figures"
R_MAX = 1e3                      # Newton runaways of residual networks dropped (as paper_classes.py)
EXACT_TOL = 1e-6                 # imposed copies of the eqc arm sit at dist_true_eq ~ 1e-16
R_LIGHT = 1e-3                   # largest radius of the light protocol
M_CLASSES = ("mlp", "mlp2", "res")
P_DESIGN = {("mlp2", 256)}
EQ_VIS = tuple(g for g in MAIN_REGIMES if g != "chua_II")   # visible state is an equilibrium
ROT_REGIMES = ("se3", "se11", "wangchen2", "wangchen2_a0.03")
DBINS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, np.inf)
NB = len(DBINS) - 1
DBIN_NAMES = ["0-0.01", "0.01-0.03", "0.03-0.1", "0.1-0.3", "0.3-1", ">1"]
OSC = ("osc", "or", "oc")
LIMIT = int(os.environ.get("FID_LIMIT", "0"))   # for a quick test: at most this many stage M files
if LIMIT:                                         # test run: write JSON and figures to FID_TEST_OUT, not to the paper
    OUT = FIG = Path(os.environ.get("FID_TEST_OUT", str(HERE / "_fid_test")))


# ============================================================================================ helpers
def wil(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def shr(flags):
    f = [bool(a) for a in flags if a is not None]
    k, n = sum(f), len(f)
    return {"k": k, "n": n, "share": (k / n) if n else None, "wilson95": wil(k, n)}


def q3(v):
    v = [float(a) for a in v if a is not None and np.isfinite(a)]
    if not v:
        return None
    return [float(np.median(v)), float(np.quantile(v, 0.25)), float(np.quantile(v, 0.75)), len(v)]


def nest(rows, keys, fn):
    if not keys:
        return fn(rows)
    g = defaultdict(list)
    for x in rows:
        g[x[keys[0]]].append(x)
    return {str(k): nest(v, keys[1:], fn) for k, v in sorted(g.items(), key=lambda t: str(t[0]))}


def main_only(rows, key="reg"):
    return [x for x in rows if x[key] not in SEPARATE]


def wind(P, C, chunk=96):
    """Winding numbers of the polyline P (n, 2) around each point of C (m, 2): sum of wrapped angle increments / 2 pi."""
    C = np.atleast_2d(C)
    out = np.empty(len(C))
    for i0 in range(0, len(C), chunk):
        c = C[i0:i0 + chunk]
        th = np.arctan2(P[None, :, 1] - c[:, None, 1], P[None, :, 0] - c[:, None, 0])
        d = np.diff(th, axis=1)
        d = (d + np.pi) % (2 * np.pi) - np.pi
        out[i0:i0 + chunk] = d.sum(1) / (2 * np.pi)
    return out


def dbin(d):
    if d is None or not np.isfinite(d):
        return None
    return int(np.clip(np.searchsorted(DBINS, d, side="right") - 1, 0, NB - 1))


def eqtype(nu, nr):
    if nu is None:
        return None
    nus = str(nu) if nu < 4 else "4+"
    return f"{nus}u" if nr is None else f"{nus}u_r{nr}"


TYPE_NOTE = ("eq type '<n_unstable>u_r<n_real_gt1>': 1u_r1 = saddle with one real eigenvalue > 1 (index -1, the type "
             "of the true saddle-foci); 1u_r0 = one real eigenvalue < -1 (flip, index +1); 2u_r0 = complex pair outside "
             "the unit circle (unstable focus with 2-D unstable manifold, index +1) or two real eigenvalues < -1; "
             "2u_r1 = one > 1 and one < -1 (index -1); 2u_r2 = two real > 1 (index +1); 3u_*: all directions unstable "
             "(for echo state networks n_unstable counts reservoir directions)")


# ============================================================================================ true systems
class TrueSys:
    def __init__(self, reg):
        m = json.loads((ROOT / "data" / f"{reg}.json").read_text(encoding="utf-8"))
        z = np.load(ROOT / "data" / f"{reg}.npz")
        self.reg, self.m = reg, m
        self.lo, self.hi = np.array(m["box_lo"], float), np.array(m["box_hi"], float)
        self.L = float(np.max(self.hi - self.lo)) / 2
        self.hidden, self.visible = list(m["hidden"]), m["visible"]
        self.eqs = []
        for j, e in enumerate(m["equilibria"] or []):
            re, im = np.array(e["eig_re"], float), np.array(e["eig_im"], float)
            npos = int(np.sum((np.abs(im) < 1e-12) & (re > 0)))
            self.eqs.append(dict(j=j, name=e["name"], point=np.array(e["point"], float), stable=bool(e["stable"]),
                                 iota=int((-1) ** npos), n_unstable_flow=int(np.sum(re > 0)),
                                 complex_unstable=bool(np.any((np.abs(im) > 1e-12) & (re > 0)))))
        self.s = sum(e["iota"] for e in self.eqs) if self.eqs else None
        self.vis_j = next((e["j"] for e in self.eqs if e["name"] == self.visible), None)
        self.labels = np.asarray(z["labels_uniform"]).astype(str)
        self.ics = np.asarray(z["ics_uniform"], float)
        self.N = len(self.labels)
        self.att = {y: self._att(np.asarray(z[f"att_{y}"], float)) for y in self.hidden}

    def _att(self, A):
        mu = A.mean(0)
        rng = np.random.default_rng(0)
        D = float(pdist(A[rng.choice(len(A), min(2000, len(A)), replace=False)]).max())
        Lm = np.cross(A[:-1] - mu, np.diff(A, axis=0)).sum(0)
        n = Lm / np.linalg.norm(Lm)
        _, S, Vt = np.linalg.svd(A - mu, full_matrices=False)
        u1 = Vt[0] - (Vt[0] @ n) * n
        if np.linalg.norm(u1) < 1e-6:
            u1 = Vt[1] - (Vt[1] @ n) * n
        u1 /= np.linalg.norm(u1)
        u2 = np.cross(n, u1)
        info = dict(mu=mu, D=D, n=n, tree=cKDTree(A), planes={}, pc_axes=Vt, pc_sd=S / np.sqrt(len(A)))
        for nm, B in (("rot", np.stack([u1, u2])), ("pc12", Vt[:2])):
            P = (A - mu) @ B.T
            lo, hi = P.min(0), P.max(0)
            G = np.array([[a, b] for a in np.linspace(lo[0], hi[0], 41) for b in np.linspace(lo[1], hi[1], 41)])
            w = wind(P, G)
            W = float(w[int(np.argmax(np.abs(w)))])
            core2 = G[(w / W) >= 0.9].mean(0)
            info["planes"][nm] = dict(B=B, P=P, W=W, core2=core2, core3=mu + core2 @ B,
                                      grid_share_encircled=float(np.mean(w / W >= 0.5)))
        d = info["tree"].query(self.ics)[0] / D
        info["ics_d"] = d
        info["ics_bin"] = np.array([dbin(v) for v in d])
        # P(true label | distance bin) for the stratified null
        pg = []
        for b in range(NB):
            sel = (info["ics_bin"] == b) & (self.labels != "unknown")
            c = Counter(self.labels[sel])
            s = sum(c.values())
            pg.append({g: v / s for g, v in c.items()} if s else {})
        info["Pg_bin"] = pg
        info["n_bin"] = [int(np.sum(info["ics_bin"] == b)) for b in range(NB)]
        return info

    def geom(self, y, X):
        X = np.atleast_2d(np.asarray(X, float))
        a = self.att[y]
        D = a["D"]
        out = dict(dist_att=a["tree"].query(X)[0] / D, centroid_dist=np.linalg.norm(X - a["mu"], axis=1) / D,
                   offplane=np.abs((X - a["mu"]) @ a["n"]) / D)
        for nm, p in a["planes"].items():
            Q = (X - a["mu"]) @ p["B"].T
            out[f"w_{nm}"] = wind(p["P"], Q) / p["W"]
            out[f"axis_dist_{nm}"] = np.linalg.norm(Q - p["core2"], axis=1) / D
        return out

    def inside(self, x):
        if x is None:
            return None
        x = np.asarray(x, float)
        if x.shape != (3,):
            return None
        return bool(np.all(x >= self.lo) and np.all(x <= self.hi))

    def summary(self):
        cnt = Counter(self.labels)
        out = {"visible": self.visible, "hidden": self.hidden, "index_sum_s": self.s, "L_half_width": self.L,
               "census_n": self.N, "census_fractions": {g: v / self.N for g, v in sorted(cnt.items())},
               "pB_table_pool": {y: self.m["attractors"][y]["p_pool"] for y in self.hidden},
               "pB_census": {y: cnt.get(y, 0) / self.N for y in self.hidden},
               "pB_census_all_hidden": sum(cnt.get(y, 0) for y in self.hidden) / self.N,
               "escape_fraction_census": cnt.get("escape", 0) / self.N,
               "equilibria": [{"name": e["name"], "iota": e["iota"], "stable": e["stable"],
                               "n_unstable_flow": e["n_unstable_flow"], "point": e["point"].tolist()} for e in self.eqs],
               "attractors": {}}
        for y, a in self.att.items():
            g = self.geom(y, np.array([e["point"] for e in self.eqs])) if self.eqs else None
            out["attractors"][y] = {
                "diameter": a["D"], "centroid": a["mu"].tolist(), "rotation_axis": a["n"].tolist(),
                "pc_sd": a["pc_sd"].tolist(),
                "turns_in_sample": {nm: abs(p["W"]) for nm, p in a["planes"].items()},
                "rotation_core": {nm: p["core3"].tolist() for nm, p in a["planes"].items()},
                "core_to_centroid_over_D": {nm: float(np.linalg.norm(p["core2"]) / a["D"]) for nm, p in a["planes"].items()},
                "grid_share_encircled": {nm: p["grid_share_encircled"] for nm, p in a["planes"].items()},
                "census_points_per_distance_bin": dict(zip(DBIN_NAMES, a["n_bin"])),
                "true_equilibria": ({e["name"]: {"dist_att_over_D": float(g["dist_att"][k]), "w_rot": float(g["w_rot"][k]),
                                                 "w_pc12": float(g["w_pc12"][k]),
                                                 "axis_dist_rot_over_D": float(g["axis_dist_rot"][k]),
                                                 "offplane_over_D": float(g["offplane"][k]),
                                                 "centroid_dist_over_D": float(g["centroid_dist"][k])}
                                     for k, e in enumerate(self.eqs)} if g is not None else None)}
        return out


# ============================================================================================ record extraction
def _rc(rc):
    """radius_counts {"i|eps": {label: count}} -> {i: {eps: (total, {a-label: count})}}."""
    out = {}
    for k, c in (rc or {}).items():
        i, e = k.split("|")
        out.setdefault(int(i), {})[float(e)] = (int(sum(c.values())), {l: int(v) for l, v in c.items() if l.startswith("a")})
    return out


def _fps(fp_list, index_src=None):
    fps, run = [], 0
    for i, x in enumerate(fp_list or []):
        xx = x.get("x")
        if xx is not None and len(xx) == 3 and np.linalg.norm(xx) > R_MAX:
            run += 1
            continue
        ind, nr = x.get("index"), x.get("n_real_gt1")
        if index_src is not None and i < len(index_src):
            ind, nr = index_src[i].get("index", ind), index_src[i].get("n_real_gt1", nr)
        d = x.get("dist_true_eq")
        fps.append(dict(i=i, x=(list(xx) if xx is not None and len(xx) == 3 else None), nu=int(x["n_unstable"]),
                        nr=None if nr is None else int(nr), ind=None if ind is None else int(ind), rho=float(x["rho"]),
                        d=9.0 if d is None else float(d), j=x.get("nearest_true_eq")))
    return fps, run


def ext_m(job):
    """Stage M / P / E-poly record (stage M protocol) -> compact dict, or None if filtered out."""
    path, stage = job
    try:
        r = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if "hidden_match" not in r or "census_confusion" not in r:
        return None
    if stage == "M" and r.get("class") not in M_CLASSES:
        return None
    if stage == "P" and (r.get("class"), r.get("H")) not in P_DESIGN:
        return None
    if stage == "E" and r.get("kind", "poly") != "poly":
        return None
    fps, run = _fps(r["fixed_points"])
    hm = {y: dict(rep=bool(mm.get("reproduced")), att=mm.get("attractor"), label=mm.get("label")) for y, mm in r["hidden_match"].items()}
    p = Path(path)
    return dict(stage=stage, reg=r["regime"], cls=r.get("class", "poly"), set=r.get("set", "E1"), H=r.get("H"),
                seed=r.get("seed"), lam=float(r["lam"]) if "lam" in r else None, thr=r.get("thr"),
                key=p.parent.name + "/" + p.stem, fps=fps, n_runaway=run,
                isum=int(sum(x["ind"] for x in fps if x["ind"] is not None)), hm=hm, conf=r["census_confusion"],
                rc=_rc(r.get("radius_counts")), proto="light")


def hidden_of(label, proto):
    if label is None:
        return None
    if proto == "light":
        return label == "not_excited"
    if label == "hidden_stat":
        return True
    if label in ("self_excited", "excited_r"):
        return False
    return None


def exciters(rec, a, rmax=R_LIGHT):
    out = set()
    for i, d in rec["rc"].items():
        if any(k.get(a, 0) > 0 for e, (t, k) in d.items() if e <= rmax * (1 + 1e-9)):
            out.add(i)
    return out


# ============================================================================================ loaders
def first_per_key(files):
    seen, out = set(), []
    for f in files:
        k = f.parent.name + "/" + f.stem
        if k in seen:
            continue
        seen.add(k)
        out.append(f)
    return out


def load_light(workers):
    files_M = first_per_key(sorted((ROOT / "gpuq-results").rglob("M/*/*.json")))
    if LIMIT:
        rng = np.random.default_rng(1)
        files_M = [files_M[i] for i in sorted(rng.choice(len(files_M), min(LIMIT, len(files_M)), replace=False))]
    files_P = [f for f in sorted((ROOT / "gpuq-results").rglob("P/*/*.json")) + sorted((ROOT / "node_results").rglob("P/*/*.json"))
               if ".failed" not in f.name]
    files_E = first_per_key([f for d in ("node_results", "gpuq-results") for f in sorted((ROOT / d).rglob("E/*/*.json"))
                             if ".failed" not in f.name])
    jobs = [(str(f), "M") for f in files_M] + [(str(f), "P") for f in files_P] + [(str(f), "E") for f in files_E]
    print(f"parsing {len(jobs)} records with {workers} workers", flush=True)
    if workers > 1:
        with Pool(workers) as pool:
            res = pool.map(ext_m, jobs, chunksize=8)
    else:
        res = [ext_m(j) for j in jobs]
    M = [r for r in res if r and r["stage"] == "M"]
    seenP, P = set(), []
    for r in res:                                      # P: first file per key that passes the filters (paper_penalty.py)
        if r and r["stage"] == "P" and r["key"] not in seenP:
            seenP.add(r["key"])
            P.append(r)
    E = [r for r in res if r and r["stage"] == "E"]
    return M, P, E


def load_C():
    """Stage C echo state networks (full protocol) with per-point census labels (npz) and indices (stage I)."""
    rec = PE.load_dedup("C")
    I = {}
    for f in (ROOT / "local_results" / "I").glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        I[r["key"]] = r["fixed_points"]
    out, n_no_npz, n_no_I = [], 0, 0
    for key, (r, path, _) in rec.items():
        if r.get("model") != "esn":
            continue
        reg = r["regime"]
        m = PD.meta(reg)
        Ifp = I.get(key)
        if Ifp is not None and (len(Ifp) != len(r["fixed_points"]) or any(
                p.get("dist_true_eq") != q.get("dist_true_eq") for p, q in zip(Ifp, r["fixed_points"]))):
            Ifp = None
        n_no_I += Ifp is None
        fps, run = _fps(r["fixed_points"], Ifp)
        lab = None
        npz = path.with_suffix(".npz")
        if npz.exists():
            z = np.load(npz)
            names = [str(s) for s in z["group_names"]]
            if "uL2" in names:
                k = names.index("uL2")
                offs, leg = z["group_offsets"], np.asarray(z["legend"]).astype(str)
                L = np.asarray(z["labels"][offs[k]:offs[k + 1]]).astype(int)
                lab = np.where((L >= 0) & (L < len(leg)), leg[np.clip(L, 0, len(leg) - 1)], "none")
        else:
            n_no_npz += 1
        hm = {}
        for y in m["hidden"]:
            b = PD.best_match_for(r, m, y)
            a = b[1] if b else None
            hm[y] = dict(rep=bool(b), att=a, label=r["hiddenness"].get(a, {}).get("label") if a else None)
        if lab is not None and len(lab) == 10000:
            tl = TS[reg].labels
            conf = defaultdict(lambda: defaultdict(int))
            for g, l in zip(tl, lab):
                conf[g][l] += 1
            conf = {g: dict(c) for g, c in conf.items()}
        else:
            conf = r.get("confusion", {}).get("uL2", {})
            lab = None
        out.append(dict(stage="C", reg=reg, cls="esn", set=r["set"], H=r.get("N", 300), seed=r.get("seed"), lam=None,
                        key=key, fps=fps, n_runaway=run, isum=(sum(p["index"] for p in Ifp) if Ifp else None), hm=hm,
                        conf=conf, rc=_rc(r.get("radius_counts")), proto="full", lab=lab, rho=r.get("rho"),
                        eta=r.get("eta"), I_ok=Ifp is not None))
    return out, {"n_no_npz": n_no_npz, "n_without_stage_I_indices": n_no_I}


def load_ngrc():
    out = []
    for key, f in PD.collect("E").items():
        r = PD.load(f)
        if not (r.get("model") == "ngrc" or "hiddenness" in r):
            continue
        reg = r["regime"]
        m = PD.meta(reg)
        fps, run = _fps(r.get("fixed_points", []))
        hm = {}
        for y in m["hidden"]:
            b = PD.best_match_for(r, m, y)
            a = b[1] if b else None
            hm[y] = dict(rep=bool(b), att=a, label=r["hiddenness"].get(a, {}).get("label") if a else None)
        out.append(dict(stage="E", reg=reg, cls="ngrc", set=r.get("set", "E1"), H=None, seed=None, lam=None, key=key,
                        fps=fps, n_runaway=run, isum=None, hm=hm, conf=r.get("confusion", {}).get("uL0", {}),
                        rc=_rc(r.get("radius_counts")), proto="full"))
    return out


def load_V():
    out = []
    for key, f in PD.collect("V").items():
        r = PD.load(f)
        for arm, a in r["arms"].items():
            fps, run = _fps(a["fixed_points"])
            hm = {y: dict(rep=bool(mm.get("reproduced")), att=mm.get("attractor"), label=mm.get("label"))
                  for y, mm in a["hidden_match"].items()}
            out.append(dict(stage="V", reg=r["regime"], cls="esn", set=arm, arm=arm, H=r["N"], seed=r["seed"], lam=None,
                            key=key, fps=fps, n_runaway=run, isum=a.get("index_sum"), hm=hm, conf=None,
                            basin={y: a.get(f"census_basin_{y}", {}) for y in hm}, cube=a.get("census_cube", {}),
                            rc=_rc(a.get("radius_counts")), proto="light", rho=r["rho"], eta=r["eta"]))
    return out


def load_N():
    out = []
    for key, (r, path, _) in PE.load_dedup("N").items():
        default = (not PE.SUFFIX_PAT.search(path.stem) and float(r.get("input_scale", 0.5)) == 0.5
                   and float(r.get("density", 0.05)) == 0.05 and float(r["rho"]) == 0.9 and r["set"] in ("HE1", "H1"))
        if not default or "hidden_match" not in r:
            continue
        fps, run = _fps(r.get("fixed_points", []))
        hm = {y: dict(rep=bool(mm.get("reproduced")), att=mm.get("attractor"), label=mm.get("label"))
              for y, mm in r["hidden_match"].items()}
        out.append(dict(stage="N", reg=r["regime"], cls="esn", set=r["set"], H=int(r["N"]), seed=r["seed"], lam=None,
                        key=key, fps=fps, n_runaway=run, isum=r.get("index_sum"), hm=hm, conf=None,
                        basin={y: r.get(f"census_basin_{y}", {}) for y in hm}, cube=r.get("census_cube", {}),
                        rc=_rc(r.get("radius_counts")), proto="light"))
    return out


# ============================================================================================ 1. basin fidelity
def visible_ok(rec, ts, vis_recall):
    fps = rec["fps"]
    if ts.vis_j is not None:
        return any(x["nu"] == 0 and x["d"] < COPY_TOL and x["j"] == ts.vis_j for x in fps)
    if ts.visible == "Eset":
        return any(x["nu"] == 0 and x["d"] < COPY_TOL and (x["x"] is None or x["x"][2] < 0) for x in fps)
    if ts.visible == "LC":
        return None if vis_recall is None else vis_recall >= 0.5
    return None


def fidelity_rows(rec, ts):
    """One row per reproduced (model, hidden attractor) with the census measures; model-level dict."""
    fps = rec["fps"]
    att_rep = {y: h["att"] for y, h in rec["hm"].items() if h["rep"] and h.get("att")}
    matched = set(att_rep.values())
    U = sum(x["nu"] > 0 for x in fps)
    U_in = sum(1 for x in fps if x["nu"] > 0 and ts.inside(x["x"]))
    conf = rec.get("conf")
    mod = dict(agree_all=None, agree_bounded=None, agree_escape=None, vis_recall=None)
    tot = {}
    copy_labels = set()
    if conf:
        stable_copy = defaultdict(set)
        for x in fps:
            if x["nu"] == 0 and x["d"] < COPY_TOL:
                if ts.eqs and x["j"] is not None:
                    stable_copy[ts.eqs[x["j"]]["name"]].add(f"f{x['i']}")
                elif not ts.eqs and (x["x"] is None or x["x"][2] < 0):
                    stable_copy["Eset"].add(f"f{x['i']}")
        copy_labels = set().union(*stable_copy.values()) if stable_copy else set()

        def correct(g, l):
            if g in ts.hidden:
                return g in att_rep and l == att_rep[g]
            if g == "escape":
                return l == "esc"
            if g == "LC":
                return l in OSC or (l.startswith("a") and l not in matched)
            return l in stable_copy.get(g, ())
        ok = {}
        for g, c in conf.items():
            if g == "unknown":
                continue
            cc = {l: v for l, v in c.items() if l not in ("lift_undefined", "none")}
            tot[g] = sum(cc.values())
            ok[g] = sum(v for l, v in cc.items() if correct(g, l))
        N = sum(tot.values())
        if N:
            Nb = N - tot.get("escape", 0)
            mod = dict(agree_all=sum(ok.values()) / N,
                       agree_bounded=(sum(v for g, v in ok.items() if g != "escape") / Nb) if Nb else None,
                       agree_escape=(ok["escape"] / tot["escape"]) if tot.get("escape") else None,
                       vis_recall=(ok[ts.visible] / tot[ts.visible]) if tot.get(ts.visible) else None)
    vis = visible_ok(rec, ts, mod["vis_recall"])
    rows = []
    for y, a in att_rep.items():
        base = dict(stage=rec["stage"], reg=rec["reg"], cls=rec["cls"], set=rec["set"], key=rec["key"], lam=rec.get("lam"),
                    H=rec.get("H"), seed=rec.get("seed"), y=y, label=rec["hm"][y]["label"],
                    hidden=hidden_of(rec["hm"][y]["label"], rec["proto"]), vis_ok=vis, U=U, U_in=U_in, isum=rec.get("isum"),
                    **mod)
        if conf and tot:
            N = sum(tot.values())
            Nb = N - tot.get("escape", 0)
            n_a = sum(conf[g].get(a, 0) for g in tot)
            n_ab = sum(conf[g].get(a, 0) for g in tot if g != "escape")
            Ny = tot.get(y, 0)
            hit = conf.get(y, {}).get(a, 0)
            base.update(pB_model=n_a / N, pB_true=Ny / N, pB_ratio=(n_a / Ny) if Ny else None,
                        pB_model_bounded=(n_ab / Nb) if Nb else None, pB_true_bounded=(Ny / Nb) if Nb else None,
                        pB_ratio_bounded=(n_ab / Ny) if Ny else None, recall=(hit / Ny) if Ny else None,
                        precision=(hit / n_a) if n_a else None, precision_bounded=(hit / n_ab) if n_ab else None,
                        escape_share_of_model_basin=(conf.get("escape", {}).get(a, 0) / n_a) if n_a else None)
            # where the rest of Q goes in the model: copies of true stable equilibria, spurious stable fixed points,
            # other oscillation, escape, other copies of hidden attractors
            comp = Counter()
            for g in tot:
                for l, v in conf[g].items():
                    if l in ("lift_undefined", "none") or l == a:
                        continue
                    if l == "esc":
                        comp["escape"] += v
                    elif l in copy_labels:
                        comp["true_stable_copy"] += v
                    elif l.startswith("f"):
                        comp["spurious_fixed_point"] += v
                    elif l in matched:
                        comp["other_reproduced_attractor"] += v
                    else:
                        comp["other_oscillation"] += v
            for k in ("true_stable_copy", "spurious_fixed_point", "other_oscillation", "escape", "other_reproduced_attractor"):
                base[f"frac_{k}"] = comp[k] / N
            base["basin_ok"] = bool(base["recall"] is not None and base["recall"] >= 0.5 and base["pB_ratio_bounded"] is not None
                                    and 0.5 <= base["pB_ratio_bounded"] <= 2)
        elif rec.get("basin") is not None:                      # stages V, N: only the true hidden basin was censused
            b = rec["basin"].get(y, {})
            nb = sum(b.values())
            cube = rec.get("cube") or {}
            nc = sum(cube.values())
            base.update(recall=(b.get(a, 0) / nb) if nb else None,
                        pB_reservoir_cube=(cube.get(a, 0) / nc) if nc else None)
        rows.append(base)
    return rows, dict(mod, vis_ok=vis, U=U, U_in=U_in, n_rep=len(att_rep))


MEAS = ("agree_all", "agree_bounded", "agree_escape", "recall", "precision", "precision_bounded", "pB_model", "pB_true",
        "pB_ratio", "pB_model_bounded", "pB_true_bounded", "pB_ratio_bounded", "escape_share_of_model_basin", "vis_recall",
        "pB_reservoir_cube", "frac_true_stable_copy", "frac_spurious_fixed_point", "frac_other_oscillation", "frac_escape",
        "frac_other_reproduced_attractor")


def summarize(rows):
    out = {"n_rows": len(rows), "hidden": shr(x["hidden"] for x in rows), "vis_ok": shr(x["vis_ok"] for x in rows),
           "basin_ok": shr(x.get("basin_ok") for x in rows),
           "hidden_and_vis_ok": shr((x["hidden"] and x["vis_ok"]) if (x["hidden"] is not None and x["vis_ok"] is not None) else None
                                    for x in rows),
           "hidden_vis_ok_basin_ok": shr((x["hidden"] and x["vis_ok"] and x["basin_ok"])
                                         if (x["hidden"] is not None and x["vis_ok"] is not None and x.get("basin_ok") is not None)
                                         else None for x in rows)}
    for k in MEAS:
        v = q3(x.get(k) for x in rows)
        if v is not None:
            out[k] = v
    return out


def fid_block(rows, gkeys):
    main = main_only(rows)
    out = {"pooled_six": nest(main, gkeys, summarize),
           "pooled_six_all": summarize(main),
           "by_regime": nest(rows, ("reg",) + gkeys, summarize),
           "by_survival_pooled_six": {h: nest([x for x in main if x["hidden"] is hv], gkeys, summarize)
                                      for h, hv in (("hidden", True), ("excited", False))},
           "by_survival_pooled_six_all": {h: summarize([x for x in main if x["hidden"] is hv])
                                          for h, hv in (("hidden", True), ("excited", False))}}
    tests = {}
    for k in ("agree_bounded", "agree_all", "recall", "precision", "pB_ratio", "pB_ratio_bounded", "vis_recall"):
        R = [dict(x, v=x.get(k)) for x in main if x["hidden"] is not None and x.get(k) is not None]
        if len(R) > 10 and len({x["hidden"] for x in R}) == 2:
            tests[k] = PD.van_elteren(R, "v", "hidden", lambda x: (x["reg"], x["cls"], x["set"], x.get("lam")))
    out["hidden_vs_excited_van_elteren"] = {"note": "stratified Wilcoxon (regime x class x set [x lam]); z > 0: larger "
                                                    "values among hidden rows", **tests}
    le1 = [x for x in rows if x["reg"] in SEPARATE]
    if le1:
        out["le1_separate"] = nest(le1, gkeys, summarize)
    return out


def pB_table(rows_by_class):
    """Model p_B of the reproduced hidden attractor (per attractor) by regime and class vs the true p_B."""
    out = {}
    for g in ALL_REGIMES:
        ts = TS[g]
        d = {"true_pB_table_pool": {y: ts.m["attractors"][y]["p_pool"] for y in ts.hidden},
             "true_pB_census": {y: float(np.mean(ts.labels == y)) for y in ts.hidden},
             "true_pB_census_bounded": {y: float(np.sum(ts.labels == y) / max(1, np.sum((ts.labels != "escape")
                                                                                     & (ts.labels != "unknown"))))
                                        for y in ts.hidden},
             "classes": {}}
        for c, R in rows_by_class.items():
            r = [x for x in R if x["reg"] == g and x.get("pB_model") is not None]
            if not r:
                continue
            both = defaultdict(list)
            for x in r:
                both[x["key"]].append(x)
            two = [v for v in both.values() if len(v) == 2]
            d["classes"][c] = {
                "n_rows": len(r), "pB_model": q3(x["pB_model"] for x in r), "pB_ratio": q3(x["pB_ratio"] for x in r),
                "pB_model_bounded": q3(x["pB_model_bounded"] for x in r),
                "pB_ratio_bounded": q3(x["pB_ratio_bounded"] for x in r),
                "share_rows_pB_model_gt_2x_true": shr((x["pB_ratio"] or 0) > 2 for x in r),
                "share_rows_pB_model_lt_half_true": shr((x["pB_ratio"] or 0) < 0.5 for x in r),
                "escape_share_of_model_basin": q3(x["escape_share_of_model_basin"] for x in r),
                "by_set": {s: {"n": len(rs), "pB_model": q3(x["pB_model"] for x in rs), "pB_ratio": q3(x["pB_ratio"] for x in rs)}
                           for s in sorted({x["set"] for x in r}) for rs in [[x for x in r if x["set"] == s]]},
                "by_survival": {h: {"n": len(rs), "pB_model": q3(x["pB_model"] for x in rs), "pB_ratio": q3(x["pB_ratio"] for x in rs)}
                                for h, hv in (("hidden", True), ("excited", False))
                                for rs in [[x for x in r if x["hidden"] is hv]]},
                "both_attractors_reproduced": ({"n_models": len(two),
                                                "pB_model_sum": q3(sum(x["pB_model"] for x in v) for v in two),
                                                "pB_true_sum": q3(sum(x["pB_true"] for x in v) for v in two)}
                                               if len(ts.hidden) == 2 else None)}
        out[g] = d
    out["note"] = ("per (model, reproduced hidden attractor); [median, q25, q75, n]; pB_model = share of the 1e4 uniform "
                   "census points of Q sent to the model's copy of the attractor; ratio to the census share of the true "
                   "basin of the same attractor (Chua: per attractor, HA_a and HA_b separately; both_attractors_reproduced "
                   "sums them for models that reproduce both). Stage V's 'basin share' (paper text 'Chua I from 0.22 to 0.13 "
                   "of the region') is a share of the reservoir cube [-1, 1]^N, not of Q, and is not comparable with Table I")
    return out


# ============================================================================================ 1(ii) visible state
def vis_block(rows, gkeys, strat_extra=()):
    R = [x for x in rows if x["reg"] in EQ_VIS and x["vis_ok"] is not None and x["hidden"] is not None]

    def f(rs):
        return {"vis_ok_share": shr(x["vis_ok"] for x in rs),
                "survival_if_vis_ok": shr(x["hidden"] for x in rs if x["vis_ok"]),
                "survival_if_not_vis_ok": shr(x["hidden"] for x in rs if not x["vis_ok"]),
                "U_in_ge1_survival_if_vis_ok": shr(x["hidden"] for x in rs if x["vis_ok"] and x["U_in"] >= 1),
                "U_in_ge1_survival_if_not_vis_ok": shr(x["hidden"] for x in rs if not x["vis_ok"] and x["U_in"] >= 1)}
    out = {"pooled_five": nest(R, gkeys, f), "pooled_five_all": f(R), "by_regime": nest(R, ("reg",) + gkeys, f)}
    mh = PE.mantel_haenszel(R, "vis_ok", "hidden", lambda x: (x["reg"], x["cls"], x["set"], min(x["U_in"], 9)) + tuple(x.get(k) for k in strat_extra))
    out["mantel_haenszel_vis_ok_vs_hidden"] = {"strata": "regime x class x set x min(U_in, 9) (U_in = 0 for stages V, N: no "
                                                         "coordinates stored)", **mh}
    mh2 = PE.mantel_haenszel(R, "vis_ok", "hidden", lambda x: (x["reg"], x["cls"], x["set"]) + tuple(x.get(k) for k in strat_extra))
    out["mantel_haenszel_vis_ok_vs_hidden_no_U"] = {"strata": "regime x class x set", **mh2}
    out["U_in_mean_by_vis_ok"] = {str(v): float(np.mean([x["U_in"] for x in R if x["vis_ok"] is v])) if any(x["vis_ok"] is v for x in R) else None
                                 for v in (True, False)}
    c2 = [x for x in rows if x["reg"] == "chua_II" and x["vis_ok"] is not None and x["hidden"] is not None]
    if c2:
        out["chua_II_cycle_proxy"] = nest(c2, gkeys, f)
    return out


# ============================================================================================ 1(iii) saddle-foci
def saddle_focus_rows(recs):
    """Per model record of Chua I / II: copies of the true unstable equilibria (saddle-foci)."""
    rows = []
    for rec in recs:
        ts = TS[rec["reg"]]
        if rec["reg"] not in ("chua_I", "chua_II"):
            continue
        att_rep = {y: h["att"] for y, h in rec["hm"].items() if h["rep"] and h.get("att")}
        exc = {y: exciters(rec, a) for y, a in att_rep.items()}
        for e in ts.eqs:
            if e["stable"]:
                continue
            cps = [x for x in rec["fps"] if x["d"] < COPY_TOL and x["j"] == e["j"]]
            near = min((x["d"] for x in rec["fps"] if x["j"] == e["j"]), default=None)
            rows.append(dict(stage=rec["stage"], reg=rec["reg"], cls=rec["cls"], set=rec["set"], lam=rec.get("lam"),
                             key=rec["key"], eq=e["name"], n_copies=len(cps), nearest_dist=near,
                             copies=[dict(nu=x["nu"], nr=x["nr"], ind=x["ind"], type=eqtype(x["nu"], x["nr"]),
                                          inside=ts.inside(x["x"]), type_ok=(x["nu"] == 1 and x["nr"] == 1),
                                          index_ok=(x["ind"] == e["iota"]) if x["ind"] is not None else None,
                                          excites={y: x["i"] in s for y, s in exc.items()}) for x in cps],
                             n_rep=len(att_rep)))
    return rows


def sf_summary(rows):
    def f(rs):
        models = defaultdict(list)
        for x in rs:
            models[x["key"]].append(x)
        cps = [c for x in rs for c in x["copies"]]
        cps_rep = [c for x in rs if x["n_rep"] for c in x["copies"]]
        return {"n_models": len(models),
                "models_with_copies_of_k_saddle_foci": dict(Counter(sum(x["n_copies"] > 0 for x in v) for v in models.values())),
                "n_copies": len(cps),
                "copy_type": dict(Counter(c["type"] for c in cps)),
                "copy_type_ok_1u_r1": shr(c["type_ok"] for c in cps),
                "copy_index_ok": shr(c["index_ok"] for c in cps),
                "copy_stable": shr(c["nu"] == 0 for c in cps),
                "copies_in_models_with_reproduced_attractor": len(cps_rep),
                "copy_excites_reproduced_attractor": shr(any(c["excites"].values()) for c in cps_rep),
                "type_ok_copy_excites": shr(any(c["excites"].values()) for c in cps_rep if c["type_ok"]),
                "nearest_model_eq_dist_over_L": q3(x["nearest_dist"] for x in rs)}
    return f(rows), nest(rows, ("reg", "stage", "cls", "set"), f)


def losses_by_source(recs):
    """Excited reproduced attractors: which equilibria excite them (saddle-focus copies, unstable copies of stable true
    equilibria, spurious equilibria inside / outside Q)."""
    rows = []
    for rec in recs:
        ts = TS[rec["reg"]]
        fpi = {x["i"]: x for x in rec["fps"]}
        for y, h in rec["hm"].items():
            if not (h["rep"] and h.get("att")):
                continue
            hid = hidden_of(h["label"], rec["proto"])
            if hid is not False:
                continue
            ex = [fpi[i] for i in exciters(rec, h["att"]) if i in fpi]
            kinds = set()
            for x in ex:
                if x["d"] < COPY_TOL:
                    if ts.eqs and x["j"] is not None:
                        kinds.add("copy_true_unstable" if not ts.eqs[x["j"]]["stable"] else "copy_true_stable")
                    else:
                        kinds.add("copy_line")
                else:
                    ins = ts.inside(x["x"])
                    kinds.add("spurious_in_Q" if ins else ("spurious_out_Q" if ins is False else "spurious"))
            rows.append(dict(stage=rec["stage"], reg=rec["reg"], cls=rec["cls"], set=rec["set"], lam=rec.get("lam"),
                             kinds=kinds, n_ex=len(ex)))
    return rows


def loss_summary(rows):
    def f(rs):
        return {"n_losses": len(rs), "no_exciter_on_record": sum(1 for x in rs if not x["n_ex"]),
                **{f"with_{k}": sum(1 for x in rs if k in x["kinds"]) for k in
                   ("copy_true_unstable", "copy_true_stable", "copy_line", "spurious_in_Q", "spurious_out_Q", "spurious")},
                "only_copy_true_unstable": sum(1 for x in rs if x["kinds"] == {"copy_true_unstable"}),
                "only_spurious": sum(1 for x in rs if x["kinds"] and x["kinds"] <= {"spurious_in_Q", "spurious_out_Q", "spurious"})}
    return f


# ============================================================================================ 2. penalty accounting
def eq_census(rec, ts):
    c = Counter()
    for x in rec["fps"]:
        ins = ts.inside(x["x"])
        where = "in" if ins else "out"
        kind = "stable" if x["nu"] == 0 else ("idx-1" if x["ind"] == -1 else "unst_idx+1")
        c[f"{where}_{kind}"] += 1
    c["K"] = len(rec["fps"])
    c["U"] = sum(x["nu"] > 0 for x in rec["fps"])
    c["U_in"] = c["in_idx-1"] + c["in_unst_idx+1"]
    c["U_out"] = c["out_idx-1"] + c["out_unst_idx+1"]
    c["K_in"] = c["U_in"] + c["in_stable"]
    return c


def penalty_accounting(P):
    keys = ("K", "U", "K_in", "U_in", "U_out", "in_idx-1", "out_idx-1", "in_unst_idx+1", "out_unst_idx+1", "in_stable",
            "out_stable")
    by = defaultdict(dict)
    for r in P:
        by[(r["reg"], r["set"], r["cls"], r["H"], r["seed"])][r["lam"]] = r
    lams = sorted({r["lam"] for r in P})

    def block(regs):
        out = {"per_lam": {}, "paired_vs_lam0": {}}
        for l in lams:
            R = [r for r in P if r["lam"] == l and r["reg"] in regs]
            C = [eq_census(r, TS[r["reg"]]) for r in R]
            out["per_lam"][f"{l:g}"] = {"n_models": len(R), **{f"total_{k}": int(sum(c[k] for c in C)) for k in keys},
                                       **{f"mean_{k}": (float(np.mean([c[k] for c in C])) if C else None) for k in keys}}
        for l in lams:
            if l == 0:
                continue
            pairs = [(d[0.0], d[l]) for k, d in by.items() if k[0] in regs and 0.0 in d and l in d]
            lost_in = gain_out = annih = 0
            per = Counter()
            loc = Counter()
            dists = []
            for a, b in pairs:
                ts = TS[a["reg"]]
                ca, cb = eq_census(a, ts), eq_census(b, ts)
                din = cb["in_idx-1"] - ca["in_idx-1"]
                dout = cb["out_idx-1"] - ca["out_idx-1"]
                lost = max(0, -din)
                g = min(max(0, dout), lost)
                lost_in += lost
                gain_out += g
                annih += lost - g
                if ca["in_idx-1"]:
                    per["models_with_idx-1_in_Q_at_lam0"] += 1
                    per["of_which_none_left_in_Q"] += cb["in_idx-1"] == 0
                    per["of_which_more_idx-1_outside_Q"] += dout > 0
                per["pairs_dK_negative"] += cb["K"] < ca["K"]
                per["pairs_dK_positive"] += cb["K"] > ca["K"]
                # location matching of every index -1 equilibrium inside Q at lam = 0
                Xb = [x for x in b["fps"] if x["x"] is not None]
                for x in a["fps"]:
                    if x["ind"] != -1 or not ts.inside(x["x"]):
                        continue
                    if not Xb:
                        loc["no_equilibrium_at_lam"] += 1
                        continue
                    dd = [np.linalg.norm(np.subtract(x["x"], z["x"])) / ts.L for z in Xb]
                    k = int(np.argmin(dd))
                    z = Xb[k]
                    dists.append(dd[k])
                    if dd[k] < 0.05:
                        typ = "stable" if z["nu"] == 0 else ("idx-1" if z["ind"] == -1 else "unst_idx+1")
                        loc[f"within_0.05L_{'in' if ts.inside(z['x']) else 'out'}_{typ}"] += 1
                    elif dd[k] < 0.2:
                        loc["nearest_0.05_0.2L"] += 1
                    else:
                        loc["nearest_beyond_0.2L"] += 1
            out["paired_vs_lam0"][f"{l:g}"] = {
                "n_pairs": len(pairs), "idx-1_in_Q_at_lam0": int(sum(eq_census(a, TS[a["reg"]])["in_idx-1"] for a, _ in pairs)),
                "idx-1_lost_from_Q": lost_in, "matched_by_new_idx-1_outside_Q_upper_bound_pushed_out": gain_out,
                "not_matched_lower_bound_annihilated": annih,
                "net_dK_pairs_removed": int(sum((eq_census(a, TS[a["reg"]])["K"] - eq_census(b, TS[b["reg"]])["K"])
                                                for a, b in pairs)) / 2,
                "model_counts": dict(per), "location_matching_of_lam0_idx-1_in_Q": dict(loc),
                "nearest_eq_distance_over_L": q3(dists)}
        return out
    res = {"note": ("stage P, mlp2 H = 256; equilibria with |x| <= 1e3; 'in' = inside Q; idx-1 = index -1 (always unstable); "
                    "unst_idx+1 = unstable with index +1. pushed_out (upper bound) = per pair min(loss of idx-1 inside Q, "
                    "gain of idx-1 outside Q); annihilated (lower bound) = the rest. Location matching: nearest "
                    "equilibrium of the penalized model (same seed) to each idx-1 equilibrium of the lam = 0 model in Q, "
                    "distance in units of L (equilibria move, so this is only indicative)"),
           "six_regimes": block(set(MAIN_REGIMES))}
    for g in ("chua_I", "chua_II", "se3", "se11", "wangchen2", "wangchen2_a0.03", "le1"):
        res[g] = block({g})
    return res


def outside_excitation(eqrows_P):
    out = {}
    for l in sorted({x["lam"] for x in eqrows_P}):
        R = [x for x in eqrows_P if x["lam"] == l and x["reg"] not in SEPARATE]
        out[f"{l:g}"] = {"unstable_outside_Q": shr(x["excited"] for x in R if x["inside"] is False),
                         "unstable_inside_Q": shr(x["excited"] for x in R if x["inside"])}
    return out


# ============================================================================================ 3. Proposition 1 on eqc
def prop1_eqc(V):
    out = {}
    for g in ALL_REGIMES:
        ts = TS[g]
        if not ts.eqs:
            continue
        res = {}
        for arm in ("eqc", "H1"):
            R = [r for r in V if r["reg"] == g and r["arm"] == arm]
            if not R:
                continue
            nets = []
            for r in R:
                imp = [x for x in r["fps"] if x["d"] < EXACT_TOL] if arm == "eqc" else \
                      [x for x in r["fps"] if x["d"] < COPY_TOL]
                byj = defaultdict(list)
                for x in imp:
                    byj[x["j"]].append(x)
                cop = [byj[j][0] if byj.get(j) else None for j in range(len(ts.eqs))]
                rest = [x for x in r["fps"] if all(x is not c for c in cop)]
                ci = [c["ind"] if c else None for c in cop]
                found_all = all(c is not None for c in cop)
                rest_idx = sum(x["ind"] for x in rest)
                nets.append(dict(found_all=found_all, ci=tuple(ci), ctype=tuple(eqtype(c["nu"], c["nr"]) if c else None for c in cop),
                                 true_idx=all(c is not None and c["ind"] == e["iota"] for c, e in zip(cop, ts.eqs)),
                                 rest_idx=rest_idx, n_rest=len(rest), n_rest_stable=sum(x["nu"] == 0 for x in rest),
                                 n_rest_unstable=sum(x["nu"] > 0 for x in rest), isum=r["isum"],
                                 prop1=(found_all and rest_idx == 1 - sum(ci)), K=len(r["fps"])))
            ti = [n for n in nets if n["true_idx"]]
            res[arm] = {
                "n_networks": len(nets), "all_true_equilibria_present": sum(n["found_all"] for n in nets),
                "index_sum_1": sum(n["isum"] == 1 for n in nets),
                "copy_indices_counts (order " + ",".join(e["name"] for e in ts.eqs) + ")": {str(k): v for k, v in Counter(n["ci"] for n in nets).items()},
                "copy_types_counts": {str(k): v for k, v in Counter(n["ctype"] for n in nets).items()},
                "copies_have_true_indices": sum(n["true_idx"] for n in nets),
                "rest_index_sum_counts": {str(k): v for k, v in sorted(Counter(n["rest_idx"] for n in nets).items())},
                "prop1_rest_equals_1_minus_sum_copy_indices": sum(n["prop1"] for n in nets),
                "with_true_indices_rest_equals_1_minus_s": [sum(n["rest_idx"] == 1 - ts.s for n in ti), len(ti)],
                "mean_n_rest": float(np.mean([n["n_rest"] for n in nets])),
                "mean_n_rest_stable": float(np.mean([n["n_rest_stable"] for n in nets])),
                "mean_n_rest_unstable": float(np.mean([n["n_rest_unstable"] for n in nets])),
                "n_rest_stable_counts": dict(sorted(Counter(n["n_rest_stable"] for n in nets).items())),
                "with_true_indices": {"n": len(ti), "mean_n_rest": float(np.mean([n["n_rest"] for n in ti])) if ti else None,
                                      "mean_n_rest_stable": float(np.mean([n["n_rest_stable"] for n in ti])) if ti else None,
                                      "networks_with_a_stable_extra": sum(n["n_rest_stable"] > 0 for n in ti)},
                "mean_K": float(np.mean([n["K"] for n in nets]))}
        out[g] = {"s": ts.s, "true_indices": {e["name"]: e["iota"] for e in ts.eqs}, **res}
    out["note"] = ("stage V (66+ networks, N = 300/700): eqc = readout constrained to W_out r* = x* for every true "
                   "equilibrium; imposed copies = equilibria with dist_true_eq < 1e-6 (the constraint residual is ~1e-15). "
                   "For comparison arm H1 (same reservoir, unconstrained) with copies at dist < 0.01. Proposition 1: if the "
                   "copies have the true indices iota_j, the other equilibria have total index 1 - s (2 for Chua I, 0 for "
                   "Chua II). copy types '<n_unstable>u_r<n_real_gt1>' (reservoir space)")
    return out


# ============================================================================================ 4. mechanism
def eq_rows_of(rec, ts):
    rows = []
    for y, h in rec["hm"].items():
        if not (h["rep"] and h.get("att")):
            continue
        a = h["att"]
        for x in rec["fps"]:
            if x["nu"] == 0:
                continue
            rc = rec["rc"].get(x["i"], {})
            reach = sorted(e for e, (t, k) in rc.items() if k.get(a, 0) > 0 and e <= R_LIGHT * (1 + 1e-9))
            share = (rc[reach[0]][1].get(a, 0) / rc[reach[0]][0]) if reach else None
            share_all = [rc[e][1].get(a, 0) / rc[e][0] for e in sorted(rc) if e <= R_LIGHT * (1 + 1e-9) and rc[e][0]]
            copy = x["d"] < COPY_TOL
            ck = None
            if copy:
                ck = ("true_unstable" if not ts.eqs[x["j"]]["stable"] else "true_stable") if (ts.eqs and x["j"] is not None) else "line"
            rows.append(dict(stage=rec["stage"], reg=rec["reg"], cls=rec["cls"], set=rec["set"], lam=rec.get("lam"),
                             key=rec["key"], y=y, a=a, inside=ts.inside(x["x"]), copy=copy, copy_kind=ck, nu=x["nu"],
                             dist_true_D=(x["d"] * ts.L / ts.att[y]["D"]) if x["d"] < 9 else None,
                             near_true=(ts.eqs[x["j"]]["name"] if (ts.eqs and x["j"] is not None) else None),
                             nr=x["nr"], ind=x["ind"], rho=x["rho"], type=eqtype(x["nu"], x["nr"]), excited=bool(reach),
                             r_min=reach[0] if reach else None, share_rmin=share,
                             share_min_all_r=min(share_all) if share_all else None,
                             share_max_all_r=max(share_all) if share_all else None, x=x["x"],
                             hidden=hidden_of(h["label"], rec["proto"]), n_radii=len(rc)))
    return rows


def add_geometry(rows):
    g = defaultdict(list)
    for k, x in enumerate(rows):
        if x["x"] is not None:
            g[(x["reg"], x["y"])].append(k)
    for (reg, y), idx in g.items():
        X = np.array([rows[k]["x"] for k in idx], float)
        geo = TS[reg].geom(y, X)
        for t, k in enumerate(idx):
            for nm, v in geo.items():
                rows[k][nm] = float(v[t])
            rows[k]["dbin"] = dbin(rows[k]["dist_att"])
            rows[k]["encircled"] = abs(rows[k]["w_rot"]) >= 0.5
            rows[k]["encircled_pc12"] = abs(rows[k]["w_pc12"]) >= 0.5


def null_vectors(rec, ts):
    """Matched nulls per reproduced y: stratified (from the confusion counts) and direct (per-point labels)."""
    out = {}
    conf = rec.get("conf") or {}
    for y, h in rec["hm"].items():
        if not (h["rep"] and h.get("att")):
            continue
        a = h["att"]
        pg = {}
        for gl, c in conf.items():
            s = sum(v for l, v in c.items() if l not in ("lift_undefined", "none"))
            if s:
                pg[gl] = c.get(a, 0) / s
        A = ts.att[y]
        strat = np.array([sum(w * pg.get(gl, 0.0) for gl, w in A["Pg_bin"][b].items()) if A["Pg_bin"][b] else np.nan
                          for b in range(NB)]) if pg else None
        direct = None
        if rec.get("lab") is not None:
            hit = rec["lab"] == a
            ok = ~np.isin(rec["lab"], ["lift_undefined", "none"])
            direct = np.array([float(np.mean(hit[(A["ics_bin"] == b) & ok])) if np.any((A["ics_bin"] == b) & ok) else np.nan
                               for b in range(NB)])
        out[y] = (strat, direct)
    return out


def mechanism(eqrows, nulls, pbmap, label):
    """eqrows: rows with geometry; nulls: {(key, y): (strat, direct)}; pbmap: {(key, y): pB_model}."""
    for x in eqrows:
        sv, dv = nulls.get((x["key"], x["y"]), (None, None))
        b = x.get("dbin")
        x["null_strat"] = float(sv[b]) if (sv is not None and b is not None and np.isfinite(sv[b])) else None
        x["null_direct"] = float(dv[b]) if (dv is not None and b is not None and np.isfinite(dv[b])) else None
        x["pB_model"] = pbmap.get((x["key"], x["y"]))
    R6 = [x for x in eqrows if x["reg"] not in SEPARATE and x.get("dist_att") is not None]

    def exc_tab(rs):
        return {"n": len(rs), "excited": shr(x["excited"] for x in rs),
                "mean_null_stratified": (float(np.mean([x["null_strat"] for x in rs if x["null_strat"] is not None]))
                                         if any(x["null_strat"] is not None for x in rs) else None),
                "mean_null_direct": (float(np.mean([x["null_direct"] for x in rs if x["null_direct"] is not None]))
                                     if any(x["null_direct"] is not None for x in rs) else None),
                "mean_pB_model_global": (float(np.mean([x["pB_model"] for x in rs if x["pB_model"] is not None]))
                                         if any(x["pB_model"] is not None for x in rs) else None)}

    def by_bin(rs):
        return {DBIN_NAMES[b]: exc_tab([x for x in rs if x["dbin"] == b]) for b in range(NB)}

    out = {"label": label, "n_rows_six": len(R6)}
    for scope, rs in (("inside_Q", [x for x in R6 if x["inside"]]), ("all", R6)):
        o = {"overall": exc_tab(rs), "by_distance_bin": by_bin(rs)}
        ex = [x for x in rs if x["excited"]]
        ne = [x for x in rs if not x["excited"]]
        cmp_ = {}
        for k in ("dist_att", "w_rot", "w_pc12", "axis_dist_rot", "centroid_dist", "offplane"):
            va = [abs(x[k]) if k.startswith("w_") else x[k] for x in ex]
            vb = [abs(x[k]) if k.startswith("w_") else x[k] for x in ne]
            t = PD.van_elteren([dict(x, v=abs(x[k]) if k.startswith("w_") else x[k]) for x in rs], "v", "excited",
                               lambda x: (x["reg"], x["cls"], x["set"]))
            cmp_[k] = {"exciting": q3(va), "non_exciting": q3(vb), "van_elteren_regime_class_set": t}
        o["exciting_vs_not"] = cmp_
        o["by_encircled_rot"] = {str(v): exc_tab([x for x in rs if x["encircled"] is v]) for v in (True, False)}
        o["by_encircled_pc12"] = {str(v): exc_tab([x for x in rs if x["encircled_pc12"] is v]) for v in (True, False)}
        o["by_type"] = {t: dict(exc_tab(v), share_encircled=shr(x["encircled"] for x in v),
                                dist_att=q3(x["dist_att"] for x in v))
                        for t, v in sorted(group_by(rs, "type").items(), key=lambda t: -len(t[1])) if len(v) >= 5}
        o["by_index"] = {str(t): exc_tab(v) for t, v in sorted(group_by(rs, "ind").items(), key=lambda t: str(t[0]))}
        o["by_copy_kind"] = {str(t): exc_tab(v) for t, v in group_by(rs, "copy_kind").items()}
        o["by_regime"] = {g: dict(exc_tab(v), encircled=shr(x["encircled"] for x in v),
                                  excited_if_encircled=shr(x["excited"] for x in v if x["encircled"]),
                                  excited_if_not_encircled=shr(x["excited"] for x in v if not x["encircled"]),
                                  share_type_2u_r0_among_encircled=shr(x["type"] == "2u_r0" for x in v if x["encircled"]),
                                  share_type_2u_r0_among_not_encircled=shr(x["type"] == "2u_r0" for x in v if not x["encircled"]))
                          for g, v in sorted(group_by(rs, "reg").items())}
        out[scope] = o
    # null check: stratified vs direct where both exist
    both = [x for x in R6 if x["null_strat"] is not None and x["null_direct"] is not None]
    if both:
        out["null_stratified_vs_direct"] = {"n": len(both), "mean_strat": float(np.mean([x["null_strat"] for x in both])),
                                            "mean_direct": float(np.mean([x["null_direct"] for x in both])),
                                            "by_bin": {DBIN_NAMES[b]: [float(np.mean([x["null_strat"] for x in both if x["dbin"] == b])),
                                                                       float(np.mean([x["null_direct"] for x in both if x["dbin"] == b])),
                                                                       sum(1 for x in both if x["dbin"] == b)]
                                                       for b in range(NB) if any(x["dbin"] == b for x in both)}}
    # rotation regimes: the virtual-centre hypothesis
    rot = {}
    for g in ROT_REGIMES:
        rs = [x for x in R6 if x["reg"] == g and x["inside"]]
        if not rs:
            continue
        ex = [x for x in rs if x["excited"]]
        ne = [x for x in rs if not x["excited"]]
        rot[g] = {"n": len(rs), "n_exciting": len(ex),
                  "exciting": {"axis_dist_rot": q3(x["axis_dist_rot"] for x in ex), "centroid_dist": q3(x["centroid_dist"] for x in ex),
                               "dist_att": q3(x["dist_att"] for x in ex), "encircled": shr(x["encircled"] for x in ex),
                               "type_2u_r0": shr(x["type"] == "2u_r0" for x in ex), "index_plus1": shr(x["ind"] == 1 for x in ex),
                               "types": dict(Counter(x["type"] for x in ex).most_common(6))},
                  "non_exciting": {"axis_dist_rot": q3(x["axis_dist_rot"] for x in ne), "centroid_dist": q3(x["centroid_dist"] for x in ne),
                                   "dist_att": q3(x["dist_att"] for x in ne), "encircled": shr(x["encircled"] for x in ne),
                                   "type_2u_r0": shr(x["type"] == "2u_r0" for x in ne), "index_plus1": shr(x["ind"] == 1 for x in ne),
                                   "types": dict(Counter(x["type"] for x in ne).most_common(6))},
                  "excited_if_encircled_and_2u_r0": shr(x["excited"] for x in rs if x["encircled"] and x["type"] == "2u_r0"),
                  "excited_if_encircled_other_type": shr(x["excited"] for x in rs if x["encircled"] and x["type"] != "2u_r0"),
                  "excited_if_not_encircled": shr(x["excited"] for x in rs if not x["encircled"]),
                  "dist_to_true_stable_eq_over_D": {"exciting": q3(x["dist_true_D"] for x in ex),
                                                    "non_exciting": q3(x["dist_true_D"] for x in ne)},
                  "excited_if_within_0.1D_of_true_eq": shr(x["excited"] for x in rs if (x["dist_true_D"] or 9) < 0.1),
                  "excited_if_farther_than_0.1D": shr(x["excited"] for x in rs if (x["dist_true_D"] or 9) >= 0.1),
                  "within_0.1D_of_true_eq_types": dict(Counter(x["type"] for x in rs if (x["dist_true_D"] or 9) < 0.1).most_common(6)),
                  "excited_if_within_0.1D_and_2u_r0": shr(x["excited"] for x in rs if (x["dist_true_D"] or 9) < 0.1 and x["type"] == "2u_r0"),
                  "share_exciting_within_0.1D_of_true_eq": shr((x["dist_true_D"] or 9) < 0.1 for x in ex),
                  "near_core_axis_dist_lt_0.1": shr(x["excited"] for x in rs if x["axis_dist_rot"] < 0.1),
                  "share_eq_near_core_lt_0.1": shr(x["axis_dist_rot"] < 0.1 for x in rs)}
    out["rotation_regimes_inside_Q"] = rot
    # logistic regression: excited ~ log10(dist_att + 1e-3) + encircled + index -1 + 2u_r0 + n_unstable >= 2 + copy + inside + regime
    rs = [x for x in R6 if x["inside"] is not None and x["ind"] is not None]
    if len(rs) > 50:
        regs = sorted({x["reg"] for x in rs})
        names = ["log10_dist_att", "encircled_rot", "index_minus1", "type_2u_r0", "n_unstable_ge2", "copy", "inside_Q"] + \
                [f"regime_{g}" for g in regs]
        X = np.array([[math.log10(x["dist_att"] + 1e-3), float(x["encircled"]), float(x["ind"] == -1),
                       float(x["type"] == "2u_r0"), float(x["nu"] >= 2), float(x["copy"]), float(bool(x["inside"]))]
                      + [float(x["reg"] == g) for g in regs] for x in rs])
        yv = np.array([float(x["excited"]) for x in rs])
        w, se, ll = PE.logit_fit(X, yv)
        _, _, ll0 = PE.logit_fit(X[:, [0, 2, 3, 4, 5, 6] + list(range(7, X.shape[1]))], yv)
        out["logit_excited"] = {"n": len(rs), "coef": {k: {"coef": float(a), "se": float(b)} for k, a, b in zip(names, w, se)
                                                         if not k.startswith("regime_")},
                                "LR_encircled": PE.lr_test(ll, ll0, 1)}
    return out


def group_by(rows, k):
    d = defaultdict(list)
    for x in rows:
        d[x.get(k)].append(x)
    return d


# ============================================================================================ 5. ball mixing
def ball_mixing(eqrows, label):
    R = [x for x in eqrows if x["reg"] not in SEPARATE and x["excited"] and x["share_rmin"] is not None]

    def f(rs):
        sh = [x["share_rmin"] for x in rs]
        return {"n_exciting": len(rs), "all_to_attractor_ge_0.99": shr(s >= 0.99 for s in sh),
                "mixed_lt_0.99": shr(s < 0.99 for s in sh), "minority_lt_0.5": shr(s < 0.5 for s in sh),
                "share_to_attractor_at_r_min": q3(sh),
                "mixed_at_every_radius": shr((x["share_max_all_r"] or 0) < 0.99 for x in rs)}
    return {"label": label, "pooled_six": f(R), "inside_Q": f([x for x in R if x["inside"]]),
            "by_type": {t: f(v) for t, v in sorted(group_by(R, "type").items(), key=lambda t: -len(t[1])) if len(v) >= 5},
            "by_regime": {g: f(v) for g, v in sorted(group_by(R, "reg").items())},
            "by_encircled": {str(v): f([x for x in R if x.get("encircled") is v]) for v in (True, False)}}


# ============================================================================================ figures
def figures(rows_by_class, mech_M, mech_C, eqM, eqC):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sys.path.insert(0, str(HERE.parents[1] / "src"))
        try:
            from hsearch import viz
            viz.setup_en()
        except Exception:  # noqa: BLE001
            pass
        plt.rcParams.update({"axes.grid": True, "axes.axisbelow": True, "grid.alpha": 0.5, "figure.facecolor": "white", "axes.facecolor": "white",
                             "savefig.facecolor": "white"})
    except Exception as e:  # noqa: BLE001
        print("figures skipped:", e)
        return []
    CLS = {"esn": ("#0072B2", "o", "ESN (stage C)"), "mlp": ("#E69F00", "s", "one-layer perceptron"),
           "mlp2": ("#009E73", "^", "two-layer perceptron"), "res": ("#D55E00", "D", "residual network")}
    REG = [("se3", "SE3"), ("se11", "SE11"), ("wangchen2", "WC"), ("wangchen2_a0.03", "WC$'$"), ("chua_I", "Chua I"),
           ("chua_II", "Chua II"), ("le1", "LE1")]
    made = []
    # ---- sm_fid_basins: (a) model p_B vs true p_B; (b) survival by reproduction of the visible equilibrium
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.69, 2.7), gridspec_kw={"width_ratios": [1.6, 1]})
    for gi, (g, gl) in enumerate(REG):
        ptrue = float(np.mean([np.mean(TS[g].labels == y) for y in TS[g].hidden]))
        a.plot([gi - 0.4, gi + 0.4], [ptrue, ptrue], color="k", lw=1.6, zorder=3)
        for ci, (c, (col, mk, lab)) in enumerate(CLS.items()):
            v = [x["pB_model"] for x in rows_by_class.get(c, []) if x["reg"] == g and x.get("pB_model") is not None]
            if len(v) < 3:
                continue
            q = np.quantile(v, [0.25, 0.5, 0.75])
            xx = gi - 0.27 + 0.18 * ci
            a.errorbar([xx], [q[1]], yerr=[[q[1] - q[0]], [q[2] - q[1]]], fmt=mk, color=col, ms=4, lw=1, capsize=0,
                       label=lab if gi == 0 else None)
    a.set_yscale("log")
    a.set_xticks(range(len(REG)))
    a.set_xticklabels([gl for _, gl in REG], rotation=30)
    a.set_ylabel("basin fraction in $Q$")
    a.plot([], [], color="k", lw=1.6, label="true system")
    a.legend(fontsize=7, loc="lower left", ncol=2, framealpha=0.9)
    a.text(-0.12, 1.03, "(a)", transform=a.transAxes, fontsize=9, fontweight="bold")
    sets = ("H1", "HE1", "E1")
    for ci, (c, (col, mk, lab)) in enumerate(CLS.items()):
        R = [x for x in rows_by_class.get(c, []) if x["reg"] in EQ_VIS and x["vis_ok"] is not None and x["hidden"] is not None]
        for si, s in enumerate(sets):
            for vi, v in enumerate((True, False)):
                rs = [x["hidden"] for x in R if x["set"] == s and x["vis_ok"] is v]
                if len(rs) < 5:
                    continue
                k, n = sum(rs), len(rs)
                lo, hi = wil(k, n)
                xx = si * 2.4 + vi * 1.1 + 0.2 * ci - 0.3
                b.errorbar([xx], [k / n], yerr=[[max(0.0, k / n - lo)], [max(0.0, hi - k / n)]], fmt=mk, color=col, ms=4, lw=1, capsize=0,
                           mfc=col if v else "white")
    b.set_xticks([si * 2.4 + vi * 1.1 for si in range(3) for vi in range(2)])
    b.set_xticklabels([f"{s}\n{'copy' if v else 'no copy'}" for s in sets for v in (True, False)], fontsize=7)
    b.set_ylabel("share kept hidden")
    b.set_ylim(-0.03, 1.03)
    b.text(-0.18, 1.03, "(b)", transform=b.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "sm_fid_basins.pdf", dpi=400)
    plt.close(fig)
    made.append(str(FIG / "sm_fid_basins.pdf"))
    # ---- sm_fid_mechanism: (a) excitation vs distance with matched null; (b) by winding and type (rotation regimes)
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.69, 2.6))
    for nm, mech, col, mk in (("perceptrons, residual nets (M)", mech_M, "#009E73", "^"), ("ESN (C)", mech_C, "#0072B2", "o")):
        if not mech:
            continue
        tab = mech["inside_Q"]["by_distance_bin"]
        xs, ys, lo_, hi_, nul = [], [], [], [], []
        for bi, bn in enumerate(DBIN_NAMES):
            t = tab[bn]
            if t["n"] < 10:
                continue
            xs.append(bi)
            ys.append(t["excited"]["share"])
            lo_.append(max(0.0, t["excited"]["share"] - t["excited"]["wilson95"][0]))
            hi_.append(max(0.0, t["excited"]["wilson95"][1] - t["excited"]["share"]))
            nul.append(t["mean_null_direct"] if t["mean_null_direct"] is not None else t["mean_null_stratified"])
        a.errorbar(xs, ys, yerr=[lo_, hi_], fmt=mk + "-", color=col, ms=4, lw=1.2, capsize=0, label=f"{nm}: excites")
        a.plot(xs, nul, mk + "--", color=col, ms=3, lw=1, mfc="white", label=f"{nm}: null")
    a.set_xticks(range(NB))
    a.set_xticklabels(DBIN_NAMES, rotation=30)
    a.set_xlabel("distance to the true attractor / diameter")
    a.set_ylabel("probability")
    a.set_ylim(-0.03, 1.03)
    a.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
    a.text(-0.14, 1.03, "(a)", transform=a.transAxes, fontsize=9, fontweight="bold")
    cats = [("encircled,\nunstable focus", lambda x: x["encircled"] and x["type"] == "2u_r0"),
            ("encircled,\nother", lambda x: x["encircled"] and x["type"] != "2u_r0"),
            ("not encircled,\nsaddle", lambda x: (not x["encircled"]) and x["type"] == "1u_r1"),
            ("not encircled,\nother", lambda x: (not x["encircled"]) and x["type"] != "1u_r1")]
    for si, (nm, eqs, col, mk) in enumerate((("M", eqM, "#009E73", "^"), ("C", eqC, "#0072B2", "o"))):
        R = [x for x in eqs if x["reg"] in ROT_REGIMES and x["inside"] and x.get("dist_att") is not None]
        for ci, (cn, fn) in enumerate(cats):
            v = [x["excited"] for x in R if fn(x)]
            if len(v) < 5:
                continue
            k, n = sum(v), len(v)
            lo, hi = wil(k, n)
            b.errorbar([ci + 0.15 * (si - 0.5)], [k / n], yerr=[[max(0.0, k / n - lo)], [max(0.0, hi - k / n)]], fmt=mk, color=col, ms=4, lw=1,
                       capsize=0, label=("perceptrons, residual nets (M)" if si == 0 else "ESN (C)") if ci == 0 else None)
            b.text(ci + 0.15 * (si - 0.5) + 0.05, k / n, f"{n}", fontsize=6, color=col, va="center")
    b.set_xticks(range(len(cats)))
    b.set_xticklabels([c for c, _ in cats], rotation=25, fontsize=7)
    b.set_ylabel("share that excites the attractor")
    b.set_ylim(-0.03, 1.03)
    b.legend(fontsize=6.5, loc="lower left", framealpha=0.9)
    b.text(-0.14, 1.03, "(b)", transform=b.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "sm_fid_mechanism.pdf", dpi=400)
    plt.close(fig)
    made.append(str(FIG / "sm_fid_mechanism.pdf"))
    return made


# ============================================================================================ main
TS = {}


def main():
    t0 = time.time()
    workers = int(os.environ.get("RMQ_CPU", "1"))
    for g in ALL_REGIMES:
        TS[g] = TrueSys(g)
    print(f"true systems ready ({time.time() - t0:.0f} s)", flush=True)
    res = {"meta": {"script": "paper2_esn/gpu/paper_fidelity.py", "generated": time.strftime("%Y-%m-%d %H:%M"),
                    "root": str(ROOT), "pooling": POOL_NOTE, "copy_tol": COPY_TOL, "limit_stage_M_files": LIMIT or None,
                    "quantiles": "[median, q25, q75, n] lists", "type_note": TYPE_NOTE},
           "definitions": __doc__.split("Definitions (also in fidelity.json[\"definitions\"]):")[1].split("Pooled statistics")[0].strip()}
    res["true_systems"] = {g: TS[g].summary() for g in ALL_REGIMES}
    print(f"true summaries ({time.time() - t0:.0f} s)", flush=True)

    M, P, E = load_light(workers)
    print(f"stage M {len(M)}, P {len(P)}, E-poly {len(E)} records ({time.time() - t0:.0f} s)", flush=True)
    C, c_info = load_C()
    NG = load_ngrc()
    V = load_V()
    N = load_N()
    print(f"stage C {len(C)} ESN, NG-RC {len(NG)}, V {len(V)} arms, N {len(N)} ({time.time() - t0:.0f} s)", flush=True)

    # ---------------------------------------------------------------- 1. basin fidelity rows
    def rows_of(recs):
        out = []
        for r in recs:
            out += fidelity_rows(r, TS[r["reg"]])[0]
        return out
    fM, fP, fE, fC, fNG, fV, fN = (rows_of(x) for x in (M, P, E, C, NG, V, N))
    for x in fV:
        x["arm"] = x["set"]
    res["1_basin_fidelity"] = {
        "stage_M": {"n_models": len(M), "n_rows_reproduced": len(fM), **fid_block(fM, ("cls", "set"))},
        "stage_C_esn": {"n_networks": len(C), "n_rows_reproduced": len(fC), "loader": c_info, **fid_block(fC, ("set",))},
        "stage_E_poly": {"n_models": len(E), "n_rows_reproduced": len(fE), **fid_block(fE, ("set",))},
        "stage_E_ngrc": {"n_models": len(NG), "n_rows_reproduced": len(fNG), **fid_block(fNG, ("set",))},
        "stage_P": {"n_models": len(P), "n_rows_reproduced": len(fP), **fid_block(fP, ("lam", "set"))},
        "stage_V": {"note": "true hidden basin census only (ics_basin_<y>[:1000] lifted); pB_reservoir_cube is a share of "
                            "the reservoir cube, not of Q", "n_arms": len(V), "n_rows_reproduced": len(fV),
                    **fid_block(fV, ("set",))},
        "stage_N": {"note": "size sweep (default settings, H1/HE1); true hidden basin census only", "n_models": len(N),
                    "n_rows_reproduced": len(fN), **fid_block(fN, ("set",))},
    }
    # paired penalty fidelity (same regime, set, seed; reproduced in both arms)
    byk = {(x["reg"], x["set"], x["seed"], x["y"], x["lam"]): x for x in fP}
    pair = {}
    for l in sorted({x["lam"] for x in fP}):
        if l == 0:
            continue
        D = [(x, byk.get((x["reg"], x["set"], x["seed"], x["y"], l))) for x in main_only(fP) if x["lam"] == 0.0]
        D = [(a, b) for a, b in D if b is not None]
        pp = {"n_pairs_reproduced_both": len(D)}
        for k in ("agree_bounded", "agree_all", "recall", "precision", "pB_ratio", "pB_ratio_bounded", "vis_recall"):
            dd = [(b[k] - a[k]) for a, b in D if a.get(k) is not None and b.get(k) is not None]
            pp[k] = {"lam0": q3(a[k] for a, _ in D), "lam": q3(b[k] for _, b in D), "median_diff": float(np.median(dd)) if dd else None,
                     "wilcoxon_p": float(stats.wilcoxon(dd).pvalue) if len(dd) > 5 and any(dd) else None}
        pp["vis_ok"] = {"lam0": shr(a["vis_ok"] for a, _ in D), "lam": shr(b["vis_ok"] for _, b in D)}
        pair[f"{l:g}"] = pp
    res["1_basin_fidelity"]["stage_P"]["paired_vs_lam0_six"] = pair
    # vis_ok over all models (not only reproductions), stage P and M
    res["1_basin_fidelity"]["stage_P"]["vis_ok_all_models_six"] = {
        f"{l:g}": {s: shr(fidelity_rows(r, TS[r["reg"]])[1]["vis_ok"] for r in P if r["lam"] == l and r["set"] == s and r["reg"] not in SEPARATE)
                   for s in ("H1", "HE1")} for l in sorted({r["lam"] for r in P})}
    print(f"fidelity rows done ({time.time() - t0:.0f} s)", flush=True)

    classes = {"mlp": [x for x in fM if x["cls"] == "mlp"], "mlp2": [x for x in fM if x["cls"] == "mlp2"],
               "res": [x for x in fM if x["cls"] == "res"], "esn": fC, "poly": fE, "ngrc": fNG,
               "mlp2_penalty_lam0": [x for x in fP if x["lam"] == 0.0],
               "mlp2_penalty_lam0.1": [x for x in fP if x["lam"] == 0.1],
               "mlp2_penalty_lam1": [x for x in fP if x["lam"] == 1.0]}
    res["1i_pB_table"] = pB_table(classes)

    # ---------------------------------------------------------------- 1(ii) visible state
    res["1ii_visible_state"] = {
        "note": "survival (hidden share) split by vis_ok among reproduced rows; five regimes whose visible state is an "
                "equilibrium; Mantel-Haenszel common odds ratio of hidden for vis_ok True vs False",
        "stage_M": vis_block(fM, ("cls", "set")), "stage_C_esn": vis_block(fC, ("set",)),
        "stage_E_poly": vis_block(fE, ("set",)), "stage_V": vis_block(fV, ("set",)), "stage_N": vis_block(fN, ("set",)),
        "stage_P": vis_block(fP, ("lam", "set"), ("lam",)),
        "share_models_with_visible_copy_all_models": {
            st: nest([dict(reg=r["reg"], cls=r["cls"], set=r["set"], v=fidelity_rows(r, TS[r["reg"]])[1]["vis_ok"])
                      for r in recs if r["reg"] in EQ_VIS], ("cls", "set"), lambda rs: shr(x["v"] for x in rs))
            for st, recs in (("M", M), ("C", C), ("E_poly", E), ("V", V), ("N", N))}}

    # ---------------------------------------------------------------- 1(iii) saddle-foci
    sfr = saddle_focus_rows(M + C + E + P + V + N)
    tot, det = sf_summary(sfr)
    lsum = loss_summary(None)
    LR = losses_by_source(M + C + E + V + N)
    LP = losses_by_source(P)
    res["1iii_saddle_foci"] = {
        "note": ("true saddle-foci: Chua I E1, E2; Chua II E0 (flow-map type: one real eigenvalue > 1 and a stable "
                 "complex pair, index -1, i.e. type 1u_r1). A copy excites when the attractor is reached from its balls "
                 "at r <= 1e-3; in the true system the saddle-foci do not excite the hidden attractors"),
        "by_regime_stage_class_set": det,
        "losses_by_exciter_kind": {"note": "excited reproduced (model, attractor) pairs; kinds of the exciting equilibria",
                                   **{g: nest([x for x in LR if x["reg"] == g], ("stage", "cls"), lsum)
                                      for g in ALL_REGIMES},
                                   "pooled_six": nest(main_only(LR), ("stage",), lsum),
                                   "stage_P_by_lam_six": nest(main_only(LP), ("lam",), lsum)}}
    print(f"saddle-foci done ({time.time() - t0:.0f} s)", flush=True)

    # ---------------------------------------------------------------- 2. penalty accounting
    eqP = []
    for r in P:
        eqP += eq_rows_of(r, TS[r["reg"]])
    res["2_penalty_accounting"] = penalty_accounting(P)
    res["2_penalty_accounting"]["excitation_by_location_per_lam_six"] = outside_excitation(eqP)
    res["2_penalty_accounting"]["losses_by_exciter_kind_per_lam_six"] = nest(main_only(LP), ("lam",), lsum)
    sfP = [x for x in sfr if x["stage"] == "P"]
    res["2_penalty_accounting"]["chua_saddle_foci_per_lam"] = {
        g: {f"{l:g}": {"models": len({x['key'] for x in sfP if x['reg'] == g and x['lam'] == l}),
                       "copies (dist < 0.01 L)": sum(x["n_copies"] for x in sfP if x["reg"] == g and x["lam"] == l),
                       "copies_unstable": sum(c["nu"] > 0 for x in sfP if x["reg"] == g and x["lam"] == l for c in x["copies"]),
                       "copies_type_ok": sum(c["type_ok"] for x in sfP if x["reg"] == g and x["lam"] == l for c in x["copies"]),
                       "nearest_model_eq_dist_over_L": q3(x["nearest_dist"] for x in sfP if x["reg"] == g and x["lam"] == l)}
            for l in sorted({x["lam"] for x in sfP})} for g in ("chua_I", "chua_II")}
    print(f"penalty accounting done ({time.time() - t0:.0f} s)", flush=True)

    # ---------------------------------------------------------------- 3. Proposition 1 on eqc
    res["3_prop1_eqc"] = prop1_eqc(V)

    # ---------------------------------------------------------------- 4. mechanism, 5. ball mixing
    eqM = []
    for r in M:
        eqM += eq_rows_of(r, TS[r["reg"]])
    eqC = []
    for r in C:
        eqC += eq_rows_of(r, TS[r["reg"]])
    eqP0 = [x for x in eqP if x["lam"] == 0.0]
    for rows in (eqM, eqC, eqP0):
        add_geometry(rows)
    print(f"geometry of {len(eqM) + len(eqC) + len(eqP0)} equilibria done ({time.time() - t0:.0f} s)", flush=True)
    nul, pbm = {}, {}
    for recs, rows in ((M, fM), (C, fC), ([r for r in P if r["lam"] == 0.0], fP)):
        for r in recs:
            for y, v in null_vectors(r, TS[r["reg"]]).items():
                nul[(r["key"], y)] = v
        for x in rows:
            pbm[(x["key"], x["y"])] = x.get("pB_model")
    mM = mechanism(eqM, nul, pbm, "stage M (mlp, mlp2, res), stratified null")
    mC = mechanism(eqC, nul, pbm, "stage C ESN (index and n_real_gt1 from stage I), direct and stratified null")
    mP = mechanism(eqP0, nul, pbm, "stage P lam = 0 (mlp2 H = 256), stratified null")
    res["4_mechanism"] = {"note": "unstable equilibria of models with a reproduced hidden attractor; one row per (equilibrium, "
                                  "reproduced attractor); excites = reached from a ball of radius <= 1e-3",
                          "stage_M": mM, "stage_C_esn": mC, "stage_P_lam0": mP,
                          "le1_separate_stage_M": {"excited_inside_Q": shr(x["excited"] for x in eqM if x["reg"] == "le1" and x["inside"])}}
    res["5_ball_mixing"] = {"note": ("share of the ball points (radius r_min, the smallest radius at which the attractor "
                                     "is reached) that end on the attractor; stage M balls: +-eps along each unstable "
                                     "direction plus 64 random points; stage C: +-eps along up to 20 directions plus 200 "
                                     "random points"),
                            "stage_M": ball_mixing(eqM, "M"), "stage_C_esn": ball_mixing(eqC, "C")}
    print(f"mechanism done ({time.time() - t0:.0f} s)", flush=True)

    try:
        res["figures"] = figures(classes, mM, mC, eqM, eqC)
    except Exception as e:  # noqa: BLE001
        res["figures"] = f"failed: {e!r}"
        print("figures failed:", repr(e))
    res["meta"]["runtime_s"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fidelity.json").write_text(json.dumps(PD.clean(res), indent=1), encoding="utf-8")
    print(f"written {OUT / 'fidelity.json'} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

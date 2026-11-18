"""Checks of the survival law: heterogeneity across regimes, clustering, link function, equilibrium-level
model, light-protocol labels, penalty calibration, mediation of the data intervention, size of the region Q, match
thresholds, and spectral radius of the reservoir.

Read-only on the row tables of paper2_esn/chaos/numbers (esn_rows.csv, esn_C_rows.csv, classes_rows.csv,
classes_eq_rows.csv, v_rows.csv, penalty_rows.csv) and, for items 2 (basin shares), 9, 10 and 11, on the raw records in
local/esn-gpu (stage M, N and C; loaded with the same selection and de-duplication as paper_classes.py and
paper_esn_law.py). Pooled statistics use the six regimes of paper_common.MAIN_REGIMES; LE1 appears only in per-regime
tables, labelled as the weakly hidden control.

Ensembles (rows = (model, hidden attractor) pairs; cluster = model, i.e. one trained network):
  esn_sweep   - ESN size sweep, light protocol: esn_rows.csv, sample n_sweep, reproduced, index sum 1; y = not_excited;
                x = U; covariates [set H1], log N, regime fixed effects (the fit of esn_law.json: -3.08 +- 0.40).
  esn_full    - ESN full-protocol ensemble (stage C): esn_C_rows.csv (one row per model); y = hidden_stat; x = U or U_in;
                covariates [set H1], [set H0], log N, [rho 0.6], [rho 1.2], regime fixed effects.
  classes     - perceptrons and residual networks (stage M): classes_rows.csv, reproduced; y = not_excited; x = U_in;
                covariates [mlp2], [res], [set HE1], [set E1], log width, regime fixed effects (classes.json: -3.94).

Items (keys of numbers/hetero.json):
  1 regime_survival     survival per regime and U (U_in) bin with counts and Wilson intervals; figure sm_regime_survival
  2 interaction         regime x log(1+U) likelihood-ratio and cluster-robust Wald tests; per-regime slopes (ML and
                        Firth); per-regime log-link slopes against log(1 - p_B) (heuristic (1 - p_B)^(beta U))
  3 main_fits_cluster   main fits with model-based, cluster-robust (CR1) and cluster-bootstrap SEs and cluster counts
    standardized        pooled bins: crude with cluster-bootstrap 95% intervals (models resampled within regime),
                        directly standardized to the overall regime mix of reproduced pairs, and model-based
                        (marginal) standardization; figure sm_standardized
  4 links               logit/log/cloglog links on U and log(1+U): log-likelihood, AIC, slopes, binned calibration;
                        all rows and U >= 1; figure sm_links
  5 equilibrium_model   P(unstable equilibrium excites the reproduced attractor) with a random intercept per model
                        (maximum likelihood, adaptive Gauss-Hermite) and GEE (exchangeable); implied survival
                        P(no equilibrium excites) vs observed by U_in, pooled and per regime; figure sm_eq_model
  6 excited_r           label excited_r (excited only at radius 1e-4 or 1e-3) by U bin; survival with excited_r as hidden
  7 penalty_calibration penalized two-layer perceptrons against stage-M two-layer perceptrons per (regime, set, U_in bin)
                        without smoothing; exact Poisson-binomial and cluster-bootstrap intervals; U_in = 0 comparison
  8 data_intervention   stage V (first hidden attractor): within-network pairs H1 -> E5/E10/E20, U changes per regime,
                        mediation of the variant effect through log(1+U) (cluster bootstrap), conditional logit
  9 q_scale             U_in with the region Q scaled about its centre by 0.5, 0.75, 1, 1.5, 2 (classes; ESN full)
 10 match_thresholds    reproduction thresholds 0.1 / 0.2 / 0.3 on delta_att and d_H (classes; ESN size sweep)
 11 esn_rho_labels      U by reservoir spectral radius (stage C) and full-protocol labels by U bin

Run (heavy: reads ~12 500 stage-M records and runs 2000 bootstrap replicates; through the resource queue):
    rmq run --cpu 2 --ram 3 --time 20m -- python -X utf8 paper2_esn/gpu/paper_hetero.py
Options: --boot B (default 2000), --only 1,2,... (subset of items), --out PATH, --no-fig, --m-limit N (debug).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace as NS  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import optimize, special, stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_common import MAIN_REGIMES, POOL_NOTE, SEPARATE  # noqa: E402

ROOT = Path(os.environ.get("ESN_DATA", Path(__file__).resolve().parents[2] / "local" / "esn-gpu"))
NUM = HERE.parent / "chaos" / "numbers"
FIG = HERE.parent / "chaos" / "figures"
INF = 10 ** 9
ESN_BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, INF)]
CLS_BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, INF)]
REGS = tuple(sorted(MAIN_REGIMES))
Z95 = 1.959964
R_MAX = 1e3
Q_SCALES = (0.5, 0.75, 1.0, 1.5, 2.0)
THRESHOLDS = (0.1, 0.2, 0.3)
CLS = ("mlp", "mlp2", "res")


# ============================================================================================== small helpers
def blabel(lo, hi):
    return f"{lo}" if lo == hi else (f">={lo}" if hi >= INF else f"{lo}-{hi}")


def bin_of(u, bins):
    u = np.asarray(u)
    out = np.full(u.shape, -1, int)
    for i, (lo, hi) in enumerate(bins):
        out[(u >= lo) & (u <= hi)] = i
    return out


def wilson(k, n, z=Z95):
    if n <= 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def clean(o):
    """JSON-safe: numpy scalars to Python, NaN/inf to None, tuples to lists."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [clean(v) for v in o.tolist()]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def rnd(x, d=4):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), d)


def codes(a):
    """Dense integer codes of an array of hashables."""
    _, inv = np.unique(np.asarray(a).astype(str), return_inverse=True)
    return inv.astype(int)


def pq(a, q):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return [float(x) for x in np.quantile(a, q)] if len(a) else [None] * len(q)


# ============================================================================================== GLM machinery
def _link(link):
    """mu = P(y = 1) as a function of eta. Returns (log mu, log(1 - mu), mu, 1 - mu, g) with g = mu'(eta) / (mu (1 - mu)),
    all computed without cancellation in 1 - mu; score factor = (y - mu) g, expected information weight = g^2 mu (1 - mu)."""
    if link == "logit":
        def f(e):
            return -np.logaddexp(0, -e), -np.logaddexp(0, e), special.expit(e), special.expit(-e), np.ones_like(e)
    elif link == "log":
        def f(e):
            en = np.minimum(e, -1e-300)
            mu = np.exp(en)
            omu = -np.expm1(en)
            with np.errstate(divide="ignore", invalid="ignore"):
                l1 = np.where(e < 0, np.log(omu), -np.inf)
            return en, l1, mu, omu, 1 / omu
    elif link == "cloglog":
        def f(e):
            ee = np.exp(np.minimum(e, 50))
            mu = -np.expm1(-ee)
            omu = np.exp(-ee)
            with np.errstate(divide="ignore", invalid="ignore"):
                lmu = np.log(np.maximum(mu, 1e-300))
                g = np.where(mu > 0, ee / np.maximum(mu, 1e-300), 1.0)
            return lmu, -ee, mu, omu, g
    else:
        raise ValueError(link)
    return f


def glm(X, y, link="logit", ridge=1e-6, beta0=None, max_iter=None):
    """Binomial GLM by Fisher scoring with step halving on the (ridge-penalized) log-likelihood. For the log link the
    iterates stay in the feasible region eta < 0 (P < 1). Returns beta, model-based covariance (inverse expected
    information, no ridge), unpenalized log-likelihood, per-row scores (for sandwich covariances), fitted mu."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    F = _link(link)
    if beta0 is not None:
        b = np.array(beta0, float)
    else:
        b = np.zeros(p)
    I = np.eye(p)

    def ll(b_):
        e = X @ b_
        if link == "log" and np.any(e >= 0):
            return -np.inf
        lmu, l1mu = F(e)[:2]
        v = y * lmu + (1 - y) * l1mu
        return float(np.sum(v)) if np.all(np.isfinite(v)) else -np.inf

    def sw(e):
        _, _, mu, omu, gg = F(e)
        r = y * omu - (1 - y) * mu                 # y - mu without cancellation
        return r * gg, gg * gg * mu * omu, mu

    if max_iter is None:
        max_iter = 3000 if link == "log" else 300
    obj = ll(b) - ridge * b @ b
    if not np.isfinite(obj):
        raise ValueError("infeasible start")
    conv = False
    it = -1
    for it in range(max_iter):
        e = X @ b
        s, w, mu = sw(e)
        g = X.T @ s - 2 * ridge * b
        H = (X * w[:, None]).T @ X + 2 * ridge * I
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        t = 1.0
        while t > 1e-10:
            nb = b + t * step
            no = ll(nb) - ridge * nb @ nb
            if np.isfinite(no) and no >= obj - 1e-12:
                break
            t /= 2
        else:
            conv = True          # no ascent direction left
            break
        d_obj = no - obj
        b, obj = nb, no
        if abs(d_obj) < 1e-10 and np.max(np.abs(t * step)) < 1e-7:
            conv = True
            break
    e = X @ b
    s, w, mu = sw(e)
    S = X * s[:, None]
    info = (X * w[:, None]).T @ X
    cov = np.linalg.pinv(info)
    gn = float(np.max(np.abs(X.T @ s - 2 * ridge * b)))
    return NS(beta=b, cov=cov, info=info, ll=ll(b), scores=S, mu=mu, eta=e, converged=bool(conv or gn < 1e-3),
              iters=it + 1, n=n, p=p, link=link, grad_norm=gn)


def logbin(X, y, beta0):
    """Log-binomial MLE (log P(y = 1) = X b, constraint X b <= 0) by a log-barrier Newton method: the log-likelihood is
    concave and the constraints linear, so the barrier path converges to the constrained maximum (which may lie on the
    boundary P = 1). SEs from the expected information at the solution (approximate when constraints are active)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    b = np.array(beta0, float)
    if np.any(X @ b >= 0):
        raise ValueError("infeasible start")

    def F(b_, mu_):
        e = X @ b_
        if np.any(e >= 0):
            return np.inf
        return -float(np.sum(y * e + (1 - y) * np.log(-np.expm1(e)))) - mu_ * float(np.sum(np.log(-e)))

    it_tot = 0
    for mu_ in 10.0 ** -np.arange(0, 13):
        f = F(b, mu_)
        for _ in range(200):
            it_tot += 1
            e = X @ b
            om = -np.expm1(e)
            ee = np.exp(e)
            g = -(X.T @ (y - (1 - y) * ee / om)) + mu_ * (X.T @ (1 / -e))
            Hm = (X * ((1 - y) * ee / om ** 2 + mu_ / e ** 2)[:, None]).T @ X
            try:
                d = -np.linalg.solve(Hm + 1e-12 * np.eye(p), g)
            except np.linalg.LinAlgError:
                d = -np.linalg.lstsq(Hm, g, rcond=None)[0]
            dec = float(-g @ d)
            if dec / 2 < 1e-12:
                break
            t = 1.0
            acc = False
            while t > 1e-14:
                nf = F(b + t * d, mu_)
                if np.isfinite(nf) and nf <= f + 0.25 * t * (g @ d):
                    acc = True
                    break
                t /= 2
            if not acc:
                break
            b, f = b + t * d, nf
    fit = glm(X, y, link="log", beta0=b, ridge=0.0, max_iter=0)
    fit.iters = it_tot
    fit.converged = True
    fit.boundary_rows = int(np.sum(X @ b > -1e-6))
    return fit


def cluster_cov(fit, cl):
    """CR1 cluster-robust covariance: G/(G-1) (n-1)/(n-p) B^-1 (sum_g s_g s_g') B^-1."""
    cl = codes(cl)
    G = cl.max() + 1
    Sg = np.zeros((G, fit.p))
    np.add.at(Sg, cl, fit.scores)
    Binv = fit.cov
    c = G / (G - 1) * (fit.n - 1) / max(fit.n - fit.p, 1)
    return c * Binv @ (Sg.T @ Sg) @ Binv, int(G)


def firth_logit(X, y, max_iter=200):
    """Firth (Jeffreys-prior penalized) logistic regression; finite estimates under separation."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    b = np.zeros(p)

    def pen(b_):
        e = X @ b_
        mu = special.expit(e)
        I_ = (X * (mu * (1 - mu))[:, None]).T @ X
        sg, ld = np.linalg.slogdet(I_)
        return float(np.sum(y * e - np.logaddexp(0, e)) + 0.5 * ld) if sg > 0 else -np.inf

    f = pen(b)
    for _ in range(max_iter):
        e = X @ b
        mu = special.expit(e)
        W = mu * (1 - mu)
        I_ = (X * W[:, None]).T @ X
        Iinv = np.linalg.pinv(I_)
        h = W * np.einsum("ij,jk,ik->i", X, Iinv, X)
        U = X.T @ (y - mu + h * (0.5 - mu))
        step = Iinv @ U
        t = 1.0
        while t > 1e-10:
            nf = pen(b + t * step)
            if nf >= f - 1e-12:
                break
            t /= 2
        b = b + t * step
        done = abs(nf - f) < 1e-10 and np.max(np.abs(t * step)) < 1e-8
        f = nf
        if done:
            break
    mu = special.expit(X @ b)
    I_ = (X * (mu * (1 - mu))[:, None]).T @ X
    cov = np.linalg.pinv(I_)
    return b, np.sqrt(np.diag(cov)), cov


def coef_entry(fit, j, cl=None, boot=None):
    out = {"coef": float(fit.beta[j]), "se_model": float(math.sqrt(max(fit.cov[j, j], 0)))}
    if cl is not None:
        V, G = cluster_cov(fit, cl)
        out["se_cluster_CR1"] = float(math.sqrt(max(V[j, j], 0)))
        out["n_clusters"] = G
    if boot is not None:
        bb = np.asarray(boot, float)
        bb = bb[np.isfinite(bb)]
        out["se_cluster_bootstrap"] = float(np.std(bb, ddof=1)) if len(bb) > 2 else None
        out["ci95_cluster_bootstrap"] = pq(bb, [0.025, 0.975])
    out["z_model"] = out["coef"] / out["se_model"] if out["se_model"] > 0 else None
    if "se_cluster_CR1" in out and out["se_cluster_CR1"] > 0:
        out["z_cluster"] = out["coef"] / out["se_cluster_CR1"]
    return out


def lr(ll1, ll0, df):
    c = max(0.0, 2 * (ll1 - ll0))
    return {"chi2": c, "df": int(df), "p": float(stats.chi2.sf(c, df))}


# ============================================================================================== ensembles
def ens_from(df, name, y, u, reg, cl, cov, cov_names, label, bins, xname, **extra):
    e = NS(name=name, y=np.asarray(y, float), u=np.asarray(u, int), reg=np.asarray(reg).astype(str),
           cl=codes(cl), cl_raw=np.asarray(cl).astype(str), cov=np.asarray(cov, float).reshape(len(y), -1),
           cov_names=list(cov_names), label=np.asarray(label).astype(str), bins=bins, xname=xname, df=df)
    for k, v in extra.items():
        setattr(e, k, v)
    return e


def load_ensembles(include_le1=False):
    keep = (lambda s: s.isin(MAIN_REGIMES)) if not include_le1 else (lambda s: s.isin(MAIN_REGIMES + SEPARATE))
    E = {}
    s = pd.read_csv(NUM / "esn_rows.csv")
    d = s[(s["sample"] == "n_sweep") & (s.reproduced == 1) & (s.index_sum == 1) & keep(s.regime)].reset_index(drop=True)
    E["esn_sweep"] = ens_from(d, "esn_sweep", d.not_excited, d.U, d.regime, d.model,
                              np.c_[(d.set == "H1").astype(float), np.log(d.N)], ["set_H1", "log_N"],
                              d.label.fillna(""), ESN_BINS, "U", N=d.N.to_numpy())
    c = pd.read_csv(NUM / "esn_C_rows.csv")
    d = c[keep(c.regime)].reset_index(drop=True)
    covC = np.c_[(d.set == "H1").astype(float), (d.set == "H0").astype(float), np.log(d.N),
                 np.isclose(d.rho, 0.6).astype(float), np.isclose(d.rho, 1.2).astype(float)]
    namesC = ["set_H1", "set_H0", "log_N", "rho_0.6", "rho_1.2"]
    E["esn_full_U"] = ens_from(d, "esn_full_U", d.hidden, d.U, d.regime, d.model, covC, namesC, d.label, ESN_BINS, "U",
                               U_in=d.U_in.to_numpy(), rho=d.rho.to_numpy())
    E["esn_full_U_in"] = ens_from(d, "esn_full_U_in", d.hidden, d.U_in, d.regime, d.model, covC, namesC, d.label,
                                  ESN_BINS, "U_in", U_all=d.U.to_numpy(), rho=d.rho.to_numpy())
    k = pd.read_csv(NUM / "classes_rows.csv")
    d = k[(k.rep == True) & keep(k.reg)].reset_index(drop=True)  # noqa: E712
    mid = d.cls + "|" + d.set + "|" + d.reg + "|" + d.H.astype(str) + "|" + d.seed.astype(str)
    covK = np.c_[(d.cls == "mlp2").astype(float), (d.cls == "res").astype(float), (d.set == "HE1").astype(float),
                 (d.set == "E1").astype(float), np.log(d.H)]
    E["classes"] = ens_from(d, "classes", d["ne"].astype(bool), d.Un, d.reg, mid, covK,
                            ["class_mlp2", "class_res", "set_HE1", "set_E1", "log_width"], d.label.fillna(""), CLS_BINS,
                            "U_in", cls=d.cls.to_numpy().astype(str), U_all=d.U.to_numpy(), set=d.set.to_numpy().astype(str))
    return E


def sub(e, mask):
    """Row subset of an ensemble (arrays only)."""
    m = np.asarray(mask)
    out = NS(**{k: v for k, v in vars(e).items()})
    for k, v in vars(e).items():
        if isinstance(v, np.ndarray) and v.shape[:1] == e.y.shape:
            setattr(out, k, v[m])
    out.cl = codes(out.cl_raw)
    nz = [j for j in range(out.cov.shape[1]) if np.any(out.cov[:, j] != out.cov[0, j])] if len(out.y) else []
    out.cov = out.cov[:, nz]          # drop covariates constant in the subset (e.g. class dummies within one class)
    out.cov_names = [e.cov_names[j] for j in nz]
    return out


def X_of(e, xcols, regs=None, idx=None, with_cov=True):
    """Design: x columns, covariates, regime fixed effects (no intercept)."""
    sel = slice(None) if idx is None else idx
    reg = e.reg[sel]
    regs = regs or sorted(set(reg))
    R = (reg[:, None] == np.array(regs)[None, :]).astype(float)
    parts = []
    if xcols is not None:
        parts.append(np.asarray(xcols, float).reshape(len(reg), -1))
    if with_cov and e.cov.shape[1]:
        parts.append(e.cov[sel])
    parts.append(R)
    return np.hstack(parts), regs


# ============================================================================================== bootstrap index
class ClusterBoot:
    """Resample clusters with replacement within strata; returns row indices of a replicate."""

    def __init__(self, strata, cl):
        cl = np.asarray(cl)
        order = np.argsort(cl, kind="stable")
        _, start, size = np.unique(cl[order], return_index=True, return_counts=True)
        self.order, self.start, self.size = order, start, size
        cs = np.asarray(strata).astype(str)[order][start]
        self.groups = [np.where(cs == g)[0] for g in sorted(set(cs))]

    def draw(self, rng):
        sel = np.concatenate([rng.choice(g, len(g)) for g in self.groups])
        sz, st = self.size[sel], self.start[sel]
        offs = np.arange(sz.sum()) - np.repeat(np.cumsum(sz) - sz, sz)
        return self.order[np.repeat(st, sz) + offs]


# ============================================================================================== item 1
def regime_bins(y, u, reg, cl, bins, regs):
    out = {}
    b = bin_of(u, bins)
    for g in regs:
        m = reg == g
        rows = []
        for i, (lo, hi) in enumerate(bins):
            mm = m & (b == i)
            n, k = int(mm.sum()), int(y[mm].sum())
            rows.append({"bin": blabel(lo, hi), "n": n, "k": k, "n_clusters": int(len(set(cl[mm]))),
                         "share": k / n if n else None, "wilson95": wilson(k, n)})
        out[g] = {"n": int(m.sum()), "k": int(y[m].sum()), "bins": rows}
    return out


def item1(E, E7):
    res = {"description": "survival (y = 1: not excited / hidden_stat) per regime and bin, with counts, clusters "
                          "(models) and Wilson 95% intervals; LE1 listed as the weakly hidden control (not pooled)",
           "bins_esn": [blabel(*b) for b in ESN_BINS], "bins_classes": [blabel(*b) for b in CLS_BINS]}
    regs7 = list(REGS) + list(SEPARATE)
    for name in ("esn_sweep", "esn_full_U", "esn_full_U_in"):
        e = E7[name]
        res[name] = regime_bins(e.y, e.u, e.reg, e.cl, ESN_BINS, regs7)
        res[name + "_bins6"] = regime_bins(e.y, e.u, e.reg, e.cl, CLS_BINS, regs7)
    e = E7["classes"]
    res["classes_all"] = regime_bins(e.y, e.u, e.reg, e.cl, CLS_BINS, regs7)
    for c in CLS:
        m = e.cls == c
        res[f"classes_{c}"] = regime_bins(e.y[m], e.u[m], e.reg[m], e.cl[m], CLS_BINS, regs7)
    return res


# ============================================================================================== item 2
def interaction(e, xfun=np.log1p, regs=REGS, name=""):
    x = xfun(e.u.astype(float))
    X0, rr = X_of(e, x[:, None])
    f0 = glm(X0, e.y)
    # per-regime slopes: x * [regime == g]
    R = (e.reg[:, None] == np.array(rr)[None, :]).astype(float)
    X1 = np.hstack([x[:, None] * R, e.cov, R])
    f1 = glm(X1, e.y)
    V1, G = cluster_cov(f1, e.cl)
    k = len(rr)
    C = np.zeros((k - 1, X1.shape[1]))
    for i in range(1, k):
        C[i - 1, 0], C[i - 1, i] = -1, 1
    cb = C @ f1.beta
    try:
        Wc = float(cb @ np.linalg.solve(C @ V1 @ C.T, cb))
        Wm = float(cb @ np.linalg.solve(C @ f1.cov @ C.T, cb))
    except np.linalg.LinAlgError:
        Wc = Wm = float("nan")
    bf, sf, Vf = firth_logit(X1, e.y)
    b0f, s0f, _ = firth_logit(X0, e.y)
    cbf = C @ bf
    Wf = float(cbf @ np.linalg.solve(C @ Vf @ C.T, cbf))
    seps = {g: separation(e.y[e.reg == g], e.u[e.reg == g]) for g in rr}
    ns = [i for i, g in enumerate(rr) if seps[g] is None]
    extra = {}
    if 1 < len(ns) < k:
        Cn = np.zeros((len(ns) - 1, X1.shape[1]))
        for j in range(1, len(ns)):
            Cn[j - 1, ns[0]], Cn[j - 1, ns[j]] = -1, 1
        cbn = Cn @ f1.beta
        Wn = float(cbn @ np.linalg.solve(Cn @ V1 @ Cn.T, cbn))
        keep = np.isin(e.reg, [rr[i] for i in ns])
        es = sub(e, keep)
        xs = xfun(es.u.astype(float))
        X0s, rrs = X_of(es, xs[:, None])
        Rs = (es.reg[:, None] == np.array(rrs)[None, :]).astype(float)
        f0s, f1s = glm(X0s, es.y), glm(np.hstack([xs[:, None] * Rs, es.cov, Rs]), es.y)
        extra = {"regimes_without_separation": [rr[i] for i in ns],
                 "wald_equal_slopes_cluster_CR1_without_separated": {"chi2": Wn, "df": len(ns) - 1,
                                                                     "p": float(stats.chi2.sf(Wn, len(ns) - 1))},
                 "LR_interaction_without_separated_regimes": lr(f1s.ll, f0s.ll, len(ns) - 1),
                 "common_slope_without_separated_regimes": coef_entry(f0s, 0, es.cl)}
    per = {}
    for i, g in enumerate(rr):
        m = e.reg == g
        yy, uu = e.y[m], e.u[m]
        sep = seps[g]
        per[g] = {"n": int(m.sum()), "k": int(yy.sum()), "n_clusters": int(len(set(e.cl[m]))),
                  "U_range": [int(uu.min()), int(uu.max())], "n_U0": int((uu == 0).sum()),
                  "slope_ML": float(f1.beta[i]), "se_model": float(math.sqrt(max(f1.cov[i, i], 0))),
                  "se_cluster_CR1": float(math.sqrt(max(V1[i, i], 0))),
                  "slope_firth": float(bf[i]), "se_firth": float(sf[i]), "separation": sep}
    return {"model": f"logit P(y) ~ {('log(1+' + e.xname + ')') if xfun is np.log1p else e.xname} x regime + "
                     f"{' + '.join(e.cov_names)} + regime fixed effects (no intercept; ridge 1e-6)",
            "n_rows": int(len(e.y)), "n_clusters": G, "n_regimes": k,
            "common_slope": coef_entry(f0, 0, e.cl), "common_slope_firth": [float(b0f[0]), float(s0f[0])],
            "LR_interaction": lr(f1.ll, f0.ll, k - 1),
            "wald_equal_slopes_cluster_CR1": {"chi2": Wc, "df": k - 1, "p": float(stats.chi2.sf(Wc, k - 1))
                                              if math.isfinite(Wc) else None},
            "wald_equal_slopes_model": {"chi2": Wm, "df": k - 1, "p": float(stats.chi2.sf(Wm, k - 1))
                                        if math.isfinite(Wm) else None},
            "wald_equal_slopes_firth": {"chi2": Wf, "df": k - 1, "p": float(stats.chi2.sf(Wf, k - 1))},
            "wald_note": "Wald tests that include a separated regime (slope -> -inf) are not reliable; use the LR test, "
                         "the Firth Wald test, or the tests without separated regimes",
            **extra, "per_regime": per}


def separation(yy, uu):
    if yy.sum() == 0:
        return "no survivors"
    if yy.sum() == len(yy):
        return "all survive"
    s_u, e_u = uu[yy == 1], uu[yy == 0]
    if s_u.max() < e_u.min():
        return "complete separation in U (all survivors below all excited)"
    if s_u.max() <= e_u.min():
        return "quasi-complete separation in U"
    return None


def heuristic_check(e, pB_by_regime):
    """Heuristic P(survive) = (1 - p_B)^(beta U): log-link slope in U per regime is b_g = beta log(1 - p_B,g). Fit the
    log-binomial model with free per-regime slopes and with the constrained slope beta * log(1 - p_B,g) * U (one
    beta), same covariates; LR test of the constraint (df = regimes - 1)."""
    rr = sorted(set(e.reg))
    R = (e.reg[:, None] == np.array(rr)[None, :]).astype(float)
    u = e.u.astype(float)
    b0 = []
    for g in rr:
        m = e.reg == g
        b0.append(math.log(min(max(e.y[m].mean(), 0.02), 0.9)) - 0.3)
    X1 = np.hstack([u[:, None] * R, e.cov, R])
    beta0 = np.r_[np.zeros(len(rr)), np.zeros(e.cov.shape[1]), b0]
    out = {"model": "log P(y) = b_g U + covariates + regime effects (log-binomial, P < 1 enforced)"}
    try:
        f1 = logbin(X1, e.y, beta0)
    except Exception as ex:   # noqa: BLE001
        return {"error": str(ex)}
    lq = np.array([math.log1p(-pB_by_regime[g]) if (pB_by_regime.get(g) is not None and
                                                     math.isfinite(pB_by_regime[g])) else np.nan for g in rr])
    if not np.all(np.isfinite(lq)):
        return {"error": "p_B missing for some regime", "p_B": pB_by_regime}
    xc = u * (R @ lq)
    X0 = np.hstack([xc[:, None], e.cov, R])
    f0 = logbin(X0, e.y, np.r_[0.0, np.zeros(e.cov.shape[1]), b0])
    V1, _ = cluster_cov(f1, e.cl)
    per = {}
    for i, g in enumerate(rr):
        pb = pB_by_regime.get(g)
        sep = separation(e.y[e.reg == g], e.u[e.reg == g])
        per[g] = {"p_B_median": pb, "log_1_minus_p_B": float(lq[i]),
                  "b_log_link": float(f1.beta[i]) if sep is None else None,
                  "se_model": float(math.sqrt(max(f1.cov[i, i], 0))) if sep is None else None,
                  "se_cluster_CR1": float(math.sqrt(max(V1[i, i], 0))) if sep is None else None,
                  "implied_beta": float(f1.beta[i] / lq[i]) if (lq[i] != 0 and sep is None) else None,
                  "separation": sep}
    ib = [v["implied_beta"] for v in per.values() if v["implied_beta"] is not None]
    lqv = [v["log_1_minus_p_B"] for v in per.values()]
    out.update({"per_regime": per, "range_log_1_minus_p_B": [min(lqv), max(lqv)],
                "ratio_max_over_min_abs_log_1_minus_p_B": min(lqv) / max(lqv) if max(lqv) else None,
                "implied_beta_range_non_separated": [min(ib), max(ib)] if ib else None,
                "note_separated": "a regime whose survivors all have smaller U than all excited pairs has b -> -inf "
                                  "(not reported); it still enters the likelihood-ratio test",
                "constrained_beta": coef_entry(f0, 0, e.cl), "loglik_free": f1.ll, "loglik_constrained": f0.ll,
                "LR_constraint": lr(f1.ll, f0.ll, len(rr) - 1), "max_fitted_P_free": float(np.max(f1.mu)),
                "converged": [bool(f1.converged), bool(f0.converged)]})
    return out


# ============================================================================================== item 3
def std_weights(e):
    rr = sorted(set(e.reg))
    w = np.array([np.mean(e.reg == g) for g in rr])
    return rr, w


def bin_stats(y, b, gi, nb, G, w, lab_y=None):
    """crude share per bin, direct standardization to weights w over regimes with data in the bin (renormalized)."""
    n = np.bincount(b, minlength=nb).astype(float)
    k = np.bincount(b, weights=y, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        crude = k / n
    idx = gi * nb + b
    ng = np.bincount(idx, minlength=G * nb).reshape(G, nb).astype(float)
    kg = np.bincount(idx, weights=y, minlength=G * nb).reshape(G, nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = kg / ng
    av = ng > 0
    num = np.nansum(np.where(av, w[:, None] * s, 0.0), axis=0)
    den = (w[:, None] * av).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dstd = num / den
    out = [crude, dstd, den]
    if lab_y is not None:
        kk = np.bincount(b, weights=lab_y, minlength=nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(kk / n)
    return out


def model_std(e, idx, b, nb, regs):
    """Marginal standardization: logit y ~ bin dummies + covariates + regime effects (ridge 1e-3); share of bin j =
    mean over all rows of the prediction with the bin set to j (overall regime and covariate mix)."""
    bb = b[idx]
    present = np.bincount(bb, minlength=nb) > 0
    cols = [j for j in range(1, nb) if present[j]]
    D = (bb[:, None] == np.array(cols)[None, :]).astype(float) if cols else np.zeros((len(idx), 0))
    X, _ = X_of(e, None, regs=regs, idx=idx)
    f = glm(np.hstack([D, X]), e.y[idx], ridge=1e-3)
    eta_rest = X @ f.beta[len(cols):]
    out = np.full(nb, np.nan)
    if present[0]:
        out[0] = special.expit(eta_rest).mean()
    for c_, j in enumerate(cols):
        out[j] = special.expit(eta_rest + f.beta[c_]).mean()
    return out


def item3(E, B, rng):
    res = {"description": "pooled survival per bin (six regimes): crude share with cluster-bootstrap 95% percentile "
                          "intervals (models resampled with replacement within regime), Wilson intervals on rows and "
                          "on the number of clusters (for bins where the bootstrap interval degenerates at 0 or 1), "
                          "direct standardization to the overall regime mix of the reproduced pairs (regimes without "
                          "data in a bin dropped and the weights renormalized; 'coverage' = total weight of regimes "
                          "present; cell_k_n_by_regime shows the cells: a regime cell with n = 1 gets the full regime "
                          "weight, so the direct estimate is unstable where cells are tiny), and model-based marginal "
                          "standardization (logit with bin dummies, covariates and regime effects, no bin x regime "
                          "interaction, averaged over all rows); excited_r_as_hidden = crude share counting the "
                          "light/full-protocol label excited_r as hidden (item 6)",
           "B": B, "main_fits": {}, "ensembles": {}}
    specs = [("esn_sweep", {"bins7": ESN_BINS, "bins6": CLS_BINS}, None),
             ("esn_full_U", {"bins7": ESN_BINS, "bins6": CLS_BINS}, None),
             ("esn_full_U_in", {"bins7": ESN_BINS, "bins6": CLS_BINS}, None),
             ("classes", {"bins6": CLS_BINS}, "cls")]
    for name, schemes, by in specs:
        e = E[name]
        regs, w = std_weights(e)
        gi = np.searchsorted(np.array(regs), e.reg)
        lab_y = np.maximum(e.y, (e.label == "excited_r").astype(float))
        subsets = {"all": np.arange(len(e.y))}
        if by:
            for c in CLS:
                subsets[c] = np.where(getattr(e, by) == c)[0]
        bschemes = {k: bin_of(e.u, v) for k, v in schemes.items()}
        xlog = np.log1p(e.u.astype(float))

        def stat(idx):
            out = {}
            for sname, sidx in subsets.items():
                mask = np.zeros(len(e.y), bool)
                mask[sidx] = True
                ii = idx[mask[idx]]
                for bk, b in bschemes.items():
                    nb = len(schemes[bk])
                    cr, ds, cov_, ex = bin_stats(e.y[ii], b[ii], gi[ii], nb, len(regs), w, lab_y[ii])
                    out[(sname, bk, "crude")] = cr
                    out[(sname, bk, "direct")] = ds
                    out[(sname, bk, "excited_r_as_hidden")] = ex
                    out[(sname, bk, "model")] = model_std(e, ii, b, nb, regs)
                X, _ = X_of(e, xlog[ii][:, None], regs=regs, idx=ii)
                out[(sname, "slope")] = np.array([glm(X, e.y[ii]).beta[0]])
            return out

        t0 = time.time()
        est = stat(np.arange(len(e.y)))
        cb = ClusterBoot(e.reg, e.cl)
        reps = defaultdict(list)
        for _ in range(B):
            r = stat(cb.draw(rng))
            for k_, v in r.items():
                reps[k_].append(v)
        reps = {k_: np.array(v) for k_, v in reps.items()}
        ens = {"x": e.xname, "regime_weights": dict(zip(regs, w.tolist())), "n_rows": int(len(e.y)),
               "n_clusters": int(len(set(e.cl))), "seconds": round(time.time() - t0, 1)}
        for sname, sidx in subsets.items():
            blk = {"n_rows": int(len(sidx)), "n_clusters": int(len(set(e.cl[sidx])))}
            for bk, b in bschemes.items():
                rows = []
                for j, (lo, hi) in enumerate(schemes[bk]):
                    m = b[sidx] == j
                    n_, k_ = int(m.sum()), int(e.y[sidx][m].sum())
                    ncl = int(len(set(e.cl[sidx][m])))
                    row = {"bin": blabel(lo, hi), "n": n_, "k": k_, "n_clusters": ncl, "share": k_ / n_ if n_ else None,
                           "wilson95_rows": wilson(k_, n_),
                           "wilson95_clusters": wilson(k_ * ncl / n_, ncl) if n_ else [None, None]}
                    for kind in ("crude", "direct", "model", "excited_r_as_hidden"):
                        v = est[(sname, bk, kind)][j]
                        bs = reps[(sname, bk, kind)][:, j]
                        row[kind] = None if not np.isfinite(v) else float(v)
                        row[kind + "_ci95_boot"] = pq(bs, [0.025, 0.975])
                        row[kind + "_boot_n_finite"] = int(np.isfinite(bs).sum())
                    _, _, cov_, _ = bin_stats(e.y[sidx], b[sidx], gi[sidx], len(schemes[bk]), len(regs), w, lab_y[sidx])
                    row["direct_coverage"] = float(cov_[j])
                    row["direct_regimes_present"] = [g for gi_, g in enumerate(regs)
                                                     if np.any((gi[sidx] == gi_) & m)]
                    row["k_excited_r_as_hidden"] = int(lab_y[sidx][m].sum())
                    row["cell_k_n_by_regime"] = {g: [int(e.y[sidx][m & (gi[sidx] == gi_)].sum()),
                                                     int((m & (gi[sidx] == gi_)).sum())] for gi_, g in enumerate(regs)}
                    rows.append(row)
                blk[bk] = rows
            blk["slope_log1p_bootstrap"] = {"est": float(est[(sname, "slope")][0]),
                                            "se_boot": float(np.std(reps[(sname, "slope")][:, 0], ddof=1)),
                                            "ci95": pq(reps[(sname, "slope")][:, 0], [0.025, 0.975])}
            ens[sname] = blk
        res["ensembles"][name] = ens
        # main fit with model-based, cluster-robust and cluster-bootstrap SEs
        X, regs_ = X_of(e, xlog[:, None])
        f = glm(X, e.y)
        names = [f"log1p_{e.xname}"] + e.cov_names + [f"regime_{g}" for g in regs_]
        Vc, G = cluster_cov(f, e.cl)
        res["main_fits"][name] = {
            "model": f"logit P(y) ~ log(1+{e.xname}) + {' + '.join(e.cov_names)} + regime fixed effects "
                     "(no intercept; ridge 1e-6)",
            "n_rows": int(len(e.y)), "n_clusters": G, "n_y1": int(e.y.sum()), "loglik": f.ll,
            "coef": {nm: {"coef": float(f.beta[j]), "se_model": float(math.sqrt(max(f.cov[j, j], 0))),
                          "se_cluster_CR1": float(math.sqrt(max(Vc[j, j], 0)))} for j, nm in enumerate(names)},
            "slope": coef_entry(f, 0, e.cl, reps[("all", "slope")][:, 0])}
        if by:
            for c in CLS:
                m = getattr(e, by) == c
                es = sub(e, m)
                Xs, _ = X_of(es, np.log1p(es.u.astype(float))[:, None])
                keepc = [j for j in range(Xs.shape[1]) if np.any(Xs[:, j] != 0)]
                fs = glm(Xs[:, keepc], es.y)
                res["main_fits"][name]["slope_" + c] = coef_entry(fs, 0, es.cl, reps[(c, "slope")][:, 0])
                res["main_fits"][name]["slope_" + c]["n_rows"] = int(m.sum())
    return res


# ============================================================================================== item 4
LINK_MODELS = [
    ("logit_log1pU", "logit", np.log1p, False),
    ("logit_U", "logit", lambda u: u, False),
    ("log_U", "log", lambda u: u, False),
    ("cloglog_exc_U", "cloglog", lambda u: u, True),
    ("cloglog_exc_log1pU", "cloglog", np.log1p, True),
]


def link_fits(e, models, bins):
    out = {}
    b = bin_of(e.u, bins)
    nb = len(bins)
    for nm, link, fx, flip in models:
        x = fx(e.u.astype(float))
        X, regs = X_of(e, x[:, None])
        yy = 1 - e.y if flip else e.y
        try:
            if link == "log":
                beta0 = np.zeros(X.shape[1])
                for j, g in enumerate(regs):
                    beta0[X.shape[1] - len(regs) + j] = math.log(min(max(yy[e.reg == g].mean(), 0.02), 0.9)) - 0.3
                f = logbin(X, yy, beta0)
            else:
                f = glm(X, yy, link=link)
        except Exception as ex:   # noqa: BLE001
            out[nm] = {"error": str(ex)}
            continue
        mu_s = 1 - f.mu if flip else f.mu
        E_ = np.bincount(b, weights=mu_s, minlength=nb)
        O_ = np.bincount(b, weights=e.y, minlength=nb)
        n_ = np.bincount(b, minlength=nb)

        def xlogx(o, ex):
            return o * math.log(o / ex) if o > 0 else 0.0
        G2 = float(sum(2 * (xlogx(O_[j], E_[j]) + xlogx(n_[j] - O_[j], n_[j] - E_[j])) for j in range(nb) if n_[j]))
        ce = coef_entry(f, 0, e.cl)
        out[nm] = {"link": link, "response": "P(excited)" if flip else "P(survive)",
                   "x": nm.split("_", 1)[1].split("_")[-1], "loglik": f.ll, "n_params": int(X.shape[1]),
                   "AIC": 2 * X.shape[1] - 2 * f.ll, "BIC": math.log(len(yy)) * X.shape[1] - 2 * f.ll, "slope": ce,
                   "converged": bool(f.converged), "iterations": int(f.iters), "max_abs_score": f.grad_norm,
                   "max_fitted_P_survive": float(np.max(mu_s)), "rows_at_P_1_log_link": getattr(f, "boundary_rows", None),
                   "binned_calibration": [{"bin": blabel(*bins[j]), "n": int(n_[j]), "observed": int(O_[j]),
                                           "expected": float(E_[j])} for j in range(nb)],
                   "binned_deviance_G2": G2}
        if nm == "cloglog_exc_logU":
            out[nm]["test_exponent_eq_1"] = {"z_model": (f.beta[0] - 1) / math.sqrt(f.cov[0, 0]),
                                             "z_cluster": (f.beta[0] - 1) / ce["se_cluster_CR1"],
                                             "note": "-log P(survive) proportional to U^c; independent equilibria with "
                                                     "a common excitation probability give c = 1"}
    best = min((v["AIC"], k) for k, v in out.items() if "AIC" in v)
    for v in out.values():
        if "AIC" in v:
            v["delta_AIC"] = v["AIC"] - best[0]
    return {"best_AIC": best[1], "models": out}


def item4(E):
    res = {"description": "same rows, covariates and regime fixed effects as the main fits; only the link and the "
                          "transform of U change. logit: P(survive); log: log P(survive) linear (log-binomial, fitted "
                          "P < 1 enforced); cloglog_exc: log(-log P(survive)) = cloglog of P(excited) linear in U or "
                          "log(1+U) (positive slope = more excitation). AIC comparable within a block (same rows). "
                          "U >= 1 block adds cloglog on log U (exponent test c = 1)."}
    for name in ("esn_sweep", "esn_full_U", "esn_full_U_in", "classes"):
        e = E[name]
        res[name] = {"all_rows": link_fits(e, LINK_MODELS, e.bins)}
        e1 = sub(e, e.u >= 1)
        mods = LINK_MODELS + [("cloglog_exc_logU", "cloglog", np.log, True)]
        r1 = link_fits(e1, mods, e.bins[1:])
        r1["n_rows"] = int(len(e1.y))
        res[name][f"{e.xname}_ge_1"] = r1
    return res


# ============================================================================================== item 5
def glmm_agq(X, y, gid, nq=15, theta0=None):
    """Logistic regression with a normal random intercept per group; ML by adaptive Gauss-Hermite quadrature
    (modes per group by Newton). Returns beta, sigma, covariance of (beta, log sigma) from a numerical Hessian, loglik."""
    G = int(gid.max()) + 1
    xk, wk = np.polynomial.hermite.hermgauss(nq)
    lw = np.log(wk) + xk ** 2

    def modes(eta0, s):
        """posterior modes of the random effects: Newton on a concave function, steps clipped to +-1 (an undamped
        step overshoots and oscillates in large groups whose equilibria all share one outcome)."""
        u = np.zeros(G)
        for _ in range(300):
            mu = special.expit(eta0 + u[gid])
            g = np.bincount(gid, y - mu, G) - u / s ** 2
            h = np.bincount(gid, mu * (1 - mu), G) + 1 / s ** 2
            st = np.clip(g / h, -1.0, 1.0)
            u += st
            if np.max(np.abs(st)) < 1e-12:
                break
        mu = special.expit(eta0 + u[gid])
        h = np.bincount(gid, mu * (1 - mu), G) + 1 / s ** 2
        return u, 1 / np.sqrt(h)

    from scipy import sparse
    n, p = X.shape
    Gm = sparse.csr_matrix((np.ones(n), (gid, np.arange(n))), shape=(G, n))
    s2pi = 0.5 * math.log(2 * math.pi)

    def parts(th, derivs):
        """negative log-likelihood (AGQ with nodes centred at the current modes) and, with derivs, its gradient and
        Hessian for nodes held fixed (analytic)."""
        b, ls = th[:-1], th[-1]
        s = math.exp(ls)
        eta0 = X @ b
        u, tau = modes(eta0, s)
        Uk = u[:, None] + math.sqrt(2) * tau[:, None] * xk[None, :]           # G x nq
        Ek = eta0[:, None] + Uk[gid]                                          # n x nq
        F = lw[None, :] + np.asarray(Gm @ (y[:, None] * Ek - np.logaddexp(0, Ek))) - 0.5 * (Uk / s) ** 2 - ls - s2pi
        L = np.log(math.sqrt(2) * tau) + special.logsumexp(F, axis=1)
        if not derivs:
            return -float(L.sum())
        P = np.exp(F - special.logsumexp(F, axis=1)[:, None])                # G x nq posterior node weights
        Pr = P[gid]
        mu = special.expit(Ek)
        r = y[:, None] - mu
        dls = (Uk / s) ** 2 - 1                                               # G x nq
        grad = np.r_[X.T @ (Pr * r).sum(1), (P * dls).sum()]
        Hm = np.zeros((p + 1, p + 1))
        Hm[:p, :p] = -(X * (Pr * mu * (1 - mu)).sum(1)[:, None]).T @ X
        Hm[p, p] = float((P * (-2 * (Uk / s) ** 2)).sum())
        Sk = np.empty((G, nq, p + 1))
        for k in range(nq):
            Sk[:, k, :p] = np.asarray(Gm @ (r[:, [k]] * X))
        Sk[:, :, p] = dls
        mean = np.einsum("gk,gkj->gj", P, Sk)
        Hm += np.einsum("gk,gki,gkj->ij", P, Sk, Sk) - mean.T @ mean
        return -float(L.sum()), -grad, -Hm

    if theta0 is None:
        f0 = glm(X, y)
        theta0 = np.r_[f0.beta, 0.0]
    th = np.array(theta0, float)
    msg = "max iterations"
    ok = False
    for it in range(100):
        f, g, H = parts(th, True)
        w_, V_ = np.linalg.eigh(H)
        step = V_ @ ((V_.T @ g) / np.maximum(w_, 1e-8 * max(1.0, w_.max())))
        t = 1.0
        while t > 1e-10:
            nf = parts(th - t * step, False)
            if nf <= f + 1e-10:
                break
            t /= 2
        th = th - t * step
        if abs(f - nf) < 1e-9 and np.max(np.abs(t * step)) < 1e-6:
            ok, msg = True, f"converged in {it + 1} Newton steps"
            break
    f, g, H = parts(th, True)
    cov = np.linalg.pinv(H)
    return NS(beta=th[:-1], sigma=math.exp(th[-1]), log_sigma=th[-1], cov=cov, ll=-f, success=ok, theta=th,
              message=msg, max_abs_grad=float(np.max(np.abs(g))), min_eig_hessian=float(np.linalg.eigvalsh(H).min()),
              newton_decrement=float(g @ cov @ g), fun=lambda t_: parts(t_, False))


def item5(E, rng):
    q = pd.read_csv(NUM / "classes_eq_rows.csv")
    q = q[q.reg.isin(MAIN_REGIMES)].reset_index(drop=True)
    q["mid"] = q.cls + "|" + q.set + "|" + q.reg + "|" + q.H.astype(str) + "|" + q.seed.astype(str)
    q["pid"] = q.mid + "|" + q.att
    y = q.excited.to_numpy().astype(float)
    regs = sorted(set(q.reg))
    cov_names = ["inside_Q", "copy", "dim_u_2", "dim_u_3", "class_mlp2", "class_res", "set_HE1", "set_E1", "log_width"]
    Xc = np.c_[q.inside, q["copy"], q.dim_u == 2, q.dim_u == 3, q.cls == "mlp2", q.cls == "res", q.set == "HE1",
               q.set == "E1", np.log(q.H)].astype(float)
    R = (q.reg.to_numpy()[:, None] == np.array(regs)[None, :]).astype(float)
    X = np.hstack([Xc, R])
    names = cov_names + [f"regime_{g}" for g in regs]
    gid = codes(q.mid)
    res = {"description": "one row per (unstable equilibrium, reproduced hidden attractor) of stage-M models, six "
                          "regimes; y = equilibrium excites the attractor at some radius (1e-6, 1e-4 or 1e-3); "
                          "random intercept per model (both attractors of a Chua model share it); covariates "
                          + ", ".join(cov_names) + " and regime effects (no intercept)",
           "n_rows": int(len(y)), "n_models": int(gid.max() + 1), "n_pairs": int(q.pid.nunique()),
           "group_size": {"mean": float(np.bincount(gid).mean()), "max": int(np.bincount(gid).max()),
                          "n_models_with_one_row": int(np.sum(np.bincount(gid) == 1))}}
    # raw shares
    res["raw_share_excites"] = {
        "all": [int(y.sum()), int(len(y)), float(y.mean())],
        "inside_Q": [int(y[q.inside == 1].sum()), int((q.inside == 1).sum()), float(y[q.inside == 1].mean())],
        "outside_Q": [int(y[q.inside == 0].sum()), int((q.inside == 0).sum()), float(y[q.inside == 0].mean())],
        "inside_Q_copies": [int(y[(q.inside == 1) & (q["copy"] == 1)].sum()), int(((q.inside == 1) & (q["copy"] == 1)).sum())],
        "inside_Q_spurious": [int(y[(q.inside == 1) & (q["copy"] == 0)].sum()), int(((q.inside == 1) & (q["copy"] == 0)).sum())],
        "by_class_inside_Q": {c: [int(y[(q.cls == c) & (q.inside == 1)].sum()), int(((q.cls == c) & (q.inside == 1)).sum())]
                              for c in CLS},
        "by_regime_inside_Q": {g: [int(y[(q.reg == g) & (q.inside == 1)].sum()), int(((q.reg == g) & (q.inside == 1)).sum())]
                               for g in regs},
        "excited_at_1e-6_only_inside_Q": float((q.r_min[q.inside == 1] == 1e-6).mean())}
    # independent logit (sigma = 0)
    f0 = glm(X, y)
    Vc, G = cluster_cov(f0, gid)
    t0 = time.time()
    m15 = glmm_agq(X, y, gid, nq=15)
    t1 = time.time() - t0
    s = m15.sigma
    se_ls = math.sqrt(max(m15.cov[-1, -1], 0))
    LRs = max(0.0, 2 * (m15.ll - f0.ll))
    icc = s * s / (s * s + math.pi ** 2 / 3)
    res["independent_logit"] = {"loglik": f0.ll, "coef": {nm: {"coef": float(f0.beta[j]),
                                                               "se_model": float(math.sqrt(f0.cov[j, j])),
                                                               "se_cluster_CR1": float(math.sqrt(Vc[j, j]))}
                                                          for j, nm in enumerate(names)}}
    xk, wk = np.polynomial.hermite.hermgauss(40)
    uk = math.sqrt(2) * s * xk
    eta = X @ m15.beta
    pa = (special.expit(eta[:, None] + uk[None, :]) @ wk) / math.sqrt(math.pi)
    res["random_intercept_model"] = {
        "method": "maximum likelihood, adaptive Gauss-Hermite quadrature (15 nodes per model, centred at the modes), "
                  "Newton steps with the analytic gradient and Hessian for nodes held at the current modes; SEs from that Hessian",
        "seconds": round(t1, 1), "converged": m15.success, "message": m15.message, "loglik": m15.ll, "max_abs_gradient": m15.max_abs_grad, "min_eig_hessian": m15.min_eig_hessian,
        "sigma": s, "sigma_ci95": [math.exp(m15.log_sigma - Z95 * se_ls), math.exp(m15.log_sigma + Z95 * se_ls)],
        "icc_latent": icc, "icc_latent_note": "sigma^2 / (sigma^2 + pi^2/3), within-model correlation of the latent "
                                              "logistic propensities",
        "median_odds_ratio": math.exp(math.sqrt(2) * s * stats.norm.ppf(0.75)),
        "LR_sigma_eq_0": {"chi2": LRs, "p_boundary": float(0.5 * stats.chi2.sf(LRs, 1)),
                          "note": "50:50 mixture of chi2(0) and chi2(1) under the null"},
        "coef": {nm: {"coef": float(m15.beta[j]), "se": float(math.sqrt(max(m15.cov[j, j], 0)))}
                 for j, nm in enumerate(names)},
        "p_excite_population_averaged": {"all_rows": float(pa.mean()), "inside_Q": float(pa[q.inside == 1].mean()),
                                         "outside_Q": float(pa[q.inside == 0].mean())},
        "p_excite_conditional_u0": {"all_rows": float(special.expit(eta).mean()),
                                    "inside_Q": float(special.expit(eta[q.inside == 1]).mean())}}
    # quadrature check: refit with 30 nodes from the 15-node optimum
    try:
        m30 = glmm_agq(X, y, gid, nq=30, theta0=m15.theta)
        res["random_intercept_model"]["check_30_nodes"] = {
            "loglik": m30.ll, "sigma": m30.sigma, "max_abs_coef_change": float(np.max(np.abs(m30.beta - m15.beta))),
            "converged": m30.success}
    except Exception as ex:   # noqa: BLE001
        res["random_intercept_model"]["check_30_nodes"] = str(ex)
    # GEE exchangeable
    try:
        import statsmodels.api as sm
        order = np.argsort(gid, kind="stable")
        gee = sm.GEE(y[order], X[order], groups=gid[order], family=sm.families.Binomial(),
                     cov_struct=sm.cov_struct.Exchangeable())
        gr = gee.fit(maxiter=100)
        res["gee_exchangeable"] = {"working_correlation_alpha": float(gee.cov_struct.dep_params),
                                   "coef": {nm: {"coef": float(gr.params[j]), "se_robust": float(gr.bse[j])}
                                            for j, nm in enumerate(names)},
                                   "note": "response-scale exchangeable correlation of the excitation indicators of "
                                           "two unstable equilibria of the same model (moment estimate)"}
    except Exception as ex:   # noqa: BLE001
        res["gee_exchangeable"] = {"error": str(ex)}
    # implied survival per reproduced pair: P(no unstable equilibrium excites) = E_u prod_i (1 - p_i(u))
    pid = codes(q.pid)
    P = int(pid.max()) + 1
    pid_key = q.pid.to_numpy()[np.unique(pid, return_index=True)[1]]
    xg, wg = np.polynomial.hermite.hermgauss(60)
    Sg = np.zeros(P)
    for k in range(60):
        lsk = np.bincount(pid, -np.logaddexp(0, eta + math.sqrt(2) * s * xg[k]), P)
        Sg += wg[k] * np.exp(lsk)
    Sg /= math.sqrt(math.pi)
    S_ind = np.exp(np.bincount(pid, -np.logaddexp(0, f0.eta), P))
    pbar = float(y.mean())
    nU = np.bincount(pid, minlength=P)
    S_geo = (1 - pbar) ** nU
    imp = pd.DataFrame({"pid": pid_key, "S_glmm": Sg, "S_indep": S_ind, "S_geom": S_geo,
                        "any_excited": np.bincount(pid, y, P) > 0, "n_unstable_rows": nU})
    e = E["classes"]
    d = e.df.copy()
    d["pid"] = d.cls + "|" + d.set + "|" + d.reg + "|" + d.H.astype(str) + "|" + d.seed.astype(str) + "|" + d.att
    d = d.merge(imp, on="pid", how="left")
    no_eq = d.S_glmm.isna()
    for c_ in ("S_glmm", "S_indep", "S_geom"):
        d.loc[no_eq, c_] = 1.0
    d.loc[no_eq, "any_excited"] = False
    d.loc[no_eq, "n_unstable_rows"] = 0
    yo = d["ne"].astype(bool).to_numpy()
    consistent = (~d.any_excited.astype(bool).to_numpy()) == yo
    res["pair_consistency"] = {
        "n_pairs": int(len(d)), "n_pairs_without_unstable_equilibria": int(no_eq.sum()),
        "pairs_label_not_excited_iff_no_equilibrium_row_excites": int(consistent.sum()),
        "mismatches": int((~consistent).sum()),
        "mismatch_note": "a reproduced pair can be excited from an equilibrium dropped as a Newton runaway (|x| > 1e3) "
                         "or vice versa; such pairs are counted as observed",
        "n_U_mismatch": int(np.sum(d.n_unstable_rows.to_numpy() != d.U.to_numpy()))}
    b = bin_of(d.Un.to_numpy(), CLS_BINS)

    def comp(mask):
        rows = []
        for j, (lo, hi) in enumerate(CLS_BINS):
            m = mask & (b == j)
            n_ = int(m.sum())
            row = {"bin": blabel(lo, hi), "n": n_, "observed": int(yo[m].sum())}
            for c_, key in (("S_glmm", "random_intercept"), ("S_indep", "independent_logit"),
                            ("S_geom", "constant_p_geometric")):
                Ev = float(d[c_].to_numpy()[m].sum())
                Vv = float((d[c_].to_numpy()[m] * (1 - d[c_].to_numpy()[m])).sum())
                row[key] = {"expected": Ev, "expected_share": Ev / n_ if n_ else None,
                            "z": (row["observed"] - Ev) / math.sqrt(Vv) if Vv > 1e-9 else None}
            rows.append(row)
        tot = {}
        for c_, key in (("S_glmm", "random_intercept"), ("S_indep", "independent_logit"),
                        ("S_geom", "constant_p_geometric")):
            zs = [r[key]["z"] for r in rows if r[key]["z"] is not None]
            X2 = float(np.sum(np.square(zs)))
            tot[key] = {"sum_z2_over_bins": X2, "n_bins": len(zs),
                        "observed_total": int(yo[mask].sum()), "expected_total": float(d[c_].to_numpy()[mask].sum())}
        return {"bins": rows, "totals": tot}

    res["implied_vs_observed_survival_by_U_in"] = {
        "note": "expected = sum over reproduced pairs of the implied probability that none of the pair's unstable "
                "equilibria (inside and outside Q) excites the attractor: random_intercept integrates the product over "
                "the model's random effect (60-node Gauss-Hermite); independent_logit uses the sigma = 0 fit; "
                "constant_p_geometric uses (1 - p)^U with p = raw share of exciting equilibria; z = (O - E)/sqrt(sum "
                "S(1-S)), which ignores parameter uncertainty and the sharing of a model by two Chua pairs",
        "constant_p": pbar, "pooled": comp(np.ones(len(d), bool)),
        "per_regime": {g: comp(d.reg.to_numpy() == g) for g in regs},
        "per_class": {c: comp(d.cls.to_numpy() == c) for c in CLS}}
    return res


# ============================================================================================== item 6
def item6(E):
    res = {"description": "label excited_r: the reproduced attractor is reached from an unstable equilibrium only at "
                          "radius 1e-4 or 1e-3, not at 1e-6 (light protocol: esn_sweep, classes; full protocol: "
                          "esn_full). survival_excited_r_as_hidden counts them with the survivors; bootstrap intervals "
                          "of these shares are under standardized.*.excited_r_as_hidden_ci95_boot"}
    for name in ("esn_sweep", "esn_full_U", "esn_full_U_in", "classes"):
        e = E[name]
        b = bin_of(e.u, e.bins)
        hid = e.y.astype(bool)
        exr = e.label == "excited_r"
        y2 = (hid | exr).astype(float)
        rows = []
        for j, (lo, hi) in enumerate(e.bins):
            m = b == j
            rows.append({"bin": blabel(lo, hi), "n": int(m.sum()), "excited_r": int(exr[m].sum()),
                         "hidden": int(hid[m].sum()), "self_excited": int((e.label[m] == "self_excited").sum()),
                         "share_hidden": float(hid[m].mean()) if m.any() else None,
                         "share_hidden_with_excited_r": float(y2[m].mean()) if m.any() else None})
        X, _ = X_of(e, np.log1p(e.u.astype(float))[:, None])
        f1 = glm(X, e.y)
        f2 = glm(X, y2)
        blk = {"n_rows": int(len(e.y)), "n_excited_r": int(exr.sum()), "labels": dict(Counter(e.label.tolist())),
               "excited_r_by_regime": {g: int(exr[e.reg == g].sum()) for g in sorted(set(e.reg))},
               "bins": rows, "slope_log1p_main": coef_entry(f1, 0, e.cl),
               "slope_log1p_excited_r_as_hidden": coef_entry(f2, 0, e.cl)}
        if name == "classes":
            blk["excited_r_by_class"] = {c: int(exr[e.cls == c].sum()) for c in CLS}
        res[name] = blk
    return res


# ============================================================================================== item 7
def poibin_pmf(p):
    pmf = np.zeros(len(p) + 1)
    pmf[0] = 1.0
    for i, pi in enumerate(p):
        pmf[1:i + 2] = pmf[1:i + 2] * (1 - pi) + pmf[0:i + 1] * pi
        pmf[0] *= (1 - pi)
    return pmf


def item7(rng, B):
    pen = pd.read_csv(NUM / "penalty_rows.csv")
    pen = pen[pen.reg.isin(MAIN_REGIMES)].reset_index(drop=True)
    pen = pen[pen.rep == True].reset_index(drop=True)  # noqa: E712
    k = pd.read_csv(NUM / "classes_rows.csv")
    ref_all = k[(k.rep == True) & (k.cls == "mlp2") & k.reg.isin(MAIN_REGIMES)].reset_index(drop=True)  # noqa: E712
    lams = sorted(pen.lam.unique())
    res = {"description": "expected survival of the penalized two-layer perceptrons (stage P, H = 256, sets H1/HE1, "
                          "six regimes) from the empirical survival k/n of stage-M two-layer perceptrons in the same "
                          "(regime, set, U_in bin) cell, no smoothing; a cell without stage-M rows is replaced by the "
                          "pooled (set, bin) cell over regimes, then by the pooled bin. exact: O ~ Poisson-binomial(p) "
                          "with the reference rates fixed; bootstrap: models of both stages resampled within (regime, "
                          "set), reference rates recomputed; diff = (O - E)/n",
           "bins": [blabel(*x) for x in CLS_BINS], "B": B}

    ref_all = ref_all[ref_all.set.isin(["H1", "HE1"])].reset_index(drop=True)   # the penalized models' sets
    nb = len(CLS_BINS)
    regs = sorted(MAIN_REGIMES)
    sets = ["H1", "HE1"]

    def calib(ref, tag):
        rb = bin_of(ref.Un.to_numpy(), CLS_BINS)
        pb = bin_of(pen.Un.to_numpy(), CLS_BINS)
        ref_key = ref.reg.to_numpy() + "|" + ref.set.to_numpy()
        pen_key = pen.reg.to_numpy() + "|" + pen.set.to_numpy()
        r_gs = np.array([regs.index(g) * 2 + sets.index(s_) for g, s_ in zip(ref.reg, ref.set)])
        p_gs = np.array([regs.index(g) * 2 + sets.index(s_) for g, s_ in zip(pen.reg, pen.set)])
        r_s = r_gs % 2
        p_s = p_gs % 2
        ref_cl = (ref.reg + "|" + ref.set + "|" + ref.H.astype(str) + "|" + ref.seed.astype(str)).to_numpy()
        pen_cl = (pen.reg + "|" + pen.set + "|" + pen.seed.astype(str) + "|" + pen.lam.astype(str)).to_numpy()
        ry = ref["ne"].astype(bool).to_numpy().astype(float)
        py_ = pen["ne"].astype(bool).to_numpy().astype(float)

        def rates(ii):
            """counts (k, n) per level: 0 = (regime, set, bin), 1 = (set, bin), 2 = bin."""
            out_ = []
            for idx, size in ((r_gs[ii] * nb + rb[ii], 12 * nb), (r_s[ii] * nb + rb[ii], 2 * nb), (rb[ii], nb)):
                out_.append((np.bincount(idx, ry[ii], size), np.bincount(idx, minlength=size)))
            return out_

        def expected(cnt, jj):
            ps = np.full(len(jj), np.nan)
            lev = np.full(len(jj), 3)
            for L_, idx in enumerate((p_gs[jj] * nb + pb[jj], p_s[jj] * nb + pb[jj], pb[jj])):
                k_, n_ = cnt[L_]
                free = (lev == 3) & (n_[idx] > 0)
                ps[free] = k_[idx][free] / n_[idx][free]
                lev[free] = L_
            return ps, lev

        cnt0 = rates(np.arange(len(ref)))
        out = {"reference": tag, "n_reference_pairs": int(len(ref)),
               "reference_cells": {f"{regs[c // (2 * nb)]}|{sets[(c // nb) % 2]}|{blabel(*CLS_BINS[c % nb])}":
                                   [int(cnt0[0][0][c]), int(cnt0[0][1][c])] for c in range(12 * nb) if cnt0[0][1][c]},
               "by_lam": {}}
        cbr = ClusterBoot(ref_key, ref_cl)
        for lam in lams:
            jj = np.where(pen.lam.to_numpy() == lam)[0]
            ps, lev = expected(cnt0, jj)
            O = int(py_[jj].sum())
            n = len(jj)
            Ev = float(np.nansum(ps))
            pmf = poibin_pmf(np.nan_to_num(ps))
            cdf = np.cumsum(pmf)
            p_lo = float(cdf[O])
            p_hi = float(1 - (cdf[O - 1] if O > 0 else 0.0))
            pi_lo = int(np.searchsorted(cdf, 0.025))
            pi_hi = int(np.searchsorted(cdf, 0.975))
            cbp = ClusterBoot(pen_key[jj], pen_cl[jj])
            diffs = []
            for _ in range(B):
                ii = cbr.draw(rng)
                kk_ = jj[cbp.draw(rng)]
                c_ = rates(ii)
                pp, _ = expected(c_, kk_)
                diffs.append((py_[kk_].sum() - np.nansum(pp)) / len(kk_))
            diffs = np.array(diffs)
            uin = pen.Un.to_numpy()[jj]
            out["by_lam"][f"{lam:g}"] = {
                "n": n, "observed": O, "expected": Ev, "share_observed": O / n, "share_expected": Ev / n,
                "cells_used": {"regime_set_bin": int((lev == 0).sum()), "pooled_set_bin": int((lev == 1).sum()),
                               "pooled_bin": int((lev == 2).sum())},
                "exact_poisson_binomial": {"P_O_le_obs": p_lo, "P_O_ge_obs": p_hi,
                                           "p_two_sided": min(1.0, 2 * min(p_lo, p_hi)),
                                           "O_95pct_range_under_reference": [pi_lo, pi_hi]},
                "bootstrap_diff_share": {"est": (O - Ev) / n, "ci95": pq(diffs, [0.025, 0.975]),
                                         "share_reps_le_0": float(np.mean(diffs <= 0))},
                "by_U_in": [{"bin": blabel(*CLS_BINS[j]), "n": int((pb[jj] == j).sum()),
                             "observed": int(py_[jj][pb[jj] == j].sum()), "expected": float(np.nansum(ps[pb[jj] == j]))}
                            for j in range(len(CLS_BINS))],
                "U_in_ge_1": {"n": int((uin >= 1).sum()), "observed": int(py_[jj][uin >= 1].sum()),
                              "expected": float(np.nansum(ps[uin >= 1]))}}
        return out

    res["reference_all_widths"] = calib(ref_all, "stage-M mlp2, all widths 8-1024")
    res["reference_H256"] = calib(ref_all[ref_all.H == 256].reset_index(drop=True), "stage-M mlp2, width 256 only")
    # U_in = 0: penalized (lam > 0) vs stage-M mlp2, by regime and set
    pen0 = pen[(pen.Un == 0) & (pen.lam > 0)]
    lam0 = pen[(pen.Un == 0) & (pen.lam == 0)]

    def tab(ref):
        ref0 = ref[ref.Un == 0]
        rows, strata = {}, []
        for g in sorted(MAIN_REGIMES):
            for st in ("H1", "HE1"):
                a = pen0[(pen0.reg == g) & (pen0.set == st)]
                r_ = ref0[(ref0.reg == g) & (ref0.set == st)]
                l0 = lam0[(lam0.reg == g) & (lam0.set == st)]
                ka, na = int(a["ne"].sum()), len(a)
                kr, nr = int(r_["ne"].sum()), len(r_)
                p = float(stats.fisher_exact([[ka, na - ka], [kr, nr - kr]], alternative="greater")[1]) \
                    if na and nr else None
                rows[f"{g}|{st}"] = {"penalized_lam_gt_0": [ka, na], "stage_M_mlp2": [kr, nr],
                                     "stage_P_lam_0": [int(l0["ne"].sum()), len(l0)], "fisher_one_sided_p": p}
                if na and nr:
                    strata.append((ka, na, kr, nr))
        # exact stratified test: sum of penalized survivors given each stratum's margins (convolved hypergeometrics)
        dist = np.array([1.0])
        obs = 0
        for ka, na, kr, nr in strata:
            K = ka + kr
            lo_, hi_ = max(0, K - nr), min(K, na)
            pm = np.zeros(hi_ + 1)
            pm[lo_:hi_ + 1] = stats.hypergeom.pmf(np.arange(lo_, hi_ + 1), na + nr, K, na)
            dist = np.convolve(dist, pm)
            obs += ka
        p_strat = float(dist[obs:].sum())
        ka, na = int(pen0["ne"].sum()), len(pen0)
        kr, nr = int(ref0["ne"].sum()), len(ref0)
        return {"by_regime_set": rows, "pooled": {"penalized_lam_gt_0": [ka, na], "stage_M_mlp2": [kr, nr],
                                                  "fisher_one_sided_p": float(stats.fisher_exact(
                                                      [[ka, na - ka], [kr, nr - kr]], alternative="greater")[1])},
                "exact_stratified_one_sided_p": p_strat,
                "P_all_penalized_survive_at_stage_M_pooled_rate": float((kr / nr) ** na) if nr else None}

    res["U_in_0_comparison"] = {"note": "reproduced pairs with U_in = 0; penalized = lam 0.1 and 1 pooled; one-sided "
                                        "tests of higher survival in the penalized models; exact stratified = "
                                        "conditional test over (regime, set) strata",
                                "all_widths": tab(ref_all), "H256": tab(ref_all[ref_all.H == 256])}
    return res


# ============================================================================================== item 8
def clogit(X, y, strata, ridge=1e-8):
    """Conditional logistic regression (strata = networks), exact enumeration of the risk sets."""
    groups = []
    for s_ in np.unique(strata):
        ii = np.where(strata == s_)[0]
        t = int(y[ii].sum())
        if t == 0 or t == len(ii):
            continue
        A = np.array([X[ii][list(c)].sum(0) for c in combinations(range(len(ii)), t)])
        groups.append((A, X[ii][y[ii] == 1].sum(0)))
    p = X.shape[1]
    b = np.zeros(p)

    def fgh(b_):
        f, g, H = 0.0, np.zeros(p), np.zeros((p, p))
        for A, xo in groups:
            a = A @ b_
            lse = special.logsumexp(a)
            w = np.exp(a - lse)
            xb = w @ A
            f -= xo @ b_ - lse
            g -= xo - xb
            H += (A * w[:, None]).T @ A - np.outer(xb, xb)
        return f + ridge * b_ @ b_, g + 2 * ridge * b_, H + 2 * ridge * np.eye(p)

    f, g, H = fgh(b)
    for _ in range(100):
        st = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-10:
            nf, ng, nH = fgh(b - t * st)
            if nf <= f + 1e-12:
                break
            t /= 2
        b = b - t * st
        done = abs(f - nf) < 1e-11
        f, g, H = nf, ng, nH
        if done:
            break
    return b, np.sqrt(np.diag(np.linalg.pinv(H))), -f, len(groups)


def item8(rng, B):
    v = pd.read_csv(NUM / "v_rows.csv")
    v = v[v.regime.isin(MAIN_REGIMES) & v.target.isin(["HA", "HA_a"])].reset_index(drop=True)
    v["net"] = v.regime + "|" + v.N.astype(str) + "|" + v.rho.astype(str) + "|" + v.eta.astype(str) + "|" + v.seed.astype(str)
    arms = ["E5", "E10", "E20"]
    res = {"description": "stage V, six regimes, first hidden attractor of each network (HA / HA_a), as "
                          "data_effects.json stage_V.first_hidden_attractor_only; network = reservoir (regime, N, rho, "
                          "eta, seed), whose readout is trained on H1 or on H1 plus 5/10/20 transients (E5/E10/E20); "
                          "analysis rows = attractor reproduced and index sum 1",
           "n_networks": int(v.net.nunique())}
    w = v.pivot_table(index="net", columns="variant", values="U", aggfunc="first")
    ina = v.pivot_table(index="net", columns="variant", values="in_analysis", aggfunc="first")
    hid = v.pivot_table(index="net", columns="variant", values="hidden_or_not_excited", aggfunc="first")
    regn = v.groupby("net").regime.first()
    paired = {}
    for a in arms:
        ok = (ina["H1"] == 1) & (ina[a] == 1)
        dU = (w[a] - w["H1"])[ok]
        dh = (hid[a] - hid["H1"])[ok]
        sgn = np.sign(dU)
        tabx = {f"{lab_h}|dU_{lab_u}": int(((dh == vh) & (sgn == vu)).sum())
                for vh, lab_h in ((1, "appeared"), (0, "unchanged"), (-1, "lost")) for vu, lab_u in ((-1, "neg"), (0, "zero"), (1, "pos"))}
        nap, nlo = int((dh == 1).sum()), int((dh == -1).sum())
        ndn, nup = int((dU < 0).sum()), int((dU > 0).sum())
        paired[f"H1->{a}"] = {
            "n_pairs": int(ok.sum()), "U_decreased": ndn, "U_same": int((dU == 0).sum()), "U_increased": nup,
            "median_dU": float(np.median(dU)), "mean_dU": float(np.mean(dU)),
            "sign_test_dU_p": float(stats.binomtest(ndn, ndn + nup).pvalue) if ndn + nup else None,
            "hidden_appeared": nap, "hidden_lost": nlo,
            "mcnemar_exact_p": float(stats.binomtest(nap, nap + nlo).pvalue) if nap + nlo else None,
            "crosstab_hidden_change_by_sign_dU": tabx,
            "by_regime": {g: {"n_pairs": int((regn[ok.index[ok]] == g).sum()),
                              "U_down_same_up": [int(((dU < 0) & (regn[dU.index] == g)).sum()),
                                                 int(((dU == 0) & (regn[dU.index] == g)).sum()),
                                                 int(((dU > 0) & (regn[dU.index] == g)).sum())],
                              "appeared_lost": [int(((dh == 1) & (regn[dh.index] == g)).sum()),
                                                int(((dh == -1) & (regn[dh.index] == g)).sum())]}
                          for g in sorted(MAIN_REGIMES)}}
    res["paired_within_network"] = paired
    # U per regime and variant: all networks (U defined whether or not the attractor is reproduced)
    uchg = {}
    for g in sorted(MAIN_REGIMES):
        nets = regn.index[regn == g]
        blk = {"n_networks": int(len(nets)), "U_mean_H1": float(w.loc[nets, "H1"].mean()),
               "U_median_H1": float(w.loc[nets, "H1"].median())}
        for a in arms:
            d_ = (w.loc[nets, a] - w.loc[nets, "H1"]).dropna()
            blk[a] = {"U_mean": float(w.loc[nets, a].mean()), "U_median": float(w.loc[nets, a].median()),
                      "dU_mean": float(d_.mean()), "dU_median": float(d_.median()),
                      "down_same_up": [int((d_ < 0).sum()), int((d_ == 0).sum()), int((d_ > 0).sum())],
                      "wilcoxon_p": float(stats.wilcoxon(d_).pvalue) if np.any(d_ != 0) else None,
                      "dlog1pU_mean": float((np.log1p(w.loc[nets, a]) - np.log1p(w.loc[nets, "H1"])).mean())}
        uchg[g] = blk
    alln = regn.index
    uchg["six_regimes"] = {a: {"dU_mean": float((w.loc[alln, a] - w.loc[alln, "H1"]).mean()),
                               "dU_median": float((w.loc[alln, a] - w.loc[alln, "H1"]).median()),
                               "down_same_up": [int(((w[a] - w["H1"]) < 0).sum()), int(((w[a] - w["H1"]) == 0).sum()),
                                                int(((w[a] - w["H1"]) > 0).sum())]} for a in arms}
    res["U_change_per_regime_all_networks"] = uchg

    # ---------------- mediation
    D = v[(v.in_analysis == 1) & v.variant.isin(["H1"] + arms)].reset_index(drop=True)
    yv = D.hidden_or_not_excited.to_numpy().astype(float)
    Mv = np.log1p(D.U.to_numpy().astype(float))
    A = np.column_stack([(D.variant == a).to_numpy().astype(float) for a in arms])
    regs = sorted(set(D.regime))
    Rg = (D.regime.to_numpy()[:, None] == np.array(regs)[None, :]).astype(float)
    lN = np.log(D.N.to_numpy().astype(float))[:, None]
    net = D.net.to_numpy()

    def med(ii, inter_out=False, inter_med=False):
        y_, M_, A_, R_, N_ = yv[ii], Mv[ii], A[ii], Rg[ii], lN[ii]
        # mediator model: OLS log(1+U) ~ arm (+ arm x regime) + log N + regime
        if inter_med:
            AR = np.hstack([A_[:, [j]] * R_ for j in range(3)])
            keep = np.where(AR.any(0))[0]
            Xm = np.hstack([AR[:, keep], N_, R_])
        else:
            Xm = np.hstack([A_, N_, R_])
        cm = np.linalg.lstsq(Xm, M_, rcond=None)[0]
        resid = M_ - Xm @ cm

        def mu_med(a):
            Aa = np.zeros_like(A_)
            if a > 0:
                Aa[:, a - 1] = 1
            if inter_med:
                ARa = np.hstack([Aa[:, [j]] * R_ for j in range(3)])[:, keep]
                return np.hstack([ARa, N_, R_]) @ cm
            return np.hstack([Aa, N_, R_]) @ cm
        # outcome models
        f_noM = glm(np.hstack([A_, N_, R_]), y_)
        Xo = np.hstack([A_, M_[:, None], N_, R_] + ([A_ * M_[:, None]] if inter_out else []))
        f_o = glm(Xo, y_)
        bo = f_o.beta
        gA, bM = bo[:3], bo[3]
        dA = bo[-3:] if inter_out else np.zeros(3)
        base = np.hstack([N_, R_]) @ bo[4:4 + 1 + R_.shape[1]]

        def P(a, ap):
            ga = gA[a - 1] if a > 0 else 0.0
            sl = bM + (dA[a - 1] if a > 0 else 0.0)
            c = base + ga + sl * mu_med(ap)
            return float(special.expit(c[:, None] + sl * resid[None, :]).mean())
        out = {}
        p00 = P(0, 0)
        out["P_H1"] = p00
        for j, a in enumerate(arms, start=1):
            paa, pa0 = P(j, j), P(j, 0)
            te, nde, nie = paa - p00, pa0 - p00, paa - pa0
            lg = lambda x: math.log(x / (1 - x))  # noqa: E731
            out[a] = {"P_a_Ma": paa, "P_a_MH1": pa0, "TE": te, "NDE": nde, "NIE": nie,
                      "prop_mediated": nie / te if abs(te) > 1e-9 else np.nan,
                      "logOR_TE": lg(paa) - lg(p00), "logOR_NDE": lg(pa0) - lg(p00), "logOR_NIE": lg(paa) - lg(pa0),
                      "alpha_arm_on_log1pU": float(cm[j - 1]) if not inter_med else float(np.mean(mu_med(j) - mu_med(0))),
                      "c_total_logit": float(f_noM.beta[j - 1]), "c_direct_logit": float(gA[j - 1]),
                      "indirect_product_logit": float((cm[j - 1] if not inter_med else np.mean(mu_med(j) - mu_med(0))) * bM),
                      "indirect_difference_logit": float(f_noM.beta[j - 1] - gA[j - 1])}
        out["beta_log1pU_logit"] = float(bM)
        return out

    def summarize(est, reps):
        o = {"P_H1": {"est": est["P_H1"], "ci95": pq([r["P_H1"] for r in reps], [0.025, 0.975])},
             "beta_log1pU_logit": {"est": est["beta_log1pU_logit"],
                                   "ci95": pq([r["beta_log1pU_logit"] for r in reps], [0.025, 0.975])}}
        for a in arms:
            o[a] = {}
            for k_ in est[a]:
                bs = np.array([r[a][k_] for r in reps], float)
                o[a][k_] = {"est": est[a][k_], "ci95": pq(bs, [0.025, 0.975]),
                            "boot_share_le_0": float(np.mean(bs[np.isfinite(bs)] <= 0)) if np.isfinite(bs).any() else None}
        return o

    cb = ClusterBoot(D.regime.to_numpy(), net)
    full = np.arange(len(D))
    variants = {"primary": dict(), "outcome_arm_x_log1pU": dict(inter_out=True),
                "mediator_arm_x_regime": dict(inter_med=True)}
    medres = {}
    for nm, kw in variants.items():
        ii0 = full
        cbx = cb
        if kw.get("inter_med"):
            # regimes with reproduced rows in all four arms (chua_II has none for E5/E10/E20)
            okr = [g for g in regs if all(((D.regime == g) & (D.variant == a)).any() for a in ["H1"] + arms)]
            ii0 = np.where(D.regime.isin(okr).to_numpy())[0]
            cbx = ClusterBoot(D.regime.to_numpy()[ii0], net[ii0])
        est = med(ii0, **kw)
        reps = []
        for _ in range(B):
            ii = ii0[cbx.draw(rng)] if kw.get("inter_med") else cbx.draw(rng)
            try:
                reps.append(med(ii, **kw))
            except Exception:   # noqa: BLE001
                continue
        medres[nm] = {"n_rows": int(len(ii0)), "n_networks": int(len(set(net[ii0]))), "B_ok": len(reps),
                      **summarize(est, reps)}
        if kw.get("inter_med"):
            medres[nm]["regimes"] = okr
    res["mediation"] = {
        "description": "exposure = data variant (E5, E10, E20 vs H1); mediator M = log(1+U); outcome = not excited "
                       "(light protocol); confounders = regime and log N. Mediator model: OLS of M on variant + log N + "
                       "regime (variant x regime in 'mediator_arm_x_regime', restricted to regimes with reproduced rows "
                       "in all four arms); outcome model: logit of y on variant + M + log N + regime (+ variant x M in "
                       "'outcome_arm_x_log1pU'). Counterfactual P(a, M(a')) = mean over rows i and empirical "
                       "mediator residuals j of expit(outcome model at variant a, M = predicted M_i(a') + resid_j) "
                       "(parametric g-formula); TE = P(a, M(a)) - P(H1, M(H1)), NDE = P(a, M(H1)) - P(H1, M(H1)), "
                       "NIE = P(a, M(a)) - P(a, M(H1)); log-odds versions logOR_*; logit-scale coefficients: c_total "
                       "(model without M), c_direct (with M), indirect_product = alpha x beta_M, indirect_difference = "
                       "c_total - c_direct (not equal under non-collapsibility). 95% percentile intervals from the "
                       "cluster bootstrap (networks resampled within regime); boot_share_le_0 = share of replicates "
                       "<= 0. Rows: analysis rows of H1, E5, E10, E20 (eqc excluded)",
        **medres}
    # conditional (within-network) logit
    Dn = v[(v.in_analysis == 1) & v.variant.isin(["H1"] + arms)].reset_index(drop=True)
    Xa = np.column_stack([(Dn.variant == a).to_numpy().astype(float) for a in arms])
    Mn = np.log1p(Dn.U.to_numpy().astype(float))[:, None]
    yn = Dn.hidden_or_not_excited.to_numpy().astype(float)
    b1, s1, l1, g1 = clogit(Xa, yn, Dn.net.to_numpy())
    b2, s2, l2, g2 = clogit(np.hstack([Xa, Mn]), yn, Dn.net.to_numpy())
    res["conditional_logit_within_network"] = {
        "description": "strata = networks; only networks with both outcomes among their analysis rows contribute; "
                       "regime and N are constant within a network",
        "n_informative_networks": g1,
        "without_U": {a: [float(b1[j]), float(s1[j])] for j, a in enumerate(arms)},
        "with_log1pU": {**{a: [float(b2[j]), float(s2[j])] for j, a in enumerate(arms)},
                        "log1pU": [float(b2[3]), float(s2[3])]},
        "LR_log1pU": lr(l2, l1, 1)}
    return res


# ============================================================================================== raw loaders (9-11)
def load_raw_M(limit=None):
    """Stage-M records exactly as paper_classes.py (gpuq-results, first file per regime/name, classes mlp/mlp2/res,
    Newton runaways |x| > 1e3 dropped). One row per (model, hidden attractor) with U_in at each scale of Q, the match
    metrics and the label, and p_B (share of the uniform census in Q ending on the model attractor)."""
    meta, rows, seen = {}, [], set()
    files = sorted((ROOT / "gpuq-results").rglob("M/*/*.json"))
    if limit:
        files = files[::max(1, len(files) // limit)]      # debug: evenly spaced subset
    for f in files:
        key = f.parent.name + "/" + f.stem
        if key in seen:
            continue
        seen.add(key)
        r = json.loads(f.read_text(encoding="utf-8"))
        if r["class"] not in CLS:
            continue
        reg = r["regime"]
        m = meta.setdefault(reg, json.loads((ROOT / "data" / f"{reg}.json").read_text(encoding="utf-8")))
        lo, hi = np.array(m["box_lo"], float), np.array(m["box_hi"], float)
        c0, h0 = (lo + hi) / 2, (hi - lo) / 2
        xs = [x["x"] for x in r["fixed_points"] if x["n_unstable"] > 0 and np.linalg.norm(x["x"]) <= R_MAX]
        U = len(xs)
        xs = np.array(xs, float).reshape(-1, 3)
        dev = np.max(np.abs(xs - c0) / h0, axis=1) if U else np.zeros(0)
        uin = {s: int(np.sum(dev <= s * (1 + 1e-12))) for s in Q_SCALES}
        conf = r.get("census_confusion", {}) or {}
        tot = sum(sum(v.values()) for v in conf.values())
        for y, mm in r["hidden_match"].items():
            a = mm.get("attractor")
            pB = (sum(v.get(a, 0) for v in conf.values()) / tot) if (a and tot) else None
            rows.append(dict(cls=r["class"], set=r["set"], reg=reg, H=int(r["H"]), seed=int(r["seed"]), att=y, U=U,
                             rep=bool(mm.get("reproduced")), label=mm.get("label") or "", dA=mm.get("delta_att"),
                             dH=mm.get("d_H"), pB=pB, **{f"Uin_{s:g}": uin[s] for s in Q_SCALES}))
    return pd.DataFrame(rows)


def load_raw_esn():
    """Stage N sweep rows and stage C ESN models through paper_esn_law (same selection and de-duplication)."""
    import paper_esn_law as L
    Nrec = L.load_dedup("N")
    srows = []
    for key, (r, path, _) in Nrec.items():
        default = (not L.SUFFIX_PAT.search(path.stem) and float(r.get("input_scale", 0.5)) == 0.5
                   and float(r.get("density", 0.05)) == 0.05 and float(r["rho"]) == 0.9 and r["set"] in ("HE1", "H1"))
        if not default:
            continue
        cube = r.get("census_cube", {}) or {}
        nc = sum(cube.values())
        for y, m in r["hidden_match"].items():
            a = m.get("attractor")
            srows.append(dict(model=key, regime=r["regime"], set=r["set"], N=int(r["N"]), U=int(r["U"]),
                              index_sum=int(r["index_sum"]), target=y, rep=bool(m.get("reproduced")),
                              label=m.get("label") or "", dA=m.get("delta_att"), dH=m.get("d_H"),
                              p_cube=(cube.get(a, 0) / nc) if (a and nc) else None))
    Crec = L.load_dedup("C")
    crows = []
    for key, (r, path, _) in Crec.items():
        if r.get("model") != "esn" or r["set"] not in ("H1", "HE1", "H0"):
            continue
        m = L.meta(r["regime"])
        lo, hi = np.array(m["box_lo"], float), np.array(m["box_hi"], float)
        c0, h0 = (lo + hi) / 2, (hi - lo) / 2
        fps = r.get("fixed_points", [])
        U = sum(1 for x in fps if x.get("n_unstable", 0) > 0)
        xs = np.array([x["x"] for x in fps if x.get("n_unstable", 0) > 0 and "x" in x], float).reshape(-1, 3)
        dev = np.max(np.abs(xs - c0) / h0, axis=1) if len(xs) else np.zeros(0)
        best = L.best_match_C(r, m["hidden"])
        lab = r["hiddenness"][best[1]]["label"] if best else ""
        crows.append(dict(model=key, regime=r["regime"], set=r["set"], N=int(r.get("N", 300)), rho=float(r["rho"]),
                          eta=float(r["eta"]), seed=int(r["seed"]), U=U, K=len(fps), rep=best is not None, label=lab,
                          **{f"Uin_{s:g}": int(np.sum(dev <= s * (1 + 1e-12))) for s in Q_SCALES}))
    return pd.DataFrame(srows), pd.DataFrame(crows)


# ============================================================================================== item 9
def fit_bins(y, u, reg, cl, cov, cov_names, bins, xname):
    if len(np.asarray(y)) < 10:
        return None
    e = NS(y=np.asarray(y, float), u=np.asarray(u, int), reg=np.asarray(reg).astype(str), cl=codes(cl),
           cov=np.asarray(cov, float), cov_names=cov_names, xname=xname)
    X, _ = X_of(e, np.log1p(e.u.astype(float))[:, None])
    f = glm(X, e.y)
    b = bin_of(e.u, bins)
    return {"n_rows": int(len(e.y)), "n_y1": int(e.y.sum()), "slope_log1p": coef_entry(f, 0, e.cl),
            "mean_x": float(e.u.mean()), "share_x0": float(np.mean(e.u == 0)),
            "bins": [{"bin": blabel(*bins[j]), "n": int((b == j).sum()), "k": int(e.y[b == j].sum()),
                      "share": float(e.y[b == j].mean()) if (b == j).any() else None,
                      "wilson95": wilson(int(e.y[b == j].sum()), int((b == j).sum()))} for j in range(len(bins))]}


def classes_cov(d):
    return np.c_[(d.cls == "mlp2").astype(float), (d.cls == "res").astype(float), (d.set == "HE1").astype(float),
                 (d.set == "E1").astype(float), np.log(d.H.astype(float))], \
        ["class_mlp2", "class_res", "set_HE1", "set_E1", "log_width"]


def item9(rawM, rawC, E7):
    res = {"description": "U_in recomputed with the region Q scaled about its centre by s (inside if |x_k - c_k| <= s "
                          "h_k in every coordinate, h = half-edges); s = 1 is the paper's U_in. Classes: stage-M raw "
                          "records (runaways |x| > 1e3 dropped), reproduced pairs, six regimes, covariates as the "
                          "classes fit; ESN full protocol: stage-C raw records, reproducing models, six regimes, "
                          "covariates as the esn_full fit", "scales": list(Q_SCALES)}
    d = rawM[rawM.rep & rawM.reg.isin(MAIN_REGIMES)].reset_index(drop=True)
    # check against classes_rows.csv
    k = E7["classes"].df
    kk = k[k.reg.isin(MAIN_REGIMES)][["cls", "set", "reg", "H", "seed", "att", "Un", "ne"]]
    mm = d.merge(kk, on=["cls", "set", "reg", "H", "seed", "att"], how="left")
    res["check_vs_classes_rows"] = {"n_rows_raw": int(len(d)), "n_rows_table": int(len(kk)),
                                    "U_in_s1_equal": int((mm["Uin_1"] == mm["Un"]).sum()),
                                    "label_not_excited_equal": int(((mm.label == "not_excited") == mm["ne"].astype(bool)).sum())}
    cov, names = classes_cov(d)
    cl = d.cls + "|" + d.set + "|" + d.reg + "|" + d.H.astype(str) + "|" + d.seed.astype(str)
    yk = (d.label == "not_excited").to_numpy()
    res["classes"] = {f"{s:g}": fit_bins(yk, d[f"Uin_{s:g}"], d.reg, cl, cov, names, CLS_BINS, "U_in") for s in Q_SCALES}
    res["classes_by_class"] = {
        f"{s:g}": {c: fit_bins(yk[(d.cls == c).to_numpy()], d[f"Uin_{s:g}"][d.cls == c], d.reg[d.cls == c],
                               cl[d.cls == c], cov[(d.cls == c).to_numpy()][:, 2:], names[2:], CLS_BINS, "U_in")
                   for c in CLS} for s in Q_SCALES}
    c = rawC[rawC.rep & rawC.regime.isin(MAIN_REGIMES)].reset_index(drop=True)
    cc = E7["esn_full_U_in"].df
    m2 = c.merge(cc[["model", "U_in", "label"]], on="model", how="left", suffixes=("", "_tab"))
    res["check_vs_esn_C_rows"] = {"n_rows_raw": int(len(c)), "n_rows_table": int(cc.regime.isin(MAIN_REGIMES).sum()),
                                  "U_in_s1_equal": int((m2["Uin_1"] == m2["U_in"]).sum()),
                                  "label_equal": int((m2.label == m2.label_tab).sum())}
    covC = np.c_[(c.set == "H1").astype(float), (c.set == "H0").astype(float), np.log(c.N),
                 np.isclose(c.rho, 0.6).astype(float), np.isclose(c.rho, 1.2).astype(float)]
    yc = (c.label == "hidden_stat").to_numpy()
    res["esn_full"] = {f"{s:g}": fit_bins(yc, c[f"Uin_{s:g}"], c.regime, c.model, covC,
                                          ["set_H1", "set_H0", "log_N", "rho_0.6", "rho_1.2"], ESN_BINS, "U_in")
                       for s in Q_SCALES}
    return res


# ============================================================================================== item 10
def item10(rawN, E7):
    res = {"description": "reproduced = delta_att < t and d_H < t (the paper uses t = 0.2); the hiddenness label is "
                          "stored for every pair whose model attractor of the right kind was found (>= 8 of 64 starts "
                          "on the true trajectory), so the split can be redone without new runs. Classes: "
                          "classes_rows.csv (columns dA, dH, label); ESN size sweep: stage-N raw records (same "
                          "selection as paper_esn_law.py), index sum 1; six regimes",
           "metrics_stored": True}
    k = pd.read_csv(NUM / "classes_rows.csv")
    k = k[k.reg.isin(MAIN_REGIMES)].reset_index(drop=True)
    cov, names = classes_cov(k)
    cl = k.cls + "|" + k.set + "|" + k.reg + "|" + k.H.astype(str) + "|" + k.seed.astype(str)
    out = {}
    for t in THRESHOLDS:
        m = (k.dA < t) & (k.dH < t)
        mv = m.to_numpy()
        r = fit_bins((k.label[m] == "not_excited").to_numpy(), k.Un[m], k.reg[m], cl[m], cov[mv], names, CLS_BINS, "U_in")
        r["n_pairs_reproduced"] = int(mv.sum())
        r["by_class"] = {c: int((mv & (k.cls == c).to_numpy()).sum()) for c in CLS}
        out[f"{t:g}"] = r
    res["classes"] = out
    s = rawN[rawN.regime.isin(MAIN_REGIMES) & (rawN.index_sum == 1)].reset_index(drop=True)
    out = {}
    for t in THRESHOLDS:
        m = (s.dA.astype(float) < t) & (s.dH.astype(float) < t)
        mv = m.fillna(False).to_numpy().astype(bool)
        d = s[mv]
        covS = np.c_[(d.set == "H1").astype(float), np.log(d.N)]
        r = fit_bins((d.label == "not_excited").to_numpy(), d.U, d.regime, d.model, covS, ["set_H1", "log_N"], ESN_BINS, "U")
        r["n_pairs_reproduced"] = int(mv.sum())
        out[f"{t:g}"] = r
    res["esn_sweep"] = out
    e = E7["esn_sweep"]
    m6 = np.isin(e.reg, MAIN_REGIMES)
    res["check_t0.2_vs_esn_rows"] = {"rows_table": int(m6.sum()), "not_excited_table": int(e.y[m6].sum()),
                                     "rows_raw": out["0.2"]["n_rows"], "not_excited_raw": out["0.2"]["n_y1"]}
    return res


# ============================================================================================== item 11
def item11(rawC, E7):
    c = rawC[rawC.regime.isin(MAIN_REGIMES)].reset_index(drop=True)
    res = {"description": "stage C ESN models (sets H1, HE1, H0), six regimes. U by spectral radius rho for all models "
                          "(reproducing or not) and for reproducing models; stratified comparison within (regime, set, "
                          "N, eta) strata and paired by seed (the same reservoir draw rescaled to another rho); "
                          "full-protocol labels by U bin for the reproducing models"}
    by = {}
    for rho in sorted(c.rho.unique()):
        m = c.rho == rho
        r_ = c[m & c.rep]
        by[f"{rho:g}"] = {"n_models": int(m.sum()), "U_mean": float(c.U[m].mean()), "U_median": float(c.U[m].median()),
                          "U_q25_q75": pq(c.U[m], [0.25, 0.75]), "n_reproducing": int(len(r_)),
                          "U_mean_reproducing": float(r_.U.mean()) if len(r_) else None,
                          "hidden_stat_among_reproducing": [int((r_.label == "hidden_stat").sum()), int(len(r_))]}
    res["by_rho"] = by
    res["by_regime_rho"] = {g: {f"{rho:g}": {"n": int(((c.regime == g) & (c.rho == rho)).sum()),
                                            "U_mean": float(c.U[(c.regime == g) & (c.rho == rho)].mean())
                                            if ((c.regime == g) & (c.rho == rho)).any() else None}
                                for rho in sorted(c.rho.unique())} for g in sorted(MAIN_REGIMES)}
    # stratified regression log(1+U) ~ rho dummies + stratum fixed effects (HC1)
    st = c.regime + "|" + c.set + "|N" + c.N.astype(str) + "|eta" + c.eta.astype(str)
    S = pd.get_dummies(st).to_numpy().astype(float)
    Xr = np.c_[np.isclose(c.rho, 0.6), np.isclose(c.rho, 1.2)].astype(float)
    X = np.hstack([Xr, S])
    yv = np.log1p(c.U.to_numpy().astype(float))
    bb = np.linalg.lstsq(X, yv, rcond=None)[0]
    e_ = yv - X @ bb
    XtXi = np.linalg.pinv(X.T @ X)
    n, p = X.shape
    V = XtXi @ (X.T * e_ ** 2) @ X @ XtXi * n / (n - np.linalg.matrix_rank(X))
    res["stratified_ols_log1pU"] = {"rho_0.6_vs_0.9": [float(bb[0]), float(math.sqrt(V[0, 0]))],
                                    "rho_1.2_vs_0.9": [float(bb[1]), float(math.sqrt(V[1, 1]))],
                                    "n_models": int(n), "n_strata": int(S.shape[1]),
                                    "note": "OLS with stratum fixed effects (regime x set x N x eta), HC1 SEs; "
                                            "coefficients are differences in mean log(1+U)"}
    # paired by seed within stratum
    c["stratum"] = st
    pv = c.pivot_table(index=["stratum", "seed"], columns="rho", values="U", aggfunc="first")
    pairs = {}
    for a, b in ((0.6, 0.9), (1.2, 0.9), (1.2, 0.6)):
        if a in pv.columns and b in pv.columns:
            d_ = (pv[a] - pv[b]).dropna()
            pairs[f"{a:g}_minus_{b:g}"] = {"n_pairs": int(len(d_)), "mean_dU": float(d_.mean()),
                                           "median_dU": float(d_.median()),
                                           "down_same_up": [int((d_ < 0).sum()), int((d_ == 0).sum()), int((d_ > 0).sum())],
                                           "wilcoxon_p": float(stats.wilcoxon(d_).pvalue) if np.any(d_ != 0) else None}
    res["paired_by_seed"] = pairs
    # labels by U bin (reproducing models; table esn_C_rows.csv)
    e = E7["esn_full_U"]
    m6 = np.isin(e.reg, MAIN_REGIMES)
    b = bin_of(e.u, ESN_BINS)
    labs = ["hidden_stat", "self_excited", "excited_r", "undetermined"]
    res["labels_by_U_bin"] = [{"bin": blabel(*ESN_BINS[j]), "n": int((m6 & (b == j)).sum()),
                               **{l_: int((m6 & (b == j) & (e.label == l_)).sum()) for l_ in labs}}
                              for j in range(len(ESN_BINS))]
    res["labels_total"] = {l_: int((m6 & (e.label == l_)).sum()) for l_ in labs}
    res["labels_by_U_bin_le1"] = [{"bin": blabel(*ESN_BINS[j]), "n": int((~m6 & (b == j)).sum()),
                                   **{l_: int((~m6 & (b == j) & (e.label == l_)).sum()) for l_ in labs}}
                                  for j in range(len(ESN_BINS))]
    # survival by rho at fixed U (fit of the main model; rho dummies)
    E6 = sub(e, m6)
    X, regs = X_of(E6, np.log1p(E6.u.astype(float))[:, None])
    f = glm(X, E6.y)
    Vc, G = cluster_cov(f, E6.cl)
    j6, j12 = 1 + E6.cov_names.index("rho_0.6"), 1 + E6.cov_names.index("rho_1.2")
    res["survival_rho_effect_given_U"] = {
        "rho_0.6": [float(f.beta[j6]), float(math.sqrt(f.cov[j6, j6])), float(math.sqrt(Vc[j6, j6]))],
        "rho_1.2": [float(f.beta[j12]), float(math.sqrt(f.cov[j12, j12])), float(math.sqrt(Vc[j12, j12]))],
        "note": "[coef, se_model, se_robust] of the rho dummies (vs 0.9) in the esn_full logit with log(1+U)"}
    return res


# ============================================================================================== figures
W1, W2 = 3.37, 6.69
CLASSC = {"ESN": ("#0072B2", "o", "ESN, size sweep"), "ESNF": ("#CC79A7", "v", "ESN, full protocol"),
          "mlp": ("#E69F00", "s", "one-layer perceptron"), "mlp2": ("#009E73", "^", "two-layer perceptron"),
          "res": ("#D55E00", "D", "residual network")}
REGNAME = {"se3": "SE3", "se11": "SE11", "le1": "LE1 (weakly hidden)", "wangchen2": "WC, a=0.006",
           "wangchen2_a0.03": "WC, a=0.03", "chua_I": "Chua I", "chua_II": "Chua II"}


def fig_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from hsearch import viz
    viz.setup()
    viz.setup_en()
    plt.rcParams.update({"axes.grid": True, "axes.axisbelow": True, "grid.alpha": 0.5, "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white"})
    return plt


def save(plt, fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", dpi=400)
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", FIG / f"{name}.pdf")


def _curve(ax, rows, x, color, marker, label, ls="-", nmin=5, key_k="k", key_n="n", ci=None):
    ok = [i for i, r in enumerate(rows) if r[key_n] >= nmin]
    if not ok:
        return
    sh = np.array([rows[i][key_k] / rows[i][key_n] for i in ok])
    if ci is None:
        lo, hi = np.array([wilson(rows[i][key_k], rows[i][key_n]) for i in ok], float).T
    else:
        lo, hi = np.array([ci[i] for i in ok], float).T
    ax.errorbar(np.asarray(x)[ok], sh, yerr=[np.clip(sh - lo, 0, None), np.clip(hi - sh, 0, None)], color=color,
                marker=marker, ms=3.2, lw=1.0, ls=ls, capsize=1.5, elinewidth=0.7, label=label)


def figures(R):
    plt = fig_setup()
    labs6 = ["0", "1", "2", "3-4", "5-8", r"$\geq$9"]
    x6 = np.arange(6)
    # ---- item 1: per-regime survival
    if "regime_survival" in R:
        rs = R["regime_survival"]
        fig, axs = plt.subplots(2, 4, figsize=(W2, 3.9), sharex=True, sharey=True)
        order = ["se3", "se11", "wangchen2", "wangchen2_a0.03", "chua_I", "chua_II", "le1"]
        lines = [("esn_sweep_bins6", "ESN", -0.16, "-"), ("esn_full_U_bins6", "ESNF", -0.08, "--"),
                 ("classes_mlp", "mlp", 0.0, "-"), ("classes_mlp2", "mlp2", 0.08, "-"), ("classes_res", "res", 0.16, "-")]
        for ax, g in zip(axs.flat, order):
            for key, c_, off, ls in lines:
                if g in rs[key]:
                    col, mk, lab = CLASSC[c_]
                    _curve(ax, rs[key][g]["bins"], x6 + off, col, mk, lab, ls=ls)
            ax.set_title(REGNAME[g], fontsize=8)
            ax.set_ylim(-0.04, 1.06)
            ax.set_xticks(x6)
            ax.set_xticklabels(labs6, fontsize=7)
        axs[1, 3].axis("off")
        h, l_ = axs[0, 0].get_legend_handles_labels()
        hl = {}
        for ax in axs.flat[:7]:
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                hl.setdefault(ll, hh)
        axs[1, 3].legend(list(hl.values()), list(hl.keys()), loc="center", fontsize=7.5, frameon=False,
                         title="cells with n $\\geq$ 5;\nWilson 95% intervals", title_fontsize=7)
        for ax in axs[:, 0]:
            ax.set_ylabel("share still hidden")
        for ax in axs[1, :3]:
            ax.set_xlabel(r"$U$ (ESN) or $U_{\mathrm{in}}$")
        axs[0, 3].set_xlabel(r"$U$ (ESN) or $U_{\mathrm{in}}$")
        axs[0, 3].xaxis.set_tick_params(labelbottom=True)
        fig.tight_layout()
        save(plt, fig, "sm_regime_survival")
    # ---- item 3 (+6): standardized pooled curves
    if "standardized" in R:
        S = R["standardized"]["ensembles"]
        fig, axs = plt.subplots(1, 3, figsize=(W2, 2.5), sharey=True)
        panels = [("esn_sweep", "bins7", "ESN size sweep (light)", "$U$"),
                  ("esn_full_U", "bins7", "ESN full protocol", "$U$"),
                  ("classes", "bins6", "perceptrons and residual nets", r"$U_{\mathrm{in}}$")]
        for i, (ax, (nm, bk, ttl, xl)) in enumerate(zip(axs, panels)):
            rows = S[nm]["all"][bk]
            x = np.arange(len(rows))
            for kind, col, mk, off, ls, lab in (("crude", "#0072B2", "o", -0.12, "-", "crude"),
                                                ("direct", "#D55E00", "s", 0.0, "--", "direct std."),
                                                ("model", "#009E73", "^", 0.12, ":", "model std.")):
                ok = [j for j, r in enumerate(rows) if r[kind] is not None and r["n"] >= 5
                      and r[kind + "_ci95_boot"][0] is not None]
                v_ = np.array([rows[j][kind] for j in ok])
                lo = np.array([rows[j][kind + "_ci95_boot"][0] for j in ok])
                hi = np.array([rows[j][kind + "_ci95_boot"][1] for j in ok])
                ax.errorbar(x[ok] + off, v_, yerr=[np.clip(v_ - lo, 0, None), np.clip(hi - v_, 0, None)], color=col,
                            marker=mk, ms=3.2, lw=1.0, ls=ls, capsize=1.5, elinewidth=0.7, label=lab)
            ok = [j for j, r in enumerate(rows) if r["n"] >= 5 and r["excited_r_as_hidden"] is not None]
            ax.plot(x[ok], [rows[j]["excited_r_as_hidden"] for j in ok], color="#555555", marker="x", ms=3.5, lw=0.0,
                    label="excited_r as hidden")
            ax.set_xticks(x)
            ax.set_xticklabels([r["bin"].replace(">=", r"$\geq$") for r in rows], fontsize=7)
            ax.set_xlabel(xl)
            ax.set_title(ttl, fontsize=8)
            ax.set_ylim(-0.04, 1.06)
            if i == 0:
                ax.set_ylabel("share still hidden")
                ax.legend(loc="upper right", fontsize=7, title="six regimes; cluster\nbootstrap 95%",
                          title_fontsize=6.5)
        fig.tight_layout()
        save(plt, fig, "sm_standardized")
    # ---- item 4: link calibration
    if "links" in R:
        L_ = R["links"]
        fig, axs = plt.subplots(1, 3, figsize=(W2, 2.5), sharey=True)
        cols = {"logit_log1pU": ("#0072B2", "-"), "logit_U": ("#56B4E9", "--"), "log_U": ("#D55E00", "-."),
                "cloglog_exc_U": ("#CC79A7", ":"), "cloglog_exc_log1pU": ("#009E73", "-")}
        for i, (ax, (nm, ttl, xl)) in enumerate(zip(axs, (("esn_sweep", "ESN size sweep", "$U$"),
                                                          ("esn_full_U", "ESN full protocol", "$U$"),
                                                          ("classes", "other classes", r"$U_{\mathrm{in}}$")))):
            mods = L_[nm]["all_rows"]["models"]
            cal = mods["logit_log1pU"]["binned_calibration"]
            x = np.arange(len(cal))
            rows = [{"k": c_["observed"], "n": c_["n"]} for c_ in cal]
            _curve(ax, rows, x, "black", "o", "observed", ls="none")
            for mname, (col, ls) in cols.items():
                if "binned_calibration" not in mods.get(mname, {}):
                    continue
                cc = mods[mname]["binned_calibration"]
                ok = [j for j, c_ in enumerate(cc) if c_["n"] >= 5]
                pretty = {"logit_log1pU": r"logit, $\log(1+U)$", "logit_U": r"logit, $U$", "log_U": r"log link, $U$",
                          "cloglog_exc_U": r"cloglog, $U$", "cloglog_exc_log1pU": r"cloglog, $\log(1+U)$"}[mname]
                ax.plot(x[ok], [cc[j]["expected"] / cc[j]["n"] for j in ok], color=col, ls=ls, lw=1.1,
                        label=f"{pretty} ($\\Delta$AIC {mods[mname]['delta_AIC']:.0f})")
            ax.set_xticks(x)
            ax.set_xticklabels([c_["bin"].replace(">=", r"$\geq$") for c_ in cal], fontsize=7)
            ax.set_xlabel(xl)
            ax.set_title(ttl, fontsize=8)
            ax.set_ylim(-0.04, 1.06)
            ax.legend(loc="upper right", fontsize=5.5)
            if i == 0:
                ax.set_ylabel("share still hidden")
        fig.tight_layout()
        save(plt, fig, "sm_links")
    # ---- item 5: implied vs observed survival
    if "equilibrium_model" in R:
        iv = R["equilibrium_model"]["implied_vs_observed_survival_by_U_in"]["pooled"]["bins"]
        fig, ax = plt.subplots(figsize=(W1, 2.5))
        x = np.arange(len(iv))
        _curve(ax, [{"k": r["observed"], "n": r["n"]} for r in iv], x, "black", "o", "observed", ls="none")
        for key, col, ls, lab in (("random_intercept", "#0072B2", "-", "random intercept per model"),
                                  ("independent_logit", "#D55E00", "--", "independent equilibria (logit)"),
                                  ("constant_p_geometric", "#999999", ":", "constant $p$, $(1-p)^U$")):
            ok = [j for j, r in enumerate(iv) if r["n"] >= 5]
            ax.plot(x[ok], [iv[j][key]["expected_share"] for j in ok], color=col, ls=ls, lw=1.2, marker=".", label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels(labs6)
        ax.set_xlabel(r"unstable equilibria in the region $U_{\mathrm{in}}$")
        ax.set_ylabel("share still hidden")
        ax.set_ylim(-0.04, 1.06)
        ax.legend(loc="upper right", fontsize=7, title="perceptrons and residual nets, six regimes", title_fontsize=6.5)
        fig.tight_layout()
        save(plt, fig, "sm_eq_model")
    # ---- item 9: Q scale
    if "q_scale" in R:
        qs = R["q_scale"]["classes"]
        fig, ax = plt.subplots(figsize=(W1, 2.5))
        cols = {"0.5": "#56B4E9", "0.75": "#0072B2", "1": "black", "1.5": "#E69F00", "2": "#D55E00"}
        for i, (s, r) in enumerate(qs.items()):
            sl = r["slope_log1p"]
            _curve(ax, r["bins"], x6 - 0.16 + 0.08 * i, cols.get(s, "grey"), "o",
                   f"$s$ = {s}: slope {sl['coef']:.2f} $\\pm$ {sl['se_cluster_CR1']:.2f}")
        ax.set_xticks(x6)
        ax.set_xticklabels(labs6)
        ax.set_xlabel(r"$U_{\mathrm{in}}$ in the region $Q$ scaled by $s$")
        ax.set_ylabel("share still hidden")
        ax.set_ylim(-0.04, 1.06)
        ax.legend(loc="upper right", fontsize=6.5)
        fig.tight_layout()
        save(plt, fig, "sm_q_scale")


# ============================================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(NUM / "hetero.json"))
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--m-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261112)
    a = ap.parse_args()
    items = {int(x) for x in a.only.split(",") if x} or set(range(1, 12))
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    E = load_ensembles()
    E7 = load_ensembles(include_le1=True)
    R = {"meta": {"script": "paper2_esn/gpu/paper_hetero.py", "generated": time.strftime("%Y-%m-%d %H:%M"),
                  "pooling": POOL_NOTE, "bootstrap_replicates": a.boot, "seed": a.seed, "items": sorted(items),
                  "ensembles": {k: {"n_rows": int(len(v.y)), "n_clusters": int(len(set(v.cl))), "n_y1": int(v.y.sum()),
                                    "x": v.xname, "covariates": v.cov_names} for k, v in E.items()}}}
    T = {}
    rawM = rawN = rawC = None
    if items & {2, 9}:
        t = time.time()
        rawM = load_raw_M(a.m_limit or None)
        T["load_raw_M"] = time.time() - t
        print(f"raw M: {len(rawM)} rows in {T['load_raw_M']:.0f} s", flush=True)
    if items & {2, 9, 10, 11}:
        t = time.time()
        rawN, rawC = load_raw_esn()
        T["load_raw_esn"] = time.time() - t
    if 1 in items:
        R["regime_survival"] = item1(E, E7)
    if 2 in items:
        t = time.time()
        inter = {"note": "LR test of regime-specific slopes against a common slope (df = regimes - 1), cluster-robust "
                         "Wald test of equal slopes; per-regime slopes from the interaction model (ML with ridge 1e-6, "
                         "model-based and CR1 SEs) and its Firth-penalized version (finite under separation)"}
        for nm in ("esn_sweep", "esn_full_U", "esn_full_U_in", "classes"):
            inter[nm] = interaction(E[nm])
        for c in CLS:
            inter[f"classes_{c}"] = interaction(sub(E["classes"], E["classes"].cls == c))
        # heuristic (1 - p_B)^(beta U)
        s = rawN[rawN.rep & (rawN.index_sum == 1) & rawN.regime.isin(MAIN_REGIMES)]
        pb_s = {g: float(np.nanmedian(s.p_cube[s.regime == g].astype(float))) for g in REGS}
        m = rawM[rawM.rep & rawM.reg.isin(MAIN_REGIMES)]
        pb_k = {g: float(np.nanmedian(m.pB[m.reg == g].astype(float))) for g in REGS}
        inter["heuristic_check"] = {
            "note": "p_B = median basin share of the reproduced model attractor per regime: ESN = share of the uniform "
                    "census of the reservoir cube (census_cube), classes = share of the uniform census of Q "
                    "(census_confusion); the heuristic predicts log-link slopes b_g = beta log(1 - p_B,g)",
            "esn_sweep": heuristic_check(E["esn_sweep"], pb_s), "classes": heuristic_check(E["classes"], pb_k),
            "p_B_quantiles_by_regime": {"esn_sweep": {g: pq(s.p_cube[s.regime == g].astype(float), [0.1, 0.5, 0.9]) for g in REGS},
                                        "classes": {g: pq(m.pB[m.reg == g].astype(float), [0.1, 0.5, 0.9]) for g in REGS}}}
        # LE1 alone (reference)
        e7 = E7["esn_sweep"]
        el = sub(e7, e7.reg == "le1")
        X, _ = X_of(el, np.log1p(el.u.astype(float))[:, None])
        inter["le1_separate_esn_sweep_slope"] = coef_entry(glm(X, el.y), 0, el.cl)
        R["interaction"] = inter
        T["item2"] = time.time() - t
    if 3 in items:
        t = time.time()
        R["standardized"] = item3(E, a.boot, rng)
        R["main_fits_cluster"] = R["standardized"].pop("main_fits")
        T["item3"] = time.time() - t
        print(f"item 3 done in {T['item3']:.0f} s", flush=True)
    if 4 in items:
        t = time.time()
        R["links"] = item4(E)
        T["item4"] = time.time() - t
    if 5 in items:
        t = time.time()
        R["equilibrium_model"] = item5(E, rng)
        T["item5"] = time.time() - t
        print(f"item 5 done in {T['item5']:.0f} s", flush=True)
    if 6 in items:
        R["excited_r"] = item6(E)
    if 7 in items:
        t = time.time()
        R["penalty_calibration"] = item7(rng, a.boot)
        T["item7"] = time.time() - t
        print(f"item 7 done in {T['item7']:.0f} s", flush=True)
    if 8 in items:
        t = time.time()
        R["data_intervention"] = item8(rng, a.boot)
        T["item8"] = time.time() - t
        print(f"item 8 done in {T['item8']:.0f} s", flush=True)
    if 9 in items:
        R["q_scale"] = item9(rawM, rawC, E7)
    if 10 in items:
        R["match_thresholds"] = item10(rawN, E7)
    if 11 in items:
        R["esn_rho_labels"] = item11(rawC, E7)
    R["meta"]["seconds"] = {k: round(v, 1) for k, v in T.items()}
    R["meta"]["seconds"]["total"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(clean(R), indent=1, ensure_ascii=False), encoding="utf-8")
    print("written", a.out, f"({R['meta']['seconds']['total']} s)")
    if not a.no_fig:
        figures(clean(R))


if __name__ == "__main__":
    main()

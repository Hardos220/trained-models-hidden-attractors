"""Certified enumeration of all fixed points of the 3-D models of stage M (mlp, mlp2, res) by the interval Krawczyk
method with bisection (check of the completeness of U and U_in).

Model (u = (x - mean) / std):  mlp  F(u) = W2 tanh(W1 u + b1) + b2
                               mlp2 F(u) = W3 tanh(W2 tanh(W1 u + b1) + b2) + b3
                               res  F(u) = u + W2 tanh(W1 u + b1) + b2
Zeros of g(u) = u - F(u) are enumerated in a box D:
  mlp, mlp2: D = the image box of F, b_last +- sum_k |W_last[:, k]| (slightly inflated); since |tanh| < 1 it contains
             every fixed point, so the enumeration is complete in R^3;
  res:       D = the region Q of the true system in u coordinates, inflated by 1% of its width on each side (the
             residual network is unbounded; completeness is claimed only in Q), and optionally the Newton start box of
             stage M, c +- (hi - lo) (twice the width of Q).
Algorithm (all boxes of a generation processed as one numpy batch): a box X is discarded if X and F(X) are disjoint
(bounded classes; X is replaced by X n F(X)) or 0 is not in g(X) (res); then the Krawczyk operator
K(X) = m - Y g(m) + (I - Y g'(X)) (X - m), Y = inverse of the midpoint of g'(X), is evaluated: K n X empty -> no zero;
K in int X -> exactly one zero in X (recorded; refined by point Newton inside K); otherwise X <- X n K, and if that
does not shrink the box by 30% it is bisected at 0.5065 of its widest edge. Boxes narrower than wmin that are still
undecided get an epsilon-inflation test around the point-Newton root (a box around it that contains the undecided box);
if that fails they are reported as undecided (count). With no undecided boxes the set of zeros in D is complete.
Interval arithmetic: float64 numpy with round-to-nearest, made outward by inflating every computed bound by a priori
rounding-error bounds (gamma_k = k u / (1 - k u), u = 2^-53, for sums of k terms; 4 ulp for tanh, 8 ulp for sech^2,
plus 1e-300 absolute). This is the documented epsilon-inflation model, not directed rounding.
g'(X): products of interval sech^2 factors with the point weight matrices (W2 diag(d1) W1, W3 diag(d2) W2 diag(d1) W1).

    python gpu/krawczyk_m.py --src mw_weights --regimes se3 [--res-start-box] [--workers 4]
Input: <src>/<regime>/<class>_<set>_H<width>_s<seed>.npz (stage_mw.py); output: $GPUQ_OUT/KR/<regime>/<same>.json
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
UR = 2.0 ** -53
EPS = 2.0 ** -52
TINY = 1e-300
SPLIT = 0.5065
I3 = np.eye(3)


def gam(k):
    return k * UR / (1 - k * UR)


def mid_rad(lo, hi):
    c = 0.5 * (lo + hi)
    r = np.maximum(hi - c, c - lo) * (1 + 4 * UR) + TINY
    return c, r


def affine(lo, hi, W, Wa, b):
    """Enclosure of {W x + b : x in [lo, hi]} (rows = boxes)."""
    c, r = mid_rad(lo, hi)
    zc = c @ W.T + b
    zr = r @ Wa.T
    err = gam(W.shape[1] + 4) * (np.abs(c) @ Wa.T + np.abs(b) + zr) + TINY
    return zc - zr - err, zc + zr + err


def tanh_iv(lo, hi):
    tl, th = np.tanh(lo), np.tanh(hi)
    return np.maximum(tl - 4 * EPS * np.abs(tl) - TINY, -1.0), np.minimum(th + 4 * EPS * np.abs(th) + TINY, 1.0)


def sech2_iv(lo, hi):
    straddle = (lo <= 0) & (hi >= 0)
    amin = np.where(straddle, 0.0, np.minimum(np.abs(lo), np.abs(hi)))
    amax = np.maximum(np.abs(lo), np.abs(hi))
    with np.errstate(over="ignore"):
        dmax = 1.0 / np.cosh(amin) ** 2
        dmin = 1.0 / np.cosh(amax) ** 2
    return np.maximum(dmin * (1 - 8 * EPS), 0.0), np.minimum(dmax * (1 + 8 * EPS) + TINY, 1.0)


class Model:
    def __init__(self, cls, w):
        self.cls = cls
        f = lambda k: np.asarray(w[k], np.float64)  # noqa: E731
        self.W1, self.b1 = f("W1"), f("b1")
        self.H = H = self.W1.shape[0]
        self.W1a, self.W1p, self.W1n = np.abs(self.W1), np.maximum(self.W1, 0), np.maximum(-self.W1, 0)
        if cls == "mlp2":
            self.W2, self.b2, self.W3, self.b3 = f("W2"), f("b2"), f("W3"), f("b3")
            self.W2a = np.abs(self.W2)
            self.Wo, self.bo = self.W3, self.b3
            P = (self.W3[:, :, None] * self.W2[None, :, :]).transpose(1, 0, 2).reshape(H, 3 * H)  # [l, (i, k)]
            self.Pp, self.Pn, self.Pa = np.maximum(P, 0), np.maximum(-P, 0), np.abs(P)
        else:
            self.W2, self.b2 = f("W2"), f("b2")
            self.Wo, self.bo = self.W2, self.b2
            C = (self.W2.T[:, :, None] * self.W1[:, None, :]).reshape(H, 9)                        # [k, (i, j)]
            self.Cp, self.Cn, self.Ca = np.maximum(C, 0), np.maximum(-C, 0), np.abs(C)
        self.Woa = np.abs(self.Wo)
        self.bounded = cls in ("mlp", "mlp2")

    # ---------------------------------------------------------------- point evaluation
    def F(self, U):
        h = np.tanh(U @ self.W1.T + self.b1)
        if self.cls == "mlp2":
            h = np.tanh(h @ self.W2.T + self.b2)
            return h @ self.W3.T + self.b3
        out = h @ self.W2.T + self.b2
        return U + out if self.cls == "res" else out

    def JF(self, U):
        z1 = U @ self.W1.T + self.b1
        d1 = 1 - np.tanh(z1) ** 2
        if self.cls == "mlp2":
            d2 = 1 - np.tanh(np.tanh(z1) @ self.W2.T + self.b2) ** 2
            M = (self.W3[None] * d2[:, None, :]) @ self.W2[None]
            J = (M * d1[:, None, :]) @ self.W1[None]
        else:
            J = (self.W2[None] * d1[:, None, :]) @ self.W1[None]
        return J + I3[None] if self.cls == "res" else J

    def newton(self, U, iters=30):
        U = np.array(U, float)
        for _ in range(iters):
            g = U - self.F(U)
            M = I3[None] - self.JF(U)
            try:
                U = U - np.linalg.solve(M, g[:, :, None])[:, :, 0]
            except np.linalg.LinAlgError:
                break
        return U, np.linalg.norm(U - self.F(U), axis=1)

    def image_box(self):
        s = self.Woa.sum(1)
        return self.bo - s, self.bo + s

    # ---------------------------------------------------------------- interval evaluation
    def forward(self, lo, hi):
        z1 = affine(lo, hi, self.W1, self.W1a, self.b1)
        h1 = tanh_iv(*z1)
        if self.cls == "mlp2":
            z2 = affine(h1[0], h1[1], self.W2, self.W2a, self.b2)
            h2 = tanh_iv(*z2)
            return affine(h2[0], h2[1], self.W3, self.Woa, self.b3), (z1, z2)
        return affine(h1[0], h1[1], self.W2, self.Woa, self.b2), (z1,)

    def g_iv(self, lo, hi, o=None):
        """Enclosure of g(u) = u - F(u) on [lo, hi]."""
        if o is None:
            o, _ = self.forward(lo, hi)
        if self.cls == "res":
            return -o[1], -o[0]
        gl, gh = lo - o[1], hi - o[0]
        return gl - 2 * UR * np.abs(gl) - TINY, gh + 2 * UR * np.abs(gh) + TINY

    def jac_g(self, zs):
        """Enclosure of g'(u) = I - F'(u) (n, 3, 3) from the pre-activation enclosures."""
        n = zs[0][0].shape[0]
        H = self.H
        a1, e1 = sech2_iv(*zs[0])
        if self.cls == "mlp2":
            a2, e2 = sech2_iv(*zs[1])
            Ml, Mh = a2 @ self.Pp - e2 @ self.Pn, e2 @ self.Pp - a2 @ self.Pn
            errM = gam(2 * H + 4) * (e2 @ self.Pa) + TINY
            Ml, Mh = (Ml - errM).reshape(n, 3, H), (Mh + errM).reshape(n, 3, H)
            A1, E1 = a1[:, None, :], e1[:, None, :]
            pl, ph = np.minimum(Ml * A1, Ml * E1), np.maximum(Mh * A1, Mh * E1)
            pl, ph = pl - 2 * UR * np.abs(pl) - TINY, ph + 2 * UR * np.abs(ph) + TINY
            Jl, Jh = pl @ self.W1p - ph @ self.W1n, ph @ self.W1p - pl @ self.W1n
            err = gam(2 * H + 4) * (np.maximum(np.abs(pl), np.abs(ph)) @ self.W1a) + TINY
            Jl, Jh = Jl - err, Jh + err
        else:
            Jl, Jh = a1 @ self.Cp - e1 @ self.Cn, e1 @ self.Cp - a1 @ self.Cn
            err = gam(2 * H + 4) * (e1 @ self.Ca) + TINY
            Jl, Jh = (Jl - err).reshape(n, 3, 3), (Jh + err).reshape(n, 3, 3)
        if self.cls == "res":                       # g' = -W2 D1 W1
            return -Jh, -Jl
        gl, gh = I3[None] - Jh, I3[None] - Jl
        return gl - 2 * UR * np.abs(gl) - TINY, gh + 2 * UR * np.abs(gh) + TINY

    def krawczyk(self, lo, hi, zs):
        c, r = mid_rad(lo, hi)
        gcl, gch = self.g_iv(c, c)
        Jl, Jh = self.jac_g(zs)
        Jc = 0.5 * (Jl + Jh)
        Jr = np.maximum(Jh - Jc, Jc - Jl) * (1 + 4 * UR) + TINY
        Y = np.broadcast_to(I3, Jc.shape).copy()
        det = np.linalg.det(Jc)
        ok = np.isfinite(det) & (np.abs(det) > 1e-250) & np.isfinite(Jc).all((1, 2))
        if ok.any():
            with np.errstate(all="ignore"):
                Yi = np.linalg.inv(Jc[ok])
            good = np.isfinite(Yi).all((1, 2))
            idx = np.where(ok)[0][good]
            Y[idx] = Yi[good]
        Ya = np.abs(Y)
        gc, gr = mid_rad(gcl, gch)
        Ygc = np.einsum("nij,nj->ni", Y, gc)
        R = I3[None] - Y @ Jc
        rad = np.einsum("nij,nj->ni", Ya, gr) + np.einsum("nij,nj->ni", np.abs(R) + Ya @ Jr, r)
        err = gam(8) * (np.einsum("nij,nj->ni", Ya, np.abs(gc)) + np.einsum("nij,nj->ni", Ya @ np.abs(Jc) + 1.0, r)
                        + rad) + 2 * UR * np.abs(c) + TINY
        Kc = c - Ygc
        return Kc - rad - err, Kc + rad + err

    # ---------------------------------------------------------------- enumeration
    def _step(self, lo, hi, wmin_rel, ver, und, nxt_lo, nxt_hi):
        """One pass over a batch of boxes: exclusion, Krawczyk test, contraction or bisection."""
        o, zs = self.forward(lo, hi)
        if self.bounded:                        # fixed points in X lie in X n F(X)
            lo, hi = np.maximum(lo, o[0]), np.minimum(hi, o[1])
            keep = np.all(lo <= hi, axis=1)
            lo, hi = lo[keep], hi[keep]
            if not len(lo):
                return
            o, zs = self.forward(lo, hi)
        gl, gh = self.g_iv(lo, hi, o)
        keep = np.all((gl <= 0) & (gh >= 0), axis=1)
        lo, hi = lo[keep], hi[keep]
        zs = tuple((z[0][keep], z[1][keep]) for z in zs)
        if not len(lo):
            return
        Kl, Kh = self.krawczyk(lo, hi, zs)
        fin = np.isfinite(Kl).all(1) & np.isfinite(Kh).all(1)
        disj = fin & np.any((Kl > hi) | (Kh < lo), axis=1)
        inside = fin & np.all((Kl > lo) & (Kh < hi), axis=1)
        if inside.any():
            ver.append(np.concatenate([Kl[inside], Kh[inside]], 1))
        rest = ~disj & ~inside
        lo, hi, Kl, Kh, fin = lo[rest], hi[rest], Kl[rest], Kh[rest], fin[rest]
        if not len(lo):
            return
        nlo = np.where(fin[:, None], np.maximum(lo, Kl), lo)
        nhi = np.where(fin[:, None], np.minimum(hi, Kh), hi)
        w_old = np.max(hi - lo, axis=1)
        w_new = np.max(nhi - nlo, axis=1)
        shrunk = w_new < 0.7 * w_old
        if shrunk.any():
            nxt_lo.append(nlo[shrunk])
            nxt_hi.append(nhi[shrunk])
        lo, hi = nlo[~shrunk], nhi[~shrunk]
        if not len(lo):
            return
        w = hi - lo
        tiny = np.max(w, axis=1) < wmin_rel * (1 + np.max(np.abs(lo), axis=1))
        if tiny.any():
            und.append(np.concatenate([lo[tiny], hi[tiny]], 1))
        lo, hi, w = lo[~tiny], hi[~tiny], w[~tiny]
        if not len(lo):
            return
        ii = np.arange(len(lo))
        k = np.argmax(w, axis=1)
        cut = lo[ii, k] + SPLIT * w[ii, k]
        hi1 = hi.copy()
        hi1[ii, k] = cut
        lo2 = lo.copy()
        lo2[ii, k] = cut
        nxt_lo += [lo, lo2]
        nxt_hi += [hi1, hi]

    def enumerate(self, lo0, hi0, wmin_rel=1e-10, chunk=4096, max_boxes=30_000_000, t_max=3600.0):
        t0 = time.time()
        gen_lo, gen_hi = np.atleast_2d(lo0).astype(float), np.atleast_2d(hi0).astype(float)
        ver, und = [], []
        n_proc = n_gen = 0
        status = "complete"
        while len(gen_lo):
            n_gen += 1
            nxt_lo, nxt_hi = [], []
            for i0 in range(0, len(gen_lo), chunk):
                if n_proc > max_boxes or time.time() - t0 > t_max:
                    status = "budget_exceeded"
                    und.append(np.concatenate([gen_lo[i0:], gen_hi[i0:]], 1))
                    nxt_lo, nxt_hi = [], []
                    break
                lo, hi = gen_lo[i0:i0 + chunk], gen_hi[i0:i0 + chunk]
                n_proc += len(lo)
                self._step(lo, hi, wmin_rel, ver, und, nxt_lo, nxt_hi)
            if status != "complete":
                break
            gen_lo = np.concatenate(nxt_lo) if nxt_lo else np.zeros((0, 3))
            gen_hi = np.concatenate(nxt_hi) if nxt_hi else np.zeros((0, 3))
        n_iter = n_gen
        ver = np.concatenate(ver) if ver else np.zeros((0, 6))
        und = np.concatenate(und) if und else np.zeros((0, 6))
        # roots of the verified boxes: point Newton inside K
        roots, boxes = [], []
        if len(ver):
            Kl, Kh = ver[:, :3], ver[:, 3:]
            X, res = self.newton(0.5 * (Kl + Kh))
            okb = np.all((X >= Kl) & (X <= Kh), axis=1)
            X = np.where(okb[:, None], X, 0.5 * (Kl + Kh))
            roots += list(X)
            boxes += list(ver)
        # epsilon inflation for undecided boxes
        n_und_in = len(und)
        left = []
        if len(und) and status == "complete":
            c0 = 0.5 * (und[:, :3] + und[:, 3:])
            X, res = self.newton(c0)
            for i in range(len(und)):
                x = X[i]
                if not (np.isfinite(x).all() and res[i] < 1e-12 * (1 + np.linalg.norm(x))):
                    left.append(und[i])
                    continue
                rad = np.maximum(np.abs(und[i, :3] - x), np.abs(und[i, 3:] - x)) * 2 + 1e-12 * (1 + np.abs(x))
                done = False
                for _ in range(8):
                    blo, bhi = (x - rad)[None], (x + rad)[None]
                    _, zs = self.forward(blo, bhi)
                    Kl, Kh = self.krawczyk(blo, bhi, zs)
                    if np.all(Kl > blo) and np.all(Kh < bhi):
                        roots.append(x)
                        boxes.append(np.concatenate([Kl[0], Kh[0]]))
                        done = True
                        break
                    rad = rad * 4
                if not done:
                    left.append(und[i])
        # distinct roots
        R, B = [], []
        for x, bx in zip(roots, boxes):
            if all(np.linalg.norm(x - y) > 1e-9 * (1 + np.linalg.norm(y)) for y in R):
                R.append(x)
                B.append(bx)
        return {"roots": np.array(R).reshape(-1, 3), "boxes": np.array(B).reshape(-1, 6), "n_boxes": int(n_proc),
                "n_generations": int(n_iter), "n_verified_boxes": int(len(ver)), "n_undecided_before_inflation": n_und_in,
                "undecided": np.array(left).reshape(-1, 6), "status": status, "time_s": round(time.time() - t0, 2)}


def classify(model, R, sig, mu, qlo, qhi):
    out = []
    if not len(R):
        return out
    J = model.JF(R)
    res = np.linalg.norm(R - model.F(R), axis=1)
    for x, Jm, rr in zip(R, J, res):
        w = np.linalg.eigvals(Jm)
        aw = np.abs(w)
        xp = x * sig + mu
        out.append({"u": x.tolist(), "x": xp.tolist(), "rho": float(aw.max()), "n_unstable": int(np.sum(aw > 1 + 1e-9)),
                    "index": int(np.sign(np.linalg.det(I3 - Jm))), "margin": float(np.min(np.abs(aw - 1))),
                    "inside": bool(np.all(xp >= qlo) and np.all(xp <= qhi)), "residual": float(rr)})
    return out


def match(A, B, tol=1e-6):
    """Indices of the rows of A that have a partner in B (and vice versa)."""
    if not len(A) or not len(B):
        return np.zeros(len(A), bool), np.zeros(len(B), bool)
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    sa = d.min(1) < tol * (1 + np.linalg.norm(A, axis=1))
    sb = d.min(0) < tol * (1 + np.linalg.norm(B, axis=1))
    return sa, sb


def summ(eq):
    return {"K": len(eq), "U": int(sum(e["n_unstable"] > 0 for e in eq)), "K_in": int(sum(e["inside"] for e in eq)),
            "U_in": int(sum(e["n_unstable"] > 0 and e["inside"] for e in eq)), "index_sum": int(sum(e["index"] for e in eq))}


def run_one(args):
    path, out_path, meta, res_start = args
    t0 = time.time()
    z = np.load(path)
    stem = Path(path).stem
    cls = stem.split("_")[0]
    w = {k: z[k] for k in z.files if k[0] in "Wb"}
    mu, sig, E = z["mu"], z["sig"], z["E"]
    model = Model(cls, w)
    qlo, qhi = np.array(meta["box_lo"]), np.array(meta["box_hi"])
    ulo, uhi = (qlo - mu) / sig, (qhi - mu) / sig
    domains = []
    if model.bounded:
        lo, hi = model.image_box()
        pad = 1e-9 * (1 + np.abs(lo) + np.abs(hi))
        domains.append(("image_box", lo - pad, hi + pad))
    else:
        pad = 0.01 * (uhi - ulo)
        domains.append(("Q", ulo - pad, uhi + pad))
        if res_start:
            c, r = (ulo + uhi) / 2, (uhi - ulo)
            domains.append(("start_box", c - r, c + r))
    rec = {"file": str(path), "class": cls, "H": model.H, "name": stem, "domains": {}}
    newton_eq = classify(model, E[np.isfinite(E).all(1)], sig, mu, qlo, qhi) if len(E) else []
    for dn, lo, hi in domains:
        r = model.enumerate(lo, hi)
        cert = classify(model, r["roots"], sig, mu, qlo, qhi)
        new_in = [e for e in newton_eq if np.all(np.array(e["u"]) >= lo) and np.all(np.array(e["u"]) <= hi)]
        Ein = np.array([e["u"] for e in new_in]).reshape(-1, 3)
        Rc = np.array([e["u"] for e in cert]).reshape(-1, 3)
        mc, mn = match(Rc, Ein)
        rec["domains"][dn] = {
            "lo": lo.tolist(), "hi": hi.tolist(), "status": r["status"], "n_boxes": r["n_boxes"],
            "n_generations": r["n_generations"], "n_verified_boxes": r["n_verified_boxes"],
            "n_undecided_before_inflation": r["n_undecided_before_inflation"], "n_undecided": int(len(r["undecided"])),
            "undecided": r["undecided"][:20].tolist(), "time_s": r["time_s"],
            "certified": summ(cert), "newton": summ(new_in),
            "missed_by_newton": [e for e, m_ in zip(cert, mc) if not m_],
            "newton_not_certified": [e for e, m_ in zip(new_in, mn) if not m_],
            "certified_eq": cert}
    rec["newton_all"] = summ(newton_eq)
    rec["time_total_s"] = round(time.time() - t0, 2)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out_path) + ".tmp"
    Path(tmp).write_text(json.dumps(rec), encoding="utf-8")
    os.replace(tmp, out_path)
    d0 = rec["domains"][domains[0][0]]
    return (stem, d0["certified"], d0["newton"], len(d0["missed_by_newton"]), d0["n_undecided"], d0["status"],
            rec["time_total_s"])


def report_progress(done, total, note=""):
    p = os.environ.get("RMQ_PROGRESS_FILE")
    if p:
        try:
            Path(p).write_text(f"{done}/{total} {note}", encoding="utf-8")
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="mw_weights")
    ap.add_argument("--regimes", default="se3")
    ap.add_argument("--pattern", default="*.npz")
    ap.add_argument("--res-start-box", action="store_true")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("RMQ_CPU", "1")))
    a = ap.parse_args()
    out_root = Path(os.environ.get("GPUQ_OUT", "out")) / "KR"
    jobs = []
    for reg in a.regimes.split(","):
        meta = json.loads((ROOT / "data" / f"{reg}.json").read_text(encoding="utf-8"))
        for f in sorted((ROOT / a.src / reg).glob(a.pattern)):
            o = out_root / reg / f"{f.stem}.json"
            if not o.exists():
                jobs.append((str(f), str(o), {"box_lo": meta["box_lo"], "box_hi": meta["box_hi"]}, a.res_start_box))
    print(f"krawczyk: {len(jobs)} models, {a.workers} workers", flush=True)
    stop = os.environ.get("RMQ_STOP_FILE")
    pool = None
    if a.workers > 1:
        os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
        import multiprocessing as mp
        pool = mp.Pool(a.workers)
        it = pool.imap_unordered(run_one_safe, jobs)
    else:
        it = (run_one_safe(j) for j in jobs)
    for done, res in enumerate(it, 1):
        print(f"[{done}/{len(jobs)}] {res}", flush=True)
        report_progress(done, len(jobs), str(res[0]))
        if stop and os.path.exists(stop):
            print("stop file: finishing", flush=True)
            if pool:
                pool.terminate()
            sys.exit(0)
    if pool:
        pool.close()
        pool.join()


def run_one_safe(args):
    try:
        return run_one(args)
    except Exception:
        tb = traceback.format_exc()
        Path(str(args[1]) + ".failed.txt").parent.mkdir(parents=True, exist_ok=True)
        Path(str(args[1]) + ".failed.txt").write_text(tb, encoding="utf-8")
        return (Path(args[0]).stem, "FAILED", tb[-300:])


if __name__ == "__main__":
    main()

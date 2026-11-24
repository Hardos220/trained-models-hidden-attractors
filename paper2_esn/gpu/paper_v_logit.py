"""Stage V regression restricted to the data variants described in the paper (H1, E5, E10, E20 and the exact-equilibria
constraint eqc), first hidden attractor of each model, rows with the attractor reproduced and index sum 1.

Logistic regression of "not excited" on regime fixed effects, log N and variant dummies (vs H1), without and with
log(1 + U). Pooled over the six main regimes (paper_common.MAIN_REGIMES); LE1 separately (one regime: the regime dummy
is the intercept) and the seven-regime fit for comparison. Input: paper2_esn/results/numbers/v_rows.csv (written by
paper_data_effects.py); output: numbers/v_logit.json.

    python -X utf8 paper2_esn/gpu/paper_v_logit.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_common import POOL_NOTE, SEPARATE  # noqa: E402

NUM = Path(__file__).resolve().parents[1] / "results" / "numbers"
ARMS = ["H1", "eqc", "E5", "E10", "E20"]
FIRST = {"HA", "HA_a"}


def fit(X, y):
    f = lambda w: np.sum(np.logaddexp(0, X @ w) - y * (X @ w)) + 1e-6 * w @ w  # noqa: E731
    r = optimize.minimize(f, np.zeros(X.shape[1]), method="BFGS")
    p = 1 / (1 + np.exp(-X @ r.x))
    H = (X * (p * (1 - p))[:, None]).T @ X
    return r.x, np.sqrt(np.diag(np.linalg.pinv(H))), r.fun


def v_logit(R):
    regs = sorted({r["regime"] for r in R})
    arms = ARMS[1:]
    Xb = np.array([[float(r["regime"] == g) for g in regs] + [np.log(float(r["N"]))] + [float(r["variant"] == a) for a in arms]
                   for r in R])
    Xu = np.hstack([Xb, np.log1p([[float(r["U"])] for r in R])])
    y = np.array([float(r["hidden_or_not_excited"]) for r in R])
    w1, s1, f1 = fit(Xb, y)
    w2, s2, f2 = fit(Xu, y)
    j0 = len(regs) + 1
    lr = 2 * (f1 - f2)
    return {"n_rows": len(R), "regimes": regs, "variants": ARMS, "N_values": sorted({int(r["N"]) for r in R}),
            "covariates": "regime dummies (no intercept), log N, variant dummies vs H1; second fit adds log(1+U)",
            "variant_coef_without_U": {a: [round(float(w1[j0 + i]), 4), round(float(s1[j0 + i]), 4)] for i, a in enumerate(arms)},
            "variant_coef_with_U": {a: [round(float(w2[j0 + i]), 4), round(float(s2[j0 + i]), 4)] for i, a in enumerate(arms)},
            "log1pU": [round(float(w2[-1]), 4), round(float(s2[-1]), 4)], "LR_U": [round(lr, 1), float(stats.chi2.sf(lr, 1))],
            "log_N_with_U": [round(float(w2[len(regs)]), 4), round(float(s2[len(regs)]), 4)],
            "share_by_variant": {a: [int(sum(y[i] for i, r in enumerate(R) if r["variant"] == a)),
                                     sum(1 for r in R if r["variant"] == a)] for a in ARMS}}


def main():
    with open(NUM / "v_rows.csv", encoding="utf-8") as fh:
        R = [r for r in csv.DictReader(fh) if r["variant"] in ARMS and r["target"] in FIRST and r["in_analysis"] == "1"]
    out = {"pooling": POOL_NOTE, **v_logit([r for r in R if r["regime"] not in SEPARATE])}
    out["le1_separate"] = v_logit([r for r in R if r["regime"] == "le1"])
    out["with_le1_reference"] = v_logit(R)
    (NUM / "v_logit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

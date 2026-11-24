"""Survival law on the six main regimes and with LE1 added, whose attractor is hidden only in the weak sense of Jafari
and Sprott (2013): it is reached from a segment of the unstable half-line of equilibria (paper_le1_line.py).

Since 2026-11-11 the main scripts (paper_esn_law.py, paper_classes.py, paper_robustness.py) already pool over the six
regimes of paper_common.MAIN_REGIMES and report LE1 separately; this script keeps the side-by-side comparison in one
place. From the row tables in paper2_esn/results/numbers: echo state networks of the size sweep (esn_rows.csv, sample
n_sweep, reproduced, index sum 1; logit on log1p(U), set, log N, regime) and the model classes (classes_rows.csv,
reproduced; logit on log1p(U_in), class, set, log width, regime, as in paper_classes.py); excitation by copies of the
line and by spurious equilibria in LE1 (classes_eq_rows.csv). Written to numbers/le1_robust.json.

    python -X utf8 paper2_esn/gpu/paper_le1_robust.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_classes import fit  # noqa: E402
from paper_common import SEPARATE  # noqa: E402

NUM = Path(__file__).resolve().parents[1] / "results" / "numbers"


def read(name):
    with open(NUM / name, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def logit(D, cols, ucol):
    regs = sorted({x["reg"] for x in D})
    X = np.array([[c(x) for c in cols] + [float(x["reg"] == g) for g in regs] for x in D])
    y = np.array([x["y"] for x in D], float)
    w, se, f1 = fit(X, y)
    _, _, f0 = fit(X[:, 1:], y)
    lr = 2 * (f0 - f1)
    return {"n": len(D), "share_not_excited": round(float(y.mean()), 3), "coef_U": [round(float(w[0]), 3), round(float(se[0]), 3)],
            "LR_U": [round(float(lr), 1), float(stats.chi2.sf(lr, 1))]}


def main():
    out = {"note": "six_regimes = pooled statistics of the paper (LE1 excluded); with_le1 = seven regimes, comparison only"}
    E = [dict(reg=x["regime"], set=x["set"], N=float(x["N"]), U=float(x["U"]), y=x["not_excited"] == "1")
         for x in read("esn_rows.csv") if x["sample"] == "n_sweep" and x["reproduced"] == "1" and x["index_sum"] == "1"]
    cols_e = [lambda x: np.log1p(x["U"]), lambda x: float(x["set"] == "H1"), lambda x: np.log(x["N"])]
    out["esn_size_sweep"] = {"six_regimes": logit([x for x in E if x["reg"] not in SEPARATE], cols_e, 0),
                             "with_le1": logit(E, cols_e, 0), "le1_rows": sum(x["reg"] == "le1" for x in E)}
    C = [dict(reg=x["reg"], cls=x["cls"], set=x["set"], H=float(x["H"]), Un=float(x["Un"]), y=x["ne"] == "True")
         for x in read("classes_rows.csv") if x["rep"] == "True"]
    cols_c = [lambda x: np.log1p(x["Un"]), lambda x: float(x["cls"] == "mlp2"), lambda x: float(x["cls"] == "res"),
              lambda x: float(x["set"] == "HE1"), lambda x: float(x["set"] == "E1"), lambda x: np.log(x["H"])]
    out["model_classes"] = {"six_regimes": logit([x for x in C if x["reg"] not in SEPARATE], cols_c, 0),
                            "with_le1": logit(C, cols_c, 0), "le1_rows": sum(x["reg"] == "le1" for x in C)}
    # survival by U_in bin, pooled classes: six regimes, LE1 alone, and seven regimes
    bins = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 10 ** 6)]

    def surv(rows):
        return [{"bin": f"{a}-{b if b < 10 ** 6 else 'inf'}", "n": len(q),
                 "share": round(float(np.mean([x["y"] for x in q])), 3) if q else None}
                for a, b in bins for q in [[x for x in rows if a <= x["Un"] <= b]]]
    out["model_classes"]["survival_six_regimes"] = surv([x for x in C if x["reg"] not in SEPARATE])
    out["model_classes"]["survival_le1"] = surv([x for x in C if x["reg"] == "le1"])
    out["model_classes"]["survival_with_le1"] = surv(C)
    # unstable equilibria inside the region, as in the per-equilibrium shares of Sec. V.C (all copies of the line lie inside)
    Q = [x for x in read("classes_eq_rows.csv") if x["reg"] == "le1" and x["inside"] == "1"]
    out["le1_equilibria"] = {k: {"n": len(q), "excite": sum(x["excited"] == "1" for x in q)}
                             for k, q in (("copies_of_line", [x for x in Q if x["copy"] == "1"]),
                                          ("spurious", [x for x in Q if x["copy"] == "0"]))}
    (NUM / "le1_robust.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

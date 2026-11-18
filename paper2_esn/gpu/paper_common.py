"""Shared definitions of the paper scripts (paper_*.py): which regimes enter pooled statistics, and the copy threshold.

LE1 has a line of equilibria x = y = 0; its attractor is not reached from the stable half-line but is reached from a
segment of the unstable one (paper_le1_line.py), so it is self-excited in the strict sense of Leonov and Kuznetsov and
hidden only in the weak sense of Jafari and Sprott (2013). Decision of 2026-11-11: all POOLED statistics (fits,
survival curves, bins, headline shares, pooled counts) are computed on the six MAIN_REGIMES; LE1 is reported separately,
as a weakly hidden control (key "le1_separate" in the JSON outputs). Per-regime outputs keep LE1, labeled as such, and
the row tables (esn_rows.csv, classes_rows.csv, classes_eq_rows.csv, v_rows.csv, penalty_rows.csv, escape_rows.csv) keep
all seven regimes. Theory checks that do not concern hiddenness (index sums, U >= (K - 1)/2) keep all regimes.
"""

ALL_REGIMES = ("se3", "se11", "le1", "wangchen2", "wangchen2_a0.03", "chua_I", "chua_II")
SEPARATE = ("le1",)
MAIN_REGIMES = tuple(g for g in ALL_REGIMES if g not in SEPARATE)
POOL_NOTE = ("pooled over the six regimes " + ", ".join(MAIN_REGIMES) + "; LE1 (attractor hidden only in the weak sense) "
             "is excluded and reported separately under le1_separate")

# A model equilibrium is a "copy" of a true equilibrium if dist_true_eq < COPY_TOL, where dist_true_eq (written by
# pipeline.fp_info and stage_m.eq_info) is |x - x*| / L in physical units, L = max(box_hi - box_lo) / 2 = half of the
# largest edge of the region Q (10 for SE3, SE11, LE1, Wang-Chen; 15 for Chua). For LE1 x* is the nearest point of the
# line x = y = 0. So COPY_TOL = 0.01 means "within 1% of the half-width of Q" (0.1 or 0.15 in physical units).
COPY_TOL = 0.01


def is_main(regime):
    return regime not in SEPARATE


def main_only(rows, key="regime"):
    """Rows of the six main regimes (rows: list of dicts with the regime under `key`)."""
    return [x for x in rows if x[key] not in SEPARATE]


def only(rows, regime, key="regime"):
    return [x for x in rows if x[key] == regime]

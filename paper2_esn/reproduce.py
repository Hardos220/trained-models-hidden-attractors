"""Recompute every number of the ESN article from the per-model records.

    python -X utf8 paper2_esn/reproduce.py              # all steps, in order
    python -X utf8 paper2_esn/reproduce.py --from paper_hetero

Data: the per-model records (folders data, gpuq-results, local_results, node_results), read from local/esn-gpu
or from the folder named by the environment variable ESN_DATA. Steps that use several processes take their number
from RMQ_CPU (default 1).

Output: results/numbers/*.json|csv (every number of the article) and the diagnostic figures results/figures/*.pdf
written by some of the analysis scripts.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# order matters: the last group reads the row tables written by the first
STEPS = [
    ("gpu/paper_classes.py", []),        # model classes: stage M          -> classes*.json|csv
    ("gpu/paper_completeness.py", []),   # completeness of equilibrium sets -> completeness.json
    ("gpu/paper_data_effects.py", []),   # data and training effects        -> data_effects.json, v_rows, escape_rows
    ("gpu/paper_e1_controls.py", []),    # controls for the excited equilibrium
    ("gpu/paper_e1_distance.py", []),    # distance of training data from E1
    ("gpu/paper_esn_law.py", []),        # echo state networks: survival law -> esn_law.json, esn_rows, esn_C_rows, esn_UN
    ("gpu/paper_fidelity.py", []),       # fidelity of the models
    ("gpu/paper_le1_line.py", []),       # LE1 regime: line of equilibria
    ("gpu/paper_lisprott.py", []),       # Li-Sprott 4D control
    ("gpu/paper_penalty.py", []),        # training penalty on unstable equilibria
    ("gpu/paper_theory.py", []),         # index sums and Proposition 1
    ("gpu/paper_hetero.py", []),         # pooled fits across classes and regimes -> hetero.json
    ("gpu/paper_robustness.py", []),     # sensitivity of the main fits
    ("gpu/paper_v_logit.py", []),        # logistic fit of survival on U
    ("gpu/paper_le1_robust.py", []),     # LE1 sensitivity
    ("gpu/paper_law_checks.py", []),     # out-of-sample tests and mechanism of the survival law -> law_checks.json
]


def main(argv):
    start = argv[argv.index("--from") + 1] if "--from" in argv else None
    run = start is None
    for script, args in STEPS:
        run = run or Path(script).stem == start
        if not run:
            continue
        t = time.time()
        print(f"== {script}", flush=True)
        subprocess.run([sys.executable, "-X", "utf8", str(HERE / script), *args], check=True)
        print(f"   {script} done in {time.time() - t:.0f} s", flush=True)
    if not run:
        raise SystemExit(f"no step named {start}")


if __name__ == "__main__":
    main(sys.argv[1:])

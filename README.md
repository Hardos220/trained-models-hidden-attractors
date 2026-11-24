# trained-models-hidden-attractors

Code, per-model records and analysis scripts for the article

> M. Elisov, *Do trained models keep attractors hidden? Forced spurious equilibria and a survival law for hiddenness*.

Archive: Zenodo, DOI [10.5281/zenodo.24513258](https://doi.org/10.5281/zenodo.24513258) (version 1.1.0). The record holds this repository (code and
every number of the article) and the per-model records, `trained-models-hidden-attractors-records-v1.0.0.zip`
(about 1 GB; 4.4 GB unpacked; unchanged since version 1.0.0).

## Contents

| Path | Contents |
|---|---|
| `src/hsearch/` | Library: test systems, integration, identification of attractors, echo state networks (`esn_pilot.py`, `reservoir_analysis.py`, `systems/reservoir.py`). |
| `paper2_esn/experiments/` | Test regimes and training sets (`e70_esn_pilot_data.py`, `e72_esn_gpu_inputs.py`); early CPU studies of echo state networks (`e30`–`e36`). |
| `paper2_esn/gpu/` | Training and labeling of the models by stage (`stage_*.py`; GPU code in `esngpu.py`, `ngrc_gpu.py`), the job chains with the arguments of every run (`*.sh`), and the analysis scripts `paper_*.py`. |
| `paper2_esn/results/numbers/` | Every number of the article as JSON, with the row tables (CSV) behind them. |
| `paper2_esn/reproduce.py` | Recomputes `results/numbers/` from the per-model records. |

## Stages

Each stage trains a set of models, finds their equilibria and labels the reproduced attractors; one JSON record (and
for some stages an `.npz` with arrays) per model.

| Stage | Script | Models |
|---|---|---|
| C | `stage_c.py` | Echo state networks and next-generation reservoir computers on the series of the seven regimes |
| N | `stage_n.py` | Echo state networks: number and stability of equilibria versus reservoir size |
| I, K | `stage_i.py`, `stage_k.py` | Indices of the equilibria of stage C models; scale of the reservoir matrix after training |
| M | `stage_m.py` | One- and two-layer perceptrons and residual networks in physical coordinates (classes `mlp`, `mlp2`, `res`) |
| P | `stage_p.py` | Perceptrons trained with and without a penalty on unstable equilibria |
| V | `stage_v.py` | The same reservoir trained on different data |
| E | `stage_e.py` | Controls trained without the hidden attractor in the data, with model classes that contain the true dynamics |
| H, X | `stage_h.py`, `stage_x.py` | Hard-tanh echo state networks with exact equilibrium sets: reference for the protocol and the theory |
| R, L | `stage_r.py`, `stage_l.py` | Readout before training; Li–Sprott control |
| S | `stage_sat.py` | Completeness of the equilibrium sets of echo state networks |
| MW, KR | `stage_mw.py`, `krawczyk_m.py` | Stage M models retrained with saved weights; certified enumeration of their fixed points (interval Krawczyk method) |

Many docstrings and comments in `paper2_esn/gpu` and `paper2_esn/experiments` are in Russian.

## Reproducing the numbers

```
pip install -r requirements.txt
# unpack trained-models-hidden-attractors-records-v1.0.0.zip in the repository root -> local/esn-gpu/
python -X utf8 paper2_esn/reproduce.py
```

The run takes about 20 minutes and 2 GB of memory; steps that use several processes take their number from the
environment variable `RMQ_CPU` (default 1). The records can also be kept elsewhere: set `ESN_DATA` to the folder that
holds `data/`, `gpuq-results/`, `local_results/` and `node_results/`. `results/numbers/<name>.json` is written by
`gpu/paper_<name>.py`; the docstring of each script states its inputs, the selection of records and the rule for
models that appear in several jobs. In a rerun only time stamps, run times, local paths, counts of skipped duplicate
records and the last digits of iterative fits change. Some analysis scripts also draw diagnostic figures into
`results/figures/`.

Version 1.1.0 adds `gpu/paper_law_checks.py` (`results/numbers/law_checks.json`): out-of-sample tests of the survival
law, its beta-binomial mechanism fitted to the models that lost hiddenness, and the comparison of the two forms of the
law.

The records keep the layout in which they were produced: `gpuq-results/<job>/out/<stage>/<regime>/` (runs on an
NVIDIA A100), `node_results/<stage>/` and `local_results/` (runs on CPUs); `data/` holds the regimes and training sets.
`MANIFEST-sha256.txt` in the zip lists every file with its SHA-256.

## Training the models

The stage scripts run from a folder that holds `gpu/`, `src/` and `data/`, as on the compute nodes, and write to
`$GPUQ_OUT` (default `out/`):

```
mkdir work
cp -r paper2_esn/gpu src local/esn-gpu/data work/
cd work
GPUQ_OUT=out bash gpu/m_chua_I.sh
```

The models were trained in float64 with PyTorch on an NVIDIA A100 and on CPUs; on a different device the random
numbers, and so the individual models, differ.

## License

MIT (see `LICENSE`).

# CORE-AVS

Code and compact experiment assets for **Looks Right, Sounds Wrong:
Counterfactual Reliability for Static Multi-Source Audio-Visual Segmentation**.

The model estimates mask risk and soundtrack condition from 21 × 13 class
tokens. The set auditor has 18,404 parameters and uses a shared 63,490-parameter
assignment expert.

## Quick start

Clone or download this repository. With Python installed:

```sh
python reproduce.py
```

On Windows, double-click **`reproduce.cmd`**. On Linux, run `bash reproduce.sh`.
The launcher creates a local Python 3.12 CPU environment, installs the pinned
dependencies, verifies the assets, evaluates the models, checks the results
and exports Excel and CSV tables. Installation requires internet access and
can take several minutes. Later runs reuse the environment.

Open `reproduction_results/<run>/metrics/tables/reproduction_results.xlsx`.
The first two worksheets correspond to Tables 1 and 2. Other worksheets contain
three-run summaries, individual runs, architecture/transfer results, numerical
statistics, confidence intervals and the reference comparison. Each worksheet
also has a CSV file with the same records.

Each run writes a new directory. A successful run ends with `Completed` and
`status.json` reports `passed`. The reference check compares every recorded
metric at three decimal places and shows full-precision differences separately.
Missing or differing results produce a nonzero exit code.

## Environment

The pinned environment uses Python 3.12, PyTorch 2.14.0+cpu, NumPy 2.3.5 and
SciPy 1.18.1. Windows CPU execution has been tested. The shell entry points
are provided for Linux; the CPU lock targets platforms with the corresponding
PyTorch wheels. For macOS or a custom CUDA installation, install the appropriate
PyTorch build and the portable dependencies, then use the active environment:

```sh
python -m pip install -r requirements.txt
python reproduce.py --current-env
```

For an existing Python 3.12 installation, `python reproduce.py --python PATH`
uses that interpreter to create the isolated environment. Otherwise the launcher
can install a local managed Python with uv. A manual setup is also available:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux: source .venv/bin/activate
python -m pip install -r requirements-lock-cpu.txt
python reproduce.py --current-env
```

`conda env create -f environment.yml` provides the same CPU configuration.

## Evaluation

```sh
python scripts/reproduce_all.py --with_ci
python eval.py --eval_seeds
python eval.py --eval_seeds --ablation_16ep
python reproduce_extended.py --output runs/extended
python -m unittest discover -s tests -v
```

The one-click launcher includes 10,000 paired-image bootstraps. Use `--skip-ci`
with `reproduce.py` for a shorter run. `--output PATH` selects a new output
directory and `--threads N` controls CPU parallelism. The lower-level evaluation
commands accept `--device cuda`.

| Configuration | Controlled AURC | Controlled J80 | Controlled AUROC-M | CORE-AVS AUROC-M |
|---|---:|---:|---:|---:|
| Main model, default run | .623 | .272 | .837 | .782 |
| Main model, mean ± SD | .633 ± .011 | .268 ± .005 | .821 ± .014 | .769 ± .011 |
| 16-epoch model, mean ± SD | .633 ± .010 | .268 ± .006 | .802 ± .012 | .754 ± .016 |

The default run uses seed 2027. Three-run statistics use seeds 2027–2029 and
sample standard deviation (`ddof=1`). Blank metric cells mean the method was
not evaluated for that task. They are not zero values.

Main and compact model results come from checkpoint inference. Reference-method
and architecture/transfer results are recalculated from per-output score arrays.
The reference reports are read only after evaluation to compare the results.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the experiment map.

To regenerate tables from an existing complete run:

```sh
python scripts/check_results.py path/to/metrics
python scripts/export_results.py path/to/metrics
```

The spreadsheet exporter uses Python's standard library and the bundled Excel
template. It requires neither Microsoft Excel nor an additional spreadsheet
package. Numeric values retain their precision; the workbook displays three
decimals for metrics and six for differences.

## Training

```sh
python train.py --epochs 25 --seed 2027 --output runs/main_2027
python train.py --ablation_16ep --seed 2027 --output runs/compact_2027
python cross_validate.py --output runs/weight_selection
python reproduce_baselines.py --output runs/reference_refit
python reproduce_extended.py --refit --device cuda --output runs/extended_refit
```

Both curricula use AdamW, learning rate 0.001, weight decay 0.0001 and batch
size 128. The main schedule has 10 mask-only epochs followed by 15 joint epochs;
the compact schedule has 4 + 12. The joint loss is
`SmoothL1(r_hat, r) + 0.10 * mean((softmax(logits) - one_hot)**2)`.
Training saves the last checkpoint. Normalization is fitted on training evidence
and stored with the model. Five-fold source-group validation selects the joint
weight from 0.10, 0.15 and 0.20 using OOF AURC, then AUROC-M, then smaller weight.

## Data and repository layout

```text
models/               Auditor, assignment expert, features and reference heads
datasets/             Cached evidence loader and prepared-asset restoration
data/manifests/       Image IDs, source groups, conditions and fingerprints
data/eval_features/   Controlled and CORE-AVS evaluation evidence
data/train_features/  Fitting evidence for 1,316 images
data/features/        AVSegFormer transfer evidence
data/reference_predictions/  Per-output reference scores
checkpoints/          Three main models, three compact models, assignment expert
reported_results/    Reference metrics used for comparison
figures/              Figure 2(a–b) code and numerical inputs
scripts/              Evaluation wrapper, result check and table export
tests/                Metrics, model, loss, asset and export tests
```

The compact assets support evaluation and auditor training from cached evidence.
Raw images, audio, video, pixel masks and upstream backbone weights are obtained
from their original providers. [data/README.md](data/README.md) describes the
manifests and upstream preparation requirements.

To regenerate Figure 2(a–b):

```sh
python -m pip install -r requirements-figures.txt
python figures/plot_fig2.py
```

## License

Original code is distributed under the [MIT license](LICENSE).
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) records the sources and separate
terms for derived data and models.

## Citation

Zelin Jin, Guorui He, Jiabin Fan. *Looks Right, Sounds Wrong: Counterfactual Reliability for Static Multi-Source Audio-Visual Segmentation*. 2026.

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

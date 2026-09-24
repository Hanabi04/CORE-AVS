# Reproduction map

| Paper evidence | Command | Inputs |
|---|---|---|
| Table 1 | `python eval.py --datasets core_a core_b core_c core --baselines --bootstrap 10000` | Main checkpoint, CORE evidence, pooled scores |
| Table 2 | `python eval.py --datasets controlled --baselines` | Main checkpoint, Controlled evidence, seven reference score files |
| Main three-run statistics | `python eval.py --eval_seeds` | Three 25-epoch checkpoints |
| 16-epoch ablation | `python eval.py --eval_seeds --ablation_16ep` | Three 4 + 12 checkpoints, joint weight 0.10 |
| Main/compact head training | `python train.py --epochs 25 --output runs/main` / `--epochs 16` | Fitting evidence from 1,316 images |
| Five-fold weight selection | `python cross_validate.py --output runs/cv` | Fitting evidence and source-group folds |
| DeepSets, gap-difference ablation, AVSegFormer transfer | `python reproduce_extended.py --output runs/extended` | Per-output reference scores and transfer evidence |
| Architecture/transfer refitting | Add `--refit` to the previous command | Fitting evidence and original reference training recipes |
| Figure 2(a–b) | `python figures/plot_fig2.py` | Numerical curves and score-comparison points |

`python reproduce.py` runs the evaluation rows, including bootstrap intervals,
checks the metrics and exports the tables. Training and figure generation have
separate commands. Use `--skip-ci` to omit bootstrap intervals.

## Metrics and data order

Each image contributes matched, silent and mismatched outputs in that order.
CORE-AVS concatenates cohorts A, B and C. AURC averages cumulative ground-truth
mask risk after a stable ascending sort by predicted risk. J80 and J50 average
IoU over the first `floor(0.8N)` and `floor(0.5N)` outputs. AUROC uses average
ranks for ties. AUROC-S compares matched with silence; AUROC-M compares matched
with non-silent mismatch. Figure 2(b) combines silence and mismatch as positives.

Confidence intervals resample complete images, preserving all three conditions.
The wrapper uses 10,000 draws with fixed bootstrap seeds. Across-training-run
variability is reported separately as sample SD over seeds 2027, 2028 and 2029.

## Reference methods

Soft Dice and DenseAV use archived per-output scores. Log-RMS was recomputed
from the original 600 prepared waveforms in matching UID order, using
`-log10(sqrt(mean(waveform**2)) + 1e-12)`.
Learned references use the original 16-epoch CUDA replay. Their fitting
environment used PyTorch 2.12.0.dev20260408+cu128 and NumPy 2.5.2; the CPU
evaluation environment is pinned in `requirements-lock-cpu.txt`.
Refitting on a different backend may produce small optimization differences.

The manuscript's **w/o gap difference** row is named `without_gap_difference`
in filenames and result keys. It zeros feature column 1, the observed-minus-silence
support difference, during training and evaluation. Column 2 is `tanh` of the
observed support gap. Reference dual heads use their original unit Brier weight;
main and compact curricula use 0.10.

## Upstream stages

This release includes the assignment expert's forward, target and loss functions
and its frozen checkpoint. Training that expert requires region candidates and
audio embeddings from the upstream pipeline. Raw-media preparation and upstream
backbone training are outside the cached-evidence commands. See
[data/README.md](data/README.md) for media sources and manifest fields.

The paper's 124.1-second, 3.66-GiB measurement describes the original full GPU
pipeline. The time recorded by this package measures cached-evidence evaluation
and is reported separately from environment installation.

## Result files

Each run contains JSON reports, process logs, `comparison.json` and `tables/`.
The Excel workbook and eight CSV files are generated from that run's reports.
The comparison uses the bundled `reported_results/` records after inference;
these records do not supply predictions. All values are retained at their
computed precision and compared at the paper's three-decimal display precision.

`ASSET_SHA256.json` inventories experiment inputs. `FILE_SHA256.json` inventories
the release files; the optional `INSTRUCTIONS_ZH.md` is excluded so it can be
removed before sharing the English package. `python verify_assets.py` checks
both inventories when present.

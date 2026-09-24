# Validation

Validated on 9 September 2026 with Windows, Python 3.12.14, PyTorch
2.14.0+cpu, NumPy 2.3.5 and SciPy 1.18.1. Evaluation used four CPU threads.

| Check | Result |
|---|---|
| Main tables, three-run curricula, architecture/transfer, bootstrap intervals | 292 numerical fields match the reference reports at three decimals |
| Complete cached-evidence evaluation | 19.27 seconds including child imports, excluding installation and unit tests |
| Fresh extraction and managed environment setup | Automatic Python installation, dependency installation and full evaluation passed |
| Unit tests | 12 passed: models, metrics, curriculum, manifests, asset alignment, result comparison and spreadsheet export |
| New 25-epoch training, seed 2027 | Controlled AURC/J80/AUROC-M .623/.272/.837; CORE AUROC-M .782 |
| New 16-epoch training, seed 2027 | Controlled AURC/J80/AUROC-M .624/.272/.793; CORE AUROC-M .741 |
| CSV and Excel | Eight tables; numeric values and unavailable-metric blanks preserved |
| Checkpoints, features and per-output scores | 32 binary assets covered by SHA-256 checks |

Run `python reproduce.py` to repeat evaluation, result checks and tests on
your machine. The return code is zero only when these steps pass. Runtime
depends on the host; the recorded duration is an observation, not a deadline.

Baseline refitting and cross-validation have separate commands and do not run
as part of the quick evaluation. Historical CUDA reference fitting used the
environment recorded in REPRODUCIBILITY.md. Linux/macOS execution and new GPU
training were not part of this Windows CPU validation.

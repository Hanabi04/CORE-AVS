# Release notes

## Initial release

Includes auditor checkpoints, cached evidence, evaluation manifests, reference predictions, training code, and CPU reproduction scripts. `python reproduce.py` evaluates the included checkpoints, recomputes the reported statistics, checks reference values, and exports Excel and CSV tables.

Baseline refitting and cross-validation are separate commands documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Raw media and upstream backbone weights must be obtained from their original providers; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

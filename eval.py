"""Evaluate released checkpoints and archived reference predictions."""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from datasets.cached import ROOT, load_cohort
from metrics import evaluate, bootstrap_audio
from runtime import predict

COLUMNS = ("AURC", "J80", "AUROC-S", "AUROC-M", "MAE", "J50")


def table(rows):
    print("| Dataset / model / seed | " + " | ".join(COLUMNS) + " |")
    print("|---|" + "---:|" * len(COLUMNS))
    for name, values in rows:
        print(
            "| "
            + name
            + " | "
            + " | ".join(
                v
                if isinstance(v := values.get(k), str)
                else f"{v:.3f}"
                if v is not None
                else "-"
                for k in COLUMNS
            )
            + " |"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--eval_seeds",
        action="store_true",
        help="Evaluate three independent checkpoints",
    )
    parser.add_argument("--ablation_16ep", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("controlled", "core", "core_a", "core_b", "core_c"),
        default=["controlled", "core"],
    )
    parser.add_argument("--data_root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--baselines",
        action="store_true",
        help="Evaluate archived reference scores, not reference models",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Paired-image bootstrap repetitions, e.g. 10000",
    )
    parser.add_argument(
        "--output", type=Path, help="New JSON file; overwrites are refused"
    )
    a = parser.parse_args()
    if a.checkpoint and a.eval_seeds:
        parser.error("--checkpoint and --eval_seeds are mutually exclusive")
    if a.bootstrap < 0:
        parser.error("--bootstrap must be nonnegative")
    torch.set_num_threads(a.threads)
    prefix = "ablation_16ep" if a.ablation_16ep else "final_dual"
    paths = (
        [a.checkpoint]
        if a.checkpoint
        else [
            ROOT / "checkpoints" / f"{prefix}_seed{s}.pth"
            for s in ([2027, 2028, 2029] if a.eval_seeds else [2027])
        ]
    )
    started = time.perf_counter()
    rows, results = [], {}
    for cohort in a.datasets:
        data = load_cohort(cohort, a.data_root)
        independent = []
        for path in paths:
            prediction, ck = predict(path, data, a.device)
            values = evaluate(data, **prediction)
            name = f"{cohort} / {path.stem} / {ck['seed']}"
            results[name] = dict(values)
            rows.append((name, values))
            independent.append(values)
            if a.bootstrap:
                index = ["controlled", "core_a", "core_b", "core_c", "core"].index(
                    cohort
                )
                results[name]["95% image-bootstrap CI"] = bootstrap_audio(
                    prediction["condition_probability"], 260906 + index, a.bootstrap
                )
        if len(independent) > 1:
            summary = {
                k: {
                    "mean": float(np.mean([v[k] for v in independent])),
                    "sample_std": float(np.std([v[k] for v in independent], ddof=1)),
                }
                for k in COLUMNS
            }
            results[f"{cohort} / mean_std"] = summary
            rows.append(
                (
                    f"{cohort} / mean +/- SD (ddof=1)",
                    {
                        k: f"{v['mean']:.3f} +/- {v['sample_std']:.3f}"
                        for k, v in summary.items()
                    },
                )
            )
        if a.baselines:
            for path in sorted(
                (a.data_root / "reference_predictions").glob(f"{cohort}_*.npz")
            ):
                reference_name = path.stem[len(cohort) + 1 :]
                if reference_name not in {
                    "small_pool",
                    "pooled_dual",
                    "confidnet",
                    "set_risk",
                    "without_gap_difference",
                    "soft_dice",
                    "denseav",
                    "log_rms",
                }:
                    continue
                with np.load(path, allow_pickle=False) as z:
                    if not np.array_equal(
                        z["arms"], data["arms"]
                    ) or not np.array_equal(z["truth_risk"], data["truth_risk"]):
                        raise ValueError(f"Reference target mismatch: {path.name}")
                    values = evaluate(
                        data,
                        **{
                            k: z[k]
                            for k in ("risk", "condition_probability", "audio_score")
                            if k in z
                        },
                    )
                provenance = (
                    "original scores"
                    if reference_name in {"soft_dice", "denseav", "log_rms"}
                    else "archived GPU replay"
                )
                name = f"{cohort} / {reference_name} ({provenance})"
                results[name] = values
                rows.append((name, values))
    table(rows)
    for name, result in results.items():
        if "95% image-bootstrap CI" in result:
            print(name, json.dumps(result["95% image-bootstrap CI"]))
    seconds = time.perf_counter() - started
    print(
        f"Elapsed evaluation time (excluding interpreter/import startup): {seconds:.2f} s"
    )
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.open("x", encoding="utf-8") as f:
            json.dump(
                {
                    "results": results,
                    "elapsed_seconds": seconds,
                    "torch_version": torch.__version__,
                    "device": a.device,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()

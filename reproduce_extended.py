"""Evaluate or refit the architecture controls and AVSegFormer transfer head."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from datasets import load_cohort
from metrics import evaluate
from models import baselines as b

ROOT = Path(__file__).resolve().parent


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    training = dict(epochs=16, learning_rate=0.001, weight_decay=0.0001, batch_size=128)
    train, test = load_cohort("train"), load_cohort("core")
    device = torch.device(args.device)
    results = {}
    archived = ROOT / "data/reference_predictions"
    for name in ("deepsets", "without_gap_difference"):
        rows = []
        for seed in (2027, 2028, 2029):
            training["seed"] = seed
            filename = f"core_{name}_{seed}.npz"
            if args.refit:
                b._seed(seed)
                if name == "deepsets":
                    prediction = b._train_dual(
                        b.ClassAwareDeepSets(),
                        train["features"],
                        train["truth_risk"],
                        train["arms"],
                        test["features"],
                        training,
                        seed,
                        device,
                    )
                else:
                    prediction = b._train(
                        architecture="set",
                        supervision="dual",
                        condition_loss="brier",
                        omit_gap_difference=True,
                        train=(train["features"], train["truth_risk"], train["arms"]),
                        test=(test["features"], test["truth_risk"], test["arms"]),
                        training=training,
                        device=device,
                    )
                scores = {
                    key: prediction[key] for key in ("risk", "condition_probability")
                }
                np.savez_compressed(
                    args.output / filename,
                    **scores,
                    truth_risk=test["truth_risk"],
                    arms=test["arms"],
                )
            else:
                saved = load_npz(archived / filename)
                scores = {key: saved[key] for key in ("risk", "condition_probability")}
                np.testing.assert_array_equal(saved["truth_risk"], test["truth_risk"])
                np.testing.assert_array_equal(saved["arms"], test["arms"])
            row = evaluate(test, **scores)
            rows.append(row)
            results[f"core/{name}/{seed}"] = row
        results[f"core/{name}/mean_std"] = {
            key: {
                "mean": float(np.mean([r[key] for r in rows])),
                "sample_std": float(np.std([r[key] for r in rows], ddof=1)),
            }
            for key in rows[0]
        }

    train = load_npz(ROOT / "data/features/avsegformer_train.npz")
    test = load_npz(ROOT / "data/features/avsegformer_test.npz")
    filename = "avsegformer_dual_2027.npz"
    if args.refit:
        training["seed"] = 2027
        prediction = b._train(
            architecture="set",
            supervision="dual",
            condition_loss="brier",
            omit_gap_difference=False,
            train=(train["features"], train["truth_risk"], train["arms"]),
            test=(test["features"], test["truth_risk"], test["arms"]),
            training=training,
            device=device,
        )
        scores = {key: prediction[key] for key in ("risk", "condition_probability")}
        np.savez_compressed(
            args.output / filename,
            **scores,
            truth_risk=test["truth_risk"],
            arms=test["arms"],
        )
    else:
        saved = load_npz(archived / filename)
        scores = {key: saved[key] for key in ("risk", "condition_probability")}
        np.testing.assert_array_equal(saved["truth_risk"], test["truth_risk"])
        np.testing.assert_array_equal(saved["arms"], test["arms"])
    results["avsegformer/dual"] = evaluate(test, **scores)
    results["avsegformer/soft_dice"] = evaluate(test, risk=test["soft_dice"])
    report = {
        "mode": "refit" if args.refit else "archived GPU replay",
        "results": results,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("| Experiment | AURC | J80 | AUROC-S | AUROC-M |")
    print("|---|---:|---:|---:|---:|")
    for name, row in results.items():
        values = []
        for key in ("AURC", "J80", "AUROC-S", "AUROC-M"):
            value = row.get(key)
            values.append(
                "—"
                if value is None
                else f"{value['mean']:.3f} ± {value['sample_std']:.3f}"
                if isinstance(value, dict)
                else f"{value:.3f}"
            )
        print("| " + name + " | " + " | ".join(values) + " |")


if __name__ == "__main__":
    main()

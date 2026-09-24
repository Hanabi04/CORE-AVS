"""Five-fold source-group selection of the joint loss weight."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from datasets.cached import ROOT, load_cohort
from metrics import evaluate
from runtime import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    data = load_cohort("train")
    rows = []
    for weight in (0.10, 0.15, 0.20):
        risk = np.empty_like(data["truth_risk"])
        probability = np.empty((len(risk), 3), dtype=np.float32)
        for fold in range(5):
            output = args.output / f"lambda{weight:.2f}_fold{fold}"
            command = [
                sys.executable,
                str(ROOT / "train.py"),
                "--epochs",
                "25",
                "--seed",
                str(2027 + fold),
                "--lambda_joint",
                str(weight),
                "--holdout_fold",
                str(fold),
                "--device",
                args.device,
                "--threads",
                str(args.threads),
                "--output",
                str(output),
            ]
            subprocess.run(command, check=True)
            keep = np.repeat(data["folds"] == fold, 3)
            predicted, _ = predict(
                output / "final.pth", {"features": data["features"][keep]}, args.device
            )
            risk[keep] = predicted["risk"]
            probability[keep] = predicted["condition_probability"]
        values = evaluate(data, risk=risk, condition_probability=probability)
        rows.append({"lambda": weight, **values})
        np.savez_compressed(
            args.output / f"oof_lambda{weight:.2f}.npz",
            risk=risk,
            condition_probability=probability,
        )
    selected = min(rows, key=lambda row: (row["AURC"], -row["AUROC-M"], row["lambda"]))
    report = {
        "candidates": rows,
        "selected_lambda": selected["lambda"],
        "selection": "minimum OOF AURC, then maximum AUROC-M, then smaller lambda",
    }
    (args.output / "selection.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

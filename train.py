"""Train a new auditor from fixed cached evidence; evaluation is separate."""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from datasets import load_cohort
from models import SetAuditor
from models.curriculum_loss import curriculum_loss
from runtime import seed_all


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, choices=(16, 25), default=25)
    p.add_argument(
        "--ablation_16ep", action="store_true", help="Use the 4 + 12 curriculum"
    )
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument(
        "--lambda_joint", type=float, choices=(0.10, 0.15, 0.20), default=0.10
    )
    p.add_argument(
        "--holdout_fold",
        type=int,
        choices=range(5),
        help="Exclude this source-group fold from fitting",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--data_root", type=Path)
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory; existing directories are refused",
    )
    a = p.parse_args()
    if a.ablation_16ep:
        a.epochs = 16
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(a.threads)
    seed_all(a.seed)
    data = load_cohort("train", a.data_root)
    if a.holdout_fold is not None:
        keep = np.repeat(data["folds"] != a.holdout_fold, 3)
        data = {
            k: (v[keep] if k in ("features", "truth_risk", "arms") else v)
            for k, v in data.items()
        }
    features = data["features"]
    mean, scale = features.mean((0, 1)), features.std((0, 1))
    scale[scale < 1e-6] = 1
    x = torch.from_numpy((features - mean) / scale).to(a.device)
    y = torch.from_numpy(data["truth_risk"]).to(a.device)
    arms = torch.from_numpy(data["arms"]).long().to(a.device)
    model = SetAuditor().to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup, weight = (10 if a.epochs == 25 else 4), a.lambda_joint
    records = []
    started = time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        model.train()
        order = np.random.default_rng(a.seed + epoch - 1).permutation(len(x))
        losses = []
        for start in range(0, len(x), 128):
            ids = torch.as_tensor(order[start : start + 128], device=a.device)
            opt.zero_grad(set_to_none=True)
            risk, logits = model(x[ids])
            loss = curriculum_loss(
                risk, logits, y[ids], arms[ids], 0 if epoch <= warmup else weight
            )
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        record = {
            "epoch": epoch,
            "lambda": 0 if epoch <= warmup else weight,
            "loss": float(np.mean(losses)),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    torch.save(
        {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "norm": {"mean": torch.from_numpy(mean), "scale": torch.from_numpy(scale)},
            "seed": a.seed,
            "epoch": a.epochs,
            "warmup": warmup,
            "lambda_joint": weight,
            "architecture": "set_auditor",
        },
        a.output / "final.pth",
    )
    (a.output / "training.json").write_text(
        json.dumps(
            {
                "epochs": records,
                "elapsed_seconds": time.perf_counter() - started,
                "optimizer": {
                    "name": "AdamW",
                    "lr": 0.001,
                    "weight_decay": 0.0001,
                    "batch_size": 128,
                },
                "torch_version": torch.__version__,
                "seed": a.seed,
                "warmup": warmup,
                "lambda_joint": weight,
                "holdout_fold": a.holdout_fold,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

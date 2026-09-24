"""Refit fixed reference heads; output scores go to a fresh directory."""

import argparse
from pathlib import Path
import numpy as np
import torch
from datasets import load_cohort
from models import baselines as b
from metrics import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    train = load_cohort("train")
    test = load_cohort("controlled")
    as_tuple = lambda d: (d["features"], d["truth_risk"], d["arms"])
    training = dict(
        epochs=16, learning_rate=0.001, weight_decay=0.0001, batch_size=128, seed=2027
    )
    device = torch.device(a.device)
    b._seed(2027)
    predictions = {
        "pooled_dual": b._train_dual(
            b.CapacityMatchedPooled(),
            b._pooled(train["features"]),
            train["truth_risk"],
            train["arms"],
            b._pooled(test["features"]),
            training,
            2027,
            device,
        )
    }
    predictions["confidnet"] = b._train_confidnet(
        train["features"], train["truth_risk"], test["features"], training, device
    )
    for name, supervision, observed in [
        ("set_risk", "risk_only", False),
        ("without_gap_difference", "dual", True),
    ]:
        predictions[name] = b._train(
            architecture="set",
            supervision=supervision,
            condition_loss="brier",
            omit_gap_difference=observed,
            train=as_tuple(train),
            test=as_tuple(test),
            training=training,
            device=device,
        )
    for name, prediction in predictions.items():
        scores = {
            k: prediction[k]
            for k in ("risk", "condition_probability")
            if k in prediction
        }
        if name == "set_risk":
            scores.pop("condition_probability", None)
        np.savez_compressed(
            a.output / f"controlled_{name}.npz",
            **scores,
            truth_risk=test["truth_risk"],
            arms=test["arms"],
        )
        print(name, evaluate(test, **scores), flush=True)
    combined = load_cohort("core")
    prediction = b._train(
        architecture="pooled",
        supervision="dual",
        condition_loss="brier",
        omit_gap_difference=False,
        train=as_tuple(train),
        test=as_tuple(combined),
        training=training,
        device=device,
    )
    start = 0
    for cohort, n in [
        ("core_a", 600),
        ("core_b", 600),
        ("core_c", 2019),
        ("core", 3219),
    ]:
        if cohort == "core":
            start = 0
        data = load_cohort(cohort)
        scores = {
            k: prediction[k][start : start + n]
            for k in ("risk", "condition_probability")
        }
        np.savez_compressed(
            a.output / f"{cohort}_small_pool.npz",
            **scores,
            truth_risk=data["truth_risk"],
            arms=data["arms"],
        )
        print(cohort, evaluate(data, **scores), flush=True)
        start += n


if __name__ == "__main__":
    main()

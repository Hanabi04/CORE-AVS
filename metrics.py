"""Selective risk and paired-image acoustic AUROC metrics."""

import numpy as np
from scipy.stats import rankdata


def auroc(negative, positive):
    n, p = len(negative), len(positive)
    if not n or not p:
        raise ValueError("Both classes are required")
    ranks = rankdata(np.concatenate([negative, positive]), method="average")
    return float((ranks[n:].sum() - p * (p + 1) / 2) / (n * p))


def evaluate(data, risk=None, condition_probability=None, audio_score=None):
    out = {}
    if risk is not None:
        order = np.argsort(risk, kind="stable")
        truth = data["truth_risk"]
        out["AURC"] = float(
            np.mean(np.cumsum(truth[order]) / np.arange(1, len(order) + 1))
        )
        for percent in (50, 80):
            out[f"J{percent}"] = float(
                np.mean(1 - truth[order[: int(percent / 100 * len(order))]])
            )
        out["MAE"] = float(np.mean(np.abs(risk - truth)))
    if condition_probability is not None:
        audio_score = 1 - condition_probability[:, 0]
    if audio_score is not None:
        arms = data["arms"]
        for label, arm in [("AUROC-S", 1), ("AUROC-M", 2)]:
            out[label] = auroc(audio_score[arms == 0], audio_score[arms == arm])
    return out


def bootstrap_audio(probability, seed, repetitions=10000):
    """Resample complete images, preserving their three paired arms."""
    scores = (1 - probability[:, 0]).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    samples = [[], []]
    n = len(scores)
    for start in range(0, repetitions, 128):
        ids = rng.integers(0, n, size=(min(128, repetitions - start), n))
        negative = scores[ids, 0]
        for j, arm in enumerate((1, 2)):
            ranks = rankdata(
                np.concatenate([negative, scores[ids, arm]], axis=1), axis=1
            )
            samples[j].extend((ranks[:, n:].sum(1) - n * (n + 1) / 2) / (n * n))
    return {
        k: np.quantile(v, [0.025, 0.975]).tolist()
        for k, v in zip(("AUROC-S", "AUROC-M"), samples)
    }

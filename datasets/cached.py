"""Cached scalar features. Rows are image-major: matched, silence, mismatch."""

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_cohort(name, data_root=None):
    root = Path(data_root) if data_root else ROOT / "data"
    if name == "core":
        parts = [load_cohort(k, root) for k in ("core_a", "core_b", "core_c")]
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    folder = "train_features" if name == "train" else "eval_features"
    with np.load(root / folder / f"{name}.npz", allow_pickle=False) as z:
        result = {k: z[k] for k in z.files}
    n = len(result["uids"])
    if result["features"].shape != (3 * n, 21, 13):
        raise ValueError("Invalid feature dimensions")
    if not np.array_equal(result["arms"], np.tile([0, 1, 2], n)):
        raise ValueError("Invalid arm order")
    if not np.isfinite(result["features"]).all():
        raise ValueError("Non-finite features")
    return result

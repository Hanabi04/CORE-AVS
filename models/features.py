import numpy as np

FEATURE_DIM = 13
ExpertRiskError = ValueError


def observable_features(
    *,
    cavp_probability: np.ndarray,
    rsra_probability: np.ndarray,
    support_gap: np.ndarray,
    silence_support_gap: np.ndarray,
    candidate_count: np.ndarray,
) -> np.ndarray:
    """Build the fixed 13-D per-class feature vector without labels."""

    cavp = np.asarray(cavp_probability, dtype=np.float32)
    rsra = np.asarray(rsra_probability, dtype=np.float32)
    if cavp.shape != rsra.shape or cavp.ndim != 3 or cavp.shape[2] != 21:
        raise ExpertRiskError(
            "expert probabilities must share HxWx21; "
            f"got CAVP={cavp.shape}, RSRA={rsra.shape}"
        )
    if support_gap.shape != (21,) or silence_support_gap.shape != (21,):
        raise ExpertRiskError("support gaps must have shape 21")
    if candidate_count.shape != (21,):
        raise ExpertRiskError("candidate count must have shape 21")
    cavp_mask = cavp >= 0.5
    rsra_mask = rsra >= 0.5
    intersection = np.sum(cavp_mask & rsra_mask, axis=(0, 1), dtype=np.float64)
    union = np.sum(cavp_mask | rsra_mask, axis=(0, 1), dtype=np.float64)
    cavp_area = np.mean(cavp_mask, axis=(0, 1), dtype=np.float64)
    rsra_area = np.mean(rsra_mask, axis=(0, 1), dtype=np.float64)
    overlap = intersection / np.maximum(union, 1.0)
    result = np.stack(
        (
            np.asarray(support_gap, dtype=np.float64),
            np.asarray(support_gap - silence_support_gap, dtype=np.float64),
            np.tanh(np.asarray(support_gap, dtype=np.float64)),
            np.log1p(np.asarray(candidate_count, dtype=np.float64)) / np.log(17.0),
            np.mean(cavp, axis=(0, 1), dtype=np.float64),
            np.max(cavp, axis=(0, 1)),
            cavp_area,
            np.mean(rsra, axis=(0, 1), dtype=np.float64),
            np.max(rsra, axis=(0, 1)),
            rsra_area,
            overlap,
            np.abs(cavp_area - rsra_area),
            np.mean(np.abs(cavp - rsra), axis=(0, 1), dtype=np.float64),
        ),
        axis=1,
    ).astype(np.float32)
    if result.shape != (21, FEATURE_DIM) or not np.isfinite(result).all():
        raise ExpertRiskError("invalid observable feature matrix")
    return result

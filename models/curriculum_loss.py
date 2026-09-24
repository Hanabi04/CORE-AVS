"""SmoothL1 warm-up followed by mean three-state Brier supervision."""

import torch.nn.functional as F


def curriculum_loss(risk, logits, target_risk, arms, audio_weight):
    loss = F.smooth_l1_loss(risk, target_risk)
    if audio_weight > 0:
        target = F.one_hot(arms.long(), 3).to(logits.dtype)
        loss = loss + audio_weight * (logits.softmax(-1) - target).square().mean()
    return loss

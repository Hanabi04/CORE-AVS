"""Relative source-to-region set assignment with an explicit dustbin."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalProposalSetError(RuntimeError):
    """Raised when the frozen set-assignment interface is violated."""


class RelativeSourceRegionMatcher(nn.Module):
    """Assign 21 silence-referenced source queries to frozen region candidates."""

    def __init__(self, visual_dim: int = 8, audio_dim: int = 256) -> None:
        super().__init__()
        self.audio = nn.Sequential(
            nn.Linear(audio_dim, 128), nn.GELU(), nn.LayerNorm(128)
        )
        self.class_embedding = nn.Embedding(21, 128)
        self.visual = nn.Sequential(
            nn.Linear(visual_dim, 64), nn.GELU(), nn.LayerNorm(64)
        )
        self.region_score = nn.Sequential(
            nn.Linear(192, 96), nn.GELU(), nn.Linear(96, 1)
        )
        self.dustbin_score = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def forward(
        self,
        audio: torch.Tensor,
        silence: torch.Tensor,
        candidate_classes: torch.Tensor,
        visual: torch.Tensor,
    ) -> torch.Tensor:
        """Return variants x 21 x (candidates + dustbin) assignment logits."""

        if audio.ndim != 2 or audio.shape[1] != 256:
            raise CausalProposalSetError("audio must be variants x 256")
        if silence.shape != (256,):
            raise CausalProposalSetError("silence must have shape 256")
        if candidate_classes.ndim != 1 or visual.shape != (
            candidate_classes.numel(),
            8,
        ):
            raise CausalProposalSetError("candidate tensor shape mismatch")
        if bool(((candidate_classes < 0) | (candidate_classes >= 21)).any()):
            raise CausalProposalSetError("class index outside [0, 20]")

        audio_delta = self.audio(audio) - self.audio(silence[None])[0]
        sources = audio_delta[:, None, :] * self.class_embedding.weight[None]
        variants = audio.shape[0]
        candidate_count = candidate_classes.numel()
        if candidate_count:
            regions = self.visual(visual)
            expanded_sources = sources[:, :, None, :].expand(
                -1, -1, candidate_count, -1
            )
            expanded_regions = regions[None, None].expand(variants, 21, -1, -1)
            region_logits = self.region_score(
                torch.cat((expanded_sources, expanded_regions), dim=-1)
            ).squeeze(-1)
            compatible = (
                torch.arange(21, device=candidate_classes.device)[:, None]
                == candidate_classes[None]
            )
            region_logits = region_logits.masked_fill(~compatible[None], -30.0)
        else:
            region_logits = audio.new_empty((variants, 21, 0))
        dustbin = self.dustbin_score(sources).squeeze(-1)[..., None]
        return torch.cat((region_logits, dustbin), dim=-1)


def assignment_targets(
    labels: torch.Tensor,
    candidate_classes: torch.Tensor,
    quality: torch.Tensor,
) -> torch.Tensor:
    """Build soft set-matching targets; unsupported sources map to dustbin."""

    if labels.ndim != 2 or labels.shape[1] != 21:
        raise CausalProposalSetError("labels must be variants x 21")
    if candidate_classes.ndim != 1 or quality.shape != candidate_classes.shape:
        raise CausalProposalSetError("candidate target shape mismatch")
    if bool(((quality < 0.0) | (quality > 1.0)).any()):
        raise CausalProposalSetError("quality must lie in [0, 1]")
    variants = labels.shape[0]
    candidate_count = candidate_classes.numel()
    target = labels.new_zeros((variants, 21, candidate_count + 1))
    for variant in range(variants):
        for class_index in range(21):
            selected = candidate_classes == class_index
            candidate_quality = quality[selected]
            if (
                labels[variant, class_index] > 0.5
                and float(candidate_quality.sum()) > 0.0
            ):
                positions = torch.nonzero(selected, as_tuple=False).flatten()
                target[variant, class_index, positions] = (
                    candidate_quality / candidate_quality.sum()
                )
            else:
                target[variant, class_index, -1] = 1.0
    return target


def set_assignment_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft cross-entropy over each source's candidate set plus dustbin."""

    if logits.shape != target.shape or logits.ndim != 3:
        raise CausalProposalSetError("assignment prediction/target mismatch")
    if not torch.allclose(target.sum(-1), torch.ones_like(target[..., 0])):
        raise CausalProposalSetError("assignment targets must sum to one")
    return -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()


def relative_set_rank_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_classes: torch.Tensor,
    quality: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Rank compatible region sets against dustbin under physical subsets."""

    if (
        logits.shape[:2] != labels.shape
        or logits.shape[2] != candidate_classes.numel() + 1
    ):
        raise CausalProposalSetError("ranking tensor shape mismatch")
    terms: list[torch.Tensor] = []
    for class_index in range(21):
        selected = candidate_classes == class_index
        has_region = bool(selected.any()) and float(quality[selected].sum()) > 0.0
        if not has_region:
            continue
        region_score = torch.logsumexp(logits[:, class_index, :-1][:, selected], dim=1)
        dustbin_score = logits[:, class_index, -1]
        signed_gap = torch.where(
            labels[:, class_index] > 0.5,
            region_score - dustbin_score,
            dustbin_score - region_score,
        )
        terms.append(F.relu(logits.new_tensor(margin) - signed_gap).mean())
    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).mean()


def decode_assignments(
    logits: torch.Tensor, masks: torch.Tensor, candidate_classes: torch.Tensor
) -> torch.Tensor:
    """Compose weighted frozen masks; dustbin probability produces background."""

    if logits.ndim != 3 or masks.ndim != 3 or candidate_classes.ndim != 1:
        raise CausalProposalSetError("invalid decode tensor rank")
    if (
        logits.shape[2] != candidate_classes.numel() + 1
        or masks.shape[0] != candidate_classes.numel()
    ):
        raise CausalProposalSetError("decode candidate count mismatch")
    dustbin = logits[:, :, -1].unsqueeze(-1)
    probability = torch.sigmoid(logits[:, :, :-1] - dustbin)
    variants, _, _ = logits.shape
    height, width = masks.shape[-2:]
    output = masks.new_zeros((variants, 21, height, width))
    for class_index in range(21):
        selected = candidate_classes == class_index
        if not bool(selected.any()):
            continue
        contribution = (
            probability[:, class_index][:, selected, None, None] * masks[selected][None]
        )
        output[:, class_index] = 1.0 - torch.prod(1.0 - contribution, dim=1)
    return output


__all__ = [
    "CausalProposalSetError",
    "RelativeSourceRegionMatcher",
    "assignment_targets",
    "decode_assignments",
    "relative_set_rank_loss",
    "set_assignment_loss",
]

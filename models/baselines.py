"""Fixed 16-epoch reference-head implementations."""

from __future__ import annotations
import random
import time
from typing import Any
import numpy as np
import torch
from torch import nn
from .set_auditor import SetAuditor


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


class CapacityMatchedPooled(nn.Module):
    """Dual-target MLP over 52 pooled class descriptors."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(52, 96),
            nn.GELU(),
            nn.LayerNorm(96),
            nn.Linear(96, 96),
            nn.GELU(),
            nn.LayerNorm(96),
            nn.Linear(96, 32),
            nn.GELU(),
            nn.LayerNorm(32),
        )
        self.risk = nn.Linear(32, 1)
        self.condition = nn.Linear(32, 3)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(value)
        return torch.sigmoid(self.risk(hidden)[:, 0]), self.condition(hidden)


class ConfidNetStyle(nn.Module):
    """Fixed pooled confidence regressor using frozen-mask descriptors only."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(value)[:, 0])


class ClassAwareDeepSets(nn.Module):
    """Class-preserving additive readout without token-to-token attention."""

    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(13, 64)
        self.class_embedding = nn.Embedding(21, 64)
        self.phi = nn.Sequential(
            nn.Linear(64, 120),
            nn.GELU(),
            nn.LayerNorm(120),
            nn.Linear(120, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )
        self.risk = nn.Linear(64, 1)
        self.condition = nn.Linear(64, 3)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        classes = torch.arange(21, device=value.device)
        tokens = self.input(value) + self.class_embedding(classes)[None]
        pooled = self.phi(tokens).mean(1)
        return torch.sigmoid(self.risk(pooled)[:, 0]), self.condition(pooled)


def _pooled(value: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (value.mean(1), value.std(1), value.min(1), value.max(1)), axis=1
    ).astype(np.float32)


def _condition_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.softmax(logits, dim=1)
    one_hot = torch.nn.functional.one_hot(target, num_classes=3).float()
    return torch.mean((probability - one_hot) ** 2)


def _train_dual(
    model: nn.Module,
    train_x: np.ndarray,
    train_risk: np.ndarray,
    train_arms: np.ndarray,
    test_x: np.ndarray,
    training: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    axes = (0,) if train_x.ndim == 2 else (0, 1)
    mean, scale = train_x.mean(axes), train_x.std(axes)
    scale[scale < 1e-6] = 1.0
    _seed(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    batch = int(training["batch_size"])
    history: list[float] = []
    started = time.perf_counter()
    for epoch in range(int(training["epochs"])):
        order = np.random.default_rng(seed + epoch).permutation(len(train_x))
        losses = []
        model.train()
        for start in range(0, len(order), batch):
            selected = order[start : start + batch]
            x = torch.from_numpy((train_x[selected] - mean) / scale).to(device)
            y = torch.from_numpy(train_risk[selected]).to(device)
            arm = torch.from_numpy(train_arms[selected]).long().to(device)
            risk, logits = model(x)
            loss = torch.nn.functional.smooth_l1_loss(risk, y) + _condition_loss(
                logits, arm
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    model.eval()
    with torch.inference_mode():
        risk, logits = model(torch.from_numpy((test_x - mean) / scale).to(device))
    return {
        "risk": risk.cpu().numpy(),
        "condition_probability": torch.softmax(logits, 1).cpu().numpy(),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "history": history,
        "seconds": float(time.perf_counter() - started),
    }


def _train_confidnet(
    train_features: np.ndarray,
    train_risk: np.ndarray,
    test_features: np.ndarray,
    training: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    # CAVP-only mean/max/area descriptors, summarized over the 21 classes.
    train_x = _pooled(train_features[:, :, 4:7])
    test_x = _pooled(test_features[:, :, 4:7])
    mean, scale = train_x.mean(0), train_x.std(0)
    scale[scale < 1e-6] = 1.0
    seed = 2027
    _seed(seed)
    model = ConfidNetStyle().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    batch = int(training["batch_size"])
    for epoch in range(int(training["epochs"])):
        order = np.random.default_rng(seed + epoch).permutation(len(train_x))
        model.train()
        for start in range(0, len(order), batch):
            selected = order[start : start + batch]
            x = torch.from_numpy((train_x[selected] - mean) / scale).to(device)
            y = torch.from_numpy(train_risk[selected]).to(device)
            loss = torch.nn.functional.smooth_l1_loss(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        risk = model(torch.from_numpy((test_x - mean) / scale).to(device)).cpu().numpy()
    return {"risk": risk, "parameters": int(sum(p.numel() for p in model.parameters()))}


class PooledReliability(nn.Module):
    """Pooled counterpart with the same two output semantics as the set model."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(52, 32), nn.GELU(), nn.LayerNorm(32))
        self.risk = nn.Linear(32, 1)
        self.condition = nn.Linear(32, 3)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(value)
        return torch.sigmoid(self.risk(hidden)[:, 0]), self.condition(hidden)


def _train(
    *,
    architecture: str,
    supervision: str,
    condition_loss: str,
    omit_gap_difference: bool,
    train: tuple[np.ndarray, ...],
    test: tuple[np.ndarray, ...],
    training: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if condition_loss != "brier":
        raise ValueError("This reference implementation uses the Brier objective")
    x, risk, arms, *_ = train
    tx = test[0]
    if omit_gap_difference:
        x = x.copy()
        tx = tx.copy()
        x[:, :, 1] = 0.0
        tx[:, :, 1] = 0.0
    pooled = architecture == "pooled"
    train_x = _pooled(x) if pooled else x
    test_x = _pooled(tx) if pooled else tx
    axes = (0,) if pooled else (0, 1)
    mean = train_x.mean(axes)
    scale = train_x.std(axes)
    scale[scale < 1e-6] = 1.0
    seed = int(training["seed"])
    _seed(seed)
    model: nn.Module = PooledReliability() if pooled else SetAuditor()
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    batch_size = int(training["batch_size"])
    history: list[float] = []
    started = time.perf_counter()
    for epoch in range(int(training["epochs"])):
        order = np.random.default_rng(seed + epoch).permutation(len(train_x))
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            value = torch.from_numpy((train_x[selected] - mean) / scale).to(device)
            target_risk = torch.from_numpy(risk[selected]).to(device)
            target_arm = torch.from_numpy(arms[selected]).long().to(device)
            predicted_risk, condition_logits = model(value)
            loss = torch.nn.functional.smooth_l1_loss(predicted_risk, target_risk)
            if supervision == "dual":
                loss = loss + _condition_loss(condition_logits, target_arm)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    elapsed = time.perf_counter() - started
    model.eval()
    with torch.inference_mode():
        value = torch.from_numpy((test_x - mean) / scale).to(device)
        predicted_risk, condition_logits = model(value)
        condition_probability = torch.softmax(condition_logits, dim=1)
    return {
        "risk": predicted_risk.cpu().numpy(),
        "condition_probability": condition_probability.cpu().numpy(),
        "parameters_total": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "parameters_receiving_supervision": int(
            sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if supervision == "dual"
                or not name.startswith(("condition", "corruption"))
            )
        ),
        "train_seconds": float(elapsed),
        "history": history,
    }

"""Decision heads: reward-regression MLPs, escalation gate, binary classifier.

Paper Sec. 3.3.3: all heads are MLPs with LayerNorm, GELU and Dropout, trained
on frozen backbone features.  Small/large routers regress the two action
rewards; the gate regresses the value of escalation; the supervised baseline
uses the same MLP with BCE instead of reward regression.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import ensure_dir, get_logger

log = get_logger("dlmem.models.heads")

try:  # torch is required for learned heads only
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    TORCH_AVAILABLE = False


def require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "torch is required for learned routers; install with `pip install -r requirements.txt`"
        )


if TORCH_AVAILABLE:

    class MLP(nn.Module):
        def __init__(
            self,
            in_dim: int,
            hidden: int = 256,
            depth: int = 2,
            dropout: float = 0.1,
            out_dim: int = 2,
        ) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            dim = in_dim
            for _ in range(max(1, depth)):
                layers += [
                    nn.Linear(dim, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
                dim = hidden
            layers.append(nn.Linear(dim, out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "best_epoch": self.best_epoch,
        }


def _as_tensor(X: np.ndarray):
    require_torch()
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[:, None]
    return torch.from_numpy(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))


def _fit_loop(
    model,
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    Xval: np.ndarray | None,
    Yval: np.ndarray | None,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    patience: int,
    loss_fn,
    seed: int,
) -> TrainHistory:
    torch.manual_seed(seed)
    Xtr_t, Ytr_t = _as_tensor(Xtr), _as_tensor(Ytr)
    if Xval is not None and len(Xval):
        Xval_t, Yval_t = _as_tensor(Xval), _as_tensor(Yval)
    else:
        Xval_t, Yval_t = None, None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = TrainHistory()
    best_val = float("inf")
    best_state = None
    n = Xtr_t.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = Xtr_t[idx], Ytr_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss.item())
            batches += 1
        history.train_loss.append(total / max(1, batches))
        if Xval_t is not None:
            model.eval()
            with torch.no_grad():
                val = float(loss_fn(model(Xval_t), Yval_t).item())
            history.val_loss.append(val)
            if val < best_val - 1e-6:
                best_val = val
                history.best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif epoch - history.best_epoch >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def train_reward_head(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    Xval: np.ndarray | None,
    Yval: np.ndarray | None,
    cfg: dict,
    loss_type: str = "mse",
    seed: int = 13,
):
    """Train a two-output reward head.  Y columns are r(W), r(D)."""
    require_torch()
    model = MLP(
        in_dim=int(Xtr.shape[1]),
        hidden=int(cfg.get("hidden", 256)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        out_dim=2,
    )
    if loss_type == "mse":

        def loss_fn(pred, target):
            return nn.functional.mse_loss(pred, target)

    elif loss_type == "margin_ranking":

        def loss_fn(pred, target):
            # target[:, 0] > 0 marks write-worthy facts
            sign = torch.where(target[:, 0] > 0, 1.0, -1.0)
            diff = pred[:, 0] - pred[:, 1]
            margin = float(cfg.get("margin", 0.1))
            return torch.relu(margin - sign * diff).mean()

    else:
        raise ValueError(f"unknown loss {loss_type}")
    history = _fit_loop(
        model,
        Xtr,
        Ytr,
        Xval,
        Yval,
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-2)),
        epochs=int(cfg.get("epochs", 40)),
        batch_size=int(cfg.get("batch_size", 256)),
        patience=int(cfg.get("patience", 8)),
        loss_fn=loss_fn,
        seed=seed,
    )
    return model, history


def train_gate_head(
    Xtr: np.ndarray,
    gtr: np.ndarray,
    Xval: np.ndarray | None,
    gval: np.ndarray | None,
    cfg: dict,
    seed: int = 13,
):
    require_torch()
    model = MLP(
        in_dim=int(Xtr.shape[1]),
        hidden=int(cfg.get("hidden", 128)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        out_dim=1,
    )
    history = _fit_loop(
        model,
        Xtr,
        np.asarray(gtr, dtype=np.float32).reshape(-1, 1),
        Xval,
        None if gval is None else np.asarray(gval, dtype=np.float32).reshape(-1, 1),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-2)),
        epochs=int(cfg.get("epochs", 40)),
        batch_size=int(cfg.get("batch_size", 256)),
        patience=int(cfg.get("patience", 8)),
        loss_fn=lambda p, t: nn.functional.mse_loss(p, t),
        seed=seed,
    )
    return model, history


def train_binary_head(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xval: np.ndarray | None,
    yval: np.ndarray | None,
    cfg: dict,
    seed: int = 13,
):
    require_torch()
    model = MLP(
        in_dim=int(Xtr.shape[1]),
        hidden=int(cfg.get("hidden", 256)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        out_dim=1,
    )
    history = _fit_loop(
        model,
        Xtr,
        np.asarray(ytr, dtype=np.float32).reshape(-1, 1),
        Xval,
        None if yval is None else np.asarray(yval, dtype=np.float32).reshape(-1, 1),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-2)),
        epochs=int(cfg.get("epochs", 40)),
        batch_size=int(cfg.get("batch_size", 256)),
        patience=int(cfg.get("patience", 8)),
        loss_fn=lambda p, t: nn.functional.binary_cross_entropy_with_logits(p, t),
        seed=seed,
    )
    return model, history


def predict(model, X: np.ndarray) -> np.ndarray:
    require_torch()
    if len(X) == 0:
        return np.zeros((0, 1), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        out = model(_as_tensor(X))
        return out.detach().cpu().numpy().astype(np.float32)


def predict_rewards(model, X: np.ndarray) -> np.ndarray:
    out = predict(model, X)
    if out.shape[1] == 1:
        out = np.concatenate([out, np.zeros_like(out)], axis=1)
    return out


def predict_gate(model, X: np.ndarray) -> np.ndarray:
    return predict(model, X).reshape(-1)


def save_head(model, path: str | Path, meta: dict[str, Any] | None = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, path)
    if meta:
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def load_head(path: str | Path, cfg: dict | None = None):
    require_torch()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    meta = blob.get("meta", {})
    cfg = cfg or {}
    in_dim = int(meta.get("in_dim", cfg.get("in_dim", 0)))
    out_dim = int(meta.get("out_dim", cfg.get("out_dim", 2)))
    model = MLP(
        in_dim=in_dim,
        hidden=int(meta.get("hidden", cfg.get("hidden", 256))),
        depth=int(meta.get("depth", cfg.get("depth", 2))),
        dropout=float(meta.get("dropout", cfg.get("dropout", 0.1))),
        out_dim=out_dim,
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, meta

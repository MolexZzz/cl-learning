"""Write routers and the cost-aware small-to-large cascade (Sec. 3.3).

A router = frozen-backbone features + an MLP head that regresses
``r(W) = rho - lambda_s`` and ``r(D) = 0``.  The cascade first consults the
small router, then a gate predicts whether escalation (large router minus
``lambda_e``) is worth it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..data.features import FeatureSet, concat_features
from ..utils import ensure_dir, get_logger, read_json, write_json
from . import heads

log = get_logger("dlmem.models.router")


def build_inputs(
    sets: Sequence[FeatureSet], mode: str
) -> np.ndarray:
    """Feature matrix for one router (supports small+large concatenation)."""
    return concat_features(list(sets), mode)


def fit_quantile_calibration(scores: np.ndarray, rho: np.ndarray) -> dict:
    """Map raw reward scores onto the validation rho distribution (monotone).

    Reward regression on small samples shrinks predictions toward the class
    mean, which destroys the absolute threshold that lambda_s encodes.  We
    therefore calibrate by rank: the predicted rho at a given score percentile
    is the validation rho at the same percentile.
    """
    scores = np.asarray(scores, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    if len(scores) < 8 or len(scores) != len(rho):
        return {}
    order = np.argsort(scores, kind="stable")
    return {
        "scores": [float(x) for x in scores[order]],
        "values": [float(x) for x in np.sort(rho)],
    }


def apply_quantile_calibration(scores: np.ndarray, calibration: dict) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    cal_scores = calibration.get("scores") if calibration else None
    cal_values = calibration.get("values") if calibration else None
    if not cal_scores or not cal_values:
        return np.clip(scores, 0.0, 1.0)
    ref = np.asarray(cal_scores, dtype=np.float64)
    vals = np.asarray(cal_values, dtype=np.float64)
    pct = np.searchsorted(ref, scores, side="right") / len(ref)
    grid = np.linspace(0.0, 1.0, len(vals))
    return np.clip(np.interp(pct, grid, vals), 0.0, 1.0)


@dataclass
class RouterPrediction:
    rewards: np.ndarray  # (N, 2)
    score: np.ndarray  # r(W) - r(D)
    actions: np.ndarray  # bool, True = write

    def __len__(self) -> int:
        return int(self.score.shape[0])


class Router:
    def __init__(
        self,
        head,
        feature_mode: str,
        source: str = "small",
        train_lambda_s: float = 0.2,
        history: dict | None = None,
        calibration: dict | None = None,
    ) -> None:
        self.head = head
        self.feature_mode = feature_mode
        self.source = source
        self.train_lambda_s = float(train_lambda_s)
        self.history = history or {}
        # quantile calibration of predicted rho on the validation split; small
        # samples otherwise shrink the MSE regressor toward the class mean
        self.calibration = calibration or {}

    # ------------------------------------------------------------------ #
    @classmethod
    def fit(
        cls,
        train_sets: Sequence[FeatureSet],
        val_sets: Sequence[FeatureSet],
        cfg: dict,
        lambda_s: float = 0.2,
        feature_mode: str | None = None,
        source: str = "small",
        seed: int = 13,
        loss: str | None = None,
    ) -> "Router":
        feature_mode = feature_mode or str(cfg.get("feature_mode", "semantic_uncertainty"))
        Xtr = build_inputs(train_sets, feature_mode)
        Xval = build_inputs(val_sets, feature_mode)
        rho_tr = _rho(train_sets[0])
        rho_val = _rho(val_sets[0])
        Ytr = reward_targets(rho_tr, lambda_s)
        Yval = reward_targets(rho_val, lambda_s)
        model, history = heads.train_reward_head(
            Xtr,
            Ytr,
            Xval,
            Yval,
            cfg,
            loss_type=loss or str(cfg.get("loss", "mse")),
            seed=seed,
        )
        val_rewards = heads.predict_rewards(model, Xval)
        raw_rho = val_rewards[:, 0] - val_rewards[:, 1] + lambda_s
        rho_val = _rho(val_sets[0])
        calibration = fit_quantile_calibration(raw_rho, rho_val)
        meta = {
            "in_dim": int(Xtr.shape[1]),
            "out_dim": 2,
            "hidden": int(cfg.get("hidden", 256)),
            "depth": int(cfg.get("depth", 2)),
            "dropout": float(cfg.get("dropout", 0.1)),
            "feature_mode": feature_mode,
            "source": source,
            "train_lambda_s": float(lambda_s),
            "loss": loss or str(cfg.get("loss", "mse")),
            "calibration": calibration,
        }
        return cls(
            model,
            feature_mode,
            source,
            lambda_s,
            history.to_dict() | {"meta": meta},
            calibration=calibration,
        )

    # ------------------------------------------------------------------ #
    def predict_sets(self, sets: Sequence[FeatureSet]) -> RouterPrediction:
        X = build_inputs(sets, self.feature_mode)
        rewards = heads.predict_rewards(self.head, X)
        raw_rho = rewards[:, 0] - rewards[:, 1] + self.train_lambda_s
        rho = apply_quantile_calibration(raw_rho, self.calibration)
        score = rho - self.train_lambda_s
        rewards = np.stack([score, np.zeros_like(score)], axis=1)
        return RouterPrediction(rewards=rewards, score=score, actions=score > 0)

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        meta = dict(self.history.get("meta", {}))
        meta.update(
            {
                "feature_mode": self.feature_mode,
                "source": self.source,
                "train_lambda_s": self.train_lambda_s,
                "calibration": self.calibration,
            }
        )
        if extra:
            meta.update(extra)
        heads.save_head(self.head, path, meta)

    @classmethod
    def load(cls, path: str | Path, cfg: dict | None = None) -> "Router":
        model, meta = heads.load_head(path, cfg)
        return cls(
            model,
            feature_mode=str(meta.get("feature_mode", "semantic_uncertainty")),
            source=str(meta.get("source", "small")),
            train_lambda_s=float(meta.get("train_lambda_s", 0.2)),
            calibration=meta.get("calibration") or {},
        )


def reward_targets(rho: np.ndarray, lambda_s: float) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float32)
    return np.stack([rho - lambda_s, np.zeros_like(rho)], axis=1)


def _rho(fs: FeatureSet) -> np.ndarray:
    if fs.rho is None:
        raise ValueError("FeatureSet.rho missing; run feature extraction with labels")
    return fs.rho


def gate_features(
    small_sets: Sequence[FeatureSet],
    small_pred: RouterPrediction,
    mode: str = "full",
) -> np.ndarray:
    """Input vector z_g of Eq. 6 and its ablations."""
    sem = build_inputs(small_sets, "semantic")
    unc = build_inputs(small_sets, "uncertainty")
    rw = small_pred.rewards[:, :1]
    rd = small_pred.rewards[:, 1:2]
    delta = np.abs(rw - rd)
    score_block = np.concatenate([rw, rd, delta], axis=1)
    if mode in {"full", "semantic_uncertainty_scores"}:
        return np.concatenate([sem, unc, score_block], axis=1)
    if mode in {"semantic_uncertainty", "both"}:
        return np.concatenate([sem, unc], axis=1)
    if mode == "semantic":
        return sem
    if mode in {"uncertainty", "uncertainty_only"}:
        return unc
    if mode == "score_only":
        return score_block
    raise ValueError(f"unknown gate feature mode {mode!r}")


def gate_targets(
    rho: np.ndarray,
    small_actions: np.ndarray,
    large_actions: np.ndarray,
    lambda_s: float,
    lambda_e: float,
) -> np.ndarray:
    """g* = V_large - V_small, with V = r(a) and escalation cost lambda_e."""
    rho = np.asarray(rho, dtype=np.float32)
    rw = rho - lambda_s
    r_small = np.where(small_actions, rw, 0.0)
    r_large = np.where(large_actions, rw, 0.0)
    return (r_large - lambda_e - r_small).astype(np.float32)


class Gate:
    def __init__(self, head, feature_mode: str, lambda_e: float = 0.008) -> None:
        self.head = head
        self.feature_mode = feature_mode
        self.lambda_e = float(lambda_e)

    @classmethod
    def fit(
        cls,
        small_sets: Sequence[FeatureSet],
        small_router: Router,
        large_router: Router,
        large_sets: Sequence[FeatureSet],
        cfg: dict,
        lambda_e: float = 0.008,
        feature_mode: str | None = None,
        seed: int = 13,
    ) -> "Gate":
        feature_mode = feature_mode or str(cfg.get("feature_mode", "full"))
        small_pred = small_router.predict_sets(small_sets)
        large_pred = large_router.predict_sets(large_sets)
        rho = _rho(small_sets[0])
        g = gate_targets(
            rho, small_pred.actions, large_pred.actions, small_router.train_lambda_s, lambda_e
        )
        X = gate_features(small_sets, small_pred, feature_mode)
        rng = np.random.default_rng(seed)
        # Escalation gains are rare (small router is usually right).  Oversample
        # positive-gain rows so the gate does not collapse to a constant
        # negative predictor.
        pos = np.flatnonzero(g > 1e-6)
        if len(pos):
            reps = int(min(20, max(1, 0.25 * len(X) / len(pos))))
            if reps > 1:
                X = np.vstack([X, np.repeat(X[pos], reps - 1, axis=0)])
                g = np.concatenate([g, np.repeat(g[pos], reps - 1)])
        perm = rng.permutation(len(X))
        X, g = X[perm], g[perm]
        n = len(X)
        val_n = max(1, int(0.15 * n)) if n >= 20 else 0
        if val_n:
            Xtr, Xval = X[:-val_n], X[-val_n:]
            gtr, gval = g[:-val_n], g[-val_n:]
        else:
            Xtr, Xval, gtr, gval = X, None, g, None
        model, history = heads.train_gate_head(Xtr, gtr, Xval, gval, cfg, seed=seed)
        gate = cls(model, feature_mode, lambda_e)
        gate.history = history.to_dict() | {
            "meta": {
                "in_dim": int(X.shape[1]),
                "out_dim": 1,
                "hidden": int(cfg.get("hidden", 128)),
                "depth": int(cfg.get("depth", 2)),
                "dropout": float(cfg.get("dropout", 0.1)),
                "feature_mode": feature_mode,
                "lambda_e": float(lambda_e),
            }
        }
        return gate

    def predict(self, small_sets: Sequence[FeatureSet], small_pred: RouterPrediction) -> np.ndarray:
        X = gate_features(small_sets, small_pred, self.feature_mode)
        return heads.predict_gate(self.head, X)

    def save(self, path: str | Path) -> None:
        heads.save_head(self.head, path, self.history.get("meta", {}))

    @classmethod
    def load(cls, path: str | Path, cfg: dict | None = None) -> "Gate":
        model, meta = heads.load_head(path, cfg)
        gate = cls(model, str(meta.get("feature_mode", "full")), float(meta.get("lambda_e", 0.008)))
        gate.history = {"meta": meta}
        return gate


@dataclass
class CascadeOutput:
    write: np.ndarray
    escalated: np.ndarray
    gate_score: np.ndarray
    small_write: np.ndarray
    large_write: np.ndarray
    cost: np.ndarray


class CascadeRouter:
    def __init__(
        self,
        small: Router,
        large: Router,
        gate: Gate | None = None,
        small_cost: float = 1.0,
        large_cost: float = 4.7,
    ) -> None:
        self.small = small
        self.large = large
        self.gate = gate
        self.small_cost = float(small_cost)
        self.large_cost = float(large_cost)

    def decide(
        self,
        small_sets: Sequence[FeatureSet],
        large_sets: Sequence[FeatureSet],
        escalation_cap: float | None = None,
    ) -> CascadeOutput:
        small_pred = self.small.predict_sets(small_sets)
        large_pred = self.large.predict_sets(large_sets)
        n = len(small_pred)
        if self.gate is None:
            g = np.zeros(n, dtype=np.float32)
        else:
            g = self.gate.predict(small_sets, small_pred)
        escalate = g > 0
        if escalation_cap is not None and 0 < escalation_cap < 1:
            cap = int(round(escalation_cap * n))
            if escalate.sum() > cap:
                order = np.argsort(-g)
                escalate = np.zeros(n, dtype=bool)
                escalate[order[:cap]] = True
        write = np.where(escalate, large_pred.actions, small_pred.actions)
        cost = self.small_cost + escalate.astype(np.float32) * self.large_cost
        return CascadeOutput(
            write=write.astype(bool),
            escalated=escalate,
            gate_score=g,
            small_write=small_pred.actions,
            large_write=large_pred.actions,
            cost=cost,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        ensure_dir(path)
        self.small.save(path / "small_head.pt")
        self.large.save(path / "large_head.pt")
        if self.gate is not None:
            self.gate.save(path / "gate_head.pt")
        write_json(
            path / "cascade.json",
            {
                "small_cost": self.small_cost,
                "large_cost": self.large_cost,
                "has_gate": self.gate is not None,
            },
        )

    @classmethod
    def load(cls, path: str | Path, cfg: dict | None = None) -> "CascadeRouter":
        path = Path(path)
        meta = read_json(path / "cascade.json") if (path / "cascade.json").exists() else {}
        small = Router.load(path / "small_head.pt", cfg)
        large = Router.load(path / "large_head.pt", cfg)
        gate = Gate.load(path / "gate_head.pt", cfg) if (path / "gate_head.pt").exists() else None
        return cls(
            small,
            large,
            gate,
            small_cost=float(meta.get("small_cost", 1.0)),
            large_cost=float(meta.get("large_cost", 4.7)),
        )

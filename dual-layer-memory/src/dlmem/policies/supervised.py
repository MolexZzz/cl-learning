"""Supervised baselines: logistic regression and a BCE-trained MLP classifier.

Both operate on the *same* frozen-backbone features as the write router; the
MLP shares the large router architecture but is trained with binary
cross-entropy instead of reward regression (Appendix B).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..data.features import FeatureSet
from ..data.schema import Fact
from ..models import heads
from ..utils import get_logger
from .base import PolicyOutput, WritePolicy

log = get_logger("dlmem.policies.supervised")


class LogisticPolicy(WritePolicy):
    name = "logistic"
    cost_label = "1 x Fwd8B"

    def __init__(self, feature_mode: str = "semantic") -> None:
        self.feature_mode = feature_mode
        self.model = None
        self.scaler: tuple[np.ndarray, np.ndarray] | None = None

    def fit(self, fs: FeatureSet) -> "LogisticPolicy":
        X = fs.matrix(self.feature_mode).astype(np.float64)
        y = np.asarray([1 if lab != "non-write" else 0 for lab in fs.labels or []])
        mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-6
        Xs = (X - mu) / sigma
        try:
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
            model.fit(Xs, y)
            weights = model.coef_.reshape(-1)
            bias = float(model.intercept_[0])
            self.model = ("sklearn", model, weights, bias)
        except Exception:
            log.warning("scikit-learn unavailable; using numpy logistic regression")
            w, b = _numpy_logreg(Xs, y)
            self.model = ("numpy", None, w, b)
        self.scaler = (mu, sigma)
        return self

    def scores(self, fs: FeatureSet) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("LogisticPolicy not fitted")
        X = fs.matrix(self.feature_mode).astype(np.float64)
        mu, sigma = self.scaler
        Xs = (X - mu) / sigma
        kind, model, w, b = self.model
        if kind == "sklearn":
            return model.predict_proba(Xs)[:, 1]
        z = Xs @ w + b
        return 1.0 / (1.0 + np.exp(-z))

    def decide_all(self, facts: Sequence[Fact], fs: FeatureSet | None = None) -> PolicyOutput:
        if fs is None:
            raise ValueError("LogisticPolicy.decide_all requires aligned features")
        s = self.scores(fs)
        return PolicyOutput(self.name, s >= 0.5, score=s, cost_label=self.cost_label)


class MLPClassifierPolicy(WritePolicy):
    name = "mlp"
    cost_label = "1 x Fwd8B"

    def __init__(self, feature_mode: str = "semantic_uncertainty", cfg: dict | None = None) -> None:
        self.feature_mode = feature_mode
        self.cfg = cfg or {}
        self.model = None

    def fit(self, fs: FeatureSet) -> "MLPClassifierPolicy":
        X = fs.matrix(self.feature_mode)
        y = np.asarray([1.0 if lab != "non-write" else 0.0 for lab in fs.labels or []])
        n = len(X)
        val_n = max(1, int(0.15 * n)) if n >= 20 else 0
        if val_n:
            model, _ = heads.train_binary_head(
                X[:-val_n], y[:-val_n], X[-val_n:], y[-val_n:], self.cfg, seed=13
            )
        else:
            model, _ = heads.train_binary_head(X, y, None, None, self.cfg, seed=13)
        self.model = model
        return self

    def scores(self, fs: FeatureSet) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("MLPClassifierPolicy not fitted")
        logits = heads.predict(self.model, fs.matrix(self.feature_mode)).reshape(-1)
        return 1.0 / (1.0 + np.exp(-logits))

    def decide_all(self, facts: Sequence[Fact], fs: FeatureSet | None = None) -> PolicyOutput:
        if fs is None:
            raise ValueError("MLPClassifierPolicy.decide_all requires aligned features")
        s = self.scores(fs)
        return PolicyOutput(self.name, s >= 0.5, score=s, cost_label=self.cost_label)


class ArrayPolicy(WritePolicy):
    """Wraps precomputed decisions (used for the router / cascade)."""

    def __init__(
        self,
        name: str,
        write: np.ndarray,
        escalate: np.ndarray | None = None,
        cost: np.ndarray | None = None,
        score: np.ndarray | None = None,
        cost_label: str = "",
        meta: dict | None = None,
    ) -> None:
        self.name = name
        self._write = np.asarray(write, dtype=bool)
        self._escalate = escalate
        self._cost = cost
        self._score = score
        self.cost_label = cost_label
        self._meta = meta or {}

    def decide_all(self, facts: Sequence[Fact]) -> PolicyOutput:
        if len(facts) != len(self._write):
            raise ValueError(
                f"ArrayPolicy {self.name}: {len(self._write)} decisions for {len(facts)} facts"
            )
        return PolicyOutput(
            self.name,
            self._write,
            self._escalate,
            self._cost,
            self._score,
            self.cost_label,
            self._meta,
        )


def _numpy_logreg(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1500):
    w = np.zeros(X.shape[1], dtype=np.float64)
    b = 0.0
    n = max(1, len(X))
    for _ in range(steps):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = X.T @ (p - y) / n + 1e-4 * w
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b

"""Write policies: static, heuristic, supervised and the cascaded write router.

All policies expose ``decide_all(facts)`` returning a :class:`PolicyOutput` with
per-fact write flags, escalation flags and relative routing cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..data.schema import Fact
from ..utils import get_logger

log = get_logger("dlmem.policies")


@dataclass
class PolicyOutput:
    name: str
    write: np.ndarray
    escalate: np.ndarray | None = None
    cost: np.ndarray | None = None
    score: np.ndarray | None = None
    cost_label: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.write = np.asarray(self.write, dtype=bool)
        n = len(self.write)
        if self.escalate is None:
            self.escalate = np.zeros(n, dtype=bool)
        if self.cost is None:
            self.cost = np.zeros(n, dtype=np.float32)
        if self.score is None:
            self.score = np.zeros(n, dtype=np.float32)

    @property
    def storage_ratio(self) -> float:
        return float(self.write.mean()) if len(self.write) else 0.0

    @property
    def mean_cost(self) -> float:
        return float(np.mean(self.cost)) if len(self.cost) else 0.0

    @property
    def escalation_rate(self) -> float:
        return float(self.escalate.mean()) if len(self.escalate) else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "storage_ratio": self.storage_ratio,
            "mean_cost": self.mean_cost,
            "escalation_rate": self.escalation_rate,
            "cost_label": self.cost_label,
        }


class WritePolicy:
    name = "policy"
    cost_label = ""

    def decide_all(self, facts: Sequence[Fact]) -> PolicyOutput:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Static policies
# --------------------------------------------------------------------------- #
class NoStorePolicy(WritePolicy):
    name = "no_store"
    cost_label = "0"

    def decide_all(self, facts):
        n = len(facts)
        return PolicyOutput(self.name, np.zeros(n, dtype=bool), score=-np.ones(n), cost_label=self.cost_label)


class FullStorePolicy(WritePolicy):
    name = "full_store"
    cost_label = "0"

    def decide_all(self, facts):
        n = len(facts)
        return PolicyOutput(self.name, np.ones(n, dtype=bool), score=np.ones(n), cost_label=self.cost_label)


class RandomStorePolicy(WritePolicy):
    cost_label = "0"

    def __init__(self, p: float = 0.5, seed: int = 13) -> None:
        self.p = float(p)
        self.name = f"random_{p:g}"
        self.seed = seed

    def decide_all(self, facts):
        rng = np.random.default_rng(self.seed)
        u = rng.random(len(facts))
        write = u < self.p
        return PolicyOutput(
            self.name, write, score=-u, cost_label=self.cost_label, meta={"p": self.p}
        )


# --------------------------------------------------------------------------- #
# Threshold utilities for score-based policies
# --------------------------------------------------------------------------- #
def threshold_for_storage(scores: np.ndarray, storage: float) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        return 0.0
    storage = min(max(storage, 0.0), 1.0)
    q = float(np.quantile(scores, 1.0 - storage))
    return q


def decide_by_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(scores) >= threshold

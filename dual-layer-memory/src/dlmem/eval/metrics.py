"""Evaluation metrics: QA quality, storage decision quality, routing cost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..data.schema import Fact
from ..utils import exact_match, is_refusal, token_f1


@dataclass
class StoreCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def storage(self) -> float:
        return (self.tp + self.fp) / self.n if self.n else 0.0

    def to_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "storage": self.storage,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def storage_counts(facts: Sequence[Fact], write: Sequence[bool]) -> StoreCounts:
    counts = StoreCounts()
    for fact, w in zip(facts, write):
        positive = fact.label != "non-write"
        if w and positive:
            counts.tp += 1
        elif w and not positive:
            counts.fp += 1
        elif not w and positive:
            counts.fn += 1
        else:
            counts.tn += 1
    return counts


def expected_em(facts: Sequence[Fact], write: Sequence[bool]) -> float:
    """Probe-weighted QA EM under an ideal retriever.

    Stored facts yield EM = (m - unsolved)/m, discarded facts EM = internal/m.
    With Full Store this equals the paper's upper bound; with No Store it equals
    the zero-shot baseline (see Table 1: 87.79% and 38.02%).
    """
    total = 0.0
    weight = 0
    for fact, w in zip(facts, write):
        em = fact.em_if_stored() if w else fact.em_if_discarded()
        total += em * fact.m
        weight += fact.m
    return total / weight if weight else 0.0


def qa_scores(predictions: Sequence[str], golds: Sequence[str]) -> dict:
    n = len(predictions)
    if n == 0:
        return {"n": 0, "em": 0.0, "f1": 0.0, "refusal_rate": 0.0}
    ems = np.asarray([exact_match(p, g) for p, g in zip(predictions, golds)])
    f1s = np.asarray([token_f1(p, g) for p, g in zip(predictions, golds)])
    refusals = np.asarray([is_refusal(p) for p in predictions])
    return {
        "n": n,
        "em": float(ems.mean()),
        "f1": float(f1s.mean()),
        "refusal_rate": float(refusals.mean()),
    }


def routing_summary(output) -> dict:
    return {
        "storage": float(output.write.mean()) if len(output.write) else 0.0,
        "escalation_rate": float(output.escalate.mean()) if len(output.escalate) else 0.0,
        "mean_cost": float(np.mean(output.cost)) if len(output.cost) else 0.0,
        "cost_label": output.cost_label,
    }


def aggregate(metrics_list: Sequence[dict]) -> dict:
    """Weighted aggregation over episodes (all episodes have equal weight here)."""
    keys = set().union(*[m.keys() for m in metrics_list]) if metrics_list else set()
    out: dict = {}
    for key in keys:
        vals = [m[key] for m in metrics_list if isinstance(m.get(key), (int, float))]
        if vals:
            out[key] = float(np.mean(vals))
    return out

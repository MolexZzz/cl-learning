"""Offline EM-storage-compute sweeps (Pareto frontiers, Appendix D).

Because a stored fact is assumed retrievable, the offline QA estimate is exact
under the labelling model: EM(stored) = (m - unsolved)/m, EM(discarded) =
internal/m.  This matches the paper's Full Store / No Store endpoints and lets
us trace frontiers cheaply; the streaming benchmark (``eval/stream.py``)
provides the end-to-end sanity check with real generation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..data.schema import Fact
from ..models.router import RouterPrediction
from .metrics import storage_counts, expected_em


def router_write(pred: RouterPrediction, train_lambda_s: float, lambda_s: float) -> np.ndarray:
    """Shift the reward threshold: write iff predicted rho > lambda_s."""
    rho_hat = pred.score + float(train_lambda_s)
    return rho_hat > float(lambda_s)


def cascade_write(
    small_pred: RouterPrediction,
    large_pred: RouterPrediction,
    gate_score: np.ndarray,
    lambda_s: float,
    lambda_e: float,
    train_lambda_s: float,
    train_lambda_e: float,
    escalation_cap: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    small_w = router_write(small_pred, train_lambda_s, lambda_s)
    large_w = router_write(large_pred, train_lambda_s, lambda_s)
    g = np.asarray(gate_score, dtype=np.float64) - (float(lambda_e) - float(train_lambda_e))
    escalate = g > 0
    if escalation_cap is not None and 0 < escalation_cap < 1:
        cap = int(round(escalation_cap * len(g)))
        if escalate.sum() > cap:
            order = np.argsort(-g)
            escalate = np.zeros(len(g), dtype=bool)
            escalate[order[:cap]] = True
    write = np.where(escalate, large_w, small_w)
    return write, escalate


def point_metrics(facts: Sequence[Fact], write: Sequence[bool], cost: float = 0.0) -> dict:
    counts = storage_counts(facts, write)
    return {
        "storage": counts.storage,
        "precision": counts.precision,
        "recall": counts.recall,
        "f1": counts.f1,
        "em": expected_em(facts, write),
        "cost": float(cost),
    }


def sweep_single(
    facts: Sequence[Fact],
    pred: RouterPrediction,
    train_lambda_s: float,
    lambdas: Sequence[float],
    cost: float,
) -> list[dict]:
    rows = []
    for lam in lambdas:
        write = router_write(pred, train_lambda_s, lam)
        row = point_metrics(facts, write, cost)
        row.update({"lambda_s": float(lam), "kind": "single"})
        rows.append(row)
    return rows


def sweep_cascade(
    facts: Sequence[Fact],
    small_pred: RouterPrediction,
    large_pred: RouterPrediction,
    gate_score: np.ndarray,
    lambda_s_values: Sequence[float],
    lambda_e: float,
    train_lambda_s: float,
    train_lambda_e: float,
    escalation_cap: float | None,
    small_cost: float = 1.0,
    large_cost: float = 4.7,
) -> list[dict]:
    rows = []
    for lam in lambda_s_values:
        write, escalate = cascade_write(
            small_pred,
            large_pred,
            gate_score,
            lam,
            lambda_e,
            train_lambda_s,
            train_lambda_e,
            escalation_cap,
        )
        cost = small_cost + float(escalate.mean()) * large_cost
        row = point_metrics(facts, write, cost)
        row.update(
            {
                "lambda_s": float(lam),
                "lambda_e": float(lambda_e),
                "cap": escalation_cap,
                "escalation_rate": float(escalate.mean()),
                "kind": "cascade",
            }
        )
        rows.append(row)
    return rows


def interp_to_storage_grid(rows: Sequence[dict], grid: Sequence[float]) -> list[dict]:
    """Interpolate a swept frontier onto fixed storage budgets (for plotting)."""
    ordered = sorted(rows, key=lambda r: r["storage"])
    xs = np.asarray([r["storage"] for r in ordered], dtype=np.float64)
    ys = np.asarray([r["em"] for r in ordered], dtype=np.float64)
    out = []
    for g in grid:
        if xs.min() <= g <= xs.max():
            em = float(np.interp(g, xs, ys))
            out.append({"storage": float(g), "em": em})
    return out


def frontier_gap(rows_a: Sequence[dict], rows_b: Sequence[dict], grid: Sequence[float]) -> dict:
    a = {r["storage"]: r["em"] for r in interp_to_storage_grid(rows_a, grid)}
    b = {r["storage"]: r["em"] for r in interp_to_storage_grid(rows_b, grid)}
    common = sorted(set(a) & set(b))
    if not common:
        return {"mean_gap": 0.0, "max_gap": 0.0, "n": 0}
    diffs = [a[s] - b[s] for s in common]
    return {
        "mean_gap": float(np.mean(diffs)),
        "max_gap": float(np.max(np.abs(diffs))),
        "n": len(common),
        "by_storage": {s: float(a[s] - b[s]) for s in common},
    }

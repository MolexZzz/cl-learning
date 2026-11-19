from __future__ import annotations

import numpy as np

from dlmem.data.features import FeatureSet
from dlmem.eval.sweep import cascade_write, router_write
from dlmem.models.router import (
    CascadeRouter,
    Gate,
    Router,
    RouterPrediction,
    apply_quantile_calibration,
    fit_quantile_calibration,
    gate_targets,
)


def _features(n=240, dim=12, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.array(["non-write", "write-new", "write-update"][:1] * 0)  # placeholder
    rho = np.where(rng.random(n) < 0.25, 0.0, rng.uniform(0.3, 0.9, n)).astype(np.float32)
    sem = rng.standard_normal((n, dim)).astype(np.float32)
    sem[rho > 0] += 0.8
    unc = rng.standard_normal((n, 4)).astype(np.float32)
    unc[:, 0] += 2.0 * (rho > 0)
    labels = ["non-write" if r < 0.01 else "write-new" for r in rho]
    return FeatureSet(
        ids=[f"f{i}" for i in range(n)], sem=sem, unc=unc, labels=labels, rho=rho
    )


def test_quantile_calibration_monotone():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    rho = np.array([0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9])
    cal = fit_quantile_calibration(scores, rho)
    out = apply_quantile_calibration(scores, cal)
    assert np.all(np.diff(out) >= -1e-8)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_router_training_and_cascade():
    train = _features(240, seed=1)
    val = _features(80, seed=2)
    cfg = {"hidden": 32, "depth": 1, "dropout": 0.0, "epochs": 8, "batch_size": 64, "lr": 0.01}
    router = Router.fit([train], [val], cfg, lambda_s=0.2, feature_mode="semantic_uncertainty")
    pred = router.predict_sets([val])
    assert pred.rewards.shape == (80, 2)
    assert set(np.unique(pred.actions)).issubset({True, False})
    assert router.calibration
    # calibrated predictions should separate the two groups on average
    rho_hat = pred.score + router.train_lambda_s
    assert rho_hat[val.rho > 0].mean() > rho_hat[val.rho == 0].mean()

    large = Router.fit([train], [val], cfg, lambda_s=0.2, feature_mode="semantic")
    gate_cfg = {"hidden": 16, "depth": 1, "dropout": 0.0, "epochs": 8, "batch_size": 64, "lr": 0.01}
    gate = Gate.fit([train], router, large, [train], gate_cfg, lambda_e=0.008, feature_mode="full")
    cascade = CascadeRouter(router, large, gate)
    out = cascade.decide([val], [val], escalation_cap=0.3)
    assert out.escalated.mean() <= 0.3 + 1e-9
    assert out.write.shape == (80,)
    assert np.all(out.cost >= 1.0)


def test_router_save_load(tmp_path):
    train = _features(120, seed=3)
    cfg = {"hidden": 16, "depth": 1, "dropout": 0.0, "epochs": 4, "batch_size": 32, "lr": 0.01}
    router = Router.fit([train], [train], cfg, lambda_s=0.2, feature_mode="semantic")
    router.save(tmp_path / "head.pt")
    loaded = Router.load(tmp_path / "head.pt", cfg)
    a = router.predict_sets([train]).score
    b = loaded.predict_sets([train]).score
    assert np.allclose(a, b, atol=1e-5)


def test_gate_targets_and_cap():
    rho = np.array([0.8, 0.0, 0.8])
    small = np.array([False, True, True])
    large = np.array([True, False, True])
    g = gate_targets(rho, small, large, lambda_s=0.2, lambda_e=0.01)
    # fact 0: small wrong, large right -> 0.6 - 0.01
    assert np.isclose(g[0], 0.6 - 0.01)
    # fact 1: small false positive, large right -> 0.2 - 0.01
    assert np.isclose(g[1], 0.2 - 0.01)
    # fact 2: no gain -> -0.01
    assert np.isclose(g[2], -0.01)

    n = 100
    sp = RouterPrediction(
        rewards=np.stack([np.full(n, 0.5), np.zeros(n)], 1),
        score=np.full(n, 0.5),
        actions=np.ones(n, dtype=bool),
    )
    lp = RouterPrediction(
        rewards=np.stack([np.full(n, 0.5), np.zeros(n)], 1),
        score=np.full(n, 0.5),
        actions=np.ones(n, dtype=bool),
    )
    gate_score = np.linspace(-1, 1, n)
    write, escalate = cascade_write(sp, lp, gate_score, 0.2, 0.01, 0.2, 0.01, escalation_cap=0.25)
    assert abs(escalate.mean() - 0.25) < 1e-9
    assert write.all()
    assert router_write(sp, 0.2, 0.2).sum() == n

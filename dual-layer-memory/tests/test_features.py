from __future__ import annotations

import numpy as np

from dlmem.data.features import (
    FeatureSet,
    interp_curve,
    load_features,
    save_features,
    uncertainty_vector,
)


def test_interp_and_uncertainty_shape():
    curve = np.linspace(1.0, 3.0, 10)
    out = interp_curve(curve, 64)
    assert out.shape == (64,)
    assert np.isclose(out[0], 1.0) and np.isclose(out[-1], 3.0)
    assert interp_curve([2.0], 8).tolist() == [2.0] * 8


def test_uncertainty_vector_uses_mean_and_length():
    class Enc:
        nll_curve = np.array([1.0, 2.0, 3.0])
        mean_nll = 2.0
        length = 3

    u = uncertainty_vector(Enc(), interp_len=8)
    assert u.shape == (10,)
    assert np.isclose(u[0], 2.0)
    assert np.isclose(u[1], np.log1p(3))


def test_feature_set_modes_and_roundtrip(tmp_path):
    n = 12
    sem = np.random.default_rng(0).standard_normal((n, 32)).astype(np.float32)
    unc = np.random.default_rng(1).standard_normal((n, 18)).astype(np.float32)
    fs = FeatureSet(
        ids=[f"f{i}" for i in range(n)],
        sem=sem,
        unc=unc,
        labels=["non-write"] * 4 + ["write-new"] * 8,
        rho=np.array([0.0] * 4 + [0.6] * 8, dtype=np.float32),
    )
    assert fs.matrix("semantic").shape[1] == 32
    assert fs.matrix("uncertainty").shape[1] == 18
    assert fs.matrix("semantic_uncertainty").shape[1] == 50
    assert len(fs) == n

    path = tmp_path / "feat.npz"
    save_features(path, fs, meta={"x": 1})
    loaded = load_features(path)
    assert loaded.ids == fs.ids
    assert np.allclose(loaded.sem, fs.sem)
    assert np.allclose(loaded.unc, fs.unc)
    assert loaded.rho is not None and np.allclose(loaded.rho, fs.rho)

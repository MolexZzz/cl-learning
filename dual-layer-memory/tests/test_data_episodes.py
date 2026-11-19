from __future__ import annotations

import collections

import numpy as np
import pytest

from dlmem.data.episodes import build_episodes, synthesize_episode, validate_episode
from factories import make_facts


def test_episode_invariants():
    for seed in range(10):
        ep = synthesize_episode(list(range(6)), m=30, turns=60, rng=np.random.default_rng(seed))
        validate_episode(ep, [30] * 6, 60)
        assert len(ep.queries) == 60 - len(ep.injections) == 54
        assert len(ep.injections) == 6
        counts = collections.Counter(slot for slot, _ in ep.queries.values())
        assert all(c >= 1 for c in counts.values())


def test_episode_determinism():
    a = synthesize_episode(list(range(5)), m=30, turns=40, rng=np.random.default_rng(7))
    b = synthesize_episode(list(range(5)), m=30, turns=40, rng=np.random.default_rng(7))
    assert a.injections == b.injections
    assert a.queries == b.queries


def test_causality_check_detects_bad_episode():
    ep = synthesize_episode(list(range(4)), m=30, turns=32, rng=np.random.default_rng(0))
    slot = next(slot for turn, slot in ep.injections.items() if turn > 0)
    inj_turn = next(turn for turn, s in ep.injections.items() if s == slot)
    ep.queries[inj_turn - 1] = (slot, 0)
    try:
        validate_episode(ep, [30] * 4, 32)
    except AssertionError:
        return
    raise AssertionError("validation failed to catch causality violation")


def test_build_episodes_split_local_indices():
    facts = make_facts(80)
    for i, f in enumerate(facts):
        f.split = "test" if i < 24 else "train"
        f.label = ["non-write", "write-new", "write-update"][i % 3]
    eps = build_episodes(
        facts, episodes=2, turns=40, facts_per_episode=8, seed=11, split="test", m=30
    )
    assert len(eps) == 2
    for ep in eps:
        assert len(ep.fact_indices) == 8
        assert 0 <= min(ep.fact_indices) and max(ep.fact_indices) < 24
        validate_episode(ep, [30] * 8, 40)


def test_episode_rejects_infeasible_turn_count():
    with pytest.raises(ValueError, match="infeasible"):
        synthesize_episode(list(range(4)), m=2, turns=8, rng=np.random.default_rng(0))

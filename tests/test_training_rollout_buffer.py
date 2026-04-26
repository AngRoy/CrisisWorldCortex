"""Tests for ``training.rollout_buffer``.

Round-trip add/get, group-sampling contract, clear-empties-buffer.
"""

from __future__ import annotations

import random

import pytest

from training.rollout_buffer import RolloutBuffer, TrajectoryStep


def _make_step(reward: float = 0.5, done: bool = False) -> TrajectoryStep:
    return TrajectoryStep(
        obs={"tick": 0, "regions": []},
        action={"kind": "no_op"},
        reward=reward,
        log_prob=None,
        done=done,
    )


def test_rollout_buffer_round_trip() -> None:
    """add_step → get_episode round-trips without data loss."""
    buf = RolloutBuffer()
    s1 = _make_step(reward=0.3)
    s2 = _make_step(reward=0.7, done=True)
    buf.add_step("ep-A", s1)
    buf.add_step("ep-A", s2)
    out = buf.get_episode("ep-A")
    assert out == [s1, s2]
    assert buf.get_episode("missing-id") == []


def test_rollout_buffer_episode_return_sums_rewards() -> None:
    """episode_return == sum of step rewards."""
    buf = RolloutBuffer()
    buf.add_step("ep", _make_step(reward=0.1))
    buf.add_step("ep", _make_step(reward=-0.2))
    buf.add_step("ep", _make_step(reward=0.5))
    assert abs(buf.episode_return("ep") - 0.4) < 1e-9


def test_rollout_buffer_sample_group_is_deterministic_with_seed() -> None:
    """Same seed → same sample; different seed → likely different sample.

    Locks GRPO group-sampling determinism for trainer reproducibility.
    """
    buf = RolloutBuffer()
    for i in range(8):
        buf.add_step(f"ep-{i}", _make_step())
    rng_a = random.Random(42)
    rng_b = random.Random(42)
    sample_a = sorted(buf.sample_group(4, rng=rng_a))
    sample_b = sorted(buf.sample_group(4, rng=rng_b))
    assert sample_a == sample_b
    assert len(set(sample_a)) == 4  # no duplicates


def test_rollout_buffer_sample_group_oversize_raises() -> None:
    """Requesting more episodes than buffer holds → ValueError."""
    buf = RolloutBuffer()
    buf.add_step("ep-0", _make_step())
    with pytest.raises(ValueError, match="exceeds buffer size"):
        buf.sample_group(5)


def test_rollout_buffer_clear_empties_state() -> None:
    """clear() empties the buffer; len → 0."""
    buf = RolloutBuffer()
    for i in range(3):
        buf.add_step(f"ep-{i}", _make_step())
    assert len(buf) == 3
    buf.clear()
    assert len(buf) == 0
    assert buf.episode_ids() == []

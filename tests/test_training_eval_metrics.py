"""Tests for ``training.eval_metrics``.

``collapse_rate`` is fully exercised against synthetic B1-style
trajectories; the three Cortex-dependent stubs (``dissent_value``,
``consensus_calibration``, ``novelty_yield``) get smoke checks that
assert their Phase-2 placeholder return value of 0.0.
"""

from __future__ import annotations

from training.eval_metrics import (
    collapse_rate,
    consensus_calibration,
    dissent_value,
    novelty_yield,
)


def _trajectory(action_kinds: list[str]) -> list[dict]:
    """Build a synthetic B1-style trajectory from a list of action kinds."""
    return [{"action": {"kind": k}} for k in action_kinds]


def test_collapse_rate_zero_when_actions_diverse() -> None:
    """Diverse trajectory (no modal action >= 80%) → 0.0."""
    trajs = [_trajectory(["no_op", "deploy_resource", "restrict_movement", "escalate"])]
    assert collapse_rate(trajs) == 0.0


def test_collapse_rate_one_when_all_actions_identical() -> None:
    """All-NoOp 12-tick episode → fully collapsed."""
    trajs = [_trajectory(["no_op"] * 12)]
    assert collapse_rate(trajs) == 1.0


def test_collapse_rate_partial_for_mixed_episodes() -> None:
    """Two episodes, one collapsed and one diverse → 0.5."""
    collapsed = _trajectory(["no_op"] * 10)
    diverse = _trajectory(["no_op", "deploy_resource", "restrict_movement", "no_op"])
    assert collapse_rate([collapsed, diverse]) == 0.5


def test_collapse_rate_short_episodes_excluded() -> None:
    """Episodes shorter than COLLAPSE_MIN_STEPS don't qualify."""
    trajs = [_trajectory(["no_op"] * 2)]  # below 3-step minimum
    assert collapse_rate(trajs) == 0.0


def test_cortex_dependent_metrics_return_zero_in_phase_2() -> None:
    """dissent/consensus/novelty stubs return 0.0 until Cortex Session 13."""
    trajs = [_trajectory(["no_op", "deploy_resource", "no_op"])]
    assert dissent_value(trajs) == 0.0
    assert consensus_calibration(trajs) == 0.0
    assert novelty_yield(trajs) == 0.0

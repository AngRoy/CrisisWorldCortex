"""Tests for ``training.reward_shaping``.

Token-budget penalty composition + episode-return terminal bonus.
"""

from __future__ import annotations

import pytest

from training.reward_shaping import (
    DEFAULT_LAMBDA_BUDGET,
    DEFAULT_TICK_BUDGET,
    compose_episode_return,
    shape_reward,
)


def test_shape_reward_zero_tokens_returns_outer_unchanged() -> None:
    """tokens_used=0 → r_total == r_outer (no penalty)."""
    assert shape_reward(0.7, tokens_used=0) == 0.7
    assert shape_reward(-0.3, tokens_used=0) == -0.3


def test_shape_reward_full_budget_applies_full_penalty() -> None:
    """tokens_used == tick_budget → penalty == lambda_budget."""
    r_outer = 1.0
    out = shape_reward(r_outer, tokens_used=DEFAULT_TICK_BUDGET)
    expected = r_outer - DEFAULT_LAMBDA_BUDGET * 1.0
    assert abs(out - expected) < 1e-9


def test_shape_reward_partial_budget_scales_linearly() -> None:
    """Half-budget consumption → half the lambda penalty."""
    r_outer = 0.5
    half = DEFAULT_TICK_BUDGET // 2
    out = shape_reward(r_outer, tokens_used=half, lambda_budget=0.4)
    expected = 0.5 - 0.4 * 0.5
    assert abs(out - expected) < 1e-9


def test_shape_reward_invalid_tick_budget_raises() -> None:
    """tick_budget <= 0 → ValueError (guard against div-by-zero)."""
    with pytest.raises(ValueError, match="positive"):
        shape_reward(0.5, tokens_used=10, tick_budget=0)


def test_compose_episode_return_terminal_bonus_success() -> None:
    """Success terminal adds +0.20 to summed rewards."""
    assert abs(compose_episode_return([0.1, 0.2, 0.3], "success") - 0.80) < 1e-9


def test_compose_episode_return_terminal_bonus_failure_and_neutral() -> None:
    """Failure → -0.20; timeout/none → no bonus."""
    base = [0.1, 0.1, 0.1]
    assert abs(compose_episode_return(base, "failure") - 0.10) < 1e-9
    assert abs(compose_episode_return(base, "timeout") - 0.30) < 1e-9
    assert abs(compose_episode_return(base, "none") - 0.30) < 1e-9

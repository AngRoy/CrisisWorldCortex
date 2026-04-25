"""Terminal bonus: +0.20 / -0.20 / 0.0 per design §14.3 / §15.

The bonus is a separate function from ``outer_reward`` so the per-tick
scalar stays inside ``[0.0, 1.0]``. The trainer composes them downstream
per design §14.3: ``episode_return = Σ_t r_outer + terminal_bonus``.

Covers each of the four ``state.terminal`` literal values plus the
exact-magnitude check, and confirms ``outer_reward`` itself is NOT
shifted by the terminal flag (separation-of-concerns invariant).
"""

from CrisisWorldCortex.models import NoOp
from CrisisWorldCortex.server.graders import outer_reward, terminal_bonus
from CrisisWorldCortex.server.graders.outer_reward import (
    TERMINAL_BONUS_FAILURE,
    TERMINAL_BONUS_SUCCESS,
)
from CrisisWorldCortex.server.simulator import load_task


def test_terminal_bonus_success_is_positive_020() -> None:
    state = load_task("outbreak_easy", episode_seed=0)
    state.terminal = "success"
    assert terminal_bonus(state) == TERMINAL_BONUS_SUCCESS == 0.20


def test_terminal_bonus_failure_is_negative_020() -> None:
    state = load_task("outbreak_easy", episode_seed=0)
    state.terminal = "failure"
    assert terminal_bonus(state) == TERMINAL_BONUS_FAILURE == -0.20


def test_terminal_bonus_timeout_is_zero() -> None:
    state = load_task("outbreak_easy", episode_seed=0)
    state.terminal = "timeout"
    assert terminal_bonus(state) == 0.0


def test_terminal_bonus_none_is_zero() -> None:
    """Mid-episode (``terminal == 'none'``) carries no bonus."""
    state = load_task("outbreak_easy", episode_seed=0)
    assert state.terminal == "none"
    assert terminal_bonus(state) == 0.0


def test_outer_reward_not_shifted_by_terminal_flag() -> None:
    """Separation invariant: the terminal flag must not bleed into the
    per-tick ``outer_reward`` value, only the bonus.

    Build two states with identical SEIR/resources/action-log state
    differing ONLY in ``state.terminal``. ``outer_reward`` must match;
    only ``terminal_bonus`` differs.
    """
    s1 = load_task("outbreak_medium", episode_seed=0)
    s2 = load_task("outbreak_medium", episode_seed=0)
    s1.terminal = "none"
    s2.terminal = "success"

    a = NoOp()
    assert outer_reward(s1, a) == outer_reward(s2, a), (
        "outer_reward must be invariant under state.terminal — bonus is "
        "composed by trainer, not bundled into the per-tick scalar"
    )
    assert terminal_bonus(s1) == 0.0
    assert terminal_bonus(s2) == 0.20

"""Training reward composition (design §17 + Phase-1 outer-reward range).

Composes the env-side ``outer_reward`` (post-Phase-1 range ``[-1.0, 1.0]``)
with a token-budget penalty into the per-tick training reward used by
GRPO:

    r_total = r_outer - lambda_budget * (tokens_used / TICK_BUDGET)

The terminal bonus (±0.20 per design §14.3) is added once per episode by
``compose_episode_return``, NOT folded into per-tick ``r_total``.

Phase-2 scaffold (Workstream B). Trainer is responsible for clamping if
it needs a bounded range; this module returns the raw composition.

Allowed under ``training/CLAUDE.md``: ``models``, stdlib, and
``server.graders`` reward-name constants. No ``server.simulator``, no
``cortex/*``, no ``baselines/*``.
"""

from __future__ import annotations

from CrisisWorldCortex.server.graders.outer_reward import (
    TERMINAL_BONUS_FAILURE,
    TERMINAL_BONUS_SUCCESS,
)

DEFAULT_TICK_BUDGET = 6000  # design §11.2; matches B2 default and Phase-A §5
DEFAULT_LAMBDA_BUDGET = 0.5  # design §17 default; tunable per training run


def shape_reward(
    outer_reward: float,
    tokens_used: int,
    *,
    tick_budget: int = DEFAULT_TICK_BUDGET,
    lambda_budget: float = DEFAULT_LAMBDA_BUDGET,
) -> float:
    """Compose per-tick training reward.

    ``r_total = r_outer - lambda_budget * (tokens_used / tick_budget)``

    The token-budget penalty drives the policy toward shorter rollouts;
    ``lambda_budget`` controls how aggressive that pressure is. Returns
    a raw scalar (no clamp) — caller decides how to bound for trainer
    consumption.

    Args:
        outer_reward: ``server.graders.outer_reward`` output, ``[-1, 1]``.
        tokens_used: LLM tokens spent on this tick (harness-counted via
            ``cortex.llm_client.LLMClient.tokens_used_for(...)``).
        tick_budget: Per-tick token cap (default ``6000``).
        lambda_budget: Penalty coefficient (default ``0.5``).

    Returns:
        Per-tick training reward (``r_total``), unbounded.
    """
    if tick_budget <= 0:
        raise ValueError(f"tick_budget must be positive; got {tick_budget!r}")
    budget_fraction = tokens_used / tick_budget
    return outer_reward - lambda_budget * budget_fraction


def compose_episode_return(
    per_tick_rewards: list[float],
    terminal_kind: str,
) -> float:
    """Sum per-tick rewards + terminal bonus per design §14.3.

    ``episode_return = sum(per_tick_rewards) + terminal_bonus(terminal_kind)``

    where ``terminal_bonus`` is ``+0.20`` on ``"success"``, ``-0.20`` on
    ``"failure"``, and ``0.0`` on ``"timeout"`` / ``"none"``.

    Per-tick rewards must already be shaped via ``shape_reward(...)`` if
    the trainer wants the token-budget penalty included. This function
    is the final aggregation step.

    Args:
        per_tick_rewards: List of per-tick training rewards.
        terminal_kind: One of ``"success"``, ``"failure"``, ``"timeout"``,
            ``"none"`` (matches ``state.terminal``).
    """
    base = sum(per_tick_rewards)
    if terminal_kind == "success":
        return base + TERMINAL_BONUS_SUCCESS
    if terminal_kind == "failure":
        return base + TERMINAL_BONUS_FAILURE
    return base


__all__ = [
    "DEFAULT_LAMBDA_BUDGET",
    "DEFAULT_TICK_BUDGET",
    "compose_episode_return",
    "shape_reward",
]

"""Outer reward varies meaningfully — not stuck on a single constant.

Required by ``server/CLAUDE.md`` ("Every grader is non-constant across
episodes") and the ``tests/CLAUDE.md`` row ``test_reward_non_constancy.py``.

Three checks: (1) reward differs across distinct synthetic states;
(2) reward differs across distinct seeds; (3) reward differs between
accepted and rejected actions on the same starting state (proves the
``r_policy`` component contributes signal).
"""

from CrisisWorldCortex.models import (
    NoOp,
    PublicCommunication,
    RestrictMovement,
)
from CrisisWorldCortex.server.graders import outer_reward
from CrisisWorldCortex.server.graders.outer_reward import (
    R_POLICY_NOOP_ACCEPTED,
    R_POLICY_REJECTED,
    W_POLICY,
)
from CrisisWorldCortex.server.simulator import apply_tick, load_task


def test_reward_varies_across_synthetic_states() -> None:
    """Same task, two hand-crafted states: rewards must differ."""
    s_low = load_task("outbreak_hard", episode_seed=0)
    for region in s_low.regions:
        region.S, region.E, region.I, region.R = 0.99, 0.0, 0.01, 0.0

    s_high = load_task("outbreak_hard", episode_seed=0)
    for region in s_high.regions:
        region.S, region.E, region.I, region.R = 0.0, 0.0, 0.90, 0.10

    r_low = outer_reward(s_low, NoOp())
    r_high = outer_reward(s_high, NoOp())
    assert r_low != r_high, f"reward did not vary: low-I={r_low!r}, high-I={r_high!r}"
    assert r_low > r_high, f"low-infection state should score higher: low={r_low!r} high={r_high!r}"


def test_reward_varies_across_episodes() -> None:
    """Different action trajectories from the same task → distinct rewards.

    Note on seeds: the current MVP simulator is deterministic given
    ``(state, action)``. Neither the SEIR step nor any of the 6 outer
    action handlers consumes the per-tick RNG to make stochastic latent
    moves — only ``make_observation`` uses noise (and graders read latent
    state directly, bypassing observation noise). So varying
    ``episode_seed`` alone with a fixed action sequence (e.g. all NoOp)
    produces identical latent trajectories and identical rewards. The
    contract from ``server/CLAUDE.md`` is "non-constant across episodes",
    which we test here by varying the trajectory itself: 3 distinct
    action policies, sharing seed and task.
    """
    state_a = load_task("outbreak_medium", episode_seed=0)
    for _ in range(5):
        state_a = apply_tick(state_a, NoOp())  # passive policy

    state_b = load_task("outbreak_medium", episode_seed=0)
    for _ in range(5):
        state_b = apply_tick(
            state_b,
            PublicCommunication(
                audience="general",
                message_class="informational",
                honesty=0.5,
            ),
        )  # all-rejected policy

    state_c = load_task("outbreak_medium", episode_seed=0)
    for _ in range(5):
        state_c = apply_tick(
            state_c,
            RestrictMovement(
                region="R1",
                severity="moderate",
            ),
        )  # active intervention policy

    rewards = [
        outer_reward(state_a, NoOp()),
        outer_reward(state_b, NoOp()),
        outer_reward(state_c, NoOp()),
    ]
    distinct = len(set(round(r, 6) for r in rewards))
    assert distinct >= 2, f"reward constant across 3 distinct policies: {rewards!r}"


def test_reward_differs_for_accepted_vs_rejected_action() -> None:
    """``r_policy`` should produce a measurable gap when an action is rejected.

    Compare two episodes from the same starting state: one issues NoOp
    (accepted), the other issues PublicCommunication (V2-rejected). The
    SEIR step runs identically; only ``r_policy`` differs. After the
    Workstream-B Phase-1 four-state contract, NoOp(accepted) →
    R_POLICY_NOOP_ACCEPTED (0.0) and PublicCommunication(honesty=0.9,
    rejected as legal-violation) → R_POLICY_REJECTED (-0.5). The exact
    gap is therefore ``(R_POLICY_NOOP_ACCEPTED - R_POLICY_REJECTED) *
    W_POLICY`` (modulo float rounding).
    """
    s_a = load_task("outbreak_easy", episode_seed=42)
    s_b = load_task("outbreak_easy", episode_seed=42)

    a_ok = NoOp()
    a_bad = PublicCommunication(
        audience="general",
        message_class="informational",
        honesty=0.9,
    )

    s_a = apply_tick(s_a, a_ok)
    s_b = apply_tick(s_b, a_bad)

    r_ok = outer_reward(s_a, a_ok)
    r_bad = outer_reward(s_b, a_bad)

    assert r_ok > r_bad, (
        f"accepted action should score higher than rejected: ok={r_ok!r} bad={r_bad!r}"
    )
    expected_gap = (R_POLICY_NOOP_ACCEPTED - R_POLICY_REJECTED) * W_POLICY
    assert abs((r_ok - r_bad) - expected_gap) < 1e-9, (
        f"r_policy gap mismatch: ok-bad={r_ok - r_bad!r}, expected {expected_gap!r}"
    )

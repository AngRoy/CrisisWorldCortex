"""Outer reward stays in ``[-1.0, 1.0]`` across diverse states.

Range relaxed by Workstream-B Phase-1 fix (M2-A): rejected actions land
``r_policy = -0.5`` and parse-failure markers land ``r_policy = -1.0``,
so the per-tick total can go negative. Upper bound stays at 1.0.

Covers:

  - Initial state (right after ``load_task``, no ticks applied).
  - Mid-episode rollouts on all 3 tasks with varied actions.
  - Adversarial states with high I across all regions.
  - Adversarial states with rejected actions (well-formed-illegal +
    legal-violation).
"""

from CrisisWorldCortex.models import (
    DeployResource,
    Escalate,
    NoOp,
    PublicCommunication,
    RestrictMovement,
)
from CrisisWorldCortex.server.graders import outer_reward
from CrisisWorldCortex.server.simulator import apply_tick, load_task

TASKS = ("outbreak_easy", "outbreak_medium", "outbreak_hard")


def test_outer_reward_in_range_at_episode_start() -> None:
    """Right after load_task, before any tick: must still be in [0,1]."""
    for name in TASKS:
        state = load_task(name, episode_seed=0)
        r = outer_reward(state, NoOp())
        assert -1.0 <= r <= 1.0, f"{name}: r={r!r} out of [-1,1] at tick 0"


def test_outer_reward_in_range_during_rollout() -> None:
    """Run a 10-tick varied-action rollout on each task; r ∈ [0,1] every tick."""
    actions = [
        NoOp(),
        DeployResource(region="R1", resource_type="test_kits", quantity=50),
        RestrictMovement(region="R1", severity="moderate"),
        NoOp(),
        DeployResource(region="R2", resource_type="vaccine_doses", quantity=100),
        Escalate(to_authority="regional"),
        RestrictMovement(region="R2", severity="light"),
        NoOp(),
        DeployResource(region="R1", resource_type="test_kits", quantity=20),
        NoOp(),
    ]
    for name in TASKS:
        state = load_task(name, episode_seed=7)
        for action in actions:
            state = apply_tick(state, action)
            r = outer_reward(state, action)
            assert -1.0 <= r <= 1.0, (
                f"{name} tick={state.tick}: r={r!r} out of [-1,1] after action kind={action.kind!r}"
            )


def test_outer_reward_in_range_with_high_infection() -> None:
    """Hand-craft a worst-case state: high I in every region.

    Even when every component should drive toward 0, the weighted sum
    must not go negative.
    """
    state = load_task("outbreak_hard", episode_seed=0)
    for region in state.regions:
        region.S, region.E, region.I, region.R = 0.0, 0.0, 0.95, 0.05
    state.tick = state.max_ticks  # r_time → 0
    r = outer_reward(state, NoOp())
    assert -1.0 <= r <= 1.0, f"high-I worst case: r={r!r}"


def test_outer_reward_in_range_with_rejected_actions() -> None:
    """Rejected actions (V2-illegal + legal-violation) keep r in [0,1]."""
    # V2-illegal: PublicCommunication.
    state = load_task("outbreak_easy", episode_seed=0)
    a_v2 = PublicCommunication(
        audience="general",
        message_class="informational",
        honesty=0.9,
    )
    state = apply_tick(state, a_v2)
    r = outer_reward(state, a_v2)
    assert -1.0 <= r <= 1.0, f"V2-rejected: r={r!r}"

    # Legal-violation: strict severity before escalate-national on hard.
    state2 = load_task("outbreak_hard", episode_seed=0)
    a_legal = RestrictMovement(region="R1", severity="strict")
    state2 = apply_tick(state2, a_legal)
    assert state2.recent_action_log[-1].accepted is False
    r2 = outer_reward(state2, a_legal)
    assert -1.0 <= r2 <= 1.0, f"legal-violation: r={r2!r}"

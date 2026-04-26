"""Reward signal-quality gates (Workstream B Phase 1).

Locks the post-fix reward function as a permanent regression gate. Any
future reward change that breaks these tests fails loudly. Targets are
the relaxed Phase-1-crunch values:

  - all_no_op mean per-tick < 0.40 on outbreak_easy
  - all_rejected mean per-tick < 0.40 on outbreak_easy
  - active_strategic mean per-tick > 0.50 on outbreak_easy
  - parse_failure on tick 1 sets done=True at tick 1
  - signal_separation = mean(active) - mean(no_op) >= 0.20

Per design §15 / §19 contract restoration:
  - r_policy ∈ {-1.0 (parse_failure), -0.5 (rejected), 0.0 (accepted no-op),
    +1.0 (accepted real action)}
  - parse_failure_marker (PublicCommunication with honesty=0.0)
    terminates the episode as state.terminal == "failure".
"""

from __future__ import annotations

from CrisisWorldCortex.models import (
    CrisisworldcortexAction,
    DeployResource,
    NoOp,
    PublicCommunication,
    RestrictMovement,
)
from CrisisWorldCortex.server.CrisisWorldCortex_environment import (
    CrisisworldcortexEnvironment,
)

NO_OP_THRESHOLD = 0.40
REJECTED_THRESHOLD = 0.40
ACTIVE_THRESHOLD = 0.50
SEPARATION_THRESHOLD = 0.20

EPISODE_TICKS = 12
TASK = "outbreak_easy"
SEED = 0


def _mean_per_tick_reward(rewards: list[float]) -> float:
    """Mean of per-tick obs.reward across an episode."""
    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)


def _run_action_sequence(actions: list) -> tuple[list[float], list[bool]]:
    """Run one episode with a fixed action sequence.

    Returns (rewards_per_tick, done_per_tick). Runs until ``max_ticks`` or
    ``done == True``; whichever comes first.
    """
    env = CrisisworldcortexEnvironment()
    env.reset(task_name=TASK, seed=SEED, max_ticks=EPISODE_TICKS)
    rewards: list[float] = []
    dones: list[bool] = []
    for action_payload in actions:
        obs = env.step(CrisisworldcortexAction(action=action_payload))
        rewards.append(obs.reward if obs.reward is not None else 0.0)
        dones.append(bool(obs.done))
        if obs.done:
            break
    return rewards, dones


def test_all_no_op_episode_scores_below_threshold() -> None:
    """12 ticks of NoOp on outbreak_easy → mean per-tick reward < 0.40.

    NoOp is a *legal* action, so r_policy = 0.0 (per Phase A M4). The
    other components stay near 1.0 on outbreak_easy because the env is
    gentle, but the reweighted W_POLICY (0.35) on a 0.0 r_policy makes
    no_op a structurally low-scoring trajectory.
    """
    actions = [NoOp() for _ in range(EPISODE_TICKS)]
    rewards, dones = _run_action_sequence(actions)
    mean_reward = _mean_per_tick_reward(rewards)
    assert mean_reward < NO_OP_THRESHOLD, (
        f"all_no_op mean reward {mean_reward:.3f} >= threshold {NO_OP_THRESHOLD}; "
        f"per-tick rewards={rewards!r}"
    )


def test_all_rejected_episode_scores_below_threshold() -> None:
    """12 ticks of well-formed-illegal RestrictMovement(R1, strict) → < 0.40.

    On outbreak_easy without prior Escalate(national), strict severity is
    legal-violation rejected. r_policy = -0.5 every tick. The reward
    drops below the no_op floor because of the explicit -0.5 penalty.
    """
    actions = [RestrictMovement(region="R1", severity="strict") for _ in range(EPISODE_TICKS)]
    rewards, dones = _run_action_sequence(actions)
    mean_reward = _mean_per_tick_reward(rewards)
    assert mean_reward < REJECTED_THRESHOLD, (
        f"all_rejected mean reward {mean_reward:.3f} >= threshold {REJECTED_THRESHOLD}; "
        f"per-tick rewards={rewards!r}"
    )


def test_active_strategic_episode_scores_above_threshold() -> None:
    """12 ticks of valid DeployResource(R1, test_kits, 100) → mean > 0.50.

    Real accepted action gets r_policy = 1.0 every tick. With infection
    suppression keeping the steepened r_infect / r_hosp components high,
    the weighted score should clear 0.50 comfortably.
    """
    actions = [
        DeployResource(region="R1", resource_type="test_kits", quantity=100)
        for _ in range(EPISODE_TICKS)
    ]
    rewards, dones = _run_action_sequence(actions)
    mean_reward = _mean_per_tick_reward(rewards)
    assert mean_reward > ACTIVE_THRESHOLD, (
        f"active_strategic mean reward {mean_reward:.3f} <= threshold {ACTIVE_THRESHOLD}; "
        f"per-tick rewards={rewards!r}"
    )


def test_parse_failure_terminates_episode() -> None:
    """Single parse-failure marker on tick 1 → done=True, reward < 0.

    The synthetic parse_failure_marker (PublicCommunication with
    honesty=0.0) is the magic-string discriminator for parse-failure
    rejection per Phase A M3-B. The env must set state.terminal =
    "failure" on this rejection, propagating to obs.done = True.
    """
    parse_failure = PublicCommunication(
        audience="general",
        message_class="informational",
        honesty=0.0,
    )
    rewards, dones = _run_action_sequence([parse_failure])
    assert len(dones) == 1, f"expected episode to terminate at tick 1, got {len(dones)} ticks"
    assert dones[0] is True, f"parse_failure tick 1 did not set done=True; dones={dones!r}"
    # r_policy = -1.0 dominates → final reward should be negative.
    assert rewards[0] < 0.0, (
        f"parse_failure reward {rewards[0]:.3f} should be negative (r_policy=-1.0 contract)"
    )


def test_signal_separation() -> None:
    """Active-vs-no_op gap >= 0.20 — locks the trainable gradient.

    This is the mathematical floor on what GRPO (or any policy-gradient
    method) can extract from the reward signal. If active and no_op score
    similarly, training has nothing to optimise toward.
    """
    no_op_actions = [NoOp() for _ in range(EPISODE_TICKS)]
    active_actions = [
        DeployResource(region="R1", resource_type="test_kits", quantity=100)
        for _ in range(EPISODE_TICKS)
    ]
    no_op_rewards, _ = _run_action_sequence(no_op_actions)
    active_rewards, _ = _run_action_sequence(active_actions)
    no_op_mean = _mean_per_tick_reward(no_op_rewards)
    active_mean = _mean_per_tick_reward(active_rewards)
    separation = active_mean - no_op_mean
    assert separation >= SEPARATION_THRESHOLD, (
        f"signal_separation {separation:.3f} < threshold {SEPARATION_THRESHOLD}; "
        f"no_op_mean={no_op_mean:.3f}, active_mean={active_mean:.3f}"
    )

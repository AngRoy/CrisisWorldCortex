"""Random-action episodes complete without exception; terminal state reached."""

import random

from CrisisWorldCortex.models import (
    DeployResource,
    Escalate,
    NoOp,
    OuterActionPayload,
    ReallocateBudget,
    RequestData,
    RestrictMovement,
)
from CrisisWorldCortex.server.simulator import (
    apply_tick,
    load_task,
    make_observation,
)


def _random_action(state, rng: random.Random) -> OuterActionPayload:
    region_ids = [r.region for r in state.regions]
    kind = rng.choice(
        [
            "no_op",
            "deploy_resource",
            "request_data",
            "restrict_movement",
            "escalate",
            "reallocate_budget",
        ]
    )
    if kind == "no_op":
        return NoOp()
    if kind == "deploy_resource":
        return DeployResource(
            region=rng.choice(region_ids),
            resource_type=rng.choice(
                [
                    "test_kits",
                    "hospital_beds",
                    "mobile_units",
                    "vaccine_doses",
                ]
            ),
            quantity=rng.randint(1, 50),
        )
    if kind == "request_data":
        return RequestData(
            region=rng.choice(region_ids),
            data_type="case_survey",
        )
    if kind == "restrict_movement":
        return RestrictMovement(
            region=rng.choice(region_ids),
            severity=rng.choice(["light", "moderate"]),
        )
    if kind == "escalate":
        return Escalate(to_authority=rng.choice(["regional", "national"]))
    if kind == "reallocate_budget":
        resources = ["test_kits", "hospital_beds", "mobile_units", "vaccine_doses"]
        return ReallocateBudget(
            from_resource=rng.choice(resources),
            to_resource=rng.choice(resources),
            amount=rng.randint(1, 20),
        )
    raise AssertionError(f"unhandled kind={kind}")


def _run_random_episode(task: str, episode_seed: int, max_ticks: int = 12) -> None:
    state = load_task(task, episode_seed=episode_seed, max_ticks=max_ticks)
    rng = random.Random(episode_seed)
    for _ in range(max_ticks + 5):  # + some slack so terminal can fire early
        action = _random_action(state, rng)
        state = apply_tick(state, action)
        obs = make_observation(state)
        # Wire shape always valid.
        assert isinstance(obs.tick, int)
        assert obs.tick == state.tick
        if state.terminal != "none":
            assert obs.done is True
            return
        assert obs.done is False
    # Should have reached at least max_ticks → terminal state set.
    assert state.terminal != "none", f"episode never terminated; state.tick={state.tick}"


def test_random_episode_outbreak_easy_completes() -> None:
    _run_random_episode("outbreak_easy", episode_seed=0)


def test_random_episode_outbreak_medium_completes() -> None:
    _run_random_episode("outbreak_medium", episode_seed=0)


def test_random_episode_outbreak_hard_completes() -> None:
    _run_random_episode("outbreak_hard", episode_seed=0)


def test_no_op_only_episode_outbreak_easy_terminates_at_max_ticks() -> None:
    state = load_task("outbreak_easy", episode_seed=0)
    for _ in range(12):
        state = apply_tick(state, NoOp())
        if state.terminal != "none":
            break
    assert state.terminal != "none"


def test_no_op_only_episode_reaches_max_ticks_or_failure() -> None:
    """outbreak_hard with NoOp likely escalates; either timeout or failure."""
    state = load_task("outbreak_hard", episode_seed=0)
    for _ in range(15):
        state = apply_tick(state, NoOp())
        if state.terminal != "none":
            break
    assert state.terminal in ("timeout", "failure"), (
        f"expected timeout or failure, got terminal={state.terminal} at tick={state.tick}"
    )
    assert make_observation(state).done is True

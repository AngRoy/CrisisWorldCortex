"""Determinism: same (state, action, seed) → identical output.

Covers both ``apply_tick`` and ``make_observation`` independent RNG streams.
"""

from CrisisWorldCortex.models import (
    CrisisworldcortexObservation,
    DeployResource,
    NoOp,
)
from CrisisWorldCortex.server.simulator import (
    apply_tick,
    load_task,
    make_observation,
)


def test_apply_tick_same_seed_same_state() -> None:
    s1 = load_task("outbreak_easy", episode_seed=42)
    s2 = load_task("outbreak_easy", episode_seed=42)
    a = NoOp()
    assert apply_tick(s1, a) == apply_tick(s2, a)


def test_apply_tick_explicit_seed_override() -> None:
    s1 = load_task("outbreak_medium", episode_seed=1)
    s2 = load_task("outbreak_medium", episode_seed=1)
    # Same explicit seed → identical result regardless of episode_seed.
    assert apply_tick(s1, NoOp(), seed=99) == apply_tick(s2, NoOp(), seed=99)


def test_make_observation_pure_under_same_seed() -> None:
    state = load_task("outbreak_hard", episode_seed=7)
    obs1 = make_observation(state)
    obs2 = make_observation(state)
    assert obs1 == obs2  # purity: same state + derived seed → same obs


def test_make_observation_different_seeds_different_noise() -> None:
    state = load_task("outbreak_hard", episode_seed=7)
    obs_a = make_observation(state, seed=1)
    obs_b = make_observation(state, seed=2)
    # Different observation seeds → different noise realizations.
    assert obs_a != obs_b


def test_apply_tick_different_episode_seeds_diverge() -> None:
    s1 = load_task("outbreak_hard", episode_seed=1)
    s2 = load_task("outbreak_hard", episode_seed=2)
    obs1 = make_observation(apply_tick(s1, NoOp()))
    obs2 = make_observation(apply_tick(s2, NoOp()))
    # Different episode seeds → different observation streams.
    assert obs1 != obs2


def test_apply_tick_with_action_deterministic() -> None:
    s1 = load_task("outbreak_easy", episode_seed=0)
    s2 = load_task("outbreak_easy", episode_seed=0)
    a = DeployResource(region="R1", resource_type="test_kits", quantity=100)
    assert apply_tick(s1, a) == apply_tick(s2, a)


def test_observation_shape_after_apply_tick() -> None:
    state = load_task("outbreak_easy", episode_seed=0)
    state = apply_tick(state, NoOp())
    obs = make_observation(state)
    assert isinstance(obs, CrisisworldcortexObservation)
    assert obs.tick == 1
    assert obs.ticks_remaining == 11
    assert obs.cognition_budget_remaining == 6000
    assert len(obs.regions) == 4
    assert obs.done is False

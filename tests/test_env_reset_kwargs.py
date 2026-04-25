"""Tests for ``CrisisworldcortexEnvironment.reset()`` kwargs plumbing (Session 7c).

The framework's ``Environment`` base class declares
``reset(seed=None, episode_id=None, **kwargs)`` and the HTTP layer
filters via ``_get_valid_kwargs(sig, kwargs)`` to forward whatever the
subclass declares. This file exercises the subclass-side surface:

  - Default (no-arg) reset preserves backward compat -> outbreak_easy.
  - ``task_name`` selects between the 3 known tasks; unknown raises.
  - ``seed`` overrides the ``_reset_count`` fallback and is deterministic.
  - ``max_ticks`` flows through to ``WorldState.max_ticks`` /
    ``obs.ticks_remaining``.
  - ``episode_id`` is forwarded to ``State.episode_id``.

In-process only — the framework's serialization layer has its own tests
upstream; this file owns the env subclass's contract.
"""

from __future__ import annotations

import pytest

from CrisisWorldCortex.server.CrisisWorldCortex_environment import (
    CrisisworldcortexEnvironment,
)


def test_reset_default_no_args_loads_outbreak_easy() -> None:
    """No-arg reset must keep loading outbreak_easy — backward compat
    for tests/test_smoke_env.py and any pre-7c caller."""
    env = CrisisworldcortexEnvironment()
    obs = env.reset()
    assert env._world_state is not None
    assert env._world_state.task_name == "outbreak_easy"
    # outbreak_easy has 4 regions (R1 hot + R2/R3/R4 quiet) per tasks.py.
    assert len(obs.regions) == 4
    assert obs.tick == 0


def test_reset_task_name_loads_specified_task() -> None:
    """task_name='outbreak_hard' loads the hard config (5 regions, chain)."""
    env = CrisisworldcortexEnvironment()
    obs = env.reset(task_name="outbreak_hard")
    assert env._world_state is not None
    assert env._world_state.task_name == "outbreak_hard"
    assert len(obs.regions) == 5  # hard task has 5 chained regions
    assert len(env._world_state.task_config.legal_constraints) == 1, (
        "hard task carries the L1 strict-blocked legal constraint"
    )

    env2 = CrisisworldcortexEnvironment()
    obs2 = env2.reset(task_name="outbreak_medium")
    assert env2._world_state.task_name == "outbreak_medium"
    assert len(obs2.regions) == 4  # medium task has 4 regions (3 hot + 1 quiet)


def test_reset_seed_override_makes_state_deterministic() -> None:
    """Two resets with the same explicit seed produce identical initial
    observations. The default no-arg path uses _reset_count which
    increments — so this checks that explicit seed wins."""
    env_a = CrisisworldcortexEnvironment()
    env_b = CrisisworldcortexEnvironment()

    obs_a = env_a.reset(task_name="outbreak_hard", seed=42)
    obs_b = env_b.reset(task_name="outbreak_hard", seed=42)

    # Initial observations under same task + same seed are deterministic.
    # (The seed feeds episode_seed -> make_observation noise RNG.)
    assert obs_a.model_dump_json() == obs_b.model_dump_json(), (
        "explicit seed must produce reproducible initial observation"
    )

    obs_c = env_b.reset(task_name="outbreak_hard", seed=99)
    assert obs_c.model_dump_json() != obs_a.model_dump_json(), (
        "different seed should produce different observation noise"
    )


def test_reset_max_ticks_override_flows_through() -> None:
    """max_ticks=20 -> obs.ticks_remaining=20 at tick 0."""
    env = CrisisworldcortexEnvironment()
    obs = env.reset(max_ticks=20)
    assert obs.ticks_remaining == 20
    assert env._world_state.max_ticks == 20

    obs_default = env.reset()  # no max_ticks -> default 12
    assert obs_default.ticks_remaining == 12
    assert env._world_state.max_ticks == 12


def test_reset_unknown_task_name_raises() -> None:
    """task_name='bogus' -> ValueError from load_task. Backward-compat
    no-arg path is unaffected."""
    env = CrisisworldcortexEnvironment()
    with pytest.raises(ValueError, match="Unknown task"):
        env.reset(task_name="bogus_task")  # type: ignore[arg-type]


def test_reset_episode_id_forwarded_to_state() -> None:
    """An explicit episode_id surfaces on env.state.episode_id."""
    env = CrisisworldcortexEnvironment()
    env.reset(episode_id="ep-deadbeef-001")
    assert env.state.episode_id == "ep-deadbeef-001"

    # Default path (no episode_id) still produces a fresh uuid each call.
    env.reset()
    assert env.state.episode_id != "ep-deadbeef-001"
    assert len(env.state.episode_id) > 0

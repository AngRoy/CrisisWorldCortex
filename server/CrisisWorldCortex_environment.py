# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Crisisworldcortex Environment Implementation.

Session 5 wiring: ``reset()`` builds a fresh ``WorldState`` via
``load_task("outbreak_easy", episode_seed=...)``; ``step()`` calls
``apply_tick`` + ``make_observation``; ``done`` is set from
``state.terminal`` (one of "none", "success", "failure", "timeout").

Default task is ``outbreak_easy`` and ``max_ticks=12``. Future sessions
will add task selection at reset time (e.g., via reset payload metadata)
and eval-mode ``max_ticks=20`` overrides.
"""

from typing import Any, Literal, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Wire types use canonical ``CrisisWorldCortex.models`` (Session 7d): the
# container's wheel install resolves this to one ``sys.modules`` entry,
# matching the deep server modules already on canonical. The previous
# dual-fallback fired ``from models import ...`` (bare) in container,
# producing two class identities per discriminated-union variant — see
# ``server/simulator/seir_model.py``'s import block for the full trap.
from CrisisWorldCortex.models import CrisisworldcortexAction, CrisisworldcortexObservation

# Internal server submodules use the dual-fallback pattern: the relative
# form works under canonical loading (dev), and the bare fallback resolves
# via ``server.<x>`` from PYTHONPATH=/app/env in the container. Both
# branches resolve within the same physical ``server/`` tree, so single
# class identity is preserved either way.
try:
    from .simulator import WorldState, apply_tick, load_task, make_observation
except ImportError:  # pragma: no cover - bare-name fallback for non-package runs
    from server.simulator import WorldState, apply_tick, load_task, make_observation

try:
    from .graders import outer_reward
except ImportError:  # pragma: no cover - bare-name fallback for non-package runs
    from server.graders import outer_reward

DEFAULT_TASK = "outbreak_easy"
DEFAULT_MAX_TICKS = 12


class CrisisworldcortexEnvironment(Environment):
    """
    CrisisWorld environment.

    Holds an internal ``WorldState`` per session. ``reset()`` constructs
    a fresh state via ``load_task``; ``step(action)`` advances one tick
    and returns an observation with ``done`` set from the simulator's
    terminal-condition check.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._world_state: WorldState | None = None

    def reset(
        self,
        *,
        task_name: Literal["outbreak_easy", "outbreak_medium", "outbreak_hard"] = DEFAULT_TASK,
        seed: Optional[int] = None,
        max_ticks: int = DEFAULT_MAX_TICKS,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CrisisworldcortexObservation:
        """Reset the environment to a fresh episode.

        Keyword-only kwargs are passed by callers (the HTTP layer filters
        ``ResetRequest`` body fields through ``_get_valid_kwargs(sig, …)``
        so only declared kwargs arrive here; ``**kwargs`` swallows any
        future framework additions).

        Args:
            task_name: Which task to load. No-arg reset preserves
                backward compat by defaulting to ``"outbreak_easy"``;
                explicit ``task_name`` is opt-in for new callers
                (Session 7c+).
            seed: Episode seed for deterministic make_observation noise.
                If ``seed=None``, reset uses ``self._reset_count`` for
                variation across resets; trainers wanting reproducibility
                MUST pass an explicit seed.
            max_ticks: Episode length cap. Default 12 (training); set to
                20 for eval mode per Q3.
            episode_id: Custom episode identifier surfaced on
                ``self.state.episode_id``. Default is a fresh ``uuid4``.
            **kwargs: Forward-compat tolerance for unknown framework kwargs.
        """
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._reset_count += 1
        effective_seed = seed if seed is not None else self._reset_count
        self._world_state = load_task(
            name=task_name,
            episode_seed=effective_seed,
            max_ticks=max_ticks,
        )
        return make_observation(self._world_state)

    def step(self, action: CrisisworldcortexAction) -> CrisisworldcortexObservation:  # type: ignore[override]
        """Advance one tick. Lazy-initializes a default world state if needed."""
        if self._world_state is None:
            self._world_state = load_task(
                DEFAULT_TASK,
                episode_seed=0,
                max_ticks=DEFAULT_MAX_TICKS,
            )
        self._state.step_count += 1
        self._world_state = apply_tick(self._world_state, action.action)
        obs = make_observation(self._world_state)
        # Per design §15: r_outer is the only env-side reward signal, in [0,1].
        # Terminal bonus (+/-0.20) is composed downstream by the trainer per
        # design §14.3 — never bundled into obs.reward.
        obs.reward = outer_reward(self._world_state, action.action)
        return obs

    @property
    def state(self) -> State:
        """OpenEnv-compatible state shim (``episode_id`` + ``step_count``)."""
        return self._state

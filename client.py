# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Crisisworldcortex Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CrisisworldcortexAction, CrisisworldcortexObservation


class CrisisworldcortexEnv(EnvClient[CrisisworldcortexAction, CrisisworldcortexObservation, State]):
    """
    Client for the Crisisworldcortex Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance gets its own dedicated environment session on the
    server.

    Example:
        >>> from CrisisWorldCortex import CrisisworldcortexAction, CrisisworldcortexEnv
        >>> from CrisisWorldCortex.models import NoOp
        >>> with CrisisworldcortexEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.tick)
        ...
        ...     result = client.step(CrisisworldcortexAction(action=NoOp()))
        ...     print(result.observation.tick)

    Example with Docker:
        >>> client = CrisisworldcortexEnv.from_docker_image("CrisisWorldCortex-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(CrisisworldcortexAction(action=NoOp()))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CrisisworldcortexAction) -> Dict[str, Any]:
        """Serialize the action wrapper for WebSocket transport.

        Pydantic v2 ``model_dump()`` handles the discriminated-union
        serialization automatically: the inner ``OuterActionPayload`` is
        emitted with its ``kind`` field, which the server re-validates via
        ``CrisisworldcortexAction.model_validate(payload)``.
        """
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CrisisworldcortexObservation]:
        """Parse server response into ``StepResult[CrisisworldcortexObservation]``.

        ``model_validate`` rebuilds the discriminated-union payload inside
        the observation's ``recent_action_log`` correctly.
        """
        obs_data = payload.get("observation", {})
        observation = CrisisworldcortexObservation.model_validate(obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Parse server response into the OpenEnv-compatible ``State`` object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

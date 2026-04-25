"""Smoke: reset/step on the in-process environment return a valid
CrisisworldcortexObservation of the new (session-4) shape.

Shape-only — no assertion on latent/simulated values — so this test
survives the stub -> real-env transition in session 5.
"""

from CrisisWorldCortex.models import (
    CrisisworldcortexAction,
    CrisisworldcortexObservation,
    NoOp,
)
from CrisisWorldCortex.server.CrisisWorldCortex_environment import (
    CrisisworldcortexEnvironment,
)


def test_reset_returns_valid_observation() -> None:
    env = CrisisworldcortexEnvironment()
    obs = env.reset()
    assert isinstance(obs, CrisisworldcortexObservation)
    assert isinstance(obs.regions, list)
    assert isinstance(obs.tick, int)
    assert obs.done is False


def test_step_returns_valid_observation() -> None:
    env = CrisisworldcortexEnvironment()
    env.reset()
    obs = env.step(CrisisworldcortexAction(action=NoOp()))
    assert isinstance(obs, CrisisworldcortexObservation)
    assert isinstance(obs.regions, list)
    assert obs.tick >= 1

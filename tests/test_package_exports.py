"""Smoke: the wire-protocol package re-exports resolve.

Guards against accidental breakage of `CrisisWorldCortex/__init__.py`
or the package-dir mapping in `pyproject.toml`.
"""

from CrisisWorldCortex import (
    CrisisworldcortexAction,
    CrisisworldcortexEnv,
    CrisisworldcortexObservation,
)


def test_action_class_resolves():
    assert CrisisworldcortexAction.__name__ == "CrisisworldcortexAction"


def test_observation_class_resolves():
    assert CrisisworldcortexObservation.__name__ == "CrisisworldcortexObservation"


def test_env_client_class_resolves():
    assert CrisisworldcortexEnv.__name__ == "CrisisworldcortexEnv"

"""Wrapper-projection tests for normalize_step_result (Phase 6c reward fix).

The OpenEnv ``StepResult`` wrapper carries ``reward: Optional[float]`` and
``done: bool``. The deployed Space's parsed observation arrives with
``reward=None``, so ``score_candidate``'s ``0.0`` fallback fires for every
candidate, group-relative advantages collapse to zero, and the router updates
against zero gradient. The fix mirrors ``inference.py:_SyncEnvAdapter._normalize``
(the H15 fix already on main): project the wrapper's ``reward`` and ``done``
onto the bare observation before returning.

Path-load pattern matches ``test_training_multi_model_skeleton.py``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from openenv.core.client_types import StepResult

from CrisisWorldCortex.models import CrisisworldcortexObservation

SCRIPT_PATH = Path(__file__).parent.parent / "training" / "scripts" / "train_cortex_multi_model.py"


def _load_module():
    os.environ.setdefault("HF_TOKEN", "test_token_static_only")
    spec = importlib.util.spec_from_file_location(
        "train_cortex_multi_model_under_test_normalize", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_step_result_projects_wrapper_reward_onto_obs() -> None:
    """Wrapper.reward must be copied onto obs.reward as a float."""
    mod = _load_module()
    obs = CrisisworldcortexObservation()
    wrapper = StepResult(observation=obs, reward=0.42, done=False)

    result = mod.normalize_step_result(wrapper)

    assert result.reward == 0.42, (
        f"wrapper.reward=0.42 must be projected onto obs.reward; got {result.reward!r}"
    )
    assert isinstance(result.reward, float)


def test_normalize_step_result_overrides_obs_done_with_wrapper_done() -> None:
    """Wrapper.done is authoritative; obs.done set from wrapper even when False."""
    mod = _load_module()
    obs = CrisisworldcortexObservation(done=True)
    wrapper = StepResult(observation=obs, reward=None, done=False)

    result = mod.normalize_step_result(wrapper)

    assert result.done is False, (
        f"wrapper.done=False must override obs.done=True; got result.done={result.done!r}"
    )


def test_normalize_step_result_passthrough_for_bare_observation() -> None:
    """When result has no .observation attr (in-process path), return as-is."""
    mod = _load_module()
    obs = CrisisworldcortexObservation(tick=5)

    result = mod.normalize_step_result(obs)

    assert result is obs
    assert result.tick == 5


def test_normalize_step_result_skips_reward_projection_when_wrapper_reward_is_none() -> None:
    """Reset path: wrapper.reward=None must NOT clobber obs.reward to 0.0."""
    mod = _load_module()
    obs = CrisisworldcortexObservation()
    wrapper = StepResult(observation=obs, reward=None, done=False)

    result = mod.normalize_step_result(wrapper)

    assert result.reward is None, (
        f"wrapper.reward=None must not be projected; obs.reward should stay None, got {result.reward!r}"
    )

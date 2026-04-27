"""inference.py --agent CLI dispatch smoke tests.

Per the user's Session 13 follow-up plus Workstream B Phase 6: argparse
with ``--agent`` choices {b1, b2, b3, b6} (default b1 for backward compat)
+ dispatch table that constructs the corresponding agent class. All agents expose the
same ``run_episode(task, seed, max_ticks, *, step_callback)`` surface
per Phase A Decision 54, so the rest of the inference loop is unchanged.
"""

from __future__ import annotations

import pytest

import inference
from baselines.cortex_fixed_router import B3CortexFixedRouter
from baselines.cortex_trained_router import B6CortexTrainedRouter
from baselines.flat_agent import B1FlatAgent
from baselines.flat_agent_matched_compute import B2MatchedComputeAgent
from CrisisWorldCortex.models import (
    CrisisworldcortexAction,
    CrisisworldcortexObservation,
)
from tests._helpers.llm_stub import StubLLMClient


class _FakeEnv:
    """Quack-duck env used only to satisfy agent constructors.

    ``run_episode`` is not called in dispatch tests; we only verify
    the right class is instantiated.
    """

    def reset(self) -> CrisisworldcortexObservation:  # pragma: no cover
        raise NotImplementedError

    def step(
        self, action: CrisisworldcortexAction
    ) -> CrisisworldcortexObservation:  # pragma: no cover
        raise NotImplementedError


# ============================================================================
# Dispatch table
# ============================================================================


def test_make_agent_b1_returns_b1_flat_agent() -> None:
    agent = inference._make_agent("b1", _FakeEnv(), StubLLMClient(scripted_responses=[]))
    assert isinstance(agent, B1FlatAgent)


def test_make_agent_b2_returns_b2_matched_compute_agent() -> None:
    agent = inference._make_agent("b2", _FakeEnv(), StubLLMClient(scripted_responses=[]))
    assert isinstance(agent, B2MatchedComputeAgent)


def test_make_agent_b3_returns_b3_cortex_fixed_router() -> None:
    agent = inference._make_agent("b3", _FakeEnv(), StubLLMClient(scripted_responses=[]))
    assert isinstance(agent, B3CortexFixedRouter)


def test_make_agent_b6_returns_b6_cortex_trained_router() -> None:
    agent = inference._make_agent(
        "b6",
        _FakeEnv(),
        StubLLMClient(scripted_responses=[]),
        cortex_router="Angshuman28/cortex-router-trained",
    )
    assert isinstance(agent, B6CortexTrainedRouter)


def test_make_agent_b6_requires_router_repo() -> None:
    with pytest.raises(ValueError, match="cortex-router"):
        inference._make_agent("b6", _FakeEnv(), StubLLMClient(scripted_responses=[]))


def test_make_agent_invalid_raises_value_error() -> None:
    with pytest.raises(ValueError):
        inference._make_agent("b99", _FakeEnv(), StubLLMClient(scripted_responses=[]))


# ============================================================================
# Argparse
# ============================================================================


def test_argparse_default_is_b1_for_backward_compat() -> None:
    parser = inference._build_argparser()
    args = parser.parse_args([])
    assert args.agent == "b1"


def test_argparse_accepts_b1_b2_b3_b6() -> None:
    parser = inference._build_argparser()
    for name in ("b1", "b2", "b3", "b6"):
        args = parser.parse_args(["--agent", name])
        assert args.agent == name


def test_argparse_accepts_cortex_router_flag() -> None:
    parser = inference._build_argparser()
    args = parser.parse_args(
        ["--agent", "b6", "--cortex-router", "Angshuman28/cortex-router-trained"]
    )
    assert args.cortex_router == "Angshuman28/cortex-router-trained"


def test_argparse_rejects_unknown_agent() -> None:
    parser = inference._build_argparser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--agent", "b99"])

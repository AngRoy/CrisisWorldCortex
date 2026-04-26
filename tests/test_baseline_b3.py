"""Session 13 - B3CortexFixedRouter baseline smoke tests.

Per Phase A docs/CORTEX_ARCHITECTURE.md Decisions 51-55 + the user's
Session 13 proposal acceptance. T11 verifies B3 runs one full episode
on outbreak_easy in-process; T12 verifies the step_callback fires once
per tick (matching B1/B2's contract per Decision 54).
"""

from __future__ import annotations

import json
from typing import List

from baselines.cortex_fixed_router import B3CortexFixedRouter
from baselines.flat_agent import B1StepEvent
from CrisisWorldCortex.models import CrisisworldcortexAction
from CrisisWorldCortex.server.CrisisWorldCortex_environment import (
    CrisisworldcortexEnvironment,
)
from tests._helpers.llm_stub import StubLLMClient


def _belief_json(brain: str) -> str:
    return json.dumps(
        {
            "brain": brain,
            "latent_estimates": {
                "R1": {
                    "estimated_infection_rate": 0.05,
                    "estimated_r_effective": 1.2,
                    "estimated_compliance": 0.85,
                    "confidence_intervals": {},
                }
            },
            "hypotheses": [{"label": "h1", "weight": 0.6, "explanation": "rising"}],
            "uncertainty": 0.4,
            "reducible_by_more_thought": 0.3,
            "evidence": [{"source": "telemetry", "ref": "R1.cases", "excerpt": "5"}],
        }
    )


def _plan_json() -> str:
    return json.dumps(
        {
            "action_sketch": "Deploy 100 test_kits to R1",
            "expected_outer_action": {
                "kind": "deploy_resource",
                "region": "R1",
                "resource_type": "test_kits",
                "quantity": 100,
            },
            "expected_value": 0.6,
            "cost": 200.0,
            "assumptions": [],
            "falsifiers": ["R1 cases drop"],
            "confidence": 0.75,
        }
    )


def _critic_json(brain: str) -> str:
    return json.dumps(
        {
            "brain": brain,
            "target_plan_id": "plan-0",
            "attacks": [],
            "missing_considerations": [],
            "would_change_mind_if": [],
            "severity": 0.3,
        }
    )


def _round_responses() -> List[str]:
    """9 valid JSON responses for one round."""
    out: List[str] = []
    for brain in ("epidemiology", "logistics", "governance"):
        out.append(_belief_json(brain))
        out.append(_plan_json())
        out.append(_critic_json(brain))
    return out


class _InProcessEnvAdapter:
    """Mirrors tests/test_baseline_b1.py:83-99. Tests use in-process env;
    production B3 uses HTTP CrisisworldcortexEnv per baselines/CLAUDE.md."""

    def __init__(self, env: CrisisworldcortexEnvironment) -> None:
        self._env = env

    def reset(self):
        return self._env.reset(task_name="outbreak_easy", seed=0)

    def step(self, action: CrisisworldcortexAction):
        return self._env.step(action)


# T11 - B3 runs one full episode on outbreak_easy
def test_b3_runs_one_full_episode_outbreak_easy() -> None:
    env = CrisisworldcortexEnvironment()
    env_adapter = _InProcessEnvAdapter(env)

    # Pre-populate enough responses for 3 ticks worth of round-1 calls.
    # DeterministicRouter emits after round 1 if agreement is high (all 3
    # brains return deploy_resource here -> agreement == 1.0 -> emit).
    # 3 ticks x 9 calls/tick = 27 responses.
    responses: List[str] = []
    for _ in range(3):
        responses.extend(_round_responses())
    stub = StubLLMClient(scripted_responses=responses)

    b3 = B3CortexFixedRouter(env=env_adapter, llm=stub)
    result = b3.run_episode(task="outbreak_easy", seed=0, max_ticks=3)

    # Trajectory dict shape (matches B1's run_episode return)
    assert "rewards" in result
    assert "action_history" in result
    assert "steps_taken" in result
    assert result["steps_taken"] >= 1
    assert all("kind" in a for a in result["action_history"])
    # At least 9 LLM calls fired (one tick's worth of round-1)
    assert stub.call_count >= 9


# T12 - step_callback fires once per tick (Decision 54)
def test_b3_step_callback_invoked_per_tick() -> None:
    env = CrisisworldcortexEnvironment()
    env_adapter = _InProcessEnvAdapter(env)

    responses: List[str] = []
    for _ in range(3):
        responses.extend(_round_responses())
    stub = StubLLMClient(scripted_responses=responses)

    events_out: List[B1StepEvent] = []

    def _capture(event: B1StepEvent) -> None:
        events_out.append(event)

    b3 = B3CortexFixedRouter(env=env_adapter, llm=stub)
    result = b3.run_episode(
        task="outbreak_easy",
        seed=0,
        max_ticks=3,
        step_callback=_capture,
    )

    # Callback fires exactly once per tick taken
    assert len(events_out) == result["steps_taken"]
    assert all(isinstance(e, B1StepEvent) for e in events_out)
    assert all(hasattr(e.action, "kind") for e in events_out)

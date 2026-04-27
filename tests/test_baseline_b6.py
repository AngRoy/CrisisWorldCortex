"""B6 trained-router baseline smoke tests.

The local router model is intentionally stubbed here. These tests verify the
agent contract and one-brain execution path without downloading HF models.
"""

from __future__ import annotations

import json
from typing import List

from baselines.cortex_trained_router import B6CortexTrainedRouter, parse_router_choice
from baselines.flat_agent import B1StepEvent
from CrisisWorldCortex.models import CrisisworldcortexAction
from CrisisWorldCortex.server.CrisisWorldCortex_environment import (
    CrisisworldcortexEnvironment,
)
from tests._helpers.llm_stub import StubLLMClient


def _belief_json() -> str:
    return json.dumps(
        {
            "brain": "epidemiology",
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


def _critic_json() -> str:
    return json.dumps(
        {
            "brain": "epidemiology",
            "target_plan_id": "plan-0",
            "attacks": [],
            "missing_considerations": [],
            "would_change_mind_if": [],
            "severity": 0.3,
        }
    )


def _brain_responses(ticks: int) -> List[str]:
    responses: List[str] = []
    for _ in range(ticks):
        responses.extend([_belief_json(), _plan_json(), _critic_json()])
    return responses


class _InProcessEnvAdapter:
    def __init__(self, env: CrisisworldcortexEnvironment) -> None:
        self._env = env

    def reset(self):
        return self._env.reset(task_name="outbreak_easy", seed=0)

    def step(self, action: CrisisworldcortexAction):
        return self._env.step(action)


class _StaticRouter:
    def __init__(self, brain: str | None) -> None:
        self.brain = brain

    def select_brain(self, *_args):
        if self.brain is None:
            return None, "not-json"
        return self.brain, f'{{"brain":"{self.brain}"}}'


def test_parse_router_choice_accepts_aliases() -> None:
    assert parse_router_choice('{"brain":"epi"}') == "epidemiology"
    assert parse_router_choice('{"brain":"epidemiology"}') == "epidemiology"
    assert parse_router_choice('{"brain":"logistics"}') == "logistics"
    assert parse_router_choice('{"brain":"governance"}') == "governance"
    assert parse_router_choice('{"brain":"finance"}') is None


def test_b6_runs_one_episode_with_stubbed_router() -> None:
    env = CrisisworldcortexEnvironment()
    env_adapter = _InProcessEnvAdapter(env)
    stub = StubLLMClient(scripted_responses=_brain_responses(ticks=3))
    events: List[B1StepEvent] = []

    b6 = B6CortexTrainedRouter(
        env=env_adapter,
        llm=stub,
        router_repo="test/router",
    )
    b6._router = _StaticRouter("epidemiology")
    result = b6.run_episode(
        task="outbreak_easy",
        seed=0,
        max_ticks=3,
        step_callback=events.append,
    )

    assert result["steps_taken"] >= 1
    assert len(events) == result["steps_taken"]
    assert all(event.error is None for event in events)
    assert stub.call_count >= 3


def test_b6_invalid_router_output_marks_parse_failure() -> None:
    env = CrisisworldcortexEnvironment()
    env_adapter = _InProcessEnvAdapter(env)
    stub = StubLLMClient(scripted_responses=[])
    events: List[B1StepEvent] = []

    b6 = B6CortexTrainedRouter(
        env=env_adapter,
        llm=stub,
        router_repo="test/router",
    )
    b6._router = _StaticRouter(None)
    result = b6.run_episode(
        task="outbreak_easy",
        seed=0,
        max_ticks=1,
        step_callback=events.append,
    )

    assert result["parse_failure_count"] == 1
    assert len(events) == 1
    assert events[0].error == "parse_failure"
    assert events[0].parse_failure is True

"""Session 12 - Council Executive integration smoke + worst-case LLM-call bound.

Phase A section 10 integration smoke gate: Council runs one tick
end-to-end with stub brains, returns a valid OuterAction.

Worst-case per cortex/CLAUDE.md: 9 round-1 + 9 round-2 + 1 cross-brain
challenge = 19 LLM calls / tick. T10 verifies the upper bound.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from cortex.brains import Brain, EpiBrain, GovernanceBrain, LogisticsBrain
from cortex.council import Council
from cortex.schemas import RoutingAction
from CrisisWorldCortex.models import (
    CrisisworldcortexAction,
    CrisisworldcortexObservation,
    NoOp,
    RegionTelemetry,
    ResourceInventory,
)
from tests._helpers.llm_stub import StubLLMClient
from tests._helpers.router_stub import ScriptedRouter


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
    out: List[str] = []
    for brain in ("epidemiology", "logistics", "governance"):
        out.append(_belief_json(brain))
        out.append(_plan_json())
        out.append(_critic_json(brain))
    return out


def _make_obs(tick: int = 3) -> CrisisworldcortexObservation:
    return CrisisworldcortexObservation(
        regions=[
            RegionTelemetry(
                region=f"R{i + 1}",
                reported_cases_d_ago=5 if i == 0 else 1,
                hospital_load=0.3 if i == 0 else 0.1,
                compliance_proxy=0.85,
            )
            for i in range(4)
        ],
        resources=ResourceInventory(
            test_kits=1000, hospital_beds_free=500, mobile_units=20, vaccine_doses=2000
        ),
        active_restrictions=[],
        legal_constraints=[],
        tick=tick,
        ticks_remaining=12 - tick,
        cognition_budget_remaining=5200,
        recent_action_log=[],
    )


def _make_council(
    extra_responses: Optional[List[str]] = None,
    router: Optional[ScriptedRouter] = None,
    tick_budget: int = 6000,
) -> Tuple[Council, StubLLMClient]:
    responses = _round_responses() + list(extra_responses or [])
    stub = StubLLMClient(scripted_responses=responses)
    brains: Dict[str, Brain] = {
        "epidemiology": EpiBrain(stub),
        "logistics": LogisticsBrain(stub),
        "governance": GovernanceBrain(stub),
    }
    return Council(brains=brains, routing_policy=router, tick_budget=tick_budget), stub


# T9 - Phase A integration smoke gate
def test_council_step_one_full_tick_returns_valid_action() -> None:
    """Single tick end-to-end: 9 LLM calls (3 brains x 3 subagents) + valid action."""
    router = ScriptedRouter([RoutingAction(kind="emit_outer_action", outer_action=NoOp())])
    council, stub = _make_council(router=router)
    action = council.step(_make_obs())

    assert isinstance(action, CrisisworldcortexAction)
    assert stub.call_count == 9, "round-1 fires exactly 9 LLM calls (3 brains x 3 roles)"


# T10 - worst-case 19 LLM calls per cortex/CLAUDE.md
def test_council_step_under_19_call_worst_case() -> None:
    """Round 1 + Cross-brain Critic + Round 2 = 9 + 1 + 9 = 19 max."""
    extra = _round_responses() + [_critic_json("logistics")]  # round 2 + cross-brain critic
    router = ScriptedRouter(
        [
            RoutingAction(kind="request_challenge", brain="logistics", target_brain="epidemiology"),
            RoutingAction(kind="switch_phase", new_phase="Divergence"),  # round 2
            RoutingAction(kind="emit_outer_action", outer_action=NoOp()),
        ]
    )
    council, stub = _make_council(extra_responses=extra, router=router)
    council.step(_make_obs())

    assert stub.call_count <= 19, (
        f"worst-case LLM calls bound is 19 per cortex/CLAUDE.md; got {stub.call_count}"
    )

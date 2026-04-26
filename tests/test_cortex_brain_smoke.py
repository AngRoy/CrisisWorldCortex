"""Session 11 - Single-brain end-to-end smoke + no-LLM-in-Python-layers test.

T9 asserts Brain.compute_perception and Brain.compute_lens are pure
Python (zero LLM calls). T10 is the integration smoke gate per Phase A
section 10: a single brain runs end-to-end on a real observation,
returns a BrainRecommendation. Three LLM calls in canonical
WorldModeler -> Planner -> Critic order with locked caller_id format.
"""

from __future__ import annotations

import json

from cortex.brains import EpiBrain
from cortex.schemas import BrainRecommendation
from CrisisWorldCortex.models import (
    CrisisworldcortexObservation,
    RegionTelemetry,
    ResourceInventory,
)
from tests._helpers.llm_stub import StubLLMClient


def _make_obs() -> CrisisworldcortexObservation:
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
            test_kits=1000,
            hospital_beds_free=500,
            mobile_units=20,
            vaccine_doses=2000,
        ),
        active_restrictions=[],
        legal_constraints=[],
        tick=3,
        ticks_remaining=9,
        cognition_budget_remaining=5200,
        recent_action_log=[],
    )


_VALID_BELIEF = json.dumps(
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
        "hypotheses": [{"label": "rising", "weight": 0.6, "explanation": "R1 cases up"}],
        "uncertainty": 0.4,
        "reducible_by_more_thought": 0.3,
        "evidence": [
            {"source": "telemetry", "ref": "R1.cases", "excerpt": "5"},
            {"source": "policy", "ref": "R1.restriction", "excerpt": "none"},
        ],
    }
)

_VALID_PLAN = json.dumps(
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
        "assumptions": ["kits inventory > 100"],
        "falsifiers": ["R1 cases drop without intervention"],
        "confidence": 0.75,
    }
)

_VALID_CRITIC = json.dumps(
    {
        "brain": "epidemiology",
        "target_plan_id": "plan-0",
        "attacks": ["ignores R3 hospital saturation"],
        "missing_considerations": [],
        "would_change_mind_if": [],
        "severity": 0.3,
    }
)


# T9
def test_brain_runs_zero_llm_calls_in_python_layers() -> None:
    """Perception and Lens are pure Python; no LLMClient invocation."""
    stub = StubLLMClient(scripted_responses=[])  # any chat() would raise
    brain = EpiBrain(stub)
    obs = _make_obs()

    perception = brain.compute_perception(obs)
    lensed = brain.compute_lens(obs, last_reward=0.0)

    assert stub.call_count == 0
    assert perception.brain == "epidemiology"
    assert lensed.brain == "epidemiology"


# T10 -- integration smoke gate (Phase A section 10)
def test_brain_smoke_one_tick_three_llm_calls_in_order() -> None:
    """Full round-1 tick: WorldModeler -> Planner -> Critic, in that order."""
    stub = StubLLMClient(scripted_responses=[_VALID_BELIEF, _VALID_PLAN, _VALID_CRITIC])
    brain = EpiBrain(stub)
    obs = _make_obs()

    rec = brain.run_tick(obs, last_reward=0.0, tick=3)

    assert isinstance(rec, BrainRecommendation)
    assert rec.brain == "epidemiology"
    assert rec.top_action.kind == "deploy_resource"
    assert stub.call_count == 3, "exactly 3 LLM calls per Phase A 'WM + Planner + Critic'"

    # Order pin per user adjustment: WM (s0) -> Planner (s1) -> Critic (s2)
    assert stub.calls[0].caller_id.endswith(":world_modeler:t3:r1:s0"), (
        f"first call must be WorldModeler, got {stub.calls[0].caller_id!r}"
    )
    assert stub.calls[1].caller_id.endswith(":planner:t3:r1:s1"), (
        f"second call must be Planner, got {stub.calls[1].caller_id!r}"
    )
    assert stub.calls[2].caller_id.endswith(":critic:t3:r1:s2"), (
        f"third call must be Critic, got {stub.calls[2].caller_id!r}"
    )

    # Brain prefix locks
    for call in stub.calls:
        assert call.caller_id.startswith("cortex:epidemiology:")

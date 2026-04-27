"""Tests for Council cap enforcement bugs H3, H4, H5.

H3: Cross-brain critic call must count against per-brain critic cap.
H4: switch_phase("Convergence") must terminate the router loop.
H5: _LLMClientLike Protocol must declare tokens_used_for.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from cortex.brains import Brain, EpiBrain, GovernanceBrain, LogisticsBrain
from cortex.council import Council
from cortex.schemas import RoutingAction
from cortex.subagents._base import _LLMClientLike
from CrisisWorldCortex.models import (
    CrisisworldcortexAction,
    NoOp,
    RegionTelemetry,
    ResourceInventory,
    CrisisworldcortexObservation,
)
from tests._helpers.llm_stub import StubLLMClient
from tests._helpers.router_stub import ScriptedRouter


# -- Fixtures (copied from test_cortex_council.py) --


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


# ============================================================================
# H3: Cross-brain critic must count against challenger's per-brain critic cap
# ============================================================================


def test_cross_brain_challenge_counts_against_critic_cap() -> None:
    """After a cross-brain challenge from logistics, logistics' critic count
    must be 1, preventing a second router-issued critic call."""
    challenge_response = _critic_json("logistics")
    extra_critic = _critic_json("logistics")
    router = ScriptedRouter(
        [
            # Step 1: cross-brain challenge (logistics challenges epidemiology)
            RoutingAction(
                kind="request_challenge",
                brain="logistics",
                target_brain="epidemiology",
            ),
            # Step 2: try to call logistics critic again — should be capped
            RoutingAction(
                kind="call_subagent", brain="logistics", subagent="critic"
            ),
            # Step 3: emit
            RoutingAction(kind="emit_outer_action", outer_action=NoOp()),
        ]
    )
    council, _ = _make_council(
        extra_responses=[challenge_response, extra_critic], router=router
    )
    council.step(_make_obs())

    ts = council.last_tick_state
    assert ts is not None
    # The cross-brain challenge should have counted against logistics' critic cap
    assert ts.critic_calls_per_brain.get("logistics", 0) >= 1, (
        "Cross-brain challenge must increment challenger's critic_calls_per_brain"
    )


# ============================================================================
# H4: switch_phase("Convergence") must terminate the router loop
# ============================================================================


def test_convergence_phase_terminates_router_loop() -> None:
    """When _enforce_caps overrides to switch_phase(Convergence),
    the router loop must terminate and return an action, not continue."""
    extra = _round_responses()  # round 2 responses
    router = ScriptedRouter(
        [
            # Round 1: switch to Divergence (triggers round 2)
            RoutingAction(kind="switch_phase", new_phase="Divergence"),
            # Round 2: try switch to Divergence again — capped to Convergence
            RoutingAction(kind="switch_phase", new_phase="Divergence"),
            # This should NOT be reached if Convergence terminates the loop
            RoutingAction(kind="emit_outer_action", outer_action=NoOp()),
        ]
    )
    council, _ = _make_council(extra_responses=extra, router=router)
    action = council.step(_make_obs())

    assert isinstance(action, CrisisworldcortexAction)
    ts = council.last_tick_state
    assert ts is not None
    # The second Divergence was capped to Convergence, which should terminate
    # the loop. The third router action (emit) should NOT have been called.
    assert router.call_count == 2, (
        f"switch_phase(Convergence) should terminate the loop; "
        f"router was called {router.call_count} times instead of 2"
    )


# ============================================================================
# H5: _LLMClientLike Protocol must include tokens_used_for
# ============================================================================


def test_llm_client_protocol_includes_tokens_used_for() -> None:
    """_LLMClientLike must declare tokens_used_for so stubs satisfy the Protocol."""
    assert hasattr(_LLMClientLike, "tokens_used_for"), (
        "_LLMClientLike Protocol is missing tokens_used_for method"
    )
    # Also verify StubLLMClient structurally satisfies the protocol
    stub = StubLLMClient(scripted_responses=["test"])
    assert hasattr(stub, "tokens_used_for")
    assert callable(stub.tokens_used_for)
    assert hasattr(stub, "chat")
    assert callable(stub.chat)

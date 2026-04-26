"""Session 12 - Council Executive phase-machine + hard cap tests.

Per Phase A docs/CORTEX_ARCHITECTURE.md sections 3-4 + Decisions 22-31.
Tests T1-T8 cover protocol ordering, deliberation cap (2 rounds),
cross-brain challenge cap (1/tick), critic-per-brain cap (1/brain/tick),
preserved dissent, emit-terminates-tick, budget exhaustion override,
and phase-to-protocol mapping (Item F).
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

# ============================================================================
# Test fixtures
# ============================================================================


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
    """9 valid JSON responses for one round: epi -> logistics -> governance, each WM -> Planner -> Critic."""
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
    """Construct 3 brains + Council with one shared stub LLM client."""
    responses = _round_responses() + list(extra_responses or [])
    stub = StubLLMClient(scripted_responses=responses)
    brains: Dict[str, Brain] = {
        "epidemiology": EpiBrain(stub),
        "logistics": LogisticsBrain(stub),
        "governance": GovernanceBrain(stub),
    }
    return Council(brains=brains, routing_policy=router, tick_budget=tick_budget), stub


# ============================================================================
# T1 - 5 protocol steps in order (phase trace)
# ============================================================================


def test_council_runs_5_protocol_steps_in_order() -> None:
    router = ScriptedRouter([RoutingAction(kind="emit_outer_action", outer_action=NoOp())])
    council, _ = _make_council(router=router)
    council.step(_make_obs())

    ts = council.last_tick_state
    assert ts is not None
    assert ts.phase_trace[0] == "Divergence"
    assert ts.phase_trace[-1] == "Convergence"
    assert "Anonymized" not in " ".join(ts.phase_trace), (
        "Step 4 (anonymized comparison) is V2-deferred per Decision 56"
    )


# ============================================================================
# T2 - cap deliberation rounds at 2
# ============================================================================


def test_council_caps_deliberation_rounds_at_2() -> None:
    extra = _round_responses()  # 9 more responses for round 2
    router = ScriptedRouter(
        [
            RoutingAction(kind="switch_phase", new_phase="Divergence"),  # -> round 2
            RoutingAction(kind="switch_phase", new_phase="Divergence"),  # -> overridden
            RoutingAction(kind="emit_outer_action", outer_action=NoOp()),
        ]
    )
    council, _ = _make_council(extra_responses=extra, router=router)
    council.step(_make_obs())

    assert council.last_tick_state is not None
    assert council.last_tick_state.deliberation_rounds_used <= 2


# ============================================================================
# T3 - cap cross-brain challenges at 1
# ============================================================================


def test_council_caps_cross_brain_challenges_at_1() -> None:
    challenge_response = _critic_json("logistics")
    router = ScriptedRouter(
        [
            RoutingAction(kind="request_challenge", brain="logistics", target_brain="epidemiology"),
            RoutingAction(
                kind="request_challenge", brain="governance", target_brain="logistics"
            ),  # overridden
            RoutingAction(kind="emit_outer_action", outer_action=NoOp()),
        ]
    )
    council, _ = _make_council(extra_responses=[challenge_response], router=router)
    council.step(_make_obs())

    assert council.last_tick_state is not None
    assert council.last_tick_state.cross_brain_challenges_used == 1


# ============================================================================
# T4 - cap critic calls per brain at 1
# ============================================================================


def test_council_caps_critic_calls_per_brain_at_1() -> None:
    extra_critic = _critic_json("epidemiology")
    router = ScriptedRouter(
        [
            RoutingAction(
                kind="call_subagent", brain="epidemiology", subagent="critic"
            ),  # overridden
            RoutingAction(kind="emit_outer_action", outer_action=NoOp()),
        ]
    )
    council, _ = _make_council(extra_responses=[extra_critic], router=router)
    council.step(_make_obs())

    assert council.last_tick_state is not None
    assert council.last_tick_state.critic_calls_per_brain["epidemiology"] == 1


# ============================================================================
# T5 - preserved dissent via routing action
# ============================================================================


def test_council_preserves_dissent_via_routing_action() -> None:
    router = ScriptedRouter(
        [
            RoutingAction(kind="preserve_dissent"),
            RoutingAction(kind="emit_outer_action", outer_action=NoOp()),
        ]
    )
    council, _ = _make_council(router=router)
    council.step(_make_obs())

    assert council.last_tick_state is not None
    assert len(council.last_tick_state.preserved_dissent) >= 1
    for tag in council.last_tick_state.preserved_dissent:
        assert len(tag) <= 80


# ============================================================================
# T6 - emit_outer_action terminates tick mid-loop
# ============================================================================


def test_council_emits_outer_action_terminates_tick() -> None:
    router = ScriptedRouter(
        [
            RoutingAction(kind="emit_outer_action", outer_action=NoOp()),
            RoutingAction(
                kind="call_subagent", brain="epidemiology", subagent="critic"
            ),  # unreached
            RoutingAction(kind="emit_outer_action", outer_action=NoOp()),
        ]
    )
    council, _ = _make_council(router=router)
    action = council.step(_make_obs())

    assert isinstance(action, CrisisworldcortexAction)
    assert router.call_count == 1


# ============================================================================
# T7 - budget exhaustion forces emit / no_op
# ============================================================================


def test_council_token_budget_exhaustion_forces_emit() -> None:
    router = ScriptedRouter(
        [
            RoutingAction(kind="call_subagent", brain="epidemiology", subagent="critic"),
            RoutingAction(kind="emit_outer_action", outer_action=NoOp()),
        ]
    )
    council, _ = _make_council(router=router, tick_budget=10)  # tiny
    action = council.step(_make_obs())

    assert isinstance(action, CrisisworldcortexAction)
    assert council.last_tick_state is not None
    assert council.last_tick_state.tick_tokens_used >= 10


# ============================================================================
# T8 - phase-to-protocol mapping (Item F)
# ============================================================================


def test_council_phase_to_protocol_mapping_locked() -> None:
    challenge_response = _critic_json("logistics")
    router = ScriptedRouter(
        [
            RoutingAction(kind="request_challenge", brain="logistics", target_brain="epidemiology"),
            RoutingAction(kind="emit_outer_action", outer_action=NoOp()),
        ]
    )
    council, _ = _make_council(extra_responses=[challenge_response], router=router)
    council.step(_make_obs())

    trace = council.last_tick_state.phase_trace
    assert "Divergence" in trace
    assert "Challenge" in trace
    assert "Convergence" in trace
    assert "Anonymized" not in " ".join(trace)

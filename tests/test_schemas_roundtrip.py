"""JSON round-trip tests for all discriminated-union / complex Pydantic
types across ``models.py`` and ``cortex/schemas.py``.

Catches discriminator-config bugs and Pydantic v2 serialization regressions
early, before simulator / cortex code starts producing instances of these
types in real flows.
"""

from cortex.schemas import (
    BeliefState,
    BrainRecommendation,
    CandidatePlan,
    CouncilDecision,
    CriticReport,
    EvidenceCitation,
    Hypothesis,
    MetacognitionState,
    PerceptionReport,
    RegionBeliefEstimate,
    RoutingAction,
    SubagentInput,
)
from CrisisWorldCortex.models import (
    CrisisworldcortexAction,
    CrisisworldcortexObservation,
    DeployResource,
    Escalate,
    ExecutedAction,
    LegalConstraint,
    NoOp,
    PublicCommunication,
    ReallocateBudget,
    RegionTelemetry,
    RequestData,
    ResourceInventory,
    Restriction,
    RestrictMovement,
)

# ---- Wire-type roundtrips -------------------------------------------------


def test_action_payload_roundtrip_all_variants() -> None:
    variants = [
        DeployResource(region="R1", resource_type="test_kits", quantity=10),
        RequestData(region="R1", data_type="case_survey"),
        RestrictMovement(region="R1", severity="moderate"),
        Escalate(to_authority="national"),
        ReallocateBudget(from_resource="test_kits", to_resource="mobile_units", amount=5),
        NoOp(),
        PublicCommunication(audience="general", message_class="informational", honesty=0.9),
    ]
    for v in variants:
        wrapped = CrisisworldcortexAction(action=v)
        restored = CrisisworldcortexAction.model_validate_json(wrapped.model_dump_json())
        assert restored == wrapped, f"roundtrip failed for kind={v.kind}"
        assert restored.action.kind == v.kind


def test_observation_roundtrip() -> None:
    obs = CrisisworldcortexObservation(
        regions=[
            RegionTelemetry(
                region="R1", reported_cases_d_ago=5, hospital_load=0.3, compliance_proxy=0.8
            ),
            RegionTelemetry(
                region="R2", reported_cases_d_ago=0, hospital_load=0.1, compliance_proxy=0.95
            ),
        ],
        resources=ResourceInventory(test_kits=100, hospital_beds_free=50),
        active_restrictions=[Restriction(region="R1", severity="moderate", ticks_remaining=3)],
        legal_constraints=[
            LegalConstraint(rule_id="L1", blocked_action="restrict_movement.strict")
        ],
        tick=3,
        ticks_remaining=9,
        cognition_budget_remaining=5200,
        recent_action_log=[ExecutedAction(tick=2, action=NoOp(), accepted=True)],
    )
    assert CrisisworldcortexObservation.model_validate_json(obs.model_dump_json()) == obs


def test_executed_action_carries_payload_discriminator() -> None:
    ea = ExecutedAction(
        tick=1,
        action=DeployResource(region="R2", resource_type="vaccine_doses", quantity=500),
        accepted=True,
    )
    restored = ExecutedAction.model_validate_json(ea.model_dump_json())
    assert restored == ea
    assert restored.action.kind == "deploy_resource"


# ---- Cortex-type roundtrips ----------------------------------------------


def test_brain_recommendation_roundtrip() -> None:
    br = BrainRecommendation(
        brain="Epidemiology",
        top_action=DeployResource(region="R1", resource_type="test_kits", quantity=100),
        top_confidence=0.7,
        minority_actions=[NoOp()],
        reasoning_summary="Growing case counts in R1",
        evidence=[EvidenceCitation(source="telemetry", ref="R1.reported_cases", excerpt="5")],
        falsifier="Cases in R1 drop below 2 next tick",
        uncertainty=0.4,
        tokens_used=450,
    )
    restored = BrainRecommendation.model_validate_json(br.model_dump_json())
    assert restored == br
    assert restored.top_action.kind == "deploy_resource"


def test_routing_action_roundtrip_all_kinds() -> None:
    actions = [
        RoutingAction(kind="call_subagent", brain="Epidemiology", subagent="world_modeler"),
        RoutingAction(kind="request_challenge", brain="Logistics", target_brain="Epidemiology"),
        RoutingAction(kind="switch_phase", new_phase="Challenge"),
        RoutingAction(kind="preserve_dissent"),
        RoutingAction(kind="emit_outer_action", outer_action=NoOp()),
        RoutingAction(kind="stop_and_no_op"),
    ]
    for ra in actions:
        restored = RoutingAction.model_validate_json(ra.model_dump_json())
        assert restored == ra, f"routing-action roundtrip failed for kind={ra.kind}"


def test_council_decision_roundtrip() -> None:
    cd = CouncilDecision(
        action=Escalate(to_authority="national"),
        rationale="R3 pattern suggests hidden superspreader; escalate to unlock strict movement",
        preserved_dissent=["Logistics: reallocation cheaper"],
        phase_trace=["Divergence", "Challenge", "Convergence"],
        rounds_used=2,
        tokens_used=5500,
    )
    assert CouncilDecision.model_validate_json(cd.model_dump_json()) == cd


def test_metacognition_state_roundtrip() -> None:
    ms = MetacognitionState(
        tick=7,
        round=2,
        phase="Challenge",
        inter_brain_agreement=0.3,
        average_confidence=0.65,
        average_evidence_support=0.8,
        novelty_yield_last_round=0.4,
        collapse_suspicion=0.1,
        budget_remaining_frac=0.25,
        urgency=0.7,
        preserved_dissent_count=1,
        challenge_used_this_tick=True,
    )
    assert MetacognitionState.model_validate_json(ms.model_dump_json()) == ms


def test_belief_state_with_typed_estimates_roundtrip() -> None:
    bs = BeliefState(
        brain="Epidemiology",
        latent_estimates={
            "R1": RegionBeliefEstimate(
                estimated_infection_rate=0.05,
                estimated_r_effective=1.2,
                estimated_compliance=0.85,
                confidence_intervals={"infection_rate": (0.03, 0.07)},
            ),
        },
        hypotheses=[
            Hypothesis(label="Hidden spread", weight=0.6, explanation="R1 telemetry lags"),
            Hypothesis(label="Surface event", weight=0.4, explanation="Cases trend with reports"),
        ],
        uncertainty=0.5,
        reducible_by_more_thought=0.3,
        evidence=[EvidenceCitation(source="telemetry", ref="R1.cases", excerpt="rising")],
    )
    assert BeliefState.model_validate_json(bs.model_dump_json()) == bs


def test_candidate_plan_with_typed_outer_action_roundtrip() -> None:
    cp = CandidatePlan(
        action_sketch="Deploy kits to R1",
        expected_outer_action=DeployResource(region="R1", resource_type="test_kits", quantity=100),
        expected_value=0.6,
        cost=200.0,
        assumptions=["kits available"],
        falsifiers=["kit efficacy drops below 0.3"],
        confidence=0.75,
    )
    restored = CandidatePlan.model_validate_json(cp.model_dump_json())
    assert restored == cp
    assert restored.expected_outer_action.kind == "deploy_resource"


def test_perception_and_critic_roundtrips() -> None:
    pr = PerceptionReport(
        brain="Epidemiology",
        salient_signals=["R1 cases rising"],
        anomalies=["R3 telemetry flat"],
        confidence=0.7,
        evidence=[EvidenceCitation(source="telemetry", ref="R3", excerpt="flat")],
    )
    assert PerceptionReport.model_validate_json(pr.model_dump_json()) == pr

    cr = CriticReport(
        brain="Logistics",
        target_plan_id="plan-1",
        attacks=["Ignores R1 kit depletion"],
        missing_considerations=["Reallocation delays"],
        would_change_mind_if=["Kit supply confirmed"],
        severity=0.6,
    )
    assert CriticReport.model_validate_json(cr.model_dump_json()) == cr


# T7 (Session 9) -- SubagentInput round-trip
def test_subagent_input_roundtrip() -> None:
    si = SubagentInput(
        brain="epidemiology",
        role="world_modeler",
        tick=3,
        round=1,
        perception=PerceptionReport(
            brain="epidemiology",
            salient_signals=["R1 cases rising"],
            anomalies=[],
            confidence=0.7,
            evidence=[EvidenceCitation(source="telemetry", ref="R1.cases", excerpt="rising")],
        ),
        prior_belief=None,
        prior_plans=[],
        target_plan_id=None,
        last_reward=0.5,
        recent_action_log_excerpt=[],
    )
    restored = SubagentInput.model_validate_json(si.model_dump_json())
    assert restored == si
    assert restored.role == "world_modeler"
    assert restored.brain == "epidemiology"

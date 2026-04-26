"""Session 11 - Brain Executive aggregation tests.

Per Phase A docs/CORTEX_ARCHITECTURE.md Decisions 15-21 + M-FR-3
(partial evidence union; CandidatePlan and CriticReport schemas have
no evidence field).
"""

from __future__ import annotations

import pytest

from cortex.brains import aggregate_brain_outputs
from cortex.schemas import (
    BeliefState,
    CandidatePlan,
    CriticReport,
    EvidenceCitation,
    Hypothesis,
    PerceptionReport,
    RegionBeliefEstimate,
)
from CrisisWorldCortex.models import (
    DeployResource,
    NoOp,
    RestrictMovement,
)


def _belief(uncertainty: float = 0.4, evidence_count: int = 1) -> BeliefState:
    return BeliefState(
        brain="epidemiology",
        latent_estimates={
            "R1": RegionBeliefEstimate(
                estimated_infection_rate=0.05,
                estimated_r_effective=1.2,
                estimated_compliance=0.85,
            ),
        },
        hypotheses=[Hypothesis(label="h1", weight=0.6, explanation="rising")],
        uncertainty=uncertainty,
        reducible_by_more_thought=0.3,
        evidence=[
            EvidenceCitation(source="telemetry", ref=f"R1.cases@{i}", excerpt=f"e{i}")
            for i in range(evidence_count)
        ],
    )


def _plan(action=None, confidence: float = 0.75, expected_value: float = 0.6) -> CandidatePlan:
    if action is None:
        action = DeployResource(region="R1", resource_type="test_kits", quantity=100)
    return CandidatePlan(
        action_sketch="Deploy 100 test_kits to R1",
        expected_outer_action=action,
        expected_value=expected_value,
        cost=200.0,
        assumptions=["kits available"],
        falsifiers=["R1 cases drop without intervention"],
        confidence=confidence,
    )


def _critic(severity: float = 0.3) -> CriticReport:
    return CriticReport(
        brain="epidemiology",
        target_plan_id="plan-0",
        attacks=["limited reach"],
        missing_considerations=[],
        would_change_mind_if=[],
        severity=severity,
    )


def _perception(evidence_count: int = 1) -> PerceptionReport:
    return PerceptionReport(
        brain="epidemiology",
        salient_signals=["R1 cases rising"],
        anomalies=[],
        confidence=0.7,
        evidence=[
            EvidenceCitation(source="telemetry", ref=f"R1.perception@{i}", excerpt=f"p{i}")
            for i in range(evidence_count)
        ],
    )


# T4
def test_brain_executive_aggregates_subagent_outputs() -> None:
    rec = aggregate_brain_outputs(
        brain_id="epidemiology",
        perception=_perception(),
        beliefs=[_belief()],
        plans=[_plan()],
        critics=[_critic()],
    )
    assert rec.brain == "epidemiology"
    assert rec.top_action.kind == "deploy_resource"
    assert rec.top_confidence > 0.0
    assert rec.tokens_used == 0


# T5 -- Decision 16
def test_brain_executive_top_confidence_includes_uncertainty() -> None:
    rec = aggregate_brain_outputs(
        brain_id="epidemiology",
        perception=_perception(),
        beliefs=[_belief(uncertainty=0.4)],
        plans=[_plan(confidence=0.75)],
        critics=[_critic()],
    )
    # D16: top_confidence == confidence x (1 - uncertainty) == 0.75 x 0.6 == 0.45
    assert rec.top_confidence == pytest.approx(0.45)


# T6 -- Decision 17
def test_brain_executive_minority_actions_excludes_top() -> None:
    plan_a = _plan(
        action=DeployResource(region="R1", resource_type="test_kits", quantity=100),
        confidence=0.8,
        expected_value=0.7,
    )
    plan_b = _plan(
        action=RestrictMovement(region="R1", severity="moderate"),
        confidence=0.5,
        expected_value=0.4,
    )
    # plan_a wins: 0.8 * 0.7 = 0.56 > 0.5 * 0.4 = 0.20

    rec = aggregate_brain_outputs(
        brain_id="epidemiology",
        perception=_perception(),
        beliefs=[_belief(), _belief(uncertainty=0.5)],
        plans=[plan_a, plan_b],
        critics=[_critic(), _critic()],
    )

    assert rec.top_action.kind == "deploy_resource"
    assert len(rec.minority_actions) == 1
    assert rec.minority_actions[0].kind == "restrict_movement"


# T7 -- Decision 20 + M-FR-3
def test_brain_executive_evidence_union() -> None:
    perception = _perception(evidence_count=1)
    belief = _belief(evidence_count=2)

    rec = aggregate_brain_outputs(
        brain_id="epidemiology",
        perception=perception,
        beliefs=[belief],
        plans=[_plan()],
        critics=[_critic()],
    )

    # M-FR-3: union of perception.evidence + belief.evidence (3 total)
    assert len(rec.evidence) == 3
    assert rec.evidence[0].ref.startswith("R1.perception")
    assert rec.evidence[1].ref.startswith("R1.cases")


# T8 -- empty fallback
def test_brain_executive_handles_empty_subagent_outputs() -> None:
    empty_belief = BeliefState(
        brain="epidemiology",
        latent_estimates={},
        hypotheses=[],
        uncertainty=1.0,
        reducible_by_more_thought=0.0,
        evidence=[],
    )
    empty_plan = CandidatePlan(
        action_sketch="(empty)",
        expected_outer_action=NoOp(),
        expected_value=0.0,
        cost=0.0,
        assumptions=[],
        falsifiers=[],
        confidence=0.0,
    )
    empty_critic = CriticReport(
        brain="epidemiology",
        target_plan_id="",
        attacks=[],
        missing_considerations=[],
        would_change_mind_if=[],
        severity=0.0,
    )
    empty_perception = PerceptionReport(
        brain="epidemiology",
        salient_signals=[],
        anomalies=[],
        confidence=0.0,
        evidence=[],
    )

    rec = aggregate_brain_outputs(
        brain_id="epidemiology",
        perception=empty_perception,
        beliefs=[empty_belief],
        plans=[empty_plan],
        critics=[empty_critic],
    )

    assert rec.top_action.kind == "no_op"
    assert rec.top_confidence == 0.0
    assert rec.uncertainty == 1.0

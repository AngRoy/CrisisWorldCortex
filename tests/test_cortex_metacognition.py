"""Session 13 - Cortex metacognition formula tests.

Per Phase A docs/CORTEX_ARCHITECTURE.md Decisions 32-36 + the user's
Session 13 proposal acceptance. Tests T1-T5 cover inter_brain_agreement
(3 cases), collapse_suspicion, and the average_confidence /
budget_remaining_frac scalars.
"""

from __future__ import annotations

from typing import Dict

from cortex.metacognition import compute_metacognition_state
from cortex.schemas import BrainRecommendation, EvidenceCitation
from CrisisWorldCortex.models import DeployResource, NoOp, RestrictMovement


def _rec(
    brain: str,
    *,
    action=None,
    top_confidence: float = 0.7,
    evidence_count: int = 1,
) -> BrainRecommendation:
    """Minimal valid BrainRecommendation for metacognition tests."""
    if action is None:
        action = NoOp()
    return BrainRecommendation(
        brain=brain,  # type: ignore[arg-type]
        top_action=action,
        top_confidence=top_confidence,
        minority_actions=[],
        reasoning_summary="brief summary.",
        evidence=[
            EvidenceCitation(source="telemetry", ref=f"{brain}.r{i}", excerpt="x")
            for i in range(evidence_count)
        ],
        falsifier="(test)",
        uncertainty=0.3,
        tokens_used=0,
    )


def _make_state(brain_recs: Dict[str, BrainRecommendation], **overrides):
    defaults = dict(
        tick=3,
        round_=1,
        phase="Divergence",
        brain_recommendations=brain_recs,
        tick_tokens_used=0,
        tick_budget=6000,
        ticks_remaining=9,
        max_ticks=12,
        worst_region_infection=0.0,
        preserved_dissent_count=0,
        challenge_used_this_tick=False,
    )
    defaults.update(overrides)
    return compute_metacognition_state(**defaults)


# T1
def test_inter_brain_agreement_all_match() -> None:
    """Decision 32: all 3 brains' top_action.kind match -> 1.0."""
    recs = {
        "epidemiology": _rec("epidemiology", action=NoOp()),
        "logistics": _rec("logistics", action=NoOp()),
        "governance": _rec("governance", action=NoOp()),
    }
    state = _make_state(recs)
    assert state.inter_brain_agreement == 1.0


# T2
def test_inter_brain_agreement_two_match() -> None:
    """Decision 32: 2/3 match -> 0.5."""
    recs = {
        "epidemiology": _rec("epidemiology", action=NoOp()),
        "logistics": _rec("logistics", action=NoOp()),
        "governance": _rec(
            "governance",
            action=DeployResource(region="R1", resource_type="test_kits", quantity=100),
        ),
    }
    state = _make_state(recs)
    assert state.inter_brain_agreement == 0.5


# T3
def test_inter_brain_agreement_all_differ() -> None:
    """Decision 32: 3 distinct kinds -> 0.0."""
    recs = {
        "epidemiology": _rec("epidemiology", action=NoOp()),
        "logistics": _rec(
            "logistics",
            action=DeployResource(region="R1", resource_type="test_kits", quantity=100),
        ),
        "governance": _rec("governance", action=RestrictMovement(region="R1", severity="moderate")),
    }
    state = _make_state(recs)
    assert state.inter_brain_agreement == 0.0


# T4
def test_collapse_suspicion_flags_identical_no_evidence() -> None:
    """Decision 35: 1.0 iff all top_actions match AND all evidence empty."""
    # All same NoOp + zero evidence -> collapse
    recs_collapse = {
        "epidemiology": _rec("epidemiology", action=NoOp(), evidence_count=0),
        "logistics": _rec("logistics", action=NoOp(), evidence_count=0),
        "governance": _rec("governance", action=NoOp(), evidence_count=0),
    }
    state = _make_state(recs_collapse)
    assert state.collapse_suspicion == 1.0

    # Same actions but with evidence -> not collapse (evidence shows reasoning)
    recs_with_ev = {
        "epidemiology": _rec("epidemiology", action=NoOp(), evidence_count=2),
        "logistics": _rec("logistics", action=NoOp(), evidence_count=2),
        "governance": _rec("governance", action=NoOp(), evidence_count=2),
    }
    state = _make_state(recs_with_ev)
    assert state.collapse_suspicion == 0.0


# T5
def test_metacognition_average_confidence_and_budget_frac() -> None:
    """Decision 33: average_confidence = mean. Phase A section 5: budget_remaining_frac."""
    recs = {
        "epidemiology": _rec("epidemiology", top_confidence=0.3),
        "logistics": _rec("logistics", top_confidence=0.6),
        "governance": _rec("governance", top_confidence=0.9),
    }
    state = _make_state(recs, tick_tokens_used=3000, tick_budget=6000)
    assert state.average_confidence == 0.6  # (0.3+0.6+0.9)/3
    assert state.budget_remaining_frac == 0.5  # (6000-3000)/6000

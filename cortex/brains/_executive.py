"""Brain Executive - deterministic Python aggregation.

Per Phase A docs/CORTEX_ARCHITECTURE.md Decisions 15-21 + M-FR-3 partial
evidence union (perception + beliefs only; CandidatePlan and CriticReport
schemas have no evidence fields).

Brain Executive runs ONCE per brain at round end. NOT router-callable
per cortex/CLAUDE.md.
"""

from __future__ import annotations

from typing import List

from cortex.schemas import (
    BeliefState,
    BrainRecommendation,
    CandidatePlan,
    CriticReport,
    EvidenceCitation,
    PerceptionReport,
)
from CrisisWorldCortex.models import NoOp

_REASONING_SUMMARY_MAX_CHARS = 400  # matches BrainRecommendation.reasoning_summary cap
_FALSIFIERS_TO_JOIN = 3
_FALSIFIER_FALLBACK = "(no falsifier provided)"
_EMPTY_REASONING = "(empty: no subagent produced a parseable plan)"


def aggregate_brain_outputs(
    brain_id: str,
    perception: PerceptionReport,
    beliefs: List[BeliefState],
    plans: List[CandidatePlan],
    critics: List[CriticReport],
    tokens_used: int = 0,
) -> BrainRecommendation:
    """Aggregate one brain's per-round subagent outputs into a recommendation.

    Decisions:
        D15: argmax over expected_value * confidence.
        D16: top_confidence = chosen.confidence * (1 - chosen_belief.uncertainty).
        D17: minority_actions = all expected_outer_actions except chosen.
        D19: reasoning_summary = chosen.action_sketch[:400].
        D20 (M-FR-3): evidence = perception.evidence + flat-union of beliefs[*].evidence.
                      CandidatePlan/CriticReport carry no evidence fields.
        D21: brain_id is lowercase per Pydantic Literal in BrainRecommendation.

    Empty fallback (M-FR-7): no plans, or chosen plan has confidence==0
    -> top_action=NoOp, top_confidence=0, uncertainty=1.0,
    reasoning_summary=_EMPTY_REASONING.

    Args:
        brain_id: lowercase brain id ("epidemiology" / "logistics" / "governance").
        perception: This brain's PerceptionReport for the tick.
        beliefs: Per-round BeliefStates. Index aligned with ``plans``.
        plans: Per-round CandidatePlans.
        critics: Per-round CriticReports (currently unused in aggregation but
            kept on the signature so the trajectory log captures the full
            chain).
        tokens_used: Total tokens billed across this brain's subagents.
    """
    if not plans:
        return _empty_recommendation(brain_id, perception, beliefs, tokens_used)

    # D15: argmax over expected_value * confidence
    chosen_idx = max(
        range(len(plans)),
        key=lambda i: plans[i].expected_value * plans[i].confidence,
    )
    chosen_plan = plans[chosen_idx]

    if chosen_plan.confidence == 0.0:
        # All plans are empty fallbacks (or the only plan is empty).
        # Brain Executive treats this as no-signal.
        return _empty_recommendation(brain_id, perception, beliefs, tokens_used)

    # D16: top_confidence = chosen.confidence * (1 - belief.uncertainty)
    if chosen_idx < len(beliefs):
        chosen_belief = beliefs[chosen_idx]
        uncertainty = chosen_belief.uncertainty
    else:
        # Defensive: parallel arrays should match. If not, treat as max uncertainty.
        uncertainty = 1.0
    top_confidence = chosen_plan.confidence * (1.0 - uncertainty)

    # D17: minority_actions = all plans except chosen
    minority_actions = [
        plans[i].expected_outer_action for i in range(len(plans)) if i != chosen_idx
    ]

    # D19: reasoning_summary
    reasoning_summary = chosen_plan.action_sketch[:_REASONING_SUMMARY_MAX_CHARS]

    # D20 (M-FR-3): evidence union from perception + beliefs only.
    # CandidatePlan and CriticReport schemas (Session 9) have no evidence fields;
    # the perception+beliefs union captures the actionable evidence chain since
    # plans/critics derive from beliefs.
    evidence: List[EvidenceCitation] = list(perception.evidence)
    for b in beliefs:
        evidence.extend(b.evidence)

    # falsifier (M-FR-6): join up to 3 falsifiers; fallback if empty.
    if chosen_plan.falsifiers:
        falsifier = "; ".join(chosen_plan.falsifiers[:_FALSIFIERS_TO_JOIN])
    else:
        falsifier = _FALSIFIER_FALLBACK

    return BrainRecommendation(
        brain=brain_id,
        top_action=chosen_plan.expected_outer_action,
        top_confidence=top_confidence,
        minority_actions=minority_actions,
        reasoning_summary=reasoning_summary,
        evidence=evidence,
        falsifier=falsifier,
        uncertainty=uncertainty,
        tokens_used=tokens_used,
    )


def _empty_recommendation(
    brain_id: str,
    perception: PerceptionReport,
    beliefs: List[BeliefState],
    tokens_used: int,
) -> BrainRecommendation:
    """M-FR-7 empty fallback: NoOp + confidence=0 + uncertainty=1."""
    evidence: List[EvidenceCitation] = list(perception.evidence)
    for b in beliefs:
        evidence.extend(b.evidence)

    return BrainRecommendation(
        brain=brain_id,
        top_action=NoOp(),
        top_confidence=0.0,
        minority_actions=[],
        reasoning_summary=_EMPTY_REASONING,
        evidence=evidence,
        falsifier=_FALSIFIER_FALLBACK,
        uncertainty=1.0,
        tokens_used=tokens_used,
    )

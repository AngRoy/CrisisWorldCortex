"""Metacognition signals (Phase A section 5 + Decisions 32-37).

Session 12 ships a minimal implementation of ``compute_metacognition_state``
sufficient to drive the Council's phase machine. Session 13 will extend
this with the full Decision-32-37 formulas + the eval-only signals
(novelty_yield_last_round, collapse_suspicion gradations).

The MetacognitionState schema itself is locked in cortex/schemas.py
(Phase A section 2). This module computes the per-tick / per-round
values from brain recommendations + tick state.
"""

from __future__ import annotations

from typing import Dict, List

from cortex.anti_hivemind import detect_collapse
from cortex.schemas import BrainRecommendation, EpistemicPhase, MetacognitionState


def compute_metacognition_state(
    *,
    tick: int,
    round_: int,
    phase: EpistemicPhase,
    brain_recommendations: Dict[str, BrainRecommendation],
    tick_tokens_used: int,
    tick_budget: int,
    ticks_remaining: int,
    max_ticks: int,
    worst_region_infection: float,
    preserved_dissent_count: int,
    challenge_used_this_tick: bool,
) -> MetacognitionState:
    """Compute the MetacognitionState for the router.

    Decisions 32-36 implemented; Decision 37 (novelty_yield) returns 0.0
    in MVP since round-1 doesn't have a prior round to diff against.
    """
    recs: List[BrainRecommendation] = list(brain_recommendations.values())

    # Decision 32: inter_brain_agreement
    if len(recs) < 2:
        inter_brain_agreement = 0.0
    else:
        kinds = [r.top_action.kind for r in recs]
        unique = set(kinds)
        if len(unique) == 1:
            inter_brain_agreement = 1.0
        elif len(unique) == 2:
            inter_brain_agreement = 0.5
        else:
            inter_brain_agreement = 0.0

    # Decision 33: average_confidence
    average_confidence = sum(r.top_confidence for r in recs) / len(recs) if recs else 0.0

    # Decision 34: average_evidence_support
    if recs:
        per_brain = []
        for r in recs:
            claims_count = max(1, len(r.reasoning_summary.split(".")) - 1)
            per_brain.append(min(1.0, len(r.evidence) / claims_count))
        average_evidence_support = sum(per_brain) / len(per_brain)
    else:
        average_evidence_support = 0.0

    # Decision 35: collapse_suspicion (binary minimal version per M-FR-2)
    collapse_suspicion = 1.0 if detect_collapse(recs) else 0.0

    # Decision 36: urgency
    time_pressure = 1.0 - (ticks_remaining / max(1, max_ticks))
    urgency = max(0.0, min(1.0, time_pressure + worst_region_infection * 0.5))

    # Phase A section 5: budget_remaining_frac
    budget_remaining_frac = max(0.0, 1.0 - tick_tokens_used / max(1, tick_budget))

    return MetacognitionState(
        tick=tick,
        round=round_,
        phase=phase,
        inter_brain_agreement=inter_brain_agreement,
        average_confidence=average_confidence,
        average_evidence_support=average_evidence_support,
        novelty_yield_last_round=0.0,  # Decision 37: round-1 always 0.0
        collapse_suspicion=collapse_suspicion,
        budget_remaining_frac=budget_remaining_frac,
        urgency=urgency,
        preserved_dissent_count=preserved_dissent_count,
        challenge_used_this_tick=challenge_used_this_tick,
    )

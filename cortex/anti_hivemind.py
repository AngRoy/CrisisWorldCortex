"""Anti-hivemind helpers: collapse detection + preserved-dissent tagging.

Per cortex/CLAUDE.md "Belongs here" -> "Anti-hivemind protocol + collapse
detectors". Phase A section 4 pins the preserved-dissent tag format
(<= 80 chars). Phase A Decision 35 pins the collapse_suspicion binary
signal (1.0 if all 3 brains return identical top_action AND empty
evidence; else 0.0).
"""

from __future__ import annotations

from typing import Iterable, List

from cortex.schemas import BrainRecommendation

_DISSENT_TAG_MAX_CHARS = 80


def format_dissent_tag(brain: str, action_kind: str, rationale: str) -> str:
    """Format a preserved-dissent tag per Phase A section 4.

    Format: ``"<brain>.<action_kind>:<short_rationale>"`` truncated to 80 chars.
    """
    base = f"{brain}.{action_kind}:"
    remaining = _DISSENT_TAG_MAX_CHARS - len(base)
    if remaining <= 0:
        return base[:_DISSENT_TAG_MAX_CHARS]
    return f"{base}{rationale[:remaining]}"


def detect_collapse(brain_recommendations: Iterable[BrainRecommendation]) -> bool:
    """Phase A Decision 35: True iff all brains' top_action is bytewise-equal
    AND all evidence lists are empty. Otherwise False.
    """
    recs: List[BrainRecommendation] = list(brain_recommendations)
    if len(recs) < 2:
        return False
    first_action_json = recs[0].top_action.model_dump_json()
    if any(r.top_action.model_dump_json() != first_action_json for r in recs[1:]):
        return False
    return all(len(r.evidence) == 0 for r in recs)

"""Session 13 - DeterministicRouter decision-table tests.

Per Phase A docs/CORTEX_ARCHITECTURE.md Decision 38 + Decision 28
post-review (conservative AGREEMENT_LOW=0.4) + the user's Session 13
proposal acceptance. Tests T6-T10 cover budget floor, low/moderate/
high-agreement branches, and bytewise determinism.
"""

from __future__ import annotations

from cortex.routing_policy import DeterministicRouter
from cortex.schemas import MetacognitionState


def _state(
    *,
    round_: int = 1,
    phase: str = "Divergence",
    inter_brain_agreement: float = 0.5,
    average_confidence: float = 0.6,
    average_evidence_support: float = 0.6,
    novelty_yield_last_round: float = 0.0,
    collapse_suspicion: float = 0.0,
    budget_remaining_frac: float = 1.0,
    urgency: float = 0.3,
    preserved_dissent_count: int = 0,
    challenge_used_this_tick: bool = False,
    tick: int = 3,
) -> MetacognitionState:
    """Construct a MetacognitionState with sensible defaults."""
    return MetacognitionState(
        tick=tick,
        round=round_,
        phase=phase,  # type: ignore[arg-type]
        inter_brain_agreement=inter_brain_agreement,
        average_confidence=average_confidence,
        average_evidence_support=average_evidence_support,
        novelty_yield_last_round=novelty_yield_last_round,
        collapse_suspicion=collapse_suspicion,
        budget_remaining_frac=budget_remaining_frac,
        urgency=urgency,
        preserved_dissent_count=preserved_dissent_count,
        challenge_used_this_tick=challenge_used_this_tick,
    )


# T6 - budget floor (Decision 38)
def test_router_emits_when_budget_below_floor() -> None:
    """budget_remaining_frac < 0.20 -> emit_outer_action regardless of agreement."""
    router = DeterministicRouter()
    state = _state(
        round_=1,
        inter_brain_agreement=0.0,  # would normally trigger challenge
        budget_remaining_frac=0.10,
        challenge_used_this_tick=False,
    )
    action = router.forward(state)
    assert action.kind == "emit_outer_action"


# T7 - low agreement + no prior challenge -> challenge
def test_router_challenges_when_agreement_low_and_no_prior_challenge() -> None:
    """agreement < 0.4 AND not challenge_used_this_tick -> request_challenge."""
    router = DeterministicRouter()
    state = _state(
        round_=1,
        inter_brain_agreement=0.0,
        budget_remaining_frac=0.8,
        challenge_used_this_tick=False,
    )
    action = router.forward(state)
    assert action.kind == "request_challenge"
    # Decision 38 post-review: dynamic pair selection deferred to Council
    assert action.brain is None
    assert action.target_brain is None


# T8 - high agreement -> emit
def test_router_emits_when_agreement_high() -> None:
    """agreement >= 0.7 -> emit_outer_action (consensus reached)."""
    router = DeterministicRouter()
    state = _state(
        round_=1,
        inter_brain_agreement=0.8,
        budget_remaining_frac=0.8,
    )
    action = router.forward(state)
    assert action.kind == "emit_outer_action"


# T9 - moderate disagreement -> round 2
def test_router_round_2_when_moderate_agreement() -> None:
    """agreement in [0.4, 0.7) AND round 1 -> switch_phase(Divergence) for round 2."""
    router = DeterministicRouter()
    state = _state(
        round_=1,
        inter_brain_agreement=0.5,
        budget_remaining_frac=0.8,
        challenge_used_this_tick=False,
    )
    action = router.forward(state)
    assert action.kind == "switch_phase"
    assert action.new_phase == "Divergence"


def test_router_emits_after_round_2_done() -> None:
    """round == 2 -> emit (round-2 cap forces convergence)."""
    router = DeterministicRouter()
    state = _state(
        round_=2,
        inter_brain_agreement=0.5,
        budget_remaining_frac=0.5,
    )
    action = router.forward(state)
    assert action.kind == "emit_outer_action"


def test_router_post_challenge_low_agreement_goes_to_round_2() -> None:
    """challenge already used + low agreement -> round 2 (cannot challenge twice)."""
    router = DeterministicRouter()
    state = _state(
        round_=1,
        inter_brain_agreement=0.0,
        budget_remaining_frac=0.8,
        challenge_used_this_tick=True,  # already challenged
    )
    action = router.forward(state)
    assert action.kind == "switch_phase"
    assert action.new_phase == "Divergence"


# T10 - determinism (eval-mode contract)
def test_router_determinism_same_state_same_action() -> None:
    """cortex/CLAUDE.md: same observation + same policy -> identical action."""
    router = DeterministicRouter()
    state = _state(
        round_=1,
        inter_brain_agreement=0.5,
        budget_remaining_frac=0.8,
    )
    a1 = router.forward(state)
    a2 = router.forward(state)
    a3 = router.forward(state)
    assert a1 == a2 == a3
    assert a1.model_dump_json() == a2.model_dump_json() == a3.model_dump_json()

"""Deterministic routing policy for the Cortex Council (Session 13).

Per Phase A docs/CORTEX_ARCHITECTURE.md Decision 38 (post-review fix) +
Decision 28 (D-FR-1 conservative threshold) + cortex/CLAUDE.md eval-mode
determinism contract.

The Council (Session 12) consults a routing policy after round-1 runs
and after each subsequent action it executes. ``DeterministicRouter``
is stateless: every ``forward(state)`` call is a pure function of
``MetacognitionState``. Session 15 (Workstream B) will swap in a
trainable router that conforms to the same Protocol.
"""

from __future__ import annotations

from cortex.schemas import MetacognitionState, RoutingAction


class DeterministicRouter:
    """Phase A Decision 38 5-branch decision table.

    Branches (mutually exclusive, evaluated in order):
        1. Budget exhausted (< BUDGET_FLOOR)             -> emit_outer_action
        2. Round 2 already ran (round == 2)              -> emit_outer_action
        3. High consensus (agreement >= AGREEMENT_HIGH)  -> emit_outer_action
        4. Low agreement + no prior challenge            -> request_challenge
        5. Otherwise (moderate disagreement OR challenge already used)
                                                         -> switch_phase(Divergence)

    request_challenge intentionally leaves brain/target_brain unset; the
    Council's _handle_cross_brain_challenge fills the dynamic pair
    (challenger=min top_confidence, target=max top_confidence) per
    Decision 38 post-review.
    """

    # M-FR-5 constants (per Decisions 28 + 38)
    AGREEMENT_HIGH: float = 0.7
    AGREEMENT_LOW: float = 0.4
    BUDGET_FLOOR: float = 0.20

    def forward(self, state: MetacognitionState) -> RoutingAction:
        # 1. Budget exhaustion: emit immediately.
        if state.budget_remaining_frac < self.BUDGET_FLOOR:
            return RoutingAction(kind="emit_outer_action")

        # 2. Round 2 already ran: emit (round-2 cap forces convergence).
        if state.round == 2:
            return RoutingAction(kind="emit_outer_action")

        # 3. High consensus: emit.
        if state.inter_brain_agreement >= self.AGREEMENT_HIGH:
            return RoutingAction(kind="emit_outer_action")

        # 4. Low agreement + no prior challenge: cross-brain challenge.
        if state.inter_brain_agreement < self.AGREEMENT_LOW and not state.challenge_used_this_tick:
            return RoutingAction(kind="request_challenge")

        # 5. Moderate disagreement OR already challenged: round 2.
        return RoutingAction(kind="switch_phase", new_phase="Divergence")

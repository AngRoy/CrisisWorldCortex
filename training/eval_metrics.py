"""Eval-only metrics for trajectories (design §20).

Four signals, NOT optimised by the trainer (per ``server/CLAUDE.md``
binding contract: training_reward and eval_metrics live in separate
trajectory-log columns and must never combine):

  - ``collapse_rate(trajectories)`` — fraction of episodes where the
    agent collapsed to the same action repeatedly (proxy for hivemind
    failure / over-exploitation).
  - ``dissent_value(trajectories)`` — Cortex-only: how often a preserved
    minority recommendation later proved correct (next-tick re-check).
    Returns ``0.0`` until Cortex Session 13 lands real preserved-dissent
    metadata; trajectory shape allows the metric to round-trip without
    breaking.
  - ``consensus_calibration(trajectories)`` — Cortex-only: correlation
    between brain-reported confidence and realised reward. Returns
    ``0.0`` until Cortex lands; placeholder.
  - ``novelty_yield(trajectories)`` — Cortex-only: fraction of round-2
    deliberations that produced a different action than round-1.
    Returns ``0.0`` until Cortex lands; placeholder.

Phase-2 scaffold (Workstream B). ``collapse_rate`` is fully implemented
against B1/B2-style trajectories (action_history with ``"action"`` dicts);
the three Cortex-dependent metrics are stubbed with TODOs marking what
Cortex Session 13 will need to populate.

Allowed under ``training/CLAUDE.md``: ``models``, stdlib only.
"""

from __future__ import annotations

from typing import Iterable

# Threshold: an episode is "collapsed" if at least this fraction of its
# steps used the modal action. 0.8 captures "policy went lazy" without
# false-flagging short episodes that legitimately repeat one action.
COLLAPSE_FRACTION_THRESHOLD = 0.8
COLLAPSE_MIN_STEPS = 3  # ignore episodes too short to be diagnostic


def _action_kind(step: dict) -> str:
    """Extract action-kind discriminator from a trajectory step dict."""
    action = step.get("action", {})
    if isinstance(action, dict):
        return str(action.get("kind", "unknown"))
    return getattr(action, "kind", "unknown")


def collapse_rate(trajectories: Iterable[list[dict]]) -> float:
    """Fraction of episodes that collapsed to a single modal action.

    Modal action share >= COLLAPSE_FRACTION_THRESHOLD in episodes with
    >= COLLAPSE_MIN_STEPS counts as a "collapsed" episode. Returns the
    fraction of input episodes that collapsed; 0.0 if no episodes
    qualify (all too short).
    """
    qualifying = 0
    collapsed = 0
    for traj in trajectories:
        if len(traj) < COLLAPSE_MIN_STEPS:
            continue
        qualifying += 1
        kinds = [_action_kind(step) for step in traj]
        modal_count = max(kinds.count(k) for k in set(kinds))
        if modal_count / len(kinds) >= COLLAPSE_FRACTION_THRESHOLD:
            collapsed += 1
    if qualifying == 0:
        return 0.0
    return collapsed / qualifying


def dissent_value(trajectories: Iterable[list[dict]]) -> float:
    """Cortex-only: preserved-dissent realised-correctness rate.

    Returns 0.0 in Phase-2; Cortex Session 13 wires the real metric.

    TODO(cortex-session-13): when ``RouterStep.preserved_dissent`` lands,
    replace this stub with: for each tick where a minority rec was
    preserved, check whether the next tick's chosen action matches the
    preserved minority. dissent_value = matches / preservations.
    """
    # Touch the iterator to surface accidental misuse (e.g. passing a
    # generator that the caller still expects to be unconsumed).
    _ = list(trajectories)
    return 0.0


def consensus_calibration(trajectories: Iterable[list[dict]]) -> float:
    """Cortex-only: correlation between confidence and realised reward.

    Returns 0.0 in Phase-2; Cortex Session 13 wires the real metric.

    TODO(cortex-session-13): when ``BrainRecommendation.top_confidence``
    is in trajectory rows, compute Pearson correlation between mean
    brain confidence and per-tick reward. Returns scalar in [-1, 1].
    """
    _ = list(trajectories)
    return 0.0


def novelty_yield(trajectories: Iterable[list[dict]]) -> float:
    """Cortex-only: round-2 action-change rate vs round-1.

    Returns 0.0 in Phase-2; Cortex Session 13 wires the real metric.

    TODO(cortex-session-13): when ``RouterStep.round`` is populated,
    compute fraction of ticks where round-2 emit_outer_action differs
    from the round-1 candidate top_action.
    """
    _ = list(trajectories)
    return 0.0


__all__ = [
    "collapse_rate",
    "consensus_calibration",
    "dissent_value",
    "novelty_yield",
]

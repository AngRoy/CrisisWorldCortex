# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Outer reward grader for CrisisWorld (design §15 + §19, post-Phase-1 fix).

Public API (re-exported via ``server/graders/__init__.py``):
- ``outer_reward(state, action) -> float`` — 6-component weighted score in
  ``[-1.0, 1.0]`` (Workstream-B Phase-1 range relaxation; was
  ``[0.0, 1.0]``). Negative values come from ``r_policy = -0.5`` on rejected
  actions and ``r_policy = -1.0`` on parse-failure markers (per design §19).
  Computed on post-``apply_tick`` state. Still the only env-side reward
  signal per ``server/CLAUDE.md``.
- ``terminal_bonus(state) -> float`` — episode-end ±0.20 / 0.0 bonus,
  composed by trainer in ``training/reward_shaping.py`` per design §14.3
  (``episode_return = Σ_t r_outer + terminal_bonus``). Kept separate from
  the per-tick scalar.

Phase-1 changes (Workstream B):
  - Steeper sensitivity on ``r_infect`` (``× 20``) and ``r_hosp`` (``× 10``)
    so the gentle outbreak_easy task still produces gradient.
  - Continuous ``r_casc`` (``1 - max(I)/0.30`` clamped) replaces binary.
  - ``r_policy`` ∈ {-1.0 (parse-failure), -0.5 (rejected), 0.0 (accepted
    no_op), +1.0 (accepted real action)} restoring §19 magnitudes.
  - Weight redistribution: W_POLICY 0.12 → 0.35 (signal-driver); W_TIME
    0.18 → 0.05 (action-independent noise); other components rebalanced.
  - Final ``[0,1]`` clamp dropped (Phase-A M2-A).

Wire-protocol imports use the absolute path ``CrisisWorldCortex.models``
because this file lives two levels deep inside ``server/`` — see
``server/simulator/seir_model.py``'s import block for the full rationale on
the dual-import / Pydantic class-identity trap. Internal server package
imports use package-relative paths so both ``CrisisWorldCortex.server.*``
and top-level ``server.*`` loading modes resolve within the same package.
"""

from __future__ import annotations

import statistics

from CrisisWorldCortex.models import OuterActionPayload

from ..simulator import (
    CATASTROPHIC_INFECTION_THRESHOLD,
    HOSPITAL_CAPACITY_FRACTION,
    HOSPITALIZATION_FRACTION_OF_I,
    WorldState,
)

# ============================================================================
# Component weights (design §15 + Phase-1 redistribution; sum to 1.00)
# ============================================================================

W_INFECT = 0.15  # was 0.35; iter-1 reduced because outbreak_easy keeps mean(I) tiny
W_TIME = 0.05  # was 0.18; action-independent, no signal value
W_HOSP = 0.10  # was 0.17; iter-1 reduction (gentle env keeps hosp_load low)
W_CASC = 0.10  # was 0.15; iter-1 reduction
W_POLICY = 0.55  # was 0.12; iter-1 dominant signal driver — accepted-real vs no_op vs rejected
W_FAIR = 0.05  # was 0.03; tiny boost

# Steepness coefficients (Phase-A M6, ONE-iteration tentative).
# r_infect ≈ 0 when mean(I) >= 0.05; near 1 when mean(I) <= 0.0 → strong gradient.
# r_hosp similarly sensitive to mean hospital_load.
R_INFECT_STEEPNESS = 20.0
R_HOSP_STEEPNESS = 10.0

# r_policy values per design §19 (Phase-1 restoration).
R_POLICY_PARSE_FAILURE = -1.0  # synthetic parse-failure marker
R_POLICY_REJECTED = -0.5  # well-formed-illegal (V2 / legal-violation)
R_POLICY_NOOP_ACCEPTED = 0.0  # accepted no-op (legal but inactive)
R_POLICY_REAL_ACCEPTED = 1.0  # accepted real intervention

# Terminal-bonus magnitudes (design §14.3 / §15).
TERMINAL_BONUS_SUCCESS = 0.20
TERMINAL_BONUS_FAILURE = -0.20

# r_casc threshold: at max(I) >= this, r_casc = 0 (catastrophe imminent).
# Matches design §6.4's catastrophic-infection threshold so cascade
# signal aligns with the failure terminal.
R_CASC_HOT_THRESHOLD = CATASTROPHIC_INFECTION_THRESHOLD

# Magic-string discriminator for parse-failure marker (Phase-A M3-B):
# baselines.flat_agent.parse_failure_marker emits PublicCommunication with
# honesty=0.0; intentional V2 attempts use honesty > 0.0.
PARSE_FAILURE_HONESTY_SENTINEL = 0.0


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _hospital_load(region_I: float) -> float:
    """Mirror the formula used in ``make_observation`` for ``hospital_load``.

    Reads latent ``I`` directly (graders see ground truth, not the delayed
    + noised observation). Saturates at ``I = HOSPITAL_CAPACITY_FRACTION /
    HOSPITALIZATION_FRACTION_OF_I`` (= 0.5 with current locked constants).
    """
    return _clamp01(region_I * HOSPITALIZATION_FRACTION_OF_I / HOSPITAL_CAPACITY_FRACTION)


def _r_policy_value(action: OuterActionPayload, accepted: bool) -> float:
    """Compute ``r_policy`` per design §19 four-state contract.

    Returns one of {-1.0, -0.5, 0.0, +1.0} based on (action.kind, accepted).
    Parse-failure detection uses the ``honesty == 0.0`` sentinel on a
    rejected ``PublicCommunication`` payload (Phase-A M3-B magic string).
    """
    if not accepted:
        # Rejected branch. Distinguish parse-failure marker from intentional
        # V2-PublicCommunication / legal-violation rejection.
        if (
            action.kind == "public_communication"
            and getattr(action, "honesty", None) == PARSE_FAILURE_HONESTY_SENTINEL
        ):
            return R_POLICY_PARSE_FAILURE
        return R_POLICY_REJECTED
    # Accepted branch.
    if action.kind == "no_op":
        return R_POLICY_NOOP_ACCEPTED
    return R_POLICY_REAL_ACCEPTED


def outer_reward(state: WorldState, action: OuterActionPayload) -> float:
    """Compute per-tick outer reward in ``[-1.0, 1.0]`` (post-Phase-1 range).

    Read post-``apply_tick`` state: ``state.regions[*].I`` is the
    just-stepped ground-truth infection fraction. ``recent_action_log[-1]``
    holds the just-dispatched action's acceptance flag.

    Six components (design §15 + Phase-1 fix):
        r_infect  = max(0, 1 - 20 × mean(I))                  # weight 0.25
        r_time    = 1 - tick / max_ticks                       # weight 0.05
        r_hosp    = max(0, 1 - 10 × mean(hospital_load))       # weight 0.15
        r_casc    = max(0, 1 - max(I) / 0.30)                  # weight 0.15
        r_policy  = {-1.0, -0.5, 0.0, +1.0} per §19            # weight 0.35
        r_fair    = 1 - pstdev(I)                              # weight 0.05

    The ``action`` argument is the action just dispatched. We read its
    acceptance flag from ``state.recent_action_log[-1]`` rather than
    re-dispatching — the simulator already recorded it, and re-dispatch
    would mutate state.
    """
    if not state.regions:
        return 0.0

    I_values = [r.I for r in state.regions]
    mean_I = sum(I_values) / len(I_values)

    # r_infect: steepened so gentle outbreak_easy still produces gradient.
    r_infect = _clamp01(1.0 - R_INFECT_STEEPNESS * mean_I)

    if state.max_ticks > 0:
        r_time = _clamp01(1.0 - state.tick / state.max_ticks)
    else:
        r_time = 0.0

    hosp_loads = [_hospital_load(I) for I in I_values]
    mean_hosp = sum(hosp_loads) / len(hosp_loads)
    r_hosp = _clamp01(1.0 - R_HOSP_STEEPNESS * mean_hosp)

    # r_casc: continuous ramp (1.0 at max(I)=0; 0.0 at max(I) >= threshold).
    max_I = max(I_values)
    if R_CASC_HOT_THRESHOLD > 0:
        r_casc = _clamp01(1.0 - max_I / R_CASC_HOT_THRESHOLD)
    else:
        r_casc = 0.0

    # r_policy: design §19 four-state contract.
    if state.recent_action_log:
        last_entry = state.recent_action_log[-1]
        r_policy = _r_policy_value(last_entry.action, last_entry.accepted)
    else:
        r_policy = R_POLICY_REAL_ACCEPTED  # No action dispatched yet → no penalty.

    if len(I_values) >= 2:
        r_fair = _clamp01(1.0 - statistics.pstdev(I_values))
    else:
        r_fair = 1.0  # Single-region task: no fairness gap to penalize.

    score = (
        W_INFECT * r_infect
        + W_TIME * r_time
        + W_HOSP * r_hosp
        + W_CASC * r_casc
        + W_POLICY * r_policy
        + W_FAIR * r_fair
    )
    # Final clamp to [-1.0, 1.0] (no longer [0, 1] — M2-A drops the floor).
    if score < -1.0:
        return -1.0
    if score > 1.0:
        return 1.0
    return score


def terminal_bonus(state: WorldState) -> float:
    """Episode-end bonus per design §14.3 / §15.

    Returns ``+0.20`` on ``success``, ``-0.20`` on ``failure``, ``0.0`` on
    ``timeout`` or any non-terminal state. Composed downstream by
    ``training/reward_shaping.py``; kept separate from ``outer_reward`` so
    the per-tick scalar stays inside ``[0.0, 1.0]``.
    """
    if state.terminal == "success":
        return TERMINAL_BONUS_SUCCESS
    if state.terminal == "failure":
        return TERMINAL_BONUS_FAILURE
    return 0.0

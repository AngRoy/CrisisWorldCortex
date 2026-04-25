# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Outer reward grader for CrisisWorld (design §15).

Public API (re-exported via ``server/graders/__init__.py``):
- ``outer_reward(state, action) -> float`` — 6-component weighted score in
  ``[0.0, 1.0]``, computed on post-``apply_tick`` state. The only env-side
  reward signal per ``server/CLAUDE.md`` (binding contract).
- ``terminal_bonus(state) -> float`` — episode-end ±0.20 / 0.0 bonus,
  composed by trainer in ``training/reward_shaping.py`` per design §14.3
  (``episode_return = Σ_t r_outer + terminal_bonus``). Kept separate so the
  per-tick ``r_outer`` stays inside ``[0.0, 1.0]``.

Wire-protocol imports use the absolute path ``CrisisWorldCortex.models``
because this file lives two levels deep inside ``server/`` — see
``server/simulator/seir_model.py``'s import block for the full rationale on
the dual-import / Pydantic class-identity trap. Internal server package
imports use ``server.*`` because the container entrypoint imports
``server.app:app`` from ``/app/env``.
"""

from __future__ import annotations

import statistics

from CrisisWorldCortex.models import OuterActionPayload
from server.simulator import (
    CATASTROPHIC_INFECTION_THRESHOLD,
    HOSPITAL_CAPACITY_FRACTION,
    HOSPITALIZATION_FRACTION_OF_I,
    WorldState,
)

# ============================================================================
# Component weights (design §15; sum to 1.00)
# ============================================================================

W_INFECT = 0.35
W_TIME = 0.18
W_HOSP = 0.17
W_CASC = 0.15
W_POLICY = 0.12
W_FAIR = 0.03

# Terminal-bonus magnitudes (design §14.3 / §15).
TERMINAL_BONUS_SUCCESS = 0.20
TERMINAL_BONUS_FAILURE = -0.20

# r_casc threshold: any region above this contributes a "hot" count;
# matches the catastrophic-region threshold from design §6.4 so the
# cascade-control signal aligns with the failure terminal.
R_CASC_HOT_THRESHOLD = CATASTROPHIC_INFECTION_THRESHOLD


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


def outer_reward(state: WorldState, action: OuterActionPayload) -> float:
    """Compute per-tick outer reward in ``[0.0, 1.0]``.

    Read post-``apply_tick`` state: ``state.regions[*].I`` is the
    just-stepped ground-truth infection fraction. ``recent_action_log[-1]``
    holds the just-dispatched action's acceptance flag.

    Six components (design §15):
        r_infect  = 1 - mean(I)                          # weight 0.35
        r_time    = 1 - tick / max_ticks                 # weight 0.18
        r_hosp    = 1 - mean(hospital_load)              # weight 0.17
        r_casc    = 1 if no region exceeds 0.30 else 0   # weight 0.15
        r_policy  = 1 if last action accepted else 0     # weight 0.12
        r_fair    = 1 - pstdev(I)                        # weight 0.03

    The ``action`` argument is the action just dispatched. We read its
    acceptance flag from ``state.recent_action_log[-1]`` rather than
    re-dispatching — the simulator already recorded it, and re-dispatch
    would mutate state. ``action`` is kept in the signature for design
    parity with ``training_reward(trajectory)`` and to support future
    action-shape-dependent shaping without a signature break.
    """
    if not state.regions:
        return 0.0

    I_values = [r.I for r in state.regions]

    r_infect = _clamp01(1.0 - sum(I_values) / len(I_values))

    if state.max_ticks > 0:
        r_time = _clamp01(1.0 - state.tick / state.max_ticks)
    else:
        r_time = 0.0

    hosp_loads = [_hospital_load(I) for I in I_values]
    r_hosp = _clamp01(1.0 - sum(hosp_loads) / len(hosp_loads))

    hot_regions = sum(1 for I in I_values if I > R_CASC_HOT_THRESHOLD)
    r_casc = 1.0 if hot_regions == 0 else 0.0

    if state.recent_action_log:
        last_entry = state.recent_action_log[-1]
        r_policy = 1.0 if last_entry.accepted else 0.0
    else:
        r_policy = 1.0  # No action dispatched yet → no penalty.

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
    # Final clamp guards against floating-point drift past 1.0.
    return _clamp01(score)


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

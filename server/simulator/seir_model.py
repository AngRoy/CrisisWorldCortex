# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SEIR-style world simulator for CrisisWorld (design §6).

Public API (re-exported via ``server/simulator/__init__.py``):
- ``apply_tick(state, action, seed=None) -> WorldState`` — advance one tick.
- ``make_observation(state, seed=None) -> CrisisworldcortexObservation``
  — project latent state to wire-format observation with telemetry delay/noise.

Internal types (``WorldState``, ``RegionLatentState``, ``TaskConfig``,
``SuperSpreaderEvent``, ``PendingEffect``, ``ChainBeta``) are defined here
so latent fields cannot be reached from anything that imports the wire
package — the wire/internal boundary is enforced structurally.

Determinism: every random draw goes through ``random.Random(seed)`` with
seed derived from ``(episode_seed, tick)``. ``apply_tick`` and
``make_observation`` use independent seed streams so observation noise
is decorrelated from dynamics randomness.

Modeling notes:
- SEIR uses 4 fractions (S/E/I/R) per region, sum to 1.0 after each step.
- ``base_R0`` per task is converted to per-tick transmission rate
  ``β = R_0 * γ`` inside ``_seir_step``. Cross-region β values are
  direct transmission rates (not R_0 conversions) per design §10.
- ``_advance_terminal_state`` mutates state (advances
  ``consecutive_safe_ticks``); name reflects this.
"""

from __future__ import annotations

import random
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

# Wire-protocol imports use the absolute path ``CrisisWorldCortex.models``
# (canonicalized form per root CLAUDE.md). The dual-import fallback used by
# server/CrisisWorldCortex_environment.py works there because ``..models`` from
# ``CrisisWorldCortex.server.<env>`` resolves to ``CrisisWorldCortex.models`` —
# but ``..models`` from ``CrisisWorldCortex.server.simulator.seir_model`` (two
# levels deep) resolves to a non-existent ``CrisisWorldCortex.server.models``,
# so the fallback would fire and load bare ``models``. That creates a second
# ``sys.modules`` entry with distinct class objects, and Pydantic's
# discriminated-union validator rejects ``isinstance`` checks against inputs
# constructed via the other path. Using the absolute path here avoids the trap.
from CrisisWorldCortex.models import (
    CrisisworldcortexObservation,
    DeployResource,
    Escalate,
    ExecutedAction,
    LegalConstraint,
    OuterActionPayload,
    ReallocateBudget,
    RegionId,
    RegionTelemetry,
    RequestData,
    ResourceInventory,
    ResourceType,
    Restriction,
    RestrictMovement,
)

# ============================================================================
# SEIR rate constants (design §1a; locked)
# ============================================================================

SIGMA = 0.4  # E -> I rate per tick (latent ~2.5 ticks)
GAMMA = 0.2  # I -> R rate per tick (infectious ~5 ticks)
HOSPITALIZATION_FRACTION_OF_I = 0.10
COMPLIANCE_DECAY_PER_STRICT_TICK = 0.03
COMPLIANCE_RECOVERY_PER_RELAXED_TICK = 0.02

# Hospital capacity threshold: I-fraction at which hospital_load saturates.
# I * 0.10 / 0.05 = I/0.5 → I=0.5 → load=1.0. Tunable; not load-bearing.
HOSPITAL_CAPACITY_FRACTION = 0.05

# Population per region (Q-Sim-1 option a; locked).
POPULATION_PER_REGION = 1000

# Severity multipliers for restrict_movement action (design proposal §10).
SEVERITY_MULTIPLIER: Dict[str, float] = {
    "none": 0.0,
    "light": 0.1,
    "moderate": 0.25,
    "strict": 0.5,
}

# Resource efficacy per unit deployed (per-tick I reduction; design §10 item 11).
EFFICACY_PER_UNIT: Dict[str, float] = {
    "test_kits": 0.00002,  # 1000 kits → -0.02 I/tick
    "hospital_beds": 0.0,  # affects hospital_load downstream, not I
    "mobile_units": 0.0,  # reserved for cross-region β reduction
    "vaccine_doses": 0.0001,  # 1000 doses → -0.10 I/tick (and S → R)
}

# Reallocation efficiency loss (5%).
REALLOCATION_EFFICIENCY = 0.95

# Action log buffer size (cap on `recent_action_log`).
ACTION_LOG_BUFFER = 8

# Terminal-condition thresholds (design §5a; locked).
CATASTROPHIC_INFECTION_THRESHOLD = 0.30
CATASTROPHIC_REGION_COUNT = 3
SAFE_INFECTION_THRESHOLD = 0.05
SAFE_CONSECUTIVE_TICKS = 3


# ============================================================================
# Internal state types (NOT on the wire)
# ============================================================================


class PendingEffect(BaseModel):
    """A queued resource effect that decays over a few ticks."""

    kind: ResourceType
    magnitude: float = Field(ge=0.0)
    ticks_remaining: int = Field(ge=0)


class SuperSpreaderEvent(BaseModel):
    """A scheduled latent perturbation (hard task only)."""

    region: RegionId
    fires_at_tick: int = Field(ge=0)
    surfaces_at_tick: int = Field(ge=0)
    magnitude_I: float = Field(ge=0.0, le=1.0)


class ChainBeta(BaseModel):
    """A directed cross-region transmission coefficient."""

    from_region: RegionId
    to_region: RegionId
    beta: float = Field(ge=0.0)


class RegionLatentState(BaseModel):
    """Per-region latent SEIR state. Never serialized over the wire."""

    region: RegionId
    S: float = Field(ge=0.0, le=1.0)
    E: float = Field(ge=0.0, le=1.0)
    I: float = Field(ge=0.0, le=1.0)
    R: float = Field(ge=0.0, le=1.0)
    true_compliance: float = Field(ge=0.0, le=1.0)
    history_I: List[float] = Field(default_factory=list)
    pending_effects: List[PendingEffect] = Field(default_factory=list)
    noise_reduction_ticks: int = Field(default=0, ge=0)


class TaskConfig(BaseModel):
    """Configuration for one CrisisWorld task (design §6.5)."""

    name: str
    region_count: int = Field(ge=1)
    max_ticks: int = Field(default=12, ge=1)
    base_R0: float = Field(ge=0.0)
    default_cross_beta: float = Field(default=0.0, ge=0.0)
    chain_betas: List[ChainBeta] = Field(default_factory=list)
    telemetry_delay_ticks: int = Field(ge=0)
    telemetry_noise_stddev_cases: float = Field(ge=0.0)
    telemetry_noise_stddev_compliance: float = Field(ge=0.0)
    cognition_budget_per_tick: int = Field(default=6000, ge=0)
    initial_resources: ResourceInventory
    initial_compliance: float = Field(ge=0.0, le=1.0)
    initial_seir_hot: Tuple[float, float, float, float]
    initial_seir_quiet: Optional[Tuple[float, float, float, float]] = None
    hot_regions: List[RegionId] = Field(default_factory=list)
    quiet_regions: List[RegionId] = Field(default_factory=list)
    superspreader_schedule: List[SuperSpreaderEvent] = Field(default_factory=list)
    legal_constraints: List[LegalConstraint] = Field(default_factory=list)


class WorldState(BaseModel):
    """Live simulator state. Holds latent fields that never reach the wire."""

    task_name: Literal["outbreak_easy", "outbreak_medium", "outbreak_hard"]
    task_config: TaskConfig
    episode_seed: int
    tick: int = Field(default=0, ge=0)
    max_ticks: int = Field(ge=1)
    regions: List[RegionLatentState]
    resources: ResourceInventory
    restrictions: Dict[RegionId, Restriction] = Field(default_factory=dict)
    legal_constraints: List[LegalConstraint] = Field(default_factory=list)
    escalation_level: int = Field(default=0, ge=0, le=2)
    escalation_unlocked_strict: bool = False
    superspreader_schedule: List[SuperSpreaderEvent] = Field(default_factory=list)
    recent_action_log: List[ExecutedAction] = Field(default_factory=list)
    consecutive_safe_ticks: int = Field(default=0, ge=0)
    terminal: Literal["none", "success", "failure", "timeout"] = "none"


# ============================================================================
# Determinism helpers
# ============================================================================


def _derive_tick_seed(episode_seed: int, tick: int) -> int:
    """RNG seed for ``apply_tick`` per (episode, tick)."""
    return (episode_seed * 1_000_003) ^ (tick * 31)


def _derive_obs_seed(episode_seed: int, tick: int) -> int:
    """Independent RNG seed for ``make_observation`` so observation noise
    is decorrelated from dynamics randomness."""
    return (episode_seed * 999_983) ^ (tick * 17)


# ============================================================================
# Action handlers — return accepted: bool; mutate state in place
# ============================================================================


def _find_region(state: WorldState, region_id: RegionId) -> Optional[RegionLatentState]:
    for r in state.regions:
        if r.region == region_id:
            return r
    return None


def _resource_attr(resource_type: ResourceType) -> str:
    """Map ResourceType literal to ResourceInventory attribute name."""
    if resource_type == "hospital_beds":
        return "hospital_beds_free"
    return resource_type  # test_kits, mobile_units, vaccine_doses


def _apply_deploy_resource(state: WorldState, a: DeployResource, rng: random.Random) -> bool:
    region = _find_region(state, a.region)
    if region is None:
        return False
    attr = _resource_attr(a.resource_type)
    available = getattr(state.resources, attr)
    if available < a.quantity:
        return False
    setattr(state.resources, attr, available - a.quantity)
    magnitude = EFFICACY_PER_UNIT[a.resource_type] * a.quantity
    if magnitude > 0:
        region.pending_effects.append(
            PendingEffect(
                kind=a.resource_type,
                magnitude=magnitude,
                ticks_remaining=2,
            )
        )
    return True


def _apply_request_data(state: WorldState, a: RequestData, rng: random.Random) -> bool:
    region = _find_region(state, a.region)
    if region is None:
        return False
    region.noise_reduction_ticks = max(region.noise_reduction_ticks, 3)
    return True


def _apply_restrict_movement(state: WorldState, a: RestrictMovement, rng: random.Random) -> bool:
    if a.severity == "strict" and not state.escalation_unlocked_strict:
        # Legal-constraint violation: action rejected, state unchanged.
        return False
    region = _find_region(state, a.region)
    if region is None:
        return False
    state.restrictions[a.region] = Restriction(
        region=a.region,
        severity=a.severity,
        ticks_remaining=4,
    )
    return True


def _apply_escalate(state: WorldState, a: Escalate, rng: random.Random) -> bool:
    if a.to_authority == "national":
        state.escalation_unlocked_strict = True
    state.escalation_level = min(state.escalation_level + 1, 2)
    return True


def _apply_reallocate_budget(state: WorldState, a: ReallocateBudget, rng: random.Random) -> bool:
    from_attr = _resource_attr(a.from_resource)
    to_attr = _resource_attr(a.to_resource)
    available = getattr(state.resources, from_attr)
    if available < a.amount:
        return False
    setattr(state.resources, from_attr, available - a.amount)
    transferred = round(a.amount * REALLOCATION_EFFICIENCY)
    setattr(state.resources, to_attr, getattr(state.resources, to_attr) + transferred)
    return True


def _dispatch_action(state: WorldState, action: OuterActionPayload, rng: random.Random) -> bool:
    """Dispatch action to its handler. Returns accepted: bool."""
    if action.kind == "no_op":
        return True
    if action.kind == "public_communication":
        return False  # V2-rejected per design §6.3 / §19
    if action.kind == "deploy_resource":
        return _apply_deploy_resource(state, action, rng)
    if action.kind == "request_data":
        return _apply_request_data(state, action, rng)
    if action.kind == "restrict_movement":
        return _apply_restrict_movement(state, action, rng)
    if action.kind == "escalate":
        return _apply_escalate(state, action, rng)
    if action.kind == "reallocate_budget":
        return _apply_reallocate_budget(state, action, rng)
    return False


# ============================================================================
# Dynamics helpers
# ============================================================================


def _apply_pending_effects(state: WorldState) -> None:
    """Apply queued resource effects to I (and S→R for vaccines)."""
    for region in state.regions:
        for effect in region.pending_effects:
            if effect.kind == "test_kits":
                shift = min(region.I, effect.magnitude)
                region.I -= shift
                region.R = min(1.0, region.R + shift)
            elif effect.kind == "vaccine_doses":
                shift = min(region.S, effect.magnitude)
                region.S -= shift
                region.R += shift


def _apply_scheduled_superspreaders(state: WorldState) -> None:
    """Inject scheduled +I perturbations at their fire tick."""
    for event in state.superspreader_schedule:
        if event.fires_at_tick == state.tick:
            region = _find_region(state, event.region)
            if region is not None:
                region.I = min(1.0, region.I + event.magnitude_I)


def _seir_step(state: WorldState, rng: random.Random) -> None:
    """Discrete SEIR update for all regions.

    Within-region: β_within = R_0_eff * γ. Cross-region: explicit β
    coefficients from task config (chain + default fallback).
    Effective R_0 reduced by restriction severity and scaled by compliance.
    """
    I_snapshot = {r.region: r.I for r in state.regions}

    for region in state.regions:
        # Restriction severity for this region (defaults to "none").
        restriction = state.restrictions.get(region.region)
        sev_mult = SEVERITY_MULTIPLIER[restriction.severity] if restriction else 0.0
        R_0_eff = state.task_config.base_R0 * (1 - sev_mult) * region.true_compliance
        beta_within = R_0_eff * GAMMA

        within = beta_within * region.S * region.I

        # Cross-region transmission: chain edges override the default.
        cross = 0.0
        for other_id, other_I in I_snapshot.items():
            if other_id == region.region:
                continue
            beta = state.task_config.default_cross_beta
            for ce in state.task_config.chain_betas:
                if ce.from_region == other_id and ce.to_region == region.region:
                    beta = ce.beta
                    break
            cross += beta * region.S * other_I

        new_infections = within + cross

        new_S = region.S - new_infections
        new_E = region.E + new_infections - SIGMA * region.E
        new_I = region.I + SIGMA * region.E - GAMMA * region.I
        new_R = region.R + GAMMA * region.I

        # Clamp + renormalize so S+E+I+R == 1.0.
        new_S = max(0.0, min(1.0, new_S))
        new_E = max(0.0, min(1.0, new_E))
        new_I = max(0.0, min(1.0, new_I))
        new_R = max(0.0, min(1.0, new_R))
        total = new_S + new_E + new_I + new_R
        if total > 0:
            region.S = new_S / total
            region.E = new_E / total
            region.I = new_I / total
            region.R = new_R / total


def _compliance_dynamics(state: WorldState) -> None:
    """Compliance decays under strict restrictions; recovers otherwise."""
    for region in state.regions:
        restriction = state.restrictions.get(region.region)
        if restriction is not None and restriction.severity == "strict":
            region.true_compliance = max(
                0.0,
                region.true_compliance - COMPLIANCE_DECAY_PER_STRICT_TICK,
            )
        else:
            region.true_compliance = min(
                1.0,
                region.true_compliance + COMPLIANCE_RECOVERY_PER_RELAXED_TICK,
            )


def _decrement_counters(state: WorldState) -> None:
    """Decrement ticks_remaining counters; remove expired entries."""
    for region in state.regions:
        if region.noise_reduction_ticks > 0:
            region.noise_reduction_ticks -= 1
        survivors: List[PendingEffect] = []
        for e in region.pending_effects:
            if e.ticks_remaining > 1:
                survivors.append(
                    PendingEffect(
                        kind=e.kind,
                        magnitude=e.magnitude,
                        ticks_remaining=e.ticks_remaining - 1,
                    )
                )
        region.pending_effects = survivors

    new_restrictions: Dict[RegionId, Restriction] = {}
    for region_id, restriction in state.restrictions.items():
        if restriction.ticks_remaining > 1:
            new_restrictions[region_id] = Restriction(
                region=restriction.region,
                severity=restriction.severity,
                ticks_remaining=restriction.ticks_remaining - 1,
            )
    state.restrictions = new_restrictions


def _advance_terminal_state(state: WorldState) -> None:
    """Update ``state.consecutive_safe_ticks`` and set ``state.terminal``.

    Mutating: this is NOT a pure predicate. It increments / resets the
    consecutive-safe counter as a side effect, then sets ``terminal``
    to one of {"none", "success", "failure", "timeout"} per design §6.4.
    Called at end of ``apply_tick`` after ``state.tick`` has advanced.
    """
    if state.tick >= state.max_ticks:
        state.terminal = "timeout"
        return

    catastrophic = sum(1 for r in state.regions if r.I > CATASTROPHIC_INFECTION_THRESHOLD)
    if catastrophic >= CATASTROPHIC_REGION_COUNT:
        state.terminal = "failure"
        return

    all_safe_now = all(r.I < SAFE_INFECTION_THRESHOLD for r in state.regions)
    if all_safe_now:
        state.consecutive_safe_ticks += 1
    else:
        state.consecutive_safe_ticks = 0

    if state.consecutive_safe_ticks >= SAFE_CONSECUTIVE_TICKS:
        state.terminal = "success"
        return

    state.terminal = "none"


# ============================================================================
# Public API: apply_tick + make_observation
# ============================================================================


def apply_tick(
    state: WorldState,
    action: OuterActionPayload,
    seed: Optional[int] = None,
) -> WorldState:
    """Advance one tick. Deterministic given (state, action, seed).

    Steps (in order):
    1. Dispatch action to handler; record acceptance flag.
    2. Append ExecutedAction to recent_action_log (capped at 8).
    3. Apply queued pending_effects (resource decay).
    4. Fire scheduled superspreader events (if any fire this tick).
    5. SEIR step (within-region + cross-region).
    6. Compliance dynamics.
    7. Decrement counters (noise_reduction, restrictions, pending_effects).
    8. Append post-step I to per-region history_I buffer.
    9. Advance ``state.tick``.
    10. Update ``state.terminal`` (mutating).

    Rejected actions still advance the tick — the SEIR step runs whether
    or not the action was accepted.
    """
    effective_seed = seed if seed is not None else _derive_tick_seed(state.episode_seed, state.tick)
    rng = random.Random(effective_seed)

    accepted = _dispatch_action(state, action, rng)

    state.recent_action_log.append(
        ExecutedAction(
            tick=state.tick,
            action=action,
            accepted=accepted,
        )
    )
    if len(state.recent_action_log) > ACTION_LOG_BUFFER:
        state.recent_action_log = state.recent_action_log[-ACTION_LOG_BUFFER:]

    _apply_pending_effects(state)
    _apply_scheduled_superspreaders(state)
    _seir_step(state, rng)
    _compliance_dynamics(state)
    _decrement_counters(state)

    for region in state.regions:
        region.history_I.append(region.I)

    state.tick += 1
    _advance_terminal_state(state)
    return state


def make_observation(
    state: WorldState,
    seed: Optional[int] = None,
) -> CrisisworldcortexObservation:
    """Project latent state to wire-format observation (pure function).

    Applies per-region telemetry delay (history-buffer indexed at
    ``tick - delay``) and Gaussian noise per task config. ``request_data``
    halves the noise stddev for ``noise_reduction_ticks > 0`` regions.

    No latent SEIR fields appear in the output — only the declared
    ``CrisisworldcortexObservation`` fields.
    """
    effective_seed = seed if seed is not None else _derive_obs_seed(state.episode_seed, state.tick)
    rng = random.Random(effective_seed)

    delay = state.task_config.telemetry_delay_ticks
    regions_obs: List[RegionTelemetry] = []
    for region in state.regions:
        # history_I[k] is I after the k-th apply_tick. At current tick=T,
        # history has T+1 entries (initial at index 0 + one per applied tick).
        # We want I from `delay` ticks ago: index max(0, T - delay).
        delayed_idx = max(0, state.tick - delay)
        if delayed_idx < len(region.history_I):
            I_delayed = region.history_I[delayed_idx]
        else:
            I_delayed = region.history_I[-1] if region.history_I else region.I

        # reported_cases ≈ noisy estimate, scaled to absolute count.
        noise_stddev_cases = state.task_config.telemetry_noise_stddev_cases
        if region.noise_reduction_ticks > 0:
            noise_stddev_cases *= 0.5
        true_cases = I_delayed * POPULATION_PER_REGION
        observed_cases = int(
            rng.gauss(
                true_cases,
                noise_stddev_cases * POPULATION_PER_REGION,
            )
        )
        observed_cases = max(0, observed_cases)

        # hospital_load: less delayed (operational signal); current I.
        hospital_load = max(
            0.0,
            min(
                1.0,
                region.I * HOSPITALIZATION_FRACTION_OF_I / HOSPITAL_CAPACITY_FRACTION,
            ),
        )

        # compliance_proxy: noisy estimate of true_compliance.
        noise_stddev_comp = state.task_config.telemetry_noise_stddev_compliance
        if region.noise_reduction_ticks > 0:
            noise_stddev_comp *= 0.5
        compliance_proxy = max(
            0.0,
            min(
                1.0,
                rng.gauss(region.true_compliance, noise_stddev_comp),
            ),
        )

        regions_obs.append(
            RegionTelemetry(
                region=region.region,
                reported_cases_d_ago=observed_cases,
                hospital_load=hospital_load,
                compliance_proxy=compliance_proxy,
            )
        )

    obs = CrisisworldcortexObservation(
        regions=regions_obs,
        resources=state.resources,
        active_restrictions=list(state.restrictions.values()),
        legal_constraints=state.legal_constraints,
        tick=state.tick,
        ticks_remaining=max(0, state.max_ticks - state.tick),
        cognition_budget_remaining=state.task_config.cognition_budget_per_tick,
        recent_action_log=list(state.recent_action_log),
    )
    obs.done = state.terminal != "none"
    return obs

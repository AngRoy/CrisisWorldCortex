"""Brain-specific observation lenses (Session 10).

Phase A docs/CORTEX_ARCHITECTURE.md Decisions 9-14 + §2 A1.

Lenses are pure-Python: no LLM, no I/O, no state. ``lens_for(brain, obs,
last_reward)`` dispatches to one of 3 brain-specific helpers and returns
a ``BrainLensedObservation``. V2 brain ids raise ``KeyError`` per
Decision 9 (post-review) -- no MVP stub functions.

The lens does NOT strip the raw observation (Decision 13); subagents
may need fields the lens didn't emphasise. ``salient_field_ids`` is a
salience map alongside ``raw_obs``, not a replacement for it.

``transmission_rate_trend`` is fixed at 0.0 in MVP (M-FR-2): the lens
sees one observation per call. Session 11 plumbs prior-tick obs into
the lens to enable real trend computation.
"""

from __future__ import annotations

from typing import Callable, Dict

from cortex.schemas import BrainLensedObservation
from CrisisWorldCortex.models import CrisisworldcortexObservation

_V2_BRAINS = frozenset({"communications", "equity"})


def lens_for(
    brain: str,
    obs: CrisisworldcortexObservation,
    last_reward: float,
) -> BrainLensedObservation:
    """Return the per-brain lensed observation.

    Args:
        brain: One of {"epidemiology", "logistics", "governance"}.
        obs: The current tick's observation.
        last_reward: Previous tick's reward (plumbed from B1's pattern,
            included on the lensed object so all 3 subagents in this
            brain see the same recency signal).

    Raises:
        KeyError: If ``brain`` is a V2-deferred brain (communications,
            equity) or unknown.
    """
    helper = _LENS_REGISTRY.get(brain)
    if helper is not None:
        return helper(obs, last_reward)
    if brain in _V2_BRAINS:
        raise KeyError(
            f"V2 brain {brain!r} deferred per Phase A Decision 9; "
            f"no MVP stub lens. See docs/CORTEX_ARCHITECTURE.md."
        )
    raise KeyError(f"unknown brain: {brain!r}")


# ============================================================================
# Per-brain lens helpers
# ============================================================================


def _epi_lens(obs: CrisisworldcortexObservation, last_reward: float) -> BrainLensedObservation:
    """Epidemiology lens (Decision 10, M-FR-4 rename: epi_pressure)."""
    n_regions = max(1, len(obs.regions))
    mean_hospital_load = sum(r.hospital_load for r in obs.regions) / n_regions
    # M-FR-4: pressure scalar correlated with R_eff but not a true R_eff
    # estimate. WorldModeler subagent computes proper R_eff during reasoning.
    epi_pressure = max(0.0, min(3.0, mean_hospital_load * 2.0))

    max_cases = max((r.reported_cases_d_ago for r in obs.regions), default=0)
    # /1000 normaliser matches the design-doc "~30 cases / 1000 pop" spec
    worst_region_infection = max(0.0, min(1.0, max_cases / 1000.0))

    return BrainLensedObservation(
        brain="epidemiology",
        raw_obs=obs,
        salient_field_ids=[
            "regions[*].reported_cases_d_ago",
            "regions[*].hospital_load",
            "regions[*].compliance_proxy",
        ],
        derived_features={
            "epi_pressure": float(epi_pressure),
            "worst_region_infection": float(worst_region_infection),
            # M-FR-2: needs history; Session 11 plumbs prior-tick obs.
            "transmission_rate_trend": 0.0,
        },
        last_reward=last_reward,
    )


def _logistics_lens(
    obs: CrisisworldcortexObservation, last_reward: float
) -> BrainLensedObservation:
    """Logistics lens (Decision 11, M-FR-3 floor 0.5, M-FR-6 flat keys)."""
    res = obs.resources
    total_inventory = float(
        res.test_kits + res.hospital_beds_free + res.mobile_units + res.vaccine_doses
    )

    hospital_load_max = (
        max((r.hospital_load for r in obs.regions), default=0.0) if obs.regions else 0.0
    )

    # Per-region feasibility flat keys (D14 + M-FR-6).
    strict_regions = {r.region for r in obs.active_restrictions if r.severity == "strict"}
    feasibility: Dict[str, float] = {}
    for r in obs.regions:
        key = f"deployment_feasibility_{r.region}"
        if total_inventory <= 0.0:
            feasibility[key] = 0.0
        elif r.region in strict_regions:
            # M-FR-3: 0.5 floor when strict restriction is in place but
            # units could still be helicoptered in; Planner does the
            # legal-check.
            feasibility[key] = 0.5
        else:
            feasibility[key] = 1.0

    derived_features: Dict[str, float] = {
        "total_inventory": total_inventory,
        "hospital_load_max": float(hospital_load_max),
        **feasibility,
    }

    return BrainLensedObservation(
        brain="logistics",
        raw_obs=obs,
        salient_field_ids=[
            "resources.test_kits",
            "resources.hospital_beds_free",
            "resources.mobile_units",
            "resources.vaccine_doses",
            "regions[*].hospital_load",
            "active_restrictions[*]",
        ],
        derived_features=derived_features,
        last_reward=last_reward,
    )


def _governance_lens(
    obs: CrisisworldcortexObservation, last_reward: float
) -> BrainLensedObservation:
    """Governance lens (Decision 12)."""
    # escalation_unlocked_strict: any accepted escalate(national) in the log
    escalation_unlocked = 0.0
    for ea in obs.recent_action_log:
        if (
            ea.accepted
            and ea.action.kind == "escalate"
            and getattr(ea.action, "to_authority", None) == "national"
        ):
            escalation_unlocked = 1.0
            break

    return BrainLensedObservation(
        brain="governance",
        raw_obs=obs,
        salient_field_ids=[
            "active_restrictions[*]",
            "legal_constraints[*]",
            "recent_action_log[*]",
        ],
        derived_features={
            "escalation_unlocked_strict": escalation_unlocked,
            "legal_constraints_count": float(len(obs.legal_constraints)),
            "restrictions_active_count": float(len(obs.active_restrictions)),
        },
        last_reward=last_reward,
    )


# ============================================================================
# Registry (defined after helpers so closures resolve cleanly)
# ============================================================================


_LENS_REGISTRY: Dict[
    str, Callable[[CrisisworldcortexObservation, float], BrainLensedObservation]
] = {
    "epidemiology": _epi_lens,
    "logistics": _logistics_lens,
    "governance": _governance_lens,
}

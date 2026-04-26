"""Session 10 - Cortex lens tests.

Per Phase A docs/CORTEX_ARCHITECTURE.md Decisions 9-14 + §2 A1 and the
user's Session 10 proposal acceptance with 8 tests + the M-FR-4 rename
(epi_pressure) and T7 tightening (non-bool float).
"""

from __future__ import annotations

from typing import Iterable

import pytest

from cortex.lenses import lens_for
from cortex.schemas import BrainLensedObservation
from CrisisWorldCortex.models import (
    CrisisworldcortexObservation,
    Escalate,
    ExecutedAction,
    LegalConstraint,
    NoOp,
    RegionTelemetry,
    ResourceInventory,
    Restriction,
)

# ============================================================================
# Test fixtures
# ============================================================================


def _make_obs(
    cases_per_region: Iterable[int] = (5, 1, 1, 1),
    hospital_loads: Iterable[float] = (0.3, 0.1, 0.1, 0.1),
    compliance_proxies: Iterable[float] = (0.85, 0.95, 0.95, 0.95),
    test_kits: int = 1000,
    hospital_beds_free: int = 500,
    mobile_units: int = 20,
    vaccine_doses: int = 2000,
    restrictions: Iterable[Restriction] = (),
    legal_constraints: Iterable[LegalConstraint] = (),
    recent_action_log: Iterable[ExecutedAction] = (),
) -> CrisisworldcortexObservation:
    cases = list(cases_per_region)
    loads = list(hospital_loads)
    comps = list(compliance_proxies)
    return CrisisworldcortexObservation(
        regions=[
            RegionTelemetry(
                region=f"R{i + 1}",
                reported_cases_d_ago=cases[i],
                hospital_load=loads[i],
                compliance_proxy=comps[i],
            )
            for i in range(4)
        ],
        resources=ResourceInventory(
            test_kits=test_kits,
            hospital_beds_free=hospital_beds_free,
            mobile_units=mobile_units,
            vaccine_doses=vaccine_doses,
        ),
        active_restrictions=list(restrictions),
        legal_constraints=list(legal_constraints),
        tick=3,
        ticks_remaining=9,
        cognition_budget_remaining=5200,
        recent_action_log=list(recent_action_log),
    )


# ============================================================================
# T1 - Epi lens emphasizes telemetry; uses epi_pressure (M-FR-4 rename)
# ============================================================================


def test_epi_lens_emphasizes_telemetry() -> None:
    obs = _make_obs()
    lensed = lens_for("epidemiology", obs, last_reward=0.5)

    assert isinstance(lensed, BrainLensedObservation)
    assert lensed.brain == "epidemiology"
    assert lensed.last_reward == 0.5

    keys = set(lensed.derived_features.keys())
    assert {"epi_pressure", "worst_region_infection", "transmission_rate_trend"} <= keys

    # M-FR-2: trend is 0.0 in MVP (no history available in single-obs lens)
    assert lensed.derived_features["transmission_rate_trend"] == 0.0

    assert "regions[*].reported_cases_d_ago" in lensed.salient_field_ids
    assert "regions[*].hospital_load" in lensed.salient_field_ids


# ============================================================================
# T2 - Logistics lens emphasizes resources
# ============================================================================


def test_logistics_lens_emphasizes_resources() -> None:
    obs = _make_obs(test_kits=100, hospital_beds_free=50, mobile_units=10, vaccine_doses=200)
    lensed = lens_for("logistics", obs, last_reward=0.5)

    assert lensed.brain == "logistics"
    keys = set(lensed.derived_features.keys())
    expected_keys = {
        "total_inventory",
        "hospital_load_max",
        "deployment_feasibility_R1",
        "deployment_feasibility_R2",
        "deployment_feasibility_R3",
        "deployment_feasibility_R4",
    }
    assert expected_keys <= keys

    # 100 + 50 + 10 + 200 = 360
    assert lensed.derived_features["total_inventory"] == 360.0

    assert "resources.test_kits" in lensed.salient_field_ids


# ============================================================================
# T3 - Governance lens emphasizes legal
# ============================================================================


def test_governance_lens_emphasizes_legal() -> None:
    obs = _make_obs(
        restrictions=[Restriction(region="R1", severity="moderate", ticks_remaining=3)],
        legal_constraints=[
            LegalConstraint(rule_id="L1", blocked_action="restrict_movement.strict")
        ],
    )
    lensed = lens_for("governance", obs, last_reward=0.5)

    assert lensed.brain == "governance"
    keys = set(lensed.derived_features.keys())
    assert {
        "escalation_unlocked_strict",
        "legal_constraints_count",
        "restrictions_active_count",
    } <= keys

    assert lensed.derived_features["legal_constraints_count"] == 1.0
    assert lensed.derived_features["restrictions_active_count"] == 1.0
    assert "active_restrictions[*]" in lensed.salient_field_ids


# ============================================================================
# T4 - Lens does NOT strip raw_obs (D13)
# ============================================================================


def test_lens_does_not_strip_raw_obs() -> None:
    obs = _make_obs(restrictions=[Restriction(region="R1", severity="moderate", ticks_remaining=3)])

    for brain in ("epidemiology", "logistics", "governance"):
        lensed = lens_for(brain, obs, last_reward=0.0)
        # Pydantic deep-equality on the full observation
        assert lensed.raw_obs == obs


# ============================================================================
# T5 - V2 brain ids raise KeyError (Decision 9, post-review)
# ============================================================================


def test_lens_for_v2_brain_raises_key_error() -> None:
    obs = _make_obs()

    for v2_brain in ("communications", "equity"):
        with pytest.raises(KeyError):
            lens_for(v2_brain, obs, last_reward=0.0)

    with pytest.raises(KeyError):
        lens_for("not_a_brain", obs, last_reward=0.0)


# ============================================================================
# T7 - All derived_features values are non-bool floats (D14 + tightened)
# ============================================================================


def test_lens_derived_features_all_floats() -> None:
    obs = _make_obs(
        restrictions=[Restriction(region="R1", severity="moderate", ticks_remaining=3)],
        legal_constraints=[
            LegalConstraint(rule_id="L1", blocked_action="restrict_movement.strict")
        ],
    )

    for brain in ("epidemiology", "logistics", "governance"):
        lensed = lens_for(brain, obs, last_reward=0.0)
        for key, value in lensed.derived_features.items():
            assert isinstance(value, float) and not isinstance(value, bool), (
                f"derived_features[{key!r}] = {value!r} ({type(value).__name__}) "
                f"is not a non-bool float"
            )


# ============================================================================
# T8 - Governance lens detects accepted escalate(national)
# ============================================================================


def test_governance_lens_detects_escalation_unlocked() -> None:
    obs_with_accepted = _make_obs(
        recent_action_log=[
            ExecutedAction(tick=1, action=NoOp(), accepted=True),
            ExecutedAction(tick=2, action=Escalate(to_authority="national"), accepted=True),
        ]
    )
    lensed = lens_for("governance", obs_with_accepted, last_reward=0.0)
    assert lensed.derived_features["escalation_unlocked_strict"] == 1.0

    obs_without = _make_obs(
        recent_action_log=[ExecutedAction(tick=1, action=NoOp(), accepted=True)]
    )
    lensed_w = lens_for("governance", obs_without, last_reward=0.0)
    assert lensed_w.derived_features["escalation_unlocked_strict"] == 0.0

    # Rejected escalate must NOT count
    obs_rejected = _make_obs(
        recent_action_log=[
            ExecutedAction(tick=1, action=Escalate(to_authority="national"), accepted=False),
        ]
    )
    lensed_r = lens_for("governance", obs_rejected, last_reward=0.0)
    assert lensed_r.derived_features["escalation_unlocked_strict"] == 0.0

    # Accepted escalate(regional) must NOT count -- only "national" unlocks strict
    obs_regional = _make_obs(
        recent_action_log=[
            ExecutedAction(tick=1, action=Escalate(to_authority="regional"), accepted=True),
        ]
    )
    lensed_re = lens_for("governance", obs_regional, last_reward=0.0)
    assert lensed_re.derived_features["escalation_unlocked_strict"] == 0.0

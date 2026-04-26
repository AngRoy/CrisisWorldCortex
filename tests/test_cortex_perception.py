"""Session 11 - Perception subagent tests (deterministic Python; no LLM).

Per cortex/CLAUDE.md binding (Perception is pure Python) and Phase A
Decisions 9 (V2 KeyError) + 63 (salient_signals cap at 5).
"""

from __future__ import annotations

from typing import Iterable

import pytest

from cortex.schemas import PerceptionReport
from cortex.subagents import perception_for
from CrisisWorldCortex.models import (
    CrisisworldcortexObservation,
    LegalConstraint,
    RegionTelemetry,
    ResourceInventory,
    Restriction,
)


def _make_obs(
    cases_per_region: Iterable[int] = (5, 1, 0, 0),
    hospital_loads: Iterable[float] = (0.7, 0.1, 0.1, 0.1),
    test_kits: int = 200,
    hospital_beds_free: int = 50,
    mobile_units: int = 2,
    vaccine_doses: int = 300,
    restrictions: Iterable[Restriction] = (),
    legal_constraints: Iterable[LegalConstraint] = (),
) -> CrisisworldcortexObservation:
    cases = list(cases_per_region)
    loads = list(hospital_loads)
    return CrisisworldcortexObservation(
        regions=[
            RegionTelemetry(
                region=f"R{i + 1}",
                reported_cases_d_ago=cases[i],
                hospital_load=loads[i],
                compliance_proxy=0.85,
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
        recent_action_log=[],
    )


# T1
def test_perception_runs_without_llm_call() -> None:
    """Perception is pure Python; the function does not take an LLMClient."""
    obs = _make_obs()
    for brain in ("epidemiology", "logistics", "governance"):
        report = perception_for(brain, obs)
        assert isinstance(report, PerceptionReport)
        assert report.brain == brain
        assert isinstance(report.confidence, float)
        assert 0.0 <= report.confidence <= 1.0


# T2
def test_perception_for_v2_brain_raises_key_error() -> None:
    obs = _make_obs()
    for v2_brain in ("communications", "equity"):
        with pytest.raises(KeyError):
            perception_for(v2_brain, obs)
    with pytest.raises(KeyError):
        perception_for("not_a_brain", obs)


# T3
@pytest.mark.parametrize("brain", ["epidemiology", "logistics", "governance"])
def test_perception_brain_specific_signals(brain: str) -> None:
    obs = _make_obs(
        cases_per_region=(20, 1, 0, 0),
        hospital_loads=(0.7, 0.1, 0.1, 0.1),
        test_kits=100,  # below threshold (300)
        hospital_beds_free=50,  # below threshold (100)
        mobile_units=2,  # below threshold (5)
        vaccine_doses=200,  # below threshold (500)
        restrictions=[
            Restriction(region="R1", severity="moderate", ticks_remaining=3),
        ],
        legal_constraints=[
            LegalConstraint(rule_id="L1", blocked_action="restrict_movement.strict"),
        ],
    )
    report = perception_for(brain, obs)

    assert report.brain == brain
    # Decision 63 / OQ-2 cap: at most 5 entries
    assert len(report.salient_signals) <= 5

    if brain == "epidemiology":
        assert any("R1" in s for s in report.salient_signals), (
            f"epi salient_signals should reference R1, got {report.salient_signals}"
        )
    elif brain == "logistics":
        joined = " ".join(report.salient_signals).lower()
        assert "kits" in joined or "mobile" in joined or "vaccine" in joined or "beds" in joined
    elif brain == "governance":
        assert any("R1" in s and "moderate" in s.lower() for s in report.salient_signals), (
            f"governance salient_signals should mention R1 moderate, got {report.salient_signals}"
        )

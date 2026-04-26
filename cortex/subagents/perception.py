"""Perception subagent - deterministic Python; not router-callable.

Per cortex/CLAUDE.md binding: Perception is pure Python; no LLM calls.
Phase A Decisions 9 (V2 KeyError) + 63 (salient_signals cap at 5) +
M-FR-1 (pinned confidence per brain).

Perception runs ONCE per brain at tick start (not router-callable).
The Council Executive (Session 12) calls ``perception_for`` once per
brain at the start of each tick; the resulting ``PerceptionReport`` is
plumbed into all subsequent SubagentInputs for that brain in that tick.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from cortex.schemas import EvidenceCitation, PerceptionReport
from CrisisWorldCortex.models import CrisisworldcortexObservation

_V2_BRAINS = frozenset({"communications", "equity"})

_SALIENT_SIGNALS_CAP = 5  # Phase A Decision 63 / OQ-2

_LOGISTICS_THRESHOLDS = {
    "test_kits": 300,
    "hospital_beds_free": 100,
    "mobile_units": 5,
    "vaccine_doses": 500,
}

_HOSPITAL_LOAD_ANOMALY_THRESHOLD = 0.6


def perception_for(brain: str, obs: CrisisworldcortexObservation) -> PerceptionReport:
    """Compute the per-brain Perception report.

    Args:
        brain: One of {"epidemiology", "logistics", "governance"}.
        obs: The current tick's observation.

    Raises:
        KeyError: If ``brain`` is V2-deferred or unknown.
    """
    helper = _PERCEPTION_REGISTRY.get(brain)
    if helper is not None:
        return helper(obs)
    if brain in _V2_BRAINS:
        raise KeyError(
            f"V2 brain {brain!r} deferred per Phase A Decision 9; no MVP stub perception."
        )
    raise KeyError(f"unknown brain: {brain!r}")


def _epi_perception(obs: CrisisworldcortexObservation) -> PerceptionReport:
    """Epidemiology perception: top-cases regions + high-hospital-load anomalies."""
    sorted_regions = sorted(obs.regions, key=lambda r: r.reported_cases_d_ago, reverse=True)
    salient_signals: List[str] = []
    evidence: List[EvidenceCitation] = []

    for r in sorted_regions[:3]:
        if r.reported_cases_d_ago > 0:
            salient_signals.append(f"{r.region}: cases={r.reported_cases_d_ago}")
            evidence.append(
                EvidenceCitation(
                    source="telemetry",
                    ref=f"{r.region}.reported_cases_d_ago",
                    excerpt=str(r.reported_cases_d_ago),
                )
            )

    if not salient_signals and obs.regions:
        # Fallback: cite the first region so we have at least one signal
        r = obs.regions[0]
        salient_signals.append(f"{r.region}: cases={r.reported_cases_d_ago}")
        evidence.append(
            EvidenceCitation(
                source="telemetry",
                ref=f"{r.region}.reported_cases_d_ago",
                excerpt=str(r.reported_cases_d_ago),
            )
        )

    salient_signals = salient_signals[:_SALIENT_SIGNALS_CAP]

    anomalies = [
        f"{r.region}: hospital_load={r.hospital_load:.2f}"
        for r in obs.regions
        if r.hospital_load > _HOSPITAL_LOAD_ANOMALY_THRESHOLD
    ]

    return PerceptionReport(
        brain="epidemiology",
        salient_signals=salient_signals,
        anomalies=anomalies,
        # M-FR-1: telemetry is delayed and noisy per mm.md; pinned proxy
        confidence=0.7,
        evidence=evidence,
    )


def _logistics_perception(obs: CrisisworldcortexObservation) -> PerceptionReport:
    """Logistics perception: low-resource flags + depleted-resource anomalies."""
    res = obs.resources
    salient_signals: List[str] = []
    evidence: List[EvidenceCitation] = []

    for resource_name, threshold in _LOGISTICS_THRESHOLDS.items():
        value = getattr(res, resource_name)
        if value < threshold:
            salient_signals.append(f"{resource_name} low: {value}")
            evidence.append(
                EvidenceCitation(
                    source="resource",
                    ref=f"resources.{resource_name}",
                    excerpt=str(value),
                )
            )

    salient_signals = salient_signals[:_SALIENT_SIGNALS_CAP]

    anomalies = []
    for resource_name in _LOGISTICS_THRESHOLDS:
        if getattr(res, resource_name) == 0:
            anomalies.append(f"{resource_name}: depleted")

    return PerceptionReport(
        brain="logistics",
        salient_signals=salient_signals,
        anomalies=anomalies,
        # M-FR-1: resource counts are deterministic, no telemetry noise
        confidence=1.0,
        evidence=evidence,
    )


def _governance_perception(obs: CrisisworldcortexObservation) -> PerceptionReport:
    """Governance perception: active restrictions + legal constraints + about-to-expire anomalies."""
    salient_signals: List[str] = []
    evidence: List[EvidenceCitation] = []

    for restr in obs.active_restrictions:
        salient_signals.append(f"{restr.region}: {restr.severity} ({restr.ticks_remaining}t)")
        evidence.append(
            EvidenceCitation(
                source="policy",
                ref=f"active_restrictions.{restr.region}",
                excerpt=f"{restr.severity}@{restr.ticks_remaining}",
            )
        )

    for lc in obs.legal_constraints:
        salient_signals.append(f"legal: {lc.rule_id} blocks {lc.blocked_action}")
        evidence.append(
            EvidenceCitation(
                source="policy",
                ref=f"legal_constraints.{lc.rule_id}",
                excerpt=lc.blocked_action,
            )
        )

    salient_signals = salient_signals[:_SALIENT_SIGNALS_CAP]

    has_recent_escalate_national = any(
        ea.accepted
        and ea.action.kind == "escalate"
        and getattr(ea.action, "to_authority", None) == "national"
        for ea in obs.recent_action_log
    )
    anomalies = []
    for restr in obs.active_restrictions:
        if (
            restr.severity == "strict"
            and restr.ticks_remaining <= 1
            and not has_recent_escalate_national
        ):
            anomalies.append(f"{restr.region}: strict expiring without escalation")

    return PerceptionReport(
        brain="governance",
        salient_signals=salient_signals,
        anomalies=anomalies,
        # M-FR-1: policy state is deterministic
        confidence=1.0,
        evidence=evidence,
    )


_PERCEPTION_REGISTRY: Dict[str, Callable[[CrisisworldcortexObservation], PerceptionReport]] = {
    "epidemiology": _epi_perception,
    "logistics": _logistics_perception,
    "governance": _governance_perception,
}

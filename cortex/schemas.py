# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Cortex-internal schemas (design doc §11.2).

These are the typed artifacts the Cortex agent produces and consumes:
brain recommendations, subagent reports, council decisions, metacognition
signals, and routing-policy actions. None of these are serialized over
the CrisisWorld wire protocol — only ``cortex.*`` and ``training.*`` read
and write them, plus loggers for trajectory buffers.
"""

from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field

# Wire types are accessed via the installed package path
# ``CrisisWorldCortex.models`` — NOT bare-name ``from models import …``.
#
# Both paths resolve to the same file (because ``pyproject.toml``
# package-dir maps ``CrisisWorldCortex`` to the repo root, and the repo
# root is also on ``sys.path``). Python's import machinery, however,
# creates a separate ``sys.modules`` entry per path, with its own
# distinct class objects. Pydantic's discriminated-union validator then
# builds its member-class table from one path; inputs constructed via
# the other path are rejected with a ``model_type`` error because the
# ``isinstance(input, member_cls)`` check fails across the two identities.
#
# Canonicalising cortex's wire-type imports through
# ``CrisisWorldCortex.models`` avoids that dual-loading trap. Cortex's
# OWN internal types (cortex.subagents, cortex.brains, etc.) continue
# to use bare-name sibling imports per Phase 1 C1 — only the cross-package
# wire boundary is canonicalised.
from CrisisWorldCortex.models import ExecutedAction, OuterActionPayload, RegionId

EpistemicPhase = Literal["Divergence", "Challenge", "Narrowing", "Convergence"]


# ============================================================================
# Evidence primitives
# ============================================================================


class EvidenceCitation(BaseModel):
    """Typed evidence-disclosure artifact per design §8.1 step 2."""

    source: Literal["telemetry", "resource", "policy", "action_log", "belief", "memory"]
    ref: str  # e.g. "region=R2.hospital_load@tick=7"
    excerpt: str  # the actual value/text being cited


# ============================================================================
# Subagent reports — Perception + the 3 LLM subagents (§7.2)
# ============================================================================


class PerceptionReport(BaseModel):
    """Output of the deterministic Python Perception subagent.

    NOT router-callable — Perception runs once per brain at tick start
    (design §7.2 execution rule). Included here for the logging contract.
    """

    brain: str
    salient_signals: List[str]
    anomalies: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[EvidenceCitation]


class RegionBeliefEstimate(BaseModel):
    """Cortex's estimate of one region's latent state.

    Distinct from ``WorldState`` (defined in ``server/simulator/seir_model.py``),
    which ``cortex/`` never imports. This is what the agent *thinks* the
    latent state is, derived from observed telemetry plus its own inference.
    """

    estimated_infection_rate: float = Field(ge=0.0, le=1.0)
    estimated_r_effective: float = Field(ge=0.0)
    estimated_compliance: float = Field(ge=0.0, le=1.0)
    confidence_intervals: Dict[str, Tuple[float, float]] = Field(default_factory=dict)


class Hypothesis(BaseModel):
    """One named hypothesis inside a ``BeliefState``, with a relative weight."""

    label: str
    weight: float = Field(ge=0.0, le=1.0)
    explanation: str


class BeliefState(BaseModel):
    """World Modeler output (LLM subagent, router-callable)."""

    brain: str
    latent_estimates: Dict[RegionId, RegionBeliefEstimate]
    hypotheses: List[Hypothesis]
    uncertainty: float = Field(ge=0.0, le=1.0)
    reducible_by_more_thought: float = Field(
        ge=0.0,
        le=1.0,
        description="0=need more data, 1=more recursion would help",
    )
    evidence: List[EvidenceCitation]


class CandidatePlan(BaseModel):
    """Planner output (LLM subagent, router-callable)."""

    action_sketch: str
    expected_outer_action: OuterActionPayload
    expected_value: float
    cost: float
    assumptions: List[str]
    falsifiers: List[str]
    confidence: float = Field(ge=0.0, le=1.0)


class CriticReport(BaseModel):
    """Critic output (LLM subagent, router-callable)."""

    brain: str
    target_plan_id: str
    attacks: List[str]
    missing_considerations: List[str]
    would_change_mind_if: List[str]
    severity: float = Field(ge=0.0, le=1.0)


SubagentReport = Union[BeliefState, CandidatePlan, CriticReport]
"""Type alias for the 3 LLM-subagent outputs. Used by loggers and
trajectory buffers that need to carry 'any subagent output' generically."""


class SubagentInput(BaseModel):
    """Typed input handed to one of the 3 LLM subagents per call.

    Per Phase A §2 A2: each subagent call receives a fully-typed input
    so prompts are deterministic and testable. ``prior_belief`` is
    ``None`` on round 1 (nothing to revise yet); on round 2 it carries
    the previous round's BeliefState (or an empty BeliefState if round 1
    failed, per Phase A Decision 62). ``prior_plans`` is empty for
    WorldModeler / Planner; populated for Critic so it can attack a
    specific plan. ``target_plan_id`` is required when ``role='critic'``.
    """

    brain: Literal["epidemiology", "logistics", "governance"]
    role: Literal["world_modeler", "planner", "critic"]
    tick: int = Field(ge=0)
    round: int = Field(ge=1, le=2, description="MVP cap: 1 or 2 only")
    perception: PerceptionReport
    prior_belief: Optional[BeliefState] = None
    prior_plans: List[CandidatePlan] = Field(default_factory=list)
    target_plan_id: Optional[str] = None
    last_reward: float
    recent_action_log_excerpt: List[ExecutedAction] = Field(default_factory=list)


# ============================================================================
# Brain output + Council decision
# ============================================================================


class BrainRecommendation(BaseModel):
    """One brain's output to the Council after one deliberation round (§11.2)."""

    brain: str
    top_action: OuterActionPayload
    top_confidence: float = Field(ge=0.0, le=1.0)
    minority_actions: List[OuterActionPayload] = Field(default_factory=list)
    reasoning_summary: str = Field(max_length=400)
    evidence: List[EvidenceCitation]
    falsifier: str
    uncertainty: float = Field(ge=0.0, le=1.0)
    tokens_used: int = Field(ge=0)
    anonymous_id: Optional[str] = None  # [V2] anonymized-comparison slot


class CouncilDecision(BaseModel):
    """What the Council Executive emits once a tick converges."""

    action: OuterActionPayload
    rationale: str = Field(max_length=600)
    preserved_dissent: List[str] = Field(default_factory=list)
    phase_trace: List[str] = Field(default_factory=list)
    rounds_used: int = Field(ge=1, le=2)
    tokens_used: int = Field(ge=0)


# ============================================================================
# Metacognition signals (design §7.4.3)
# ============================================================================


class MetacognitionState(BaseModel):
    """Signals computed each deliberation round.

    Consumed by the routing policy; some fields are eval-only and never fed
    into the training reward (see ``cortex/CLAUDE.md`` for the training-vs-eval
    split).
    """

    tick: int = Field(ge=0)
    round: int = Field(ge=1, le=2, description="MVP cap: 1 or 2 only")
    phase: EpistemicPhase
    inter_brain_agreement: float = Field(ge=0.0, le=1.0)
    average_confidence: float = Field(ge=0.0, le=1.0)
    average_evidence_support: float = Field(ge=0.0, le=1.0)
    novelty_yield_last_round: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of belief-or-plan change above threshold after the last "
            "extra pass, normalized to [0, 1]. Eval-only signal; not used by "
            "the MVP routing policy."
        ),
    )
    collapse_suspicion: float = Field(
        ge=0.0,
        le=1.0,
        description="0=healthy, 1=likely collapse (eval-only signal)",
    )
    budget_remaining_frac: float = Field(ge=0.0, le=1.0)
    urgency: float = Field(ge=0.0, le=1.0)
    preserved_dissent_count: int = Field(ge=0)
    challenge_used_this_tick: bool = False


# ============================================================================
# Routing policy action (6 kinds; design §7.4.4 / §11.2)
# ============================================================================


class RoutingAction(BaseModel):
    """What the learned routing policy emits every deliberation step.

    The six kinds correspond to design §7.4.4. ``recurse_in`` is [V2] and
    is NOT in the MVP action space. Fields other than ``kind`` are populated
    per-kind (see comments below).
    """

    kind: Literal[
        "call_subagent",
        "request_challenge",
        "switch_phase",
        "preserve_dissent",
        "emit_outer_action",
        "stop_and_no_op",
    ]
    # For kind == "call_subagent":
    brain: Optional[str] = None
    # Only the 3 LLM subagents are router-callable; Perception and Brain
    # Executive are deterministic Python and must not be invoked by the router.
    subagent: Optional[Literal["world_modeler", "planner", "critic"]] = None
    # For kind == "request_challenge":
    target_brain: Optional[str] = None
    # For kind == "switch_phase":
    new_phase: Optional[EpistemicPhase] = None
    # For kind == "emit_outer_action":
    outer_action: Optional[OuterActionPayload] = None

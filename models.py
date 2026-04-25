# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Wire-protocol schemas for the CrisisWorld environment.

Implements design doc §11.1 — CrisisWorld-facing OpenEnv interface:
- Outer action variants (6 MVP + 1 V2-declared-rejected) as a discriminated
  union on the ``kind`` field.
- CrisisworldcortexAction: OpenEnv wire wrapper carrying the payload.
- CrisisworldcortexObservation: full observation shape (regions, resources,
  policy, meta, action log).
- Supporting atoms (RegionTelemetry, Restriction, LegalConstraint,
  ResourceInventory, ExecutedAction) and literal type aliases.

Latent world state is deliberately NOT declared here — it lives in
``server/simulator/seir_model.py`` (session 5+) so the wire/internal
boundary is enforced structurally (nothing that imports ``models`` can
see latent state).
"""

from typing import Annotated, List, Literal, Union

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field

# ============================================================================
# Shared vocabulary
# ============================================================================

ResourceType = Literal["test_kits", "hospital_beds", "mobile_units", "vaccine_doses"]
RegionId = str  # e.g. "R1", "R2", ...
Severity = Literal["none", "light", "moderate", "strict"]


# ============================================================================
# Outer-action variants — 6 MVP + 1 V2 (design §6.3)
# ============================================================================


class DeployResource(BaseModel):
    """Deploy a quantity of a resource type to a region."""

    kind: Literal["deploy_resource"] = "deploy_resource"
    region: RegionId
    resource_type: ResourceType
    quantity: int = Field(ge=0)


class RequestData(BaseModel):
    """Reduce telemetry noise for a region for a few ticks (costs cognition budget)."""

    kind: Literal["request_data"] = "request_data"
    region: RegionId
    data_type: Literal["case_survey", "hospital_audit", "compliance_check"]


class RestrictMovement(BaseModel):
    """Apply a restriction severity to a region.

    Strict severity may be blocked by a legal constraint until
    ``escalate(national)`` is invoked (design §6.5).
    """

    kind: Literal["restrict_movement"] = "restrict_movement"
    region: RegionId
    severity: Severity


class Escalate(BaseModel):
    """Escalate to a higher authority. Unlocks additional action classes."""

    kind: Literal["escalate"] = "escalate"
    to_authority: Literal["regional", "national"]


class ReallocateBudget(BaseModel):
    """Transfer resource units between resource types (small efficiency loss)."""

    kind: Literal["reallocate_budget"] = "reallocate_budget"
    from_resource: ResourceType
    to_resource: ResourceType
    amount: int = Field(ge=0)


class NoOp(BaseModel):
    """Advance the tick without intervention."""

    kind: Literal["no_op"] = "no_op"


class PublicCommunication(BaseModel):
    """[V2] Declared for forward-compatibility.

    Rejected at runtime in MVP per design §6.3 / §19: env marks
    ``accepted=False`` in the action log; the training-reward grader
    applies the -0.1 well-formed-illegal penalty.
    """

    kind: Literal["public_communication"] = "public_communication"
    audience: Literal["general", "workers", "leaders"]
    message_class: Literal["informational", "reassurance", "directive"]
    honesty: float = Field(ge=0.0, le=1.0)


OuterActionPayload = Annotated[
    Union[
        DeployResource,
        RequestData,
        RestrictMovement,
        Escalate,
        ReallocateBudget,
        NoOp,
        PublicCommunication,
    ],
    Field(discriminator="kind"),
]


# ============================================================================
# Wire-protocol Action (frozen class name) — wraps the discriminated union
# ============================================================================


class CrisisworldcortexAction(Action):
    """OpenEnv wire wrapper carrying an ``OuterActionPayload``.

    JSON shape::

        {"action": {"kind": "deploy_resource", "region": "R1", ...}, "metadata": {}}

    The class name is frozen by the OpenEnv template and must not be renamed.
    """

    action: OuterActionPayload


# ============================================================================
# Observation atoms
# ============================================================================


class RegionTelemetry(BaseModel):
    """Per-region observed telemetry (delayed + noised derivative of latent state)."""

    region: RegionId
    reported_cases_d_ago: int = Field(ge=0)
    hospital_load: float = Field(ge=0.0, le=1.0)
    compliance_proxy: float = Field(ge=0.0, le=1.0)


class Restriction(BaseModel):
    region: RegionId
    severity: Severity
    ticks_remaining: int = Field(ge=0)


class LegalConstraint(BaseModel):
    rule_id: str
    blocked_action: str  # e.g. "restrict_movement.strict"
    unlock_via: Literal["escalate"] = "escalate"


class ResourceInventory(BaseModel):
    test_kits: int = Field(default=0, ge=0)
    hospital_beds_free: int = Field(default=0, ge=0)
    mobile_units: int = Field(default=0, ge=0)
    vaccine_doses: int = Field(default=0, ge=0)


class ExecutedAction(BaseModel):
    """Entry in the recent-action log.

    ``accepted=False`` signals an illegal action (e.g. PublicCommunication
    in MVP) that the env rejected — grader reads this to apply the -0.1
    well-formed-illegal penalty.
    """

    tick: int = Field(ge=0)
    action: OuterActionPayload
    accepted: bool


# ============================================================================
# Wire-protocol Observation (frozen class name; expanded fields)
# ============================================================================


class CrisisworldcortexObservation(Observation):
    """Full CrisisWorld observation per design §11.1.

    The class name is frozen by the OpenEnv template. Inherits ``done`` and
    ``reward`` from ``openenv.core.env_server.types.Observation``.
    """

    regions: List[RegionTelemetry] = Field(default_factory=list)
    resources: ResourceInventory = Field(default_factory=ResourceInventory)
    active_restrictions: List[Restriction] = Field(default_factory=list)
    legal_constraints: List[LegalConstraint] = Field(default_factory=list)
    tick: int = Field(default=0, ge=0)
    ticks_remaining: int = Field(default=0, ge=0)
    cognition_budget_remaining: int = Field(default=0, ge=0)
    recent_action_log: List[ExecutedAction] = Field(default_factory=list)

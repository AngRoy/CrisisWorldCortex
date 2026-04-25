"""Action round-trip: 6 MVP outer actions accepted; PublicCommunication rejected.

Verifies the dispatch table in ``apply_tick`` matches design §6.3 (6 MVP
variants legal at runtime, ``public_communication`` declared but rejected
with ``accepted=False`` per §19's well-formed-illegal rule).
"""

import pytest

from CrisisWorldCortex.models import (
    DeployResource,
    Escalate,
    NoOp,
    OuterActionPayload,
    PublicCommunication,
    ReallocateBudget,
    RequestData,
    RestrictMovement,
)
from CrisisWorldCortex.server.simulator import apply_tick, load_task


def _last_log_entry(state):
    assert state.recent_action_log, "no entries in recent_action_log"
    return state.recent_action_log[-1]


@pytest.mark.parametrize(
    "action",
    [
        DeployResource(region="R1", resource_type="test_kits", quantity=10),
        RequestData(region="R1", data_type="case_survey"),
        RestrictMovement(region="R1", severity="moderate"),
        Escalate(to_authority="regional"),
        ReallocateBudget(from_resource="test_kits", to_resource="mobile_units", amount=5),
        NoOp(),
    ],
)
def test_mvp_action_accepted(action: OuterActionPayload) -> None:
    state = load_task("outbreak_easy", episode_seed=0)
    state = apply_tick(state, action)
    log = _last_log_entry(state)
    assert log.accepted is True, f"action kind={action.kind!r} should be accepted but was rejected"
    assert log.action.kind == action.kind


def test_public_communication_rejected() -> None:
    """V2-declared action; env rejects with accepted=False per §19."""
    state = load_task("outbreak_easy", episode_seed=0)
    a = PublicCommunication(
        audience="general",
        message_class="informational",
        honesty=0.9,
    )
    state = apply_tick(state, a)
    log = _last_log_entry(state)
    assert log.accepted is False, "PublicCommunication is V2-illegal in MVP — must be rejected"
    assert log.action.kind == "public_communication"


def test_rejected_action_still_advances_tick() -> None:
    """Per §19, well-formed-illegal actions don't terminate the episode —
    the tick still advances and SEIR still runs."""
    state = load_task("outbreak_easy", episode_seed=0)
    initial_tick = state.tick
    a = PublicCommunication(
        audience="general",
        message_class="informational",
        honesty=0.5,
    )
    state = apply_tick(state, a)
    assert state.tick == initial_tick + 1, "tick must advance even when action is rejected"

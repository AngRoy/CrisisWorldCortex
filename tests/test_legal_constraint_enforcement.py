"""Legal constraint: strict severity blocked until escalate-national (hard task).

Verifies design §6.5's locked rule: ``outbreak_hard`` carries one
``LegalConstraint`` (rule_id=L1) blocking ``restrict_movement.strict``
until ``Escalate(to_authority="national")`` is invoked. Other severities
and the same severity on tasks without the constraint are unaffected.
"""

from CrisisWorldCortex.models import Escalate, RestrictMovement
from CrisisWorldCortex.server.simulator import apply_tick, load_task


def _last_log(state):
    return state.recent_action_log[-1]


def test_strict_blocked_before_escalation() -> None:
    state = load_task("outbreak_hard", episode_seed=0)
    state = apply_tick(state, RestrictMovement(region="R1", severity="strict"))
    log = _last_log(state)
    assert log.accepted is False
    assert log.action.kind == "restrict_movement"
    # No restriction was applied to R1.
    assert state.restrictions.get("R1") is None
    # escalation_unlocked_strict still False.
    assert state.escalation_unlocked_strict is False


def test_strict_allowed_after_escalate_national() -> None:
    state = load_task("outbreak_hard", episode_seed=0)
    # Escalate to national first.
    state = apply_tick(state, Escalate(to_authority="national"))
    assert _last_log(state).accepted is True
    assert state.escalation_unlocked_strict is True
    # Now strict should be accepted.
    state = apply_tick(state, RestrictMovement(region="R1", severity="strict"))
    assert _last_log(state).accepted is True
    assert state.restrictions["R1"].severity == "strict"


def test_regional_escalation_alone_does_not_unlock_strict() -> None:
    state = load_task("outbreak_hard", episode_seed=0)
    state = apply_tick(state, Escalate(to_authority="regional"))
    assert _last_log(state).accepted is True
    assert state.escalation_unlocked_strict is False
    state = apply_tick(state, RestrictMovement(region="R1", severity="strict"))
    assert _last_log(state).accepted is False, "regional escalation must NOT unlock strict severity"


def test_non_strict_severities_accepted_without_escalation() -> None:
    """light/moderate/none don't require escalation even on hard."""
    for severity in ("none", "light", "moderate"):
        s = load_task("outbreak_hard", episode_seed=0)
        s = apply_tick(s, RestrictMovement(region="R1", severity=severity))
        assert _last_log(s).accepted is True, (
            f"severity={severity!r} should be allowed without escalation"
        )

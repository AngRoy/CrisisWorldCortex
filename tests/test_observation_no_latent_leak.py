"""Latent state must not leak into the wire-format observation.

Plant identifiable sentinels in WorldState's latent fields, project to
an observation via ``make_observation``, dump to JSON, and assert no
sentinel literal or latent field name appears in the JSON.

Closes the partial-observability invariant from ``server/CLAUDE.md``:
"Observations expose only the observed layer. Latent fields leave
``server/`` only as grader input."
"""

from CrisisWorldCortex.models import NoOp
from CrisisWorldCortex.server.simulator import (
    apply_tick,
    load_task,
    make_observation,
)

# Distinctive sentinel literals chosen to be (a) inside Pydantic Field
# constraints (S/E/I/R bounded to [0,1]) and (b) extremely unlikely to
# arise from natural SEIR dynamics or noise draws. 5-decimal patterns
# minimize accidental matches in observation values.
SENTINEL_S = 0.91111
SENTINEL_E = 0.92222
SENTINEL_I = 0.93333
SENTINEL_R = 0.94444
SENTINEL_COMPLIANCE = 0.95555
SENTINEL_ESCALATION = 99
SENTINEL_CONSECUTIVE_SAFE = 42


def _plant_sentinels(state) -> None:
    """Mutate state's latent fields to carry sentinel values.

    Pydantic v2 doesn't validate-on-assignment by default in our config,
    so direct attribute assignment skips bound checks.
    """
    for region in state.regions:
        region.S = SENTINEL_S
        region.E = SENTINEL_E
        region.I = SENTINEL_I
        region.R = SENTINEL_R
        region.true_compliance = SENTINEL_COMPLIANCE
    state.escalation_level = SENTINEL_ESCALATION
    state.escalation_unlocked_strict = True
    state.consecutive_safe_ticks = SENTINEL_CONSECUTIVE_SAFE


def test_observation_json_has_no_latent_sentinel_values() -> None:
    state = load_task("outbreak_hard", episode_seed=0)
    # Run a few real ticks so history_I and recent_action_log get populated;
    # then plant sentinels just before observing.
    for _ in range(3):
        state = apply_tick(state, NoOp())
    _plant_sentinels(state)

    obs = make_observation(state)
    obs_json = obs.model_dump_json()

    # The literal sentinel values must not appear in the JSON.
    assert "0.91111" not in obs_json, "S leaked into observation JSON"
    assert "0.92222" not in obs_json, "E leaked into observation JSON"
    assert "0.94444" not in obs_json, "R leaked into observation JSON"
    assert "0.93333" not in obs_json, "I leaked into observation JSON (raw)"
    assert "0.95555" not in obs_json, "true_compliance leaked into observation JSON"


def test_observation_has_no_latent_field_names() -> None:
    """Schema-level guarantee: observation Pydantic shape doesn't carry
    latent field names regardless of state contents."""
    state = load_task("outbreak_hard", episode_seed=0)
    obs_json = make_observation(state).model_dump_json()

    # Latent SEIR / policy field names — must NEVER appear as JSON keys.
    for forbidden_key in (
        '"true_compliance"',
        '"history_I"',
        '"pending_effects"',
        '"noise_reduction_ticks"',
        '"escalation_level"',
        '"escalation_unlocked_strict"',
        '"superspreader_schedule"',
        '"consecutive_safe_ticks"',
        '"task_config"',
        '"episode_seed"',
        '"max_ticks"',
        '"task_name"',
        '"fires_at_tick"',
        '"surfaces_at_tick"',
        '"magnitude_I"',
    ):
        assert forbidden_key not in obs_json, (
            f"latent field {forbidden_key!r} leaked into observation JSON"
        )


def test_observation_only_carries_declared_wire_fields() -> None:
    """Positive check: observation JSON contains the declared
    ``CrisisworldcortexObservation`` keys."""
    state = load_task("outbreak_easy", episode_seed=0)
    obs_json = make_observation(state).model_dump_json()
    for required_key in (
        '"regions"',
        '"resources"',
        '"active_restrictions"',
        '"legal_constraints"',
        '"tick"',
        '"ticks_remaining"',
        '"cognition_budget_remaining"',
        '"recent_action_log"',
    ):
        assert required_key in obs_json, (
            f"required wire key {required_key!r} missing from observation"
        )

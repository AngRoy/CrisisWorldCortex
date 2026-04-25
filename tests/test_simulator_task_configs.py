"""Task-config loading: 3 tasks load with the expected shape."""

import pytest

from CrisisWorldCortex.server.simulator import TASK_CONFIGS, load_task


def test_load_task_outbreak_easy() -> None:
    s = load_task("outbreak_easy")
    assert s.task_name == "outbreak_easy"
    assert len(s.regions) == 4
    assert {r.region for r in s.regions} == {"R1", "R2", "R3", "R4"}
    assert s.tick == 0
    assert s.max_ticks == 12
    assert s.terminal == "none"
    assert s.task_config.telemetry_delay_ticks == 1
    assert s.task_config.base_R0 == 1.5
    assert s.task_config.cognition_budget_per_tick == 6000
    assert s.resources.test_kits == 1000
    assert s.resources.hospital_beds_free == 500
    assert len(s.superspreader_schedule) == 0
    assert len(s.legal_constraints) == 0
    # R1 is hot (I=0.03), others quiet (I=0.001)
    r1 = next(r for r in s.regions if r.region == "R1")
    r2 = next(r for r in s.regions if r.region == "R2")
    assert r1.I == 0.03
    assert r2.I == 0.001
    # history_I pre-populated with initial I
    assert r1.history_I == [0.03]


def test_load_task_outbreak_medium() -> None:
    s = load_task("outbreak_medium")
    assert s.task_name == "outbreak_medium"
    assert len(s.regions) == 4
    assert s.task_config.telemetry_delay_ticks == 2
    assert s.task_config.base_R0 == 2.0
    assert s.task_config.cognition_budget_per_tick == 6000
    assert s.resources.test_kits == 500
    assert len(s.superspreader_schedule) == 0
    assert len(s.legal_constraints) == 0
    # 3 hot regions (R1-R3) + 1 quiet (R4)
    hot = [r for r in s.regions if r.I == 0.05]
    quiet = [r for r in s.regions if r.I == 0.001]
    assert len(hot) == 3
    assert len(quiet) == 1


def test_load_task_outbreak_hard() -> None:
    s = load_task("outbreak_hard")
    assert s.task_name == "outbreak_hard"
    assert len(s.regions) == 5
    assert {r.region for r in s.regions} == {"R1", "R2", "R3", "R4", "R5"}
    assert s.task_config.telemetry_delay_ticks == 3
    assert s.task_config.base_R0 == 2.5
    assert s.task_config.cognition_budget_per_tick == 6000
    assert s.resources.test_kits == 200
    assert len(s.task_config.chain_betas) == 4
    # Superspreader scheduled at tick 7 in R3
    assert len(s.superspreader_schedule) == 1
    event = s.superspreader_schedule[0]
    assert event.region == "R3"
    assert event.fires_at_tick == 7
    assert event.surfaces_at_tick == 9
    # Legal constraint blocking strict severity
    assert len(s.legal_constraints) == 1
    assert s.legal_constraints[0].blocked_action == "restrict_movement.strict"
    # All 5 regions hot (no quiet regions for hard)
    for r in s.regions:
        assert r.I == 0.04


def test_load_task_max_ticks_override() -> None:
    s = load_task("outbreak_easy", max_ticks=20)
    assert s.max_ticks == 20
    assert s.task_config.max_ticks == 20


def test_load_task_episode_seed_recorded() -> None:
    s = load_task("outbreak_easy", episode_seed=99)
    assert s.episode_seed == 99


def test_load_task_unknown_name_raises() -> None:
    with pytest.raises(ValueError):
        load_task("nonexistent_task")  # type: ignore[arg-type]


def test_task_configs_dict_has_three_entries() -> None:
    assert set(TASK_CONFIGS.keys()) == {
        "outbreak_easy",
        "outbreak_medium",
        "outbreak_hard",
    }


def test_load_task_compliance_initialized() -> None:
    s = load_task("outbreak_easy")
    for r in s.regions:
        assert r.true_compliance == 0.95
    s = load_task("outbreak_hard")
    for r in s.regions:
        assert r.true_compliance == 0.75

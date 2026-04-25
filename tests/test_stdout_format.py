"""Tests for ``inference.py``'s stdout formatter (Session 7b).

Pure-function tests only — no env, no LLM, no in-process bypass.
The integration path ("does ``uv run inference.py`` produce valid output
end-to-end?") happens via manual run + the hackathon validator, not here.

Format strings must match triagesieve_env's ``inference.py`` byte-for-byte
so the hackathon validator regex picks them up:
    [START] task=<task> env=<env> model=<model>
    [STEP] step=<N> action=<str> reward=<r:.2f> done=<true|false> error=<error|null>
    [END] success=<true|false> steps=<N> score=<s:.3f> rewards=<r1:.2f,r2:.2f,...>
"""

from __future__ import annotations

import re

import pytest

from CrisisWorldCortex.models import (
    DeployResource,
    Escalate,
    NoOp,
    PublicCommunication,
    ReallocateBudget,
    RequestData,
    RestrictMovement,
)
from CrisisWorldCortex.server.simulator import load_task
from inference import (
    BENCHMARK,
    SUCCESS_THRESHOLD,
    StepRecord,
    _format_end_line,
    _format_start_line,
    _format_step_line,
    action_to_str,
    compute_score,
    format_episode_trace,
)

# ============================================================================
# _format_start_line
# ============================================================================


def test_format_start_line_byte_for_byte() -> None:
    out = _format_start_line(
        task_name="outbreak_easy",
        env_name="CrisisWorldCortex",
        model_name="Qwen/Qwen2.5-72B-Instruct",
    )
    assert out == "[START] task=outbreak_easy env=CrisisWorldCortex model=Qwen/Qwen2.5-72B-Instruct"


def test_format_start_line_uses_benchmark_const_in_inference() -> None:
    """``BENCHMARK`` is used as the env field in production calls."""
    assert BENCHMARK == "CrisisWorldCortex"


# ============================================================================
# _format_step_line
# ============================================================================


def test_format_step_line_byte_for_byte_no_error() -> None:
    record = StepRecord(
        step=1,
        action_str="no_op",
        reward=0.42,
        done=False,
        error=None,
    )
    out = _format_step_line(record)
    assert out == "[STEP] step=1 action=no_op reward=0.42 done=false error=null"


def test_format_step_line_done_lowercased() -> None:
    record = StepRecord(
        step=7,
        action_str="no_op",
        reward=0.50,
        done=True,
        error=None,
    )
    out = _format_step_line(record)
    assert " done=true " in out
    assert "True" not in out


def test_format_step_line_error_field_is_literal_null_when_none() -> None:
    """The string 'null', not Python's 'None'."""
    record = StepRecord(step=1, action_str="no_op", reward=0.0, done=False, error=None)
    out = _format_step_line(record)
    assert " error=null" in out
    assert "error=None" not in out


def test_format_step_line_error_field_passes_through_when_set() -> None:
    record = StepRecord(
        step=3,
        action_str="no_op",
        reward=0.1,
        done=False,
        error="connection_timeout",
    )
    out = _format_step_line(record)
    assert " error=connection_timeout" in out


def test_format_step_line_reward_two_decimals() -> None:
    record = StepRecord(step=1, action_str="no_op", reward=0.123456, done=False, error=None)
    out = _format_step_line(record)
    assert " reward=0.12 " in out
    assert "0.123" not in out


# ============================================================================
# _format_end_line
# ============================================================================


def test_format_end_line_byte_for_byte() -> None:
    out = _format_end_line(
        success=True,
        steps=3,
        score=0.751,
        rewards=[0.40, 0.50, 0.60],
    )
    assert out == "[END] success=true steps=3 score=0.751 rewards=0.40,0.50,0.60"


def test_format_end_line_success_lowercased() -> None:
    out = _format_end_line(success=False, steps=12, score=0.123, rewards=[0.1] * 12)
    assert " success=false " in out
    assert "False" not in out


def test_format_end_line_score_three_decimals() -> None:
    out = _format_end_line(success=True, steps=3, score=0.123456789, rewards=[0.5])
    assert " score=0.123 " in out
    assert "0.1234" not in out


def test_format_end_line_rewards_two_decimals_each() -> None:
    out = _format_end_line(success=True, steps=4, score=0.5, rewards=[0.111, 0.222, 0.333, 0.444])
    assert "rewards=0.11,0.22,0.33,0.44" in out


def test_format_end_line_empty_rewards_emits_empty_string() -> None:
    out = _format_end_line(success=False, steps=0, score=0.001, rewards=[])
    assert out.endswith("rewards=")


# ============================================================================
# compute_score
# ============================================================================


def test_compute_score_basic_case() -> None:
    """mean=0.5, bonus=0.0 -> (0.5 + 0.0 + 0.20)/1.40 = 0.5."""
    assert compute_score([0.5] * 10, terminal_bonus_value=0.0) == pytest.approx(0.5)


def test_compute_score_natural_max_maps_to_open_one() -> None:
    """mean=1.0, bonus=+0.20 -> (1.0 + 0.20 + 0.20)/1.40 = 1.0 -> clamped to 1-1e-3."""
    score = compute_score([1.0] * 5, terminal_bonus_value=0.20)
    assert score == 1.0 - 1e-3


def test_compute_score_natural_min_maps_to_open_zero() -> None:
    """mean=0.0, bonus=-0.20 -> (0.0 - 0.20 + 0.20)/1.40 = 0.0 -> clamped to 1e-3."""
    score = compute_score([0.0] * 5, terminal_bonus_value=-0.20)
    assert score == 1e-3


def test_compute_score_clamps_extreme_high() -> None:
    """Even unphysical mean=2.0 cannot exceed 1-1e-3 after clamp."""
    score = compute_score([2.0] * 3, terminal_bonus_value=0.20)
    assert score == 1.0 - 1e-3


def test_compute_score_clamps_extreme_low() -> None:
    """Even unphysical mean=-1.0 cannot fall below 1e-3 after clamp."""
    score = compute_score([-1.0] * 3, terminal_bonus_value=-0.20)
    assert score == 1e-3


def test_compute_score_empty_rewards_returns_lower_clamp() -> None:
    """Coarse failure signal — episode produced zero rewards. Session 14
    will refine 'env-failed-to-reset' vs 'agent-did-nothing'."""
    assert compute_score([], terminal_bonus_value=0.0) == 1e-3


# ============================================================================
# action_to_str — compact action summaries for [STEP] lines
# ============================================================================


@pytest.mark.parametrize(
    "action,expected",
    [
        (NoOp(), "no_op"),
        (
            DeployResource(region="R1", resource_type="test_kits", quantity=50),
            "deploy_resource:R1:test_kits",
        ),
        (RestrictMovement(region="R2", severity="moderate"), "restrict_movement:R2:moderate"),
        (Escalate(to_authority="national"), "escalate:national"),
        (RequestData(region="R3", data_type="case_survey"), "request_data:R3:case_survey"),
        (
            ReallocateBudget(from_resource="test_kits", to_resource="mobile_units", amount=10),
            "reallocate_budget:test_kits:mobile_units",
        ),
        (
            PublicCommunication(audience="general", message_class="informational", honesty=0.0),
            "public_communication",
        ),
    ],
)
def test_action_to_str_compact_form(action, expected: str) -> None:
    """Compact action strings for the [STEP] line. Quantity/amount/honesty
    are intentionally dropped to keep the line short."""
    assert action_to_str(action) == expected


# ============================================================================
# format_episode_trace — full-block test
# ============================================================================


def test_format_episode_trace_full_block_shape() -> None:
    """The pure formatter renders [START] + N x [STEP] + [END]."""
    state = load_task("outbreak_easy", episode_seed=0)
    state.terminal = "success"  # +0.20 terminal_bonus

    steps = [
        StepRecord(step=1, action_str="no_op", reward=0.50, done=False, error=None),
        StepRecord(step=2, action_str="no_op", reward=0.60, done=False, error=None),
        StepRecord(step=3, action_str="no_op", reward=0.70, done=True, error=None),
    ]
    out = format_episode_trace(
        task_name="outbreak_easy",
        model_name="test-model",
        steps=steps,
        final_state=state,
    )

    lines = out.split("\n")
    assert len(lines) == 5  # 1 START + 3 STEP + 1 END
    assert lines[0].startswith("[START] task=outbreak_easy env=CrisisWorldCortex model=test-model")
    assert lines[1].startswith("[STEP] step=1 ")
    assert lines[2].startswith("[STEP] step=2 ")
    assert lines[3].startswith("[STEP] step=3 ")
    assert lines[4].startswith("[END] ")


def test_format_episode_trace_uses_terminal_bonus_from_state() -> None:
    """Different state.terminal -> different score in the [END] line."""
    state_succ = load_task("outbreak_easy", episode_seed=0)
    state_succ.terminal = "success"
    state_fail = load_task("outbreak_easy", episode_seed=0)
    state_fail.terminal = "failure"

    steps = [
        StepRecord(step=1, action_str="no_op", reward=0.50, done=True, error=None),
    ]
    out_succ = format_episode_trace(
        task_name="t",
        model_name="m",
        steps=steps,
        final_state=state_succ,
    )
    out_fail = format_episode_trace(
        task_name="t",
        model_name="m",
        steps=steps,
        final_state=state_fail,
    )

    end_succ = [ln for ln in out_succ.split("\n") if ln.startswith("[END]")][0]
    end_fail = [ln for ln in out_fail.split("\n") if ln.startswith("[END]")][0]
    score_succ = float(re.search(r"score=(\d\.\d{3})", end_succ).group(1))
    score_fail = float(re.search(r"score=(\d\.\d{3})", end_fail).group(1))
    assert score_succ > score_fail, (
        f"success-terminal score {score_succ} should exceed failure-terminal {score_fail}"
    )


def test_format_episode_trace_success_threshold() -> None:
    """``success`` is computed by score >= SUCCESS_THRESHOLD."""
    assert SUCCESS_THRESHOLD == 0.5

    state = load_task("outbreak_easy", episode_seed=0)
    state.terminal = "success"

    high_steps = [
        StepRecord(step=i, action_str="no_op", reward=0.95, done=False, error=None)
        for i in range(1, 6)
    ]
    out_high = format_episode_trace("t", "m", high_steps, state)
    end_high = [ln for ln in out_high.split("\n") if ln.startswith("[END]")][0]
    assert " success=true " in end_high

    state.terminal = "failure"
    low_steps = [
        StepRecord(step=i, action_str="no_op", reward=0.05, done=False, error=None)
        for i in range(1, 6)
    ]
    out_low = format_episode_trace("t", "m", low_steps, state)
    end_low = [ln for ln in out_low.split("\n") if ln.startswith("[END]")][0]
    assert " success=false " in end_low

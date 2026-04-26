"""Session 9 - Cortex subagent tests (WorldModeler, Planner, Critic).

Per Phase A docs/CORTEX_ARCHITECTURE.md Decisions 1-8 + 62 and the user's
proposal acceptance with 11 tests total. RED-tests-first.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import pytest

from cortex.schemas import (
    BeliefState,
    CandidatePlan,
    CriticReport,
    EvidenceCitation,
    PerceptionReport,
    SubagentInput,
)
from cortex.subagents import (
    PROMPTS_DIR,
    CriticSubagent,
    PlannerSubagent,
    WorldModelerSubagent,
)
from tests._helpers.llm_stub import StubLLMClient

# ============================================================================
# Test fixtures
# ============================================================================


_VALID_BELIEF_PAYLOAD: Dict[str, Any] = {
    "brain": "epidemiology",
    "latent_estimates": {
        "R1": {
            "estimated_infection_rate": 0.05,
            "estimated_r_effective": 1.2,
            "estimated_compliance": 0.85,
            "confidence_intervals": {},
        },
    },
    "hypotheses": [{"label": "rising-r1", "weight": 0.6, "explanation": "telemetry trending up"}],
    "uncertainty": 0.4,
    "reducible_by_more_thought": 0.3,
    "evidence": [
        {"source": "telemetry", "ref": "R1.reported_cases@t3", "excerpt": "rising"},
        {"source": "policy", "ref": "R1.restriction", "excerpt": "moderate"},
    ],
}


_VALID_PLAN_PAYLOAD: Dict[str, Any] = {
    "action_sketch": "Deploy 100 test_kits to R1",
    "expected_outer_action": {
        "kind": "deploy_resource",
        "region": "R1",
        "resource_type": "test_kits",
        "quantity": 100,
    },
    "expected_value": 0.6,
    "cost": 200.0,
    "assumptions": ["kits inventory > 100"],
    "falsifiers": ["R1 cases drop without intervention"],
    "confidence": 0.75,
}


_VALID_CRITIC_PAYLOAD: Dict[str, Any] = {
    "brain": "logistics",
    "target_plan_id": "plan-1",
    "attacks": ["ignores R3 hospital saturation"],
    "missing_considerations": ["compliance decay over 4 ticks"],
    "would_change_mind_if": ["new R3 telemetry shows under-utilisation"],
    "severity": 0.6,
}


def _valid_json_for(role: str, brain: str = "epidemiology") -> str:
    """Return a valid JSON-schema response for ``role``, brain-substituted."""
    if role == "world_modeler":
        payload = dict(_VALID_BELIEF_PAYLOAD)
        payload["brain"] = brain
        return json.dumps(payload)
    if role == "planner":
        return json.dumps(_VALID_PLAN_PAYLOAD)
    if role == "critic":
        payload = dict(_VALID_CRITIC_PAYLOAD)
        payload["brain"] = brain
        return json.dumps(payload)
    raise ValueError(f"unknown role: {role}")


def _make_subagent_input(
    brain: str = "epidemiology",
    role: str = "world_modeler",
    tick: int = 3,
    round_: int = 1,
    target_plan_id: Optional[str] = None,
) -> SubagentInput:
    """Minimal valid SubagentInput. ``round_`` arg avoids shadowing builtin."""
    return SubagentInput(
        brain=brain,
        role=role,
        tick=tick,
        round=round_,
        perception=PerceptionReport(
            brain=brain,
            salient_signals=["R1 cases rising"],
            anomalies=[],
            confidence=0.7,
            evidence=[EvidenceCitation(source="telemetry", ref="R1.cases", excerpt="rising")],
        ),
        prior_belief=None,
        prior_plans=[],
        target_plan_id=target_plan_id,
        last_reward=0.5,
        recent_action_log_excerpt=[],
    )


# ============================================================================
# T1 - WorldModeler emits BeliefState
# ============================================================================


def test_world_modeler_emits_belief_state() -> None:
    stub = StubLLMClient(scripted_responses=[_valid_json_for("world_modeler", "epidemiology")])
    agent = WorldModelerSubagent(llm_client=stub)
    input_data = _make_subagent_input(brain="epidemiology", role="world_modeler")

    result = agent.run(input_data, step_idx=0)

    assert isinstance(result, BeliefState)
    assert result.brain == "epidemiology"
    assert "R1" in result.latent_estimates
    assert len(result.evidence) >= 1
    assert stub.call_count == 1


# ============================================================================
# T2 - Planner emits CandidatePlan
# ============================================================================


def test_planner_emits_candidate_plan() -> None:
    stub = StubLLMClient(scripted_responses=[_valid_json_for("planner", "epidemiology")])
    agent = PlannerSubagent(llm_client=stub)
    input_data = _make_subagent_input(brain="epidemiology", role="planner")

    result = agent.run(input_data, step_idx=1)

    assert isinstance(result, CandidatePlan)
    assert result.expected_outer_action.kind == "deploy_resource"
    assert result.confidence == 0.75


# ============================================================================
# T3 - Critic emits CriticReport
# ============================================================================


def test_critic_emits_critic_report() -> None:
    stub = StubLLMClient(scripted_responses=[_valid_json_for("critic", "logistics")])
    agent = CriticSubagent(llm_client=stub)
    input_data = _make_subagent_input(brain="logistics", role="critic", target_plan_id="plan-1")

    result = agent.run(input_data, step_idx=2)

    assert isinstance(result, CriticReport)
    assert result.brain == "logistics"
    assert result.target_plan_id == "plan-1"
    assert result.severity == 0.6


# ============================================================================
# T4 - Parse failure then retry succeeds (2 LLM calls)
# ============================================================================


def test_subagent_parse_failure_then_retry_succeeds() -> None:
    stub = StubLLMClient(
        scripted_responses=["not-json-garbage", _valid_json_for("world_modeler", "epidemiology")]
    )
    agent = WorldModelerSubagent(llm_client=stub)

    result = agent.run(_make_subagent_input(), step_idx=0)

    assert isinstance(result, BeliefState)
    assert result.brain == "epidemiology"
    assert stub.call_count == 2, "expected 1 initial call + 1 retry"


# ============================================================================
# T5 - Parse failure twice -> empty fallback (no third LLM call)
# ============================================================================


def test_subagent_parse_failure_then_retry_fails_returns_empty() -> None:
    stub = StubLLMClient(scripted_responses=["garbage-1", "garbage-2"])
    agent = WorldModelerSubagent(llm_client=stub)

    result = agent.run(_make_subagent_input(), step_idx=0)

    assert isinstance(result, BeliefState)
    assert result.brain == "epidemiology"
    assert result.latent_estimates == {}
    assert result.hypotheses == []
    assert result.evidence == []
    assert result.uncertainty == 1.0
    assert result.reducible_by_more_thought == 0.0
    assert stub.call_count == 2, "must NOT make a 3rd call after retry failure"


# ============================================================================
# T6 - caller_id format matches Phase A Decision 7
# ============================================================================


_CALLER_ID_RE = re.compile(
    r"^cortex:(epidemiology|logistics|governance):"
    r"(world_modeler|planner|critic):"
    r"t\d+:r[12]:s\d+$"
)


@pytest.mark.parametrize(
    "role_cls,brain,role_name",
    [
        (WorldModelerSubagent, "epidemiology", "world_modeler"),
        (PlannerSubagent, "logistics", "planner"),
        (CriticSubagent, "governance", "critic"),
    ],
)
def test_subagent_caller_id_format(
    role_cls: type,
    brain: str,
    role_name: str,
) -> None:
    stub = StubLLMClient(scripted_responses=[_valid_json_for(role_name, brain)])
    agent = role_cls(llm_client=stub)
    input_data = _make_subagent_input(
        brain=brain,
        role=role_name,
        tick=7,
        round_=2,
        target_plan_id="plan-X" if role_name == "critic" else None,
    )

    agent.run(input_data, step_idx=4)

    caller_id = stub.calls[0].caller_id
    assert _CALLER_ID_RE.match(caller_id), (
        f"caller_id={caller_id!r} does not match the locked format"
    )
    assert caller_id == f"cortex:{brain}:{role_name}:t7:r2:s4"


# ============================================================================
# T8 - SYS prompt loaded from prompts/<role>.txt and brain-formatted
# (folds in the prompt-formatting refinement: format() must not raise)
# ============================================================================


@pytest.mark.parametrize(
    "role_cls,role_name",
    [
        (WorldModelerSubagent, "world_modeler"),
        (PlannerSubagent, "planner"),
        (CriticSubagent, "critic"),
    ],
)
def test_subagent_uses_loaded_prompt_from_file(role_cls: type, role_name: str) -> None:
    raw = (PROMPTS_DIR / f"{role_name}.txt").read_text(encoding="utf-8")

    # Refinement: format() must not raise even with extra kwargs ignored.
    # Catches {{/}}-escape regressions in JSON-schema sections of the prompts.
    formatted = raw.format(brain="epidemiology", target_plan_id="plan-X")
    assert isinstance(formatted, str)
    assert "epidemiology" in formatted

    stub = StubLLMClient(scripted_responses=[_valid_json_for(role_name, "epidemiology")])
    agent = role_cls(llm_client=stub)
    input_data = _make_subagent_input(
        brain="epidemiology",
        role=role_name,
        target_plan_id="plan-X" if role_name == "critic" else None,
    )

    agent.run(input_data, step_idx=0)

    sys_msg = stub.calls[0].messages[0]
    assert sys_msg.role == "system"
    assert sys_msg.content == formatted


# ============================================================================
# T9 - Token counter is billed to the expected caller_id
# ============================================================================


def test_subagent_token_counter_billed_correctly() -> None:
    stub = StubLLMClient(scripted_responses=[_valid_json_for("world_modeler", "epidemiology")])
    agent = WorldModelerSubagent(llm_client=stub)
    input_data = _make_subagent_input(brain="epidemiology", role="world_modeler", tick=3, round_=1)

    agent.run(input_data, step_idx=0)

    expected_caller_id = "cortex:epidemiology:world_modeler:t3:r1:s0"
    assert stub.tokens_used_for(expected_caller_id) > 0, (
        "tokens must be billed to the per-role caller_id, not silently lost"
    )
    assert stub.tokens_used_for("never:called") == 0


# ============================================================================
# T10 - empty_fallback shape locked per Phase A Decisions 6 + 62
# ============================================================================


def test_subagent_empty_fallback_shape_locked() -> None:
    # WorldModeler: empty BeliefState
    bs = WorldModelerSubagent.empty_fallback("epidemiology")
    assert isinstance(bs, BeliefState)
    assert bs.brain == "epidemiology"
    assert bs.latent_estimates == {}
    assert bs.hypotheses == []
    assert bs.uncertainty == 1.0
    assert bs.reducible_by_more_thought == 0.0
    assert bs.evidence == []

    # Planner: empty CandidatePlan with NoOp + confidence=0
    cp = PlannerSubagent.empty_fallback("epidemiology")
    assert isinstance(cp, CandidatePlan)
    assert cp.expected_outer_action.kind == "no_op"
    assert cp.expected_value == 0.0
    assert cp.cost == 0.0
    assert cp.assumptions == []
    assert cp.falsifiers == []
    assert cp.confidence == 0.0

    # Critic: empty CriticReport with severity=0
    cr = CriticSubagent.empty_fallback("epidemiology", target_plan_id="plan-X")
    assert isinstance(cr, CriticReport)
    assert cr.brain == "epidemiology"
    assert cr.target_plan_id == "plan-X"
    assert cr.attacks == []
    assert cr.missing_considerations == []
    assert cr.would_change_mind_if == []
    assert cr.severity == 0.0


# ============================================================================
# T11 - retry call uses chat-history continuation (sys + usr + bad + retry)
# ============================================================================


def test_subagent_run_uses_chat_history_on_retry() -> None:
    stub = StubLLMClient(
        scripted_responses=["bad-json", _valid_json_for("world_modeler", "epidemiology")]
    )
    agent = WorldModelerSubagent(llm_client=stub)

    agent.run(_make_subagent_input(brain="epidemiology", role="world_modeler"), step_idx=0)

    assert stub.call_count == 2, "expected 2 LLM calls (initial + retry)"
    call1, call2 = stub.calls

    # call 1 has the original sys + user (2 messages).
    assert len(call1.messages) == 2
    assert call1.messages[0].role == "system"
    assert call1.messages[1].role == "user"

    # call 2 must contain: original sys + original user + assistant(bad-json) + retry-user.
    # Without chat-history continuation the LLM loses schema context on retry.
    assert len(call2.messages) == 4, "retry must append to the chat history, not start fresh"
    assert call2.messages[0].role == "system"
    assert call2.messages[0].content == call1.messages[0].content
    assert call2.messages[1].role == "user"
    assert call2.messages[1].content == call1.messages[1].content
    assert call2.messages[2].role == "assistant"
    assert call2.messages[2].content == "bad-json"
    assert call2.messages[3].role == "user"
    assert "failed to parse" in call2.messages[3].content.lower()

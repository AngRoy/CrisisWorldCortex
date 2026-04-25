"""Tests for the B1 flat-agent baseline (Session 7a).

Per the approved spec:
  - System prompt section 4 is domain-neutral (no strategy hand-crafting).
  - Single LLM call per tick; prompt-token sanity check on first call.
  - Parse failure → submit a synthetic ``PublicCommunication`` to the env
    so the env rejects with ``accepted=False`` and r_policy=0 lands as
    the reward-signal penalty. Episode does NOT terminate.
  - Uses ``cortex.llm_client.LLMClient`` (token-counted harness), via
    a dependency-injected stub in tests.

The HTTP-client constraint in baselines/CLAUDE.md applies to *production*
B1 (it must speak to the env over HTTP, not in-process). Tests
intentionally use the in-process env via a thin adapter — the test, not
B1, instantiates ``CrisisworldcortexEnvironment``.
"""

from __future__ import annotations

import json
from typing import List

import pytest

from baselines.flat_agent import (
    B1FlatAgent,
    parse_action,
    serialize_observation,
)
from cortex.llm_client import ChatResponse
from CrisisWorldCortex.models import CrisisworldcortexAction
from CrisisWorldCortex.server.CrisisWorldCortex_environment import (
    CrisisworldcortexEnvironment,
)

# ============================================================================
# Stubs
# ============================================================================


class _StubLLMClient:
    """Quacks-like-LLMClient: ``.chat(caller_id=..., messages=...) -> ChatResponse``.

    Cycles through a list of canned response strings. Records caller_ids
    so tests can verify B1 uses the expected prefix.
    """

    def __init__(self, scripted_contents: List[str]) -> None:
        self._scripted = list(scripted_contents)
        self.calls_made: List[str] = []  # caller_ids
        self._counters: dict[str, int] = {}

    def chat(self, caller_id: str, messages, max_tokens=None, temperature=None):
        self.calls_made.append(caller_id)
        if not self._scripted:
            content = '{"kind": "no_op"}'
        else:
            content = self._scripted.pop(0)
        prompt_tokens = sum(len(m.content) for m in messages) // 4
        completion_tokens = max(1, len(content) // 4)
        self._counters[caller_id] = (
            self._counters.get(caller_id, 0) + prompt_tokens + completion_tokens
        )
        return ChatResponse(
            content=content,
            finish_reason="stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def tokens_used_for(self, caller_id: str) -> int:
        return self._counters.get(caller_id, 0)

    def reset_counters(self, caller_id_prefix=None) -> None:
        if caller_id_prefix is None:
            self._counters.clear()
        else:
            for k in list(self._counters):
                if k.startswith(caller_id_prefix):
                    self._counters[k] = 0


class _InProcessEnvAdapter:
    """Adapts ``CrisisworldcortexEnvironment`` (returns observations
    directly) to B1's expected env interface — sync ``reset()`` /
    ``step(action)`` returning a ``CrisisworldcortexObservation``.

    Production B1 will receive a thin wrapper around the HTTP
    ``CrisisworldcortexEnv`` client (which returns ``StepResult``).
    """

    def __init__(self, env: CrisisworldcortexEnvironment) -> None:
        self._env = env

    def reset(self):
        return self._env.reset()

    def step(self, action: CrisisworldcortexAction):
        return self._env.step(action)


# ============================================================================
# Parser tests
# ============================================================================


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "no_op"},
        {"kind": "deploy_resource", "region": "R1", "resource_type": "test_kits", "quantity": 50},
        {"kind": "request_data", "region": "R2", "data_type": "case_survey"},
        {"kind": "restrict_movement", "region": "R1", "severity": "moderate"},
        {"kind": "escalate", "to_authority": "regional"},
        {
            "kind": "reallocate_budget",
            "from_resource": "test_kits",
            "to_resource": "mobile_units",
            "amount": 10,
        },
    ],
)
def test_parser_accepts_each_mvp_kind(payload: dict) -> None:
    parsed = parse_action(json.dumps(payload))
    assert parsed is not None, f"parser rejected valid payload: {payload!r}"
    assert parsed.kind == payload["kind"]


def test_parser_strips_codeblock_fences() -> None:
    """Triple-backtick ```json ... ``` wrapping (common LLM habit)."""
    text = '```json\n{"kind": "no_op"}\n```'
    parsed = parse_action(text)
    assert parsed is not None
    assert parsed.kind == "no_op"

    text2 = '```\n{"kind": "escalate", "to_authority": "national"}\n```'
    parsed2 = parse_action(text2)
    assert parsed2 is not None
    assert parsed2.kind == "escalate"


def test_parser_finds_brace_block_in_prose() -> None:
    """LLM emitted prose with a JSON block in the middle — extract it."""
    text = 'Sure, here is my action:\n{"kind": "no_op"}\nHope this helps!'
    parsed = parse_action(text)
    assert parsed is not None
    assert parsed.kind == "no_op"


@pytest.mark.parametrize(
    "garbage",
    [
        "",
        "   ",
        "I cannot determine an action.",
        "{",
        "{not valid json}",
        '{"kind": "not_a_real_kind"}',
        '{"region": "R1"}',  # missing kind
    ],
)
def test_parser_returns_none_on_garbage(garbage: str) -> None:
    assert parse_action(garbage) is None, f"parser accepted garbage: {garbage!r}"


# ============================================================================
# Serializer test
# ============================================================================


def test_serialize_observation_includes_required_sections() -> None:
    """Sanity: the rendered prompt has the documented section markers."""
    env = CrisisworldcortexEnvironment()
    obs = env.reset()
    rendered = serialize_observation(obs, last_reward=0.0)
    for marker in (
        "Tick",
        "Resources",
        "Regions",
        "Active restrictions",
        "Legal constraints",
        "Recent actions",
    ):
        assert marker in rendered, f"missing section marker: {marker!r}"


# ============================================================================
# Episode-runner tests (in-process env via adapter)
# ============================================================================


def test_b1_runs_episode_with_valid_json() -> None:
    """Stub LLM always emits ``NoOp`` JSON; B1 runs without any rejections.

    Episode length is whatever the env decides. ``outbreak_easy`` with
    all-NoOp typically terminates early on a success terminal (3
    consecutive safe ticks). What matters: at least one tick ran,
    no parse failures, every reward is in [0,1], and the action log
    shows NoOp acceptance throughout.
    """
    env = _InProcessEnvAdapter(CrisisworldcortexEnvironment())
    llm = _StubLLMClient(['{"kind": "no_op"}'] * 20)
    agent = B1FlatAgent(env=env, llm=llm)

    trajectory = agent.run_episode(task="outbreak_easy", seed=0, max_ticks=12)

    assert trajectory["steps_taken"] >= 1, "episode produced zero ticks"
    assert trajectory["steps_taken"] <= 12, "episode exceeded max_ticks"
    assert trajectory["parse_failure_count"] == 0
    assert len(trajectory["rewards"]) == trajectory["steps_taken"]
    for r in trajectory["rewards"]:
        assert 0.0 <= r <= 1.0
    for entry in trajectory["action_history"]:
        assert entry["submitted_kind"] == "no_op"
        assert entry["parse_failure"] is False


def test_b1_parse_failure_submits_synthetic_rejection() -> None:
    """When the LLM emits unparseable text, B1 submits a synthetic
    PublicCommunication so the env rejects with accepted=False — landing
    r_policy=0 in outer_reward, and the action log shows the rejection.
    """
    env_inner = CrisisworldcortexEnvironment()
    env = _InProcessEnvAdapter(env_inner)
    llm = _StubLLMClient(
        [
            "I cannot help with that.",
            "Sorry, no JSON.",
        ]
        + ['{"kind": "no_op"}'] * 20
    )
    agent = B1FlatAgent(env=env, llm=llm)

    trajectory = agent.run_episode(task="outbreak_easy", seed=0, max_ticks=5)

    # Both parse failures were detected and counted.
    assert trajectory["parse_failure_count"] == 2

    # B1 did NOT crash on parse failure — at least 3 ticks ran, even
    # though the env may then have hit a terminal (success-on-3-safe-ticks
    # or otherwise). What's binding: parse failure does not raise.
    assert trajectory["steps_taken"] >= 3, (
        f"steps_taken={trajectory['steps_taken']!r} - parse failure "
        f"shouldn't kill the agent before tick 3"
    )

    # The first two action-log entries show V2 rejection (synthetic
    # public_communication was submitted; env returned accepted=False).
    log = env_inner._world_state.recent_action_log
    assert len(log) >= 2
    assert log[0].action.kind == "public_communication"
    assert log[0].accepted is False, "parse-failure synthetic must be rejected by env"
    assert log[1].action.kind == "public_communication"
    assert log[1].accepted is False

    # The third entry should be the first parsed NoOp.
    assert log[2].action.kind == "no_op"
    assert log[2].accepted is True

    # B1's local trajectory carries the raw snippets for forensic use.
    assert trajectory["action_history"][0]["parse_failure"] is True
    assert trajectory["action_history"][0]["raw_llm"] == "I cannot help with that."
    assert trajectory["action_history"][1]["parse_failure"] is True
    assert trajectory["action_history"][1]["raw_llm"] == "Sorry, no JSON."
    assert trajectory["action_history"][2]["parse_failure"] is False


def test_b1_caller_id_format_short_colon_separated() -> None:
    """Caller IDs follow the ``b1:t<N>`` short form approved in §4."""
    env = _InProcessEnvAdapter(CrisisworldcortexEnvironment())
    llm = _StubLLMClient(['{"kind": "no_op"}'] * 5)
    agent = B1FlatAgent(env=env, llm=llm)
    agent.run_episode(task="outbreak_easy", seed=0, max_ticks=3)

    assert len(llm.calls_made) == 3
    for i, caller_id in enumerate(llm.calls_made, start=1):
        assert caller_id == f"b1:t{i}", f"expected 'b1:t{i}', got {caller_id!r}"


def test_b1_resets_counters_at_episode_start() -> None:
    """Two episodes back-to-back: token counter for b1:* resets at start."""
    env = _InProcessEnvAdapter(CrisisworldcortexEnvironment())
    llm = _StubLLMClient(['{"kind": "no_op"}'] * 10)
    agent = B1FlatAgent(env=env, llm=llm)

    agent.run_episode(task="outbreak_easy", seed=0, max_ticks=3)
    assert llm.tokens_used_for("b1:t1") > 0  # tokens were billed

    agent.run_episode(task="outbreak_easy", seed=1, max_ticks=2)
    assert llm.tokens_used_for("b1:t3") == 0, (
        "stale counter from previous episode must be zeroed on episode start"
    )

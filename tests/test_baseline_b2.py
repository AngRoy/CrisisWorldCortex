"""Tests for the B2 matched-compute self-revision baseline (Session 8).

Per design §20.1.1:
  - Mechanism: 1 initial + N x (critique, revision) until token budget
    is exhausted or nearly exhausted.
  - Final pass emits exactly one OuterAction. Never mid-revision drafts.
  - Identical model + identical per-tick token budget + identical
    observation/history access as Cortex (the matched-compute control).

These tests pin the implementation contract using stub LLMs with
token-aware responses; no real-LLM matched-compute statistical
assertion (deferred to Session 14's eval harness).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from baselines.flat_agent import B1StepEvent
from baselines.flat_agent_matched_compute import B2MatchedComputeAgent
from cortex.llm_client import ChatResponse
from CrisisWorldCortex.models import CrisisworldcortexAction
from CrisisWorldCortex.server.CrisisWorldCortex_environment import (
    CrisisworldcortexEnvironment,
)

# ============================================================================
# Stubs
# ============================================================================


@dataclass
class _StubCall:
    """One scripted LLM response with explicit token costs.

    B2's budget tracking accumulates ``response.prompt_tokens +
    completion_tokens`` after each call, so the stub must report honest
    per-call token costs to exercise the budget cap.
    """

    content: str
    prompt_tokens: int = 200
    completion_tokens: int = 100  # ~300 tokens/call default


class _StubLLMClient:
    """Quacks-like-LLMClient with scripted responses + per-caller counters.

    Tests script the response queue; B2 consumes it call-by-call. Caller
    IDs are recorded in ``calls_made`` for assertion against B2's
    expected ``b2:t<tick>:p<n>:<role>`` format.
    """

    def __init__(self, scripted: List[_StubCall]) -> None:
        self._queue = list(scripted)
        self.calls_made: List[str] = []
        self._counters: Dict[str, int] = {}

    def chat(self, caller_id: str, messages, max_tokens=None, temperature=None):
        self.calls_made.append(caller_id)
        if not self._queue:
            raise RuntimeError(f"stub exhausted at {caller_id}")
        call = self._queue.pop(0)
        self._counters[caller_id] = (
            self._counters.get(caller_id, 0) + call.prompt_tokens + call.completion_tokens
        )
        return ChatResponse(
            content=call.content,
            finish_reason="stop",
            prompt_tokens=call.prompt_tokens,
            completion_tokens=call.completion_tokens,
        )

    def tokens_used_for(self, caller_id: str) -> int:
        return self._counters.get(caller_id, 0)

    def reset_counters(self, caller_id_prefix: Optional[str] = None) -> None:
        if caller_id_prefix is None:
            self._counters.clear()
            return
        for k in list(self._counters):
            if k.startswith(caller_id_prefix):
                self._counters[k] = 0


class _InProcessEnvAdapter:
    """Same shape as B1's adapter (tests/test_baseline_b1.py): forwards
    reset()/step() to an in-process CrisisworldcortexEnvironment."""

    def __init__(self, env: CrisisworldcortexEnvironment) -> None:
        self._env = env

    def reset(self):
        return self._env.reset()

    def step(self, action: CrisisworldcortexAction):
        return self._env.step(action)


_NOOP_JSON = '{"kind": "no_op"}'
_DEPLOY_JSON = (
    '{"kind": "deploy_resource", "region": "R1", "resource_type": "test_kits", "quantity": 50}'
)
_CRITIQUE_PROSE = (
    "The proposed action ignores R1's hospital_load creep. Consider deploying test_kits."
)


def _new_agent(llm, *, tick_budget: Optional[int] = None) -> Any:
    env = _InProcessEnvAdapter(CrisisworldcortexEnvironment())
    if tick_budget is None:
        return B2MatchedComputeAgent(env=env, llm=llm)
    return B2MatchedComputeAgent(env=env, llm=llm, tick_budget=tick_budget)


# ============================================================================
# Tests
# ============================================================================


def test_b2_initial_only_when_budget_blocks_first_revision() -> None:
    """When tick_budget is too small to fit a (critique, revision) pair,
    B2 emits the initial candidate unchanged. No revisions.

    With each call ~300 tokens and the dynamic estimate at 600 per call,
    a pair costs ~1200. tick_budget=300 means initial fits but no pair
    can start.
    """
    llm = _StubLLMClient([_StubCall(_NOOP_JSON, 200, 100)] * 5)
    agent = _new_agent(llm, tick_budget=300)
    traj = agent.run_episode(task="outbreak_easy", seed=0, max_ticks=2)

    # Tick 1 fired exactly one LLM call (initial only).
    assert llm.calls_made[0] == "b2:t1:p0:initial"
    # No critique or revision in tick 1.
    tick1_calls = [c for c in llm.calls_made if c.startswith("b2:t1:")]
    assert len(tick1_calls) == 1, f"expected 1 call in tick 1 (initial only); got {tick1_calls}"
    # Pass count for tick 1 is 0 (no revisions completed).
    assert traj["pass_counts"][0] == 0


def test_b2_one_revision_when_budget_allows_one_pair() -> None:
    """tick_budget large enough for initial + 1 pair, not 2.

    Initial ~300 + critique ~300 + revision ~300 = 900 tokens. Budget
    1200 leaves ~300 remaining — less than another estimated pair (1200).
    Result: exactly 1 revision pass.
    """
    llm = _StubLLMClient(
        [
            _StubCall(_NOOP_JSON, 200, 100),  # initial
            _StubCall(_CRITIQUE_PROSE, 200, 100),  # critique
            _StubCall(_DEPLOY_JSON, 200, 100),  # revision
            _StubCall(_NOOP_JSON, 200, 100),  # spare (unused if max_ticks=1)
        ]
    )
    agent = _new_agent(llm, tick_budget=1200)
    traj = agent.run_episode(task="outbreak_easy", seed=0, max_ticks=1)

    tick1_calls = [c for c in llm.calls_made if c.startswith("b2:t1:")]
    assert tick1_calls == [
        "b2:t1:p0:initial",
        "b2:t1:p1:critique",
        "b2:t1:p1:revision",
    ], f"expected initial+critique+revision; got {tick1_calls}"
    assert traj["pass_counts"][0] == 1
    # Submitted action == revised candidate (deploy_resource), not initial NoOp.
    assert traj["action_history"][0]["submitted_kind"] == "deploy_resource"


def test_b2_n_revisions_until_budget_cap() -> None:
    """Large budget exhausted by repeated (critique, revision) pairs.

    Each call ~300 tokens. tick_budget=3000. Initial=300, then pairs
    cost ~600 each. The dynamic estimate over the moving average pushes
    the cutoff a bit conservative; expect pass_count to be in [3, 5].
    """
    queue = [_StubCall(_NOOP_JSON, 200, 100)]  # initial
    for _ in range(20):
        queue.append(_StubCall(_CRITIQUE_PROSE, 200, 100))
        queue.append(_StubCall(_DEPLOY_JSON, 200, 100))

    llm = _StubLLMClient(queue)
    agent = _new_agent(llm, tick_budget=3000)
    traj = agent.run_episode(task="outbreak_easy", seed=0, max_ticks=1)

    pass_count = traj["pass_counts"][0]
    assert 3 <= pass_count <= 5, f"expected 3-5 passes within budget; got {pass_count}"
    # Caller-IDs cover initial + 2*pass_count critique/revision pairs.
    expected_calls = 1 + 2 * pass_count
    tick1_calls = [c for c in llm.calls_made if c.startswith("b2:t1:")]
    assert len(tick1_calls) == expected_calls
    # Final submitted action == latest revision (deploy_resource).
    assert traj["action_history"][0]["submitted_kind"] == "deploy_resource"


def test_b2_caller_ids_match_b2_t_p_role_format() -> None:
    """Pin caller-ID format: b2:t<tick>:p<pass>:<role>."""
    llm = _StubLLMClient(
        [
            _StubCall(_NOOP_JSON, 100, 50),
            _StubCall(_CRITIQUE_PROSE, 100, 50),
            _StubCall(_DEPLOY_JSON, 100, 50),
        ]
        + [_StubCall(_NOOP_JSON, 100, 50)] * 5
    )
    agent = _new_agent(llm, tick_budget=600)
    agent.run_episode(task="outbreak_easy", seed=0, max_ticks=1)

    # First three calls are tick 1's initial+pair-1.
    assert llm.calls_made[0] == "b2:t1:p0:initial"
    assert llm.calls_made[1] == "b2:t1:p1:critique"
    assert llm.calls_made[2] == "b2:t1:p1:revision"


def test_b2_initial_parse_failure_falls_back_to_synthetic_marker() -> None:
    """Garbage initial + budget too small for revision -> submit synthetic
    PublicCommunication marker; tick recorded as parse_failure."""
    llm = _StubLLMClient(
        [
            _StubCall("Sorry I cannot help with that.", 100, 50),
            _StubCall(_NOOP_JSON, 100, 50),
            _StubCall(_NOOP_JSON, 100, 50),
        ]
    )
    agent = _new_agent(llm, tick_budget=300)  # initial only, no pair
    traj = agent.run_episode(task="outbreak_easy", seed=0, max_ticks=1)

    # Exactly one call (initial); no revision attempted.
    tick1_calls = [c for c in llm.calls_made if c.startswith("b2:t1:")]
    assert len(tick1_calls) == 1
    # Synthetic V2-rejected marker submitted.
    assert traj["action_history"][0]["submitted_kind"] == "public_communication"
    assert traj["action_history"][0]["parse_failure"] is True


def test_b2_revision_parse_failure_keeps_prior_candidate() -> None:
    """When the revision response can't be parsed, B2 emits the most
    recent fully-parsed candidate (initial in this test). Per design
    §20.1.1: "Never emits mid-revision drafts."
    """
    llm = _StubLLMClient(
        [
            _StubCall(_DEPLOY_JSON, 200, 100),  # initial: VALID deploy_resource
            _StubCall(_CRITIQUE_PROSE, 200, 100),  # critique
            _StubCall("garbage that won't parse", 200, 100),  # revision: PARSE FAIL
        ]
    )
    agent = _new_agent(llm, tick_budget=1200)  # ~1 pair fits
    traj = agent.run_episode(task="outbreak_easy", seed=0, max_ticks=1)

    # The submitted action must be the INITIAL deploy_resource (prior candidate),
    # NOT the parse-failed revision and NOT the synthetic marker.
    assert traj["action_history"][0]["submitted_kind"] == "deploy_resource"
    # Trajectory records that a pass was attempted but its revision failed.
    assert traj["pass_counts"][0] == 1
    # parse_failure_count tracks ticks where the FINAL submitted action was the
    # synthetic marker. Falling back to a prior valid candidate is NOT a parse
    # failure from the trajectory's perspective.
    assert traj["parse_failure_count"] == 0


def test_b2_runs_full_episode_smoke() -> None:
    """Stub LLM cycles {valid, prose, valid, prose, ...}; B2 runs a full
    multi-tick episode without crashing. Trajectory dict is well-formed.
    """
    queue = []
    for _ in range(50):  # plenty for any episode length
        queue.append(_StubCall(_NOOP_JSON, 100, 50))  # initial
        for _ in range(5):
            queue.append(_StubCall(_CRITIQUE_PROSE, 100, 50))  # critique
            queue.append(_StubCall(_NOOP_JSON, 100, 50))  # revision

    llm = _StubLLMClient(queue)
    agent = _new_agent(llm)  # default budget = 6000
    traj = agent.run_episode(task="outbreak_easy", seed=0, max_ticks=3)

    assert traj["steps_taken"] >= 1
    assert len(traj["rewards"]) == traj["steps_taken"]
    assert len(traj["pass_counts"]) == traj["steps_taken"]
    assert len(traj["tick_token_totals"]) == traj["steps_taken"]
    assert traj["tokens_total"] > 0
    assert traj["tokens_total"] == sum(traj["tick_token_totals"])
    for r in traj["rewards"]:
        assert 0.0 <= r <= 1.0


def test_b2_token_total_does_not_exceed_budget_cap() -> None:
    """Per-tick total tokens stay within budget (allowing one trailing
    pair's overshoot — bounded by max_tokens=512 + prompt growth)."""
    # Each call ~600 tokens (200 prompt + 400 completion).
    queue = [_StubCall(_NOOP_JSON, 200, 400)]
    for _ in range(20):
        queue.append(_StubCall(_CRITIQUE_PROSE, 200, 400))
        queue.append(_StubCall(_NOOP_JSON, 200, 400))

    llm = _StubLLMClient(queue)
    agent = _new_agent(llm, tick_budget=6000)
    traj = agent.run_episode(task="outbreak_easy", seed=0, max_ticks=1)

    # Per-tick total respects the budget within a small slop margin.
    # One in-flight pair could push us slightly over; allow ~1 pair's
    # overshoot since the dynamic estimate is a moving average.
    assert traj["tick_token_totals"][0] <= 6000 + 1200, (
        f"tick token total {traj['tick_token_totals'][0]!r} far exceeds "
        f"budget cap 6000 — budget guard not working"
    )


def test_b2_emits_step_event_per_tick() -> None:
    """B2 honors the same step_callback contract as B1. Exactly one
    B1StepEvent per tick, fired after the FINAL submitted action."""
    queue = []
    for _ in range(10):
        queue.append(_StubCall(_NOOP_JSON, 100, 50))
        queue.append(_StubCall(_CRITIQUE_PROSE, 100, 50))
        queue.append(_StubCall(_NOOP_JSON, 100, 50))

    llm = _StubLLMClient(queue)
    agent = _new_agent(llm, tick_budget=1200)
    events: List[B1StepEvent] = []
    traj = agent.run_episode(
        task="outbreak_easy",
        seed=0,
        max_ticks=3,
        step_callback=events.append,
    )

    assert len(events) == traj["steps_taken"]
    for i, ev in enumerate(events, start=1):
        assert ev.tick == i
        assert ev.action.kind in {
            "no_op",
            "deploy_resource",
            "request_data",
            "restrict_movement",
            "escalate",
            "reallocate_budget",
            "public_communication",
        }


# Suppress unused-pytest-import warning if the imports above don't exercise pytest.
_ = pytest

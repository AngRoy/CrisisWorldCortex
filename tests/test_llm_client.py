"""Tests for ``cortex.llm_client.LLMClient`` (Session 7a).

Per the approved spec:
  - Cumulative per-caller token counter (sum across all calls).
  - Source: API ``response.usage.prompt_tokens + .completion_tokens``;
    if ``usage`` is missing, increment by 0 and warn (no local tokenizer).
  - Caller IDs: short colon-separated strings ("inference:t3", "b1:t3",
    "cortex:epi:planner:t3"). Passed explicitly per call, not thread-local.
  - ``reset_counters`` is harness-driven: never auto-reset by the client.

The OpenAI SDK is stubbed via dependency injection: ``LLMClient`` accepts
an optional ``client`` kwarg in ``__init__`` so tests can pass a fake
without monkey-patching the global OpenAI module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from cortex.llm_client import ChatMessage, ChatResponse, LLMClient

# ============================================================================
# Stub OpenAI client — minimal subset of the SDK surface used by LLMClient
# ============================================================================


@dataclass
class _StubUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _StubMessage:
    content: str


@dataclass
class _StubChoice:
    message: _StubMessage
    finish_reason: str = "stop"


@dataclass
class _StubCompletion:
    choices: List[_StubChoice]
    usage: Optional[_StubUsage]


class _StubChatCompletions:
    def __init__(self, responses: List[_StubCompletion]) -> None:
        self._responses = list(responses)
        self.calls: List[dict] = []

    def create(self, **kwargs) -> _StubCompletion:
        self.calls.append(kwargs)
        if not self._responses:
            raise RuntimeError("stub exhausted - test wants more responses than configured")
        return self._responses.pop(0)


class _StubChat:
    def __init__(self, completions: _StubChatCompletions) -> None:
        self.completions = completions


class _StubOpenAI:
    """Quacks-like-OpenAI: ``client.chat.completions.create(...)``."""

    def __init__(self, responses: List[_StubCompletion]) -> None:
        self.chat = _StubChat(_StubChatCompletions(responses))


def _resp(
    content: str, prompt_tokens: int, completion_tokens: int, usage: bool = True
) -> _StubCompletion:
    return _StubCompletion(
        choices=[_StubChoice(message=_StubMessage(content=content))],
        usage=_StubUsage(prompt_tokens, completion_tokens) if usage else None,
    )


# ============================================================================
# Tests
# ============================================================================


def test_token_counter_increments_per_call() -> None:
    """Single caller, two calls: counter is the cumulative sum of usages."""
    stub = _StubOpenAI(
        [
            _resp("ok-1", prompt_tokens=50, completion_tokens=30),
            _resp("ok-2", prompt_tokens=70, completion_tokens=10),
        ]
    )
    client = LLMClient(model="stub-model", client=stub)

    client.chat(caller_id="b1:t1", messages=[ChatMessage(role="user", content="hi")])
    assert client.tokens_used_for("b1:t1") == 80, "after 1 call: 50+30 = 80"

    client.chat(caller_id="b1:t1", messages=[ChatMessage(role="user", content="hi")])
    assert client.tokens_used_for("b1:t1") == 160, "after 2 calls: 80 + 70+10 = 160"


def test_token_counter_isolates_callers() -> None:
    """Two distinct caller_ids accumulate independently."""
    stub = _StubOpenAI(
        [
            _resp("a", prompt_tokens=100, completion_tokens=20),
            _resp("b", prompt_tokens=5, completion_tokens=5),
        ]
    )
    client = LLMClient(model="stub-model", client=stub)

    client.chat(caller_id="inference:t1", messages=[ChatMessage(role="user", content="x")])
    client.chat(caller_id="b1:t1", messages=[ChatMessage(role="user", content="y")])

    assert client.tokens_used_for("inference:t1") == 120
    assert client.tokens_used_for("b1:t1") == 10
    assert client.tokens_used_for("never:called") == 0, "unknown caller_id reads as 0, not KeyError"


def test_reset_counters_zeroes_callers() -> None:
    """``reset_counters()`` with no args clears all; with a prefix clears
    only matching keys. No automatic reset happens — the client never
    zeroes counters on its own."""
    stub = _StubOpenAI(
        [
            _resp("a", 30, 10),
            _resp("b", 50, 10),
            _resp("c", 5, 5),
        ]
    )
    client = LLMClient(model="stub-model", client=stub)

    client.chat(caller_id="inference:t1", messages=[ChatMessage(role="user", content="x")])
    client.chat(caller_id="b1:t1", messages=[ChatMessage(role="user", content="y")])
    client.chat(caller_id="b1:t2", messages=[ChatMessage(role="user", content="z")])
    assert client.tokens_used_for("inference:t1") == 40
    assert client.tokens_used_for("b1:t1") == 60
    assert client.tokens_used_for("b1:t2") == 10

    # Prefix-scoped reset: only b1:* keys zero, inference:* survives.
    client.reset_counters(caller_id_prefix="b1:")
    assert client.tokens_used_for("b1:t1") == 0
    assert client.tokens_used_for("b1:t2") == 0
    assert client.tokens_used_for("inference:t1") == 40, (
        "prefix reset must not touch unrelated caller_ids"
    )

    # Full reset.
    client.reset_counters()
    assert client.tokens_used_for("inference:t1") == 0


def test_chat_returns_typed_response_and_handles_missing_usage() -> None:
    """``ChatResponse`` carries content + finish_reason + token fields.

    When a provider returns no ``usage`` block, the counter must NOT
    fall over — increment by 0 and surface zero token fields, so callers
    can still read the content. (No local tokenizer fallback in MVP.)
    """
    stub = _StubOpenAI(
        [
            _resp("hello world", prompt_tokens=12, completion_tokens=5),
            _resp("no usage", prompt_tokens=0, completion_tokens=0, usage=False),
        ]
    )
    client = LLMClient(model="stub-model", client=stub)

    r1 = client.chat(
        caller_id="inference:t1",
        messages=[ChatMessage(role="user", content="hello")],
    )
    assert isinstance(r1, ChatResponse)
    assert r1.content == "hello world"
    assert r1.finish_reason == "stop"
    assert r1.prompt_tokens == 12
    assert r1.completion_tokens == 5
    assert client.tokens_used_for("inference:t1") == 17

    # Provider with no usage block: don't crash.
    r2 = client.chat(
        caller_id="inference:t2",
        messages=[ChatMessage(role="user", content="hi")],
    )
    assert r2.content == "no usage"
    assert r2.prompt_tokens == 0
    assert r2.completion_tokens == 0
    assert client.tokens_used_for("inference:t2") == 0


def test_chat_passes_messages_and_temperature_to_sdk() -> None:
    """Messages and temperature must reach the SDK call verbatim."""
    stub = _StubOpenAI([_resp("ok", 1, 1)])
    client = LLMClient(model="stub-model", client=stub, temperature=0.7, max_tokens=128)
    client.chat(
        caller_id="b1:t1",
        messages=[
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="usr"),
        ],
    )
    call = stub.chat.completions.calls[0]
    assert call["model"] == "stub-model"
    assert call["temperature"] == 0.7
    assert call["max_tokens"] == 128
    assert call["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]

"""LLMClient-level test double.

Drop-in replacement for ``cortex.llm_client.LLMClient`` for tests that
exercise consumers of the client (subagents, brains, council). Differs
from the SDK-level stub in ``tests/test_llm_client.py``: that one
intercepts the OpenAI SDK; this one intercepts the ``LLMClient.chat``
surface directly, which is what subagents and harnesses see.

Reused by sessions 9-13 — do not bury role-specific test logic here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from cortex.llm_client import ChatMessage, ChatResponse


@dataclass
class RecordedCall:
    """One captured ``chat()`` invocation. Tests assert on these."""

    caller_id: str
    messages: List[ChatMessage]
    max_tokens: Optional[int]
    temperature: Optional[float]


@dataclass
class StubLLMClient:
    """Quacks like ``LLMClient`` for the ``chat()`` and counter surface.

    Args:
        scripted_responses: Yielded one per ``chat()`` call, in order.
            Each entry is the response ``content`` string.
        prompt_tokens_per_call: Fake prompt-token billing per call.
        completion_tokens_per_call: Fake completion-token billing per call.
    """

    scripted_responses: List[str]
    prompt_tokens_per_call: int = 50
    completion_tokens_per_call: int = 30
    calls: List[RecordedCall] = field(default_factory=list)
    _counters: Dict[str, int] = field(default_factory=dict)

    def chat(
        self,
        caller_id: str,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> ChatResponse:
        if not self.scripted_responses:
            raise RuntimeError(
                f"StubLLMClient exhausted: caller_id={caller_id!r}; "
                f"add more scripted_responses or assert call_count earlier."
            )
        content = self.scripted_responses.pop(0)
        self.calls.append(
            RecordedCall(
                caller_id=caller_id,
                messages=list(messages),
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
        billed = self.prompt_tokens_per_call + self.completion_tokens_per_call
        self._counters[caller_id] = self._counters.get(caller_id, 0) + billed
        return ChatResponse(
            content=content,
            finish_reason="stop",
            prompt_tokens=self.prompt_tokens_per_call,
            completion_tokens=self.completion_tokens_per_call,
        )

    def tokens_used_for(self, caller_id: str) -> int:
        return self._counters.get(caller_id, 0)

    @property
    def call_count(self) -> int:
        return len(self.calls)

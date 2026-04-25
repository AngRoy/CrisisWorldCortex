# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
LLM client wrapper with per-caller token counting (cortex/CLAUDE.md APIs).

This is the shared LLM surface for harnesses outside ``server/``:
``inference.py``, ``baselines/*``, future ``training/train_router.py``,
and Cortex subagents (sessions 9+). Per the Q1 decision in root
``CLAUDE.md``, ``r_budget`` is harness-tracked from this module's token
counters, never from env state.

Design notes:
  - Backed by the OpenAI Python SDK against an OpenAI-compatible endpoint
    (HF Router default: ``https://router.huggingface.co/v1``). The HF
    Router accepts the same chat-completions schema as openai.com.
  - Token counting reads ``response.usage.{prompt_tokens, completion_tokens}``
    only — no local tokenizer fallback. If a provider omits ``usage``,
    the counter increments by 0 and a one-line warning hits stderr; the
    caller still gets the response content.
  - Caller IDs are short colon-separated strings ("inference:t3",
    "b1:t3", "cortex:epi:planner:t3"), passed explicitly per call. Not
    thread-local — robust to async / concurrent rollouts.
  - ``reset_counters`` is harness-driven: harnesses call it at episode
    boundaries. The client never auto-resets.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

# OpenAI SDK is approved as a prod dep (Session 7a). The HF Router and
# OpenAI's own API both speak this protocol; switching providers is a
# base_url + api_key change, not a code change.
try:
    from openai import OpenAI as _OpenAI
except ImportError:  # pragma: no cover - dep listed in pyproject.toml
    _OpenAI = None  # type: ignore[assignment]


__all__ = ["LLMClient", "ChatMessage", "ChatResponse"]


# ============================================================================
# Defaults — match Session 7b inference.py spec
# ============================================================================

DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 512


# ============================================================================
# Typed message and response shapes
# ============================================================================


@dataclass(frozen=True)
class ChatMessage:
    """One chat-completions message. ``role`` is the OpenAI chat role."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class ChatResponse:
    """Decoded LLM response with token-usage fields surfaced.

    ``finish_reason`` mirrors the SDK's value (typically ``"stop"`` /
    ``"length"`` / ``"content_filter"``). Token fields default to 0 if
    the provider didn't include a ``usage`` block.
    """

    content: str
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ============================================================================
# Client
# ============================================================================


class LLMClient:
    """Per-caller token-counting wrapper around OpenAI chat-completions.

    Args:
        api_base_url: Endpoint URL. Falls back to ``$API_BASE_URL`` then
            ``DEFAULT_API_BASE_URL``.
        api_key: API key. Falls back to ``$HF_TOKEN`` then ``$OPENAI_API_KEY``.
        model: Model identifier. Falls back to ``$MODEL_NAME`` then
            ``DEFAULT_MODEL``.
        temperature: Sampling temperature. 0.0 for reproducibility.
        max_tokens: Per-call output cap.
        client: Pre-built SDK client. Tests inject a stub here; production
            leaves it ``None`` and the OpenAI SDK is constructed from
            ``api_base_url`` + ``api_key``.
    """

    def __init__(
        self,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Optional[object] = None,
    ) -> None:
        self.api_base_url = api_base_url or os.getenv(
            "API_BASE_URL",
            DEFAULT_API_BASE_URL,
        )
        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("MODEL_NAME", DEFAULT_MODEL)
        self.temperature = temperature
        self.max_tokens = max_tokens

        if client is not None:
            self._client = client
        else:
            if _OpenAI is None:
                raise RuntimeError(
                    "openai SDK is not installed but no test client was passed. "
                    "Install with `uv sync` (openai>=1.0 is in pyproject.toml)."
                )
            if not self.api_key:
                raise ValueError(
                    "LLMClient requires an api_key. Set HF_TOKEN or OPENAI_API_KEY, "
                    "or pass api_key=... explicitly."
                )
            self._client = _OpenAI(base_url=self.api_base_url, api_key=self.api_key)

        self._token_counters: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        caller_id: str,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> ChatResponse:
        """Call chat-completions; bill prompt+completion tokens to ``caller_id``.

        Per-call ``max_tokens`` and ``temperature`` overrides are accepted
        for harnesses that want finer control without constructing a new
        client.
        """
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            stream=False,
        )

        # Defensive extraction — SDK returns rich objects, but tests use
        # dataclasses with the same attribute shape.
        choice = completion.choices[0]
        content = (choice.message.content or "").strip()
        finish_reason = getattr(choice, "finish_reason", "stop") or "stop"

        usage = getattr(completion, "usage", None)
        if usage is None:
            prompt_tokens = 0
            completion_tokens = 0
            print(
                f"[WARN] llm_client: response missing .usage for caller_id={caller_id!r}",
                file=sys.stderr,
                flush=True,
            )
        else:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        # Cumulative — defaults to 0 for new caller_ids.
        self._token_counters[caller_id] = (
            self._token_counters.get(caller_id, 0) + prompt_tokens + completion_tokens
        )

        return ChatResponse(
            content=content,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def tokens_used_for(self, caller_id: str) -> int:
        """Cumulative prompt+completion tokens billed to ``caller_id``.

        Unknown caller_ids read as 0 (not a KeyError). Harnesses use this
        to compose ``r_budget`` per design §14.3.
        """
        return self._token_counters.get(caller_id, 0)

    def reset_counters(self, caller_id_prefix: Optional[str] = None) -> None:
        """Zero counters whose key starts with ``caller_id_prefix``.

        With no prefix, clears all counters. Harnesses call this at
        episode boundaries (B1, inference.py, future training loops).
        The client never auto-resets — counters are sticky until cleared
        explicitly.
        """
        if caller_id_prefix is None:
            self._token_counters.clear()
            return
        for key in list(self._token_counters.keys()):
            if key.startswith(caller_id_prefix):
                self._token_counters[key] = 0

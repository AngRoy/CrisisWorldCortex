# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
B2 - matched-compute self-revision baseline (design §20.1.1).

Per-tick mechanism:

    1. Generate a proposed action with reasoning. (1 LLM call)
    2. Critique the proposal. (1 LLM call)
    3. Revise. (1 LLM call)
    4. Repeat steps 2-3 up to K passes; K is chosen so the per-tick
       token budget is exhausted or nearly exhausted.
    5. The final pass emits exactly one ``OuterAction``.

Stop rule (§20.1.1): if the model emits a stop signal (deferred to
Session 14 — see §9.5 in this session's proposal) or the per-tick
budget is exhausted, the **last revised action** is emitted. Never
emits mid-revision drafts.

Token-budget enforcement:

    - Per-tick budget defaults to ``TICK_BUDGET = 6000`` (matches
      ``TaskConfig.cognition_budget_per_tick`` from the design's
      §6.5 task config). Override via ``B2MatchedComputeAgent(...,
      tick_budget=...)`` so future Session 14 evaluation can
      re-calibrate to Cortex's actual measured per-tick usage.

    - After each LLM call, accumulate ``response.prompt_tokens +
      completion_tokens`` into the per-tick total. Before starting a
      new (critique, revision) pair, compare the remaining budget
      against an estimate of the next pair's cost (``2 *
      _estimate_call_cost(...)``). If we can't afford it, stop and
      emit ``current_candidate`` (the last fully-parsed action).

    - No safety margin (Session 8 §9.3): the existing ``parse_action +
      current_candidate`` fallback already handles truncated revision
      responses cleanly. Submitting a parse-failed revision falls back
      to the prior pass's candidate; submitting on initial parse
      failure falls back to the synthetic V2-rejected marker.

Sharing with B1 / Cortex:

    - System prompt for the initial generation is B1's
      ``build_system_prompt()``. Per §20.1.1 "shares the LLM client
      and action schema with Cortex" — reusing B1's prompt directly
      defends the matched-compute claim.
    - ``parse_action``, ``serialize_observation``, and
      ``parse_failure_marker`` are imported from
      ``baselines.flat_agent`` (public API after Session 8 rename).
    - The per-tick callback contract (``B1StepEvent`` /
      ``StepCallback``) is shared with B1: B2 fires the callback once
      per tick AFTER the final action is submitted to the env. Mid-
      revision drafts produce no events.
"""

from __future__ import annotations

import sys
import textwrap
from typing import Any, Dict, List, Optional

from baselines.flat_agent import (
    B1StepEvent,
    ErrorKind,
    StepCallback,
    build_system_prompt,
    parse_action,
    parse_failure_marker,
    serialize_observation,
)
from cortex.llm_client import ChatMessage
from CrisisWorldCortex.models import (
    CrisisworldcortexAction,
    OuterActionPayload,
)

__all__ = ["B2MatchedComputeAgent"]


# ============================================================================
# Constants
# ============================================================================

# Per-design §6.5: every TaskConfig declares cognition_budget_per_tick=6000.
# This is the per-tick LLM-token envelope B2 must match for the matched-
# compute claim to hold against Cortex (which uses the same envelope).
_DEFAULT_TICK_BUDGET = 6000

# Initial estimate before any in-tick LLM call has reported actual cost.
# Used both for the first "can we afford a pair?" check and as the reset
# value at the start of every tick.
_INITIAL_CALL_COST_ESTIMATE = 600

# Moving-average window over the most recent in-tick LLM calls (Session 8 §9.4).
_ESTIMATE_WINDOW = 3

# Hard cap on revision passes per tick — defensive against pathological
# cost estimates that never converge. With every-call-300-tokens and
# budget=6000, the natural exhaustion fires around K=9. 64 is far above
# any realistic K; tripping it indicates a bug, not a budget exhaustion.
_MAX_PASSES_PER_TICK = 64


# ============================================================================
# B2 prompt construction
# ============================================================================

_CRITIC_SYSTEM_PROMPT = textwrap.dedent("""
    You are reviewing a proposed action for an outbreak-control simulator.
    You will see:
      1. The current observation (regions, resources, restrictions, etc.).
      2. The action just proposed (one JSON object).

    Identify weaknesses: wrong region, wrong severity, missed cascade
    signal, depleted resources, blocked legal constraint, etc. Output
    PROSE (no JSON, no markdown). Be concise: bullet points listing
    concrete concerns. If the action is sound, say so in one line and
    stop.

    Do NOT propose a new action. The reviser will do that.
""").strip()

_REVISER_SYSTEM_PROMPT_HEADER = textwrap.dedent("""
    You are revising your earlier proposed action in light of a critique.

    You will see:
      1. The current observation.
      2. Your previous proposed action.
      3. The critique of that proposal.

    Emit ONE JSON action object — no markdown fences, no prose, no
    explanation. The action schema is unchanged from the initial pass:
""").strip()


def _build_critic_prompt() -> str:
    """Critic system prompt — no action schema (saves ~400 tokens vs
    re-including B1's full schema). The critic only emits prose."""
    return _CRITIC_SYSTEM_PROMPT


def _build_reviser_prompt() -> str:
    """Reviser system prompt = critic-orientation header + B1's full
    action schema (the reviser emits JSON, so it needs the schema)."""
    return _REVISER_SYSTEM_PROMPT_HEADER + "\n\n" + build_system_prompt()


def _action_to_json_summary(action: OuterActionPayload) -> str:
    """Serialize an action for the critic / reviser user message."""
    return action.model_dump_json()


# ============================================================================
# Budget helpers
# ============================================================================


def _estimate_call_cost(recent_call_tokens: List[int]) -> int:
    """Simple moving average over the last 3 LLM calls within this tick.

    Resets to the initial 600-token estimate at the start of each tick
    (when ``recent_call_tokens`` is empty). Stable but tracks the
    recent cost trajectory — if the model starts emitting longer
    responses (prompts grow as critique chain accumulates), the
    estimate adapts so the budget check stays honest.

    Per Session 8 §9.4: window = 3 calls, reset per tick. Documented
    here so future maintenance doesn't re-derive the choice.
    """
    if not recent_call_tokens:
        return _INITIAL_CALL_COST_ESTIMATE
    window = recent_call_tokens[-_ESTIMATE_WINDOW:]
    return int(sum(window) / len(window))


# ============================================================================
# Agent
# ============================================================================


class B2MatchedComputeAgent:
    """Matched-compute self-revision baseline.

    Args:
        env: Object exposing ``reset()`` /
            ``step(CrisisworldcortexAction) -> CrisisworldcortexObservation``.
            Production: a sync-wrapped ``CrisisworldcortexEnv`` HTTP
            client. Tests: an in-process adapter.
        llm: An ``LLMClient``-shaped object (``chat(caller_id,
            messages) -> ChatResponse``, ``tokens_used_for(caller_id)``,
            ``reset_counters(caller_id_prefix)``).
        tick_budget: Per-tick LLM-token cap. Defaults to ``TICK_BUDGET``
            (6000 — matches Cortex's design envelope). Override for
            Session 14 evaluation when Cortex's actual measured per-
            tick consumption is known.
    """

    CALLER_ID_PREFIX = "b2"
    TICK_BUDGET = _DEFAULT_TICK_BUDGET

    def __init__(self, env: Any, llm: Any, *, tick_budget: Optional[int] = None) -> None:
        self._env = env
        self._llm = llm
        self._tick_budget = tick_budget if tick_budget is not None else self.TICK_BUDGET
        self._initial_system_prompt = build_system_prompt()
        self._critic_system_prompt = _build_critic_prompt()
        self._reviser_system_prompt = _build_reviser_prompt()

    def run_episode(
        self,
        task: str,
        seed: int,
        max_ticks: int = 12,
        *,
        step_callback: Optional[StepCallback] = None,
    ) -> Dict[str, Any]:
        """Run one episode. Returns a trajectory dict.

        Trajectory shape:
            task, seed, steps_taken,
            rewards: List[float],
            action_history: List[dict] (one per tick — submitted_kind,
                parse_failure, pass_count, tick_tokens_used, raw_initial,
                raw_revisions),
            pass_counts: List[int] (revision passes per tick),
            tick_token_totals: List[int] (tokens consumed per tick),
            tokens_total: int,
            parse_failure_count: int.
        """
        self._llm.reset_counters(caller_id_prefix=f"{self.CALLER_ID_PREFIX}:")

        obs = self._env.reset()
        last_reward = 0.0

        rewards: List[float] = []
        action_history: List[Dict[str, Any]] = []
        pass_counts: List[int] = []
        tick_token_totals: List[int] = []
        parse_failure_count = 0
        steps_taken = 0

        for tick in range(1, max_ticks + 1):
            steps_taken = tick

            decision = self._decide_action_for_tick(tick=tick, obs=obs, last_reward=last_reward)
            payload = decision["payload"]
            tick_parse_failure = decision["parse_failure"]
            tick_pass_count = decision["pass_count"]
            tick_tokens_used = decision["tokens_used"]
            tick_error: Optional[ErrorKind] = decision["error"]

            if tick_parse_failure:
                parse_failure_count += 1

            obs = self._env.step(CrisisworldcortexAction(action=payload))
            last_reward = obs.reward if obs.reward is not None else 0.0
            rewards.append(last_reward)

            action_history.append(
                {
                    "tick": tick,
                    "submitted_kind": payload.kind,
                    "parse_failure": tick_parse_failure,
                    "pass_count": tick_pass_count,
                    "tick_tokens_used": tick_tokens_used,
                    "raw_initial": decision["raw_initial"],
                    "raw_revisions": decision["raw_revisions"],
                }
            )
            pass_counts.append(tick_pass_count)
            tick_token_totals.append(tick_tokens_used)

            if step_callback is not None:
                step_callback(
                    B1StepEvent(
                        tick=tick,
                        action=payload,
                        reward=last_reward,
                        done=bool(obs.done),
                        error=tick_error,
                        parse_failure=tick_parse_failure,
                        raw_llm=decision["raw_initial"],
                    )
                )

            if obs.done:
                break

        return {
            "task": task,
            "seed": seed,
            "steps_taken": steps_taken,
            "rewards": rewards,
            "action_history": action_history,
            "pass_counts": pass_counts,
            "tick_token_totals": tick_token_totals,
            "tokens_total": sum(tick_token_totals),
            "parse_failure_count": parse_failure_count,
        }

    # ------------------------------------------------------------------
    # Per-tick orchestration
    # ------------------------------------------------------------------

    def _decide_action_for_tick(self, *, tick: int, obs: Any, last_reward: float) -> Dict[str, Any]:
        """Run the initial+revision loop for one tick. Return the chosen
        action plus per-tick telemetry.

        Returns a dict with keys: payload, parse_failure (bool),
        pass_count (int), tokens_used (int), error (Optional[ErrorKind]),
        raw_initial (str), raw_revisions (List[str]).
        """
        observation_text = serialize_observation(obs, last_reward=last_reward)
        recent_call_tokens: List[int] = []
        tick_tokens_used = 0
        tick_error: Optional[ErrorKind] = None

        # ---------- Initial pass ----------
        initial_caller_id = f"{self.CALLER_ID_PREFIX}:t{tick}:p0:initial"
        raw_initial, initial_tokens, initial_error = self._safe_chat(
            caller_id=initial_caller_id,
            messages=[
                ChatMessage(role="system", content=self._initial_system_prompt),
                ChatMessage(role="user", content=observation_text),
            ],
        )
        if initial_error is not None:
            tick_error = initial_error
        recent_call_tokens.append(initial_tokens)
        tick_tokens_used += initial_tokens

        current_candidate: Optional[OuterActionPayload] = parse_action(raw_initial)
        if current_candidate is None and tick_error is None:
            tick_error = "parse_failure"

        raw_revisions: List[str] = []

        # ---------- Revision loop ----------
        pass_count = 0
        for pass_idx in range(1, _MAX_PASSES_PER_TICK + 1):
            remaining = self._tick_budget - tick_tokens_used
            estimated_pair = 2 * _estimate_call_cost(recent_call_tokens)
            if remaining < estimated_pair:
                break  # not enough budget for another (critique, revision) pair

            # Critique
            critique_caller_id = f"{self.CALLER_ID_PREFIX}:t{tick}:p{pass_idx}:critique"
            critique_user_text = (
                observation_text
                + "\n\n=== Proposed action ===\n"
                + (
                    _action_to_json_summary(current_candidate)
                    if current_candidate is not None
                    else "<no parseable proposal yet>"
                )
            )
            raw_critique, critique_tokens, critique_error = self._safe_chat(
                caller_id=critique_caller_id,
                messages=[
                    ChatMessage(role="system", content=self._critic_system_prompt),
                    ChatMessage(role="user", content=critique_user_text),
                ],
            )
            if critique_error is not None and tick_error is None:
                tick_error = critique_error
            recent_call_tokens.append(critique_tokens)
            tick_tokens_used += critique_tokens

            # Revision
            revision_caller_id = f"{self.CALLER_ID_PREFIX}:t{tick}:p{pass_idx}:revision"
            revision_user_text = (
                observation_text
                + "\n\n=== Previous proposal ===\n"
                + (
                    _action_to_json_summary(current_candidate)
                    if current_candidate is not None
                    else "<no parseable proposal yet>"
                )
                + "\n\n=== Critique ===\n"
                + raw_critique
            )
            raw_revision, revision_tokens, revision_error = self._safe_chat(
                caller_id=revision_caller_id,
                messages=[
                    ChatMessage(role="system", content=self._reviser_system_prompt),
                    ChatMessage(role="user", content=revision_user_text),
                ],
            )
            if revision_error is not None and tick_error is None:
                tick_error = revision_error
            recent_call_tokens.append(revision_tokens)
            tick_tokens_used += revision_tokens
            raw_revisions.append(raw_revision)

            new_candidate = parse_action(raw_revision)
            if new_candidate is not None:
                # Only update current_candidate when the revision parses
                # cleanly. Per design §20.1.1: never emit mid-revision drafts.
                current_candidate = new_candidate
                # If a prior pass had set tick_error="parse_failure" but a
                # later revision succeeded, we now have a parseable
                # candidate — clear the error so the final state reflects
                # the actual submitted action's source.
                if tick_error == "parse_failure":
                    tick_error = None

            pass_count = pass_idx

        # ---------- Submit ----------
        if current_candidate is None:
            # No pass produced a parseable action. Use the synthetic
            # V2-rejected marker so r_policy=0 lands on this tick.
            payload: OuterActionPayload = parse_failure_marker()
            tick_parse_failure = True
            self._log_parse_failure(tick=tick, raw=raw_initial)
            if tick_error is None:
                tick_error = "parse_failure"
        else:
            payload = current_candidate
            tick_parse_failure = False

        return {
            "payload": payload,
            "parse_failure": tick_parse_failure,
            "pass_count": pass_count,
            "tokens_used": tick_tokens_used,
            "error": tick_error,
            "raw_initial": raw_initial,
            "raw_revisions": raw_revisions,
        }

    # ------------------------------------------------------------------
    # Internal: LLM call + error handling
    # ------------------------------------------------------------------

    def _safe_chat(
        self, *, caller_id: str, messages: List[ChatMessage]
    ) -> tuple[str, int, Optional[ErrorKind]]:
        """Call llm.chat with try/except. Returns (content, tokens, error).

        Mirrors B1's parse-failure-as-rejection contract: on LLM call
        failure, return empty content, zero tokens, and
        ``error="llm_call_failed"``. The caller's parse step then trips
        and the synthetic marker (or a prior revision's candidate)
        flows through as the submitted action.
        """
        try:
            response = self._llm.chat(caller_id=caller_id, messages=messages)
            tokens = int(response.prompt_tokens) + int(response.completion_tokens)
            return response.content, tokens, None
        except Exception as exc:  # pragma: no cover - exercised manually
            print(
                f"[WARN] b2: llm.chat failed at caller={caller_id!r}: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
            return "", 0, "llm_call_failed"

    @staticmethod
    def _log_parse_failure(*, tick: int, raw: str) -> None:
        snippet = (raw or "").strip().replace("\n", " ")
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."
        print(
            f"[WARN] b2: parse_failure at tick={tick} raw={snippet!r}",
            file=sys.stderr,
            flush=True,
        )

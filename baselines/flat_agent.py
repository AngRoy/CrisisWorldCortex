# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
B1 - flat single-LLM-call baseline (design §20.1; baselines/CLAUDE.md).

One LLM call per tick. No conversation memory; the env's
``recent_action_log`` already gives the model an 8-deep history.
Production use: pass a ``CrisisworldcortexEnv`` HTTP client (per
baselines/CLAUDE.md, baselines never instantiate the env directly).
Tests pass an in-process adapter - see ``tests/test_baseline_b1.py``.

Parse-failure policy (Session 7a §6):
    Parse-failure-as-rejection. When the LLM emits unparseable text,
    B1 logs a [WARN] to stderr, then SUBMITS a synthetic
    ``PublicCommunication`` to the env. The env rejects it with
    ``accepted=False``, which lands as ``r_policy=0`` in
    ``outer_reward`` - making the reward signal punish parse failures
    appropriately. Episode does NOT terminate; the agent gets a chance
    to recover next tick. (The §19 parser-fail-terminate rule is for
    wire-protocol malformations; this is client-side text-extraction
    failure.)

    Note on the user's "comment in recent_action_log" wording: the
    forensic raw-text snippet is captured in B1's local trajectory
    log (returned from ``run_episode``), not on the env's
    ``ExecutedAction`` (which has no note field - adding one would
    touch the frozen wire-protocol class).
"""

from __future__ import annotations

import json
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional, Protocol

from pydantic import TypeAdapter, ValidationError

from cortex.llm_client import ChatMessage
from CrisisWorldCortex.models import (
    CrisisworldcortexAction,
    CrisisworldcortexObservation,
    OuterActionPayload,
    PublicCommunication,
)

__all__ = [
    "B1FlatAgent",
    "parse_action",
    "serialize_observation",
    "build_system_prompt",
]


# ============================================================================
# Constants
# ============================================================================

# Approx tokens-per-char for the prompt-size sanity check. ~4 chars/token
# is the long-standing rule of thumb for English under BPE tokenizers.
_CHARS_PER_TOKEN_ESTIMATE = 4
# Approved Session 7a: warn if rendered prompt exceeds 1500 tokens for
# a 4-region observation. If it does, the prompt format needs trimming.
_PROMPT_TOKEN_WARN_THRESHOLD = 1500


def _parse_failure_marker() -> PublicCommunication:
    """Synthetic V2-rejected action used to surface parse failures
    through the env's reward signal as r_policy=0."""
    return PublicCommunication(
        audience="general",
        message_class="informational",
        honesty=0.0,
    )


# ============================================================================
# Prompt construction
# ============================================================================

_SYSTEM_PROMPT_BODY = textwrap.dedent("""
    You are an agent operating one outbreak-control simulator. You receive
    an observation each tick and must respond with EXACTLY ONE JSON object —
    no markdown fences, no prose around it, just the JSON.

    == ACTION TYPES (kind + required fields) ==

    1. {"kind": "no_op"}
       Advance the tick without intervention.

    2. {"kind": "deploy_resource", "region": "<id>",
        "resource_type": "<type>", "quantity": <int>}
       Deploy units of a resource to a region.

    3. {"kind": "request_data", "region": "<id>",
        "data_type": "case_survey" | "hospital_audit" | "compliance_check"}
       Reduce telemetry noise for that region for a few ticks.

    4. {"kind": "restrict_movement", "region": "<id>",
        "severity": "none" | "light" | "moderate" | "strict"}
       Apply a movement restriction. "strict" may be blocked by a
       legal_constraints rule until escalate(national) has been invoked.

    5. {"kind": "escalate", "to_authority": "regional" | "national"}
       Escalate to a higher authority. Escalating to "national" unlocks
       any LegalConstraint with rule_id mentioning strict severity.

    6. {"kind": "reallocate_budget", "from_resource": "<type>",
        "to_resource": "<type>", "amount": <int>}
       Move resource units between types (small efficiency loss).

    == ENUM VALUES ==

    region:        whatever ids appear in the observation (e.g. R1, R2, ...)
    resource_type: test_kits, hospital_beds, mobile_units, vaccine_doses
    severity:      none, light, moderate, strict
    to_authority:  regional, national
    data_type:     case_survey, hospital_audit, compliance_check

    == OBSERVATION FIELDS ==

    Each tick you see per-region telemetry that is DELAYED by a few ticks
    and noisy: reported_cases_d_ago is what was happening some ticks ago,
    not now. hospital_load is current and operational. compliance_proxy
    is a noisy estimate of how well restrictions are being followed.
    Resources, active_restrictions, legal_constraints, and the recent
    action log are all reported as-is.

    == OUTPUT CONTRACT ==

    Respond with ONLY the JSON action object. No explanation, no
    surrounding text, no markdown.

    == STRATEGY ==

    Respond to the situation as it unfolds. Trade off across regions and
    resource types as needed.
""").strip()


def build_system_prompt() -> str:
    """Return the static system prompt. Single source of truth — both
    B1 and a future inference.py harness can import this."""
    return _SYSTEM_PROMPT_BODY


def serialize_observation(
    obs: CrisisworldcortexObservation,
    last_reward: float,
) -> str:
    """Render an observation as a compact text prompt body.

    Format markers: "Tick", "Resources", "Regions", "Active restrictions",
    "Legal constraints", "Recent actions".
    """
    parts: List[str] = []
    parts.append(
        f"Tick {obs.tick} | Ticks remaining: {obs.ticks_remaining} | Last reward: {last_reward:.2f}"
    )

    r = obs.resources
    parts.append(
        "=== Resources ===\n"
        f"test_kits={r.test_kits} hospital_beds_free={r.hospital_beds_free} "
        f"mobile_units={r.mobile_units} vaccine_doses={r.vaccine_doses}"
    )

    region_lines = ["=== Regions ==="]
    for region in obs.regions:
        region_lines.append(
            f"- {region.region}: cases_d_ago={region.reported_cases_d_ago} "
            f"hospital_load={region.hospital_load:.2f} "
            f"compliance_proxy={region.compliance_proxy:.2f}"
        )
    parts.append("\n".join(region_lines))

    restr_lines = ["=== Active restrictions ==="]
    if obs.active_restrictions:
        for restr in obs.active_restrictions:
            restr_lines.append(
                f"- {restr.region}: severity={restr.severity} "
                f"ticks_remaining={restr.ticks_remaining}"
            )
    else:
        restr_lines.append("(none)")
    parts.append("\n".join(restr_lines))

    legal_lines = ["=== Legal constraints ==="]
    if obs.legal_constraints:
        for lc in obs.legal_constraints:
            legal_lines.append(
                f"- {lc.rule_id}: blocks {lc.blocked_action} (unlock via {lc.unlock_via})"
            )
    else:
        legal_lines.append("(none)")
    parts.append("\n".join(legal_lines))

    log_lines = ["=== Recent actions (last 8) ==="]
    if obs.recent_action_log:
        for entry in obs.recent_action_log:
            kind = entry.action.kind
            extra = _action_summary(entry.action)
            log_lines.append(f"- tick={entry.tick} {kind}{extra} accepted={entry.accepted}")
    else:
        log_lines.append("(none yet)")
    parts.append("\n".join(log_lines))

    return "\n\n".join(parts)


def _action_summary(action: OuterActionPayload) -> str:
    """One-shot summary of an action's salient fields for the log."""
    kind = action.kind
    if kind == "deploy_resource":
        return f"({action.region}, {action.resource_type}, qty={action.quantity})"
    if kind == "request_data":
        return f"({action.region}, {action.data_type})"
    if kind == "restrict_movement":
        return f"({action.region}, {action.severity})"
    if kind == "escalate":
        return f"({action.to_authority})"
    if kind == "reallocate_budget":
        return f"({action.from_resource} -> {action.to_resource}, amount={action.amount})"
    return ""


# ============================================================================
# Parsing
# ============================================================================

_PAYLOAD_ADAPTER: TypeAdapter = TypeAdapter(OuterActionPayload)


def parse_action(raw_text: str) -> Optional[OuterActionPayload]:
    """Extract a typed ``OuterActionPayload`` from raw LLM text.

    Pipeline:
      1. Strip ```json ... ``` codeblock fences.
      2. ``json.loads`` directly on the stripped text.
      3. On JSON failure: brace-match (find the first balanced ``{...}``
         block in the text and try again).
      4. Validate via Pydantic ``TypeAdapter(OuterActionPayload)``.

    Returns ``None`` on any failure. Caller decides recovery.
    """
    if not raw_text or not raw_text.strip():
        return None

    text = raw_text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    data: Optional[Dict[str, Any]] = None
    try:
        candidate = json.loads(text)
        if isinstance(candidate, dict):
            data = candidate
    except json.JSONDecodeError:
        pass

    if data is None:
        start = text.find("{")
        if start == -1:
            return None
        depth, end = 0, -1
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            return None
        try:
            candidate = json.loads(text[start : end + 1])
            if isinstance(candidate, dict):
                data = candidate
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict) or "kind" not in data:
        return None

    try:
        return _PAYLOAD_ADAPTER.validate_python(data)
    except ValidationError:
        return None


# ============================================================================
# Env protocol — duck-typed minimal interface
# ============================================================================


class _EnvLike(Protocol):
    """Sync env interface B1 expects: ``reset()`` /
    ``step(action) -> CrisisworldcortexObservation``.

    Production callers wrap ``CrisisworldcortexEnv`` (HTTP client) so
    ``.step(action)`` returns an observation directly. Tests pass an
    in-process adapter (see ``_InProcessEnvAdapter`` in test_baseline_b1).
    """

    def reset(self) -> CrisisworldcortexObservation: ...
    def step(self, action: CrisisworldcortexAction) -> CrisisworldcortexObservation: ...


# ============================================================================
# Agent
# ============================================================================


class B1FlatAgent:
    """B1 baseline: one LLM call per tick, no conversation memory."""

    CALLER_ID_PREFIX = "b1"

    def __init__(self, env: _EnvLike, llm: Any) -> None:
        self._env = env
        self._llm = llm
        self._system_prompt = build_system_prompt()
        self._first_call_logged = False

    # TODO(session-8): when B2 lands, refactor run_episode to take a
    # step_callback parameter so inference.py and B2's matched-compute
    # tracer both consume per-tick output without duplicating loop control.
    # See inference.py:_run_episode for the parallel duplication.
    def run_episode(
        self,
        task: str,
        seed: int,
        max_ticks: int = 12,
    ) -> Dict[str, Any]:
        """Run one episode. Returns a trajectory dict.

        Side effects: calls ``self._llm.reset_counters(prefix='b1:')`` at
        the start so per-episode token counts don't accumulate across
        episodes. Per Session 7a §4: harness-driven reset, not auto.

        ``task`` and ``seed`` are recorded in the trajectory dict.
        Forwarding them to env.reset(...) is forward-compat with a
        future env that learns task selection at reset time.
        """
        self._llm.reset_counters(caller_id_prefix=f"{self.CALLER_ID_PREFIX}:")
        self._first_call_logged = False

        obs = self._env.reset()
        last_reward = 0.0

        rewards: List[float] = []
        action_history: List[Dict[str, Any]] = []
        parse_failure_count = 0
        steps_taken = 0

        for tick in range(1, max_ticks + 1):
            steps_taken = tick

            user_prompt = serialize_observation(obs, last_reward=last_reward)
            self._maybe_warn_prompt_size(self._system_prompt, user_prompt)

            messages = [
                ChatMessage(role="system", content=self._system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ]
            caller_id = f"{self.CALLER_ID_PREFIX}:t{tick}"
            response = self._llm.chat(caller_id=caller_id, messages=messages)

            payload = parse_action(response.content)
            if payload is None:
                parse_failure_count += 1
                snippet = (response.content or "").strip().replace("\n", " ")
                if len(snippet) > 80:
                    snippet = snippet[:77] + "..."
                print(
                    f"[WARN] b1: parse_failure at tick={tick} caller={caller_id!r} raw={snippet!r}",
                    file=sys.stderr,
                    flush=True,
                )
                payload = _parse_failure_marker()
                action_history.append(
                    {
                        "tick": tick,
                        "submitted_kind": payload.kind,
                        "parse_failure": True,
                        "raw_llm": response.content,
                    }
                )
            else:
                action_history.append(
                    {
                        "tick": tick,
                        "submitted_kind": payload.kind,
                        "parse_failure": False,
                        "raw_llm": response.content,
                    }
                )

            obs = self._env.step(CrisisworldcortexAction(action=payload))
            last_reward = obs.reward if obs.reward is not None else 0.0
            rewards.append(last_reward)

            if obs.done:
                break

        return {
            "task": task,
            "seed": seed,
            "steps_taken": steps_taken,
            "rewards": rewards,
            "action_history": action_history,
            "parse_failure_count": parse_failure_count,
            "tokens_total": sum(
                self._llm.tokens_used_for(f"{self.CALLER_ID_PREFIX}:t{i}")
                for i in range(1, steps_taken + 1)
            ),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_warn_prompt_size(self, system_prompt: str, user_prompt: str) -> None:
        """One-shot prompt-size check on the first LLM call of an episode."""
        if self._first_call_logged:
            return
        self._first_call_logged = True
        approx_tokens = (len(system_prompt) + len(user_prompt)) // _CHARS_PER_TOKEN_ESTIMATE
        if approx_tokens > _PROMPT_TOKEN_WARN_THRESHOLD:
            print(
                f"[WARN] b1: prompt approx_tokens={approx_tokens} exceeds "
                f"{_PROMPT_TOKEN_WARN_THRESHOLD} - consider trimming the format",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"[INFO] b1: prompt approx_tokens={approx_tokens}",
                file=sys.stderr,
                flush=True,
            )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Inference harness for CrisisWorldCortex (Session 7b).

Runs B1 (single-LLM-call-per-tick) against the env over the HTTP client,
emits the byte-for-byte stdout protocol the hackathon validator expects:
  [START] task=<task> env=<env> model=<model>
  [STEP] step=<N> action=<str> reward=<r:.2f> done=<true|false> error=<error|null>
  [END] success=<true|false> steps=<N> score=<s:.3f> rewards=<r1:.2f,r2:.2f,...>

Required env vars:
  HF_TOKEN          - HF Router / OpenAI API key. No default.
  LOCAL_IMAGE_NAME  - Docker image (Docker mode), OR
  ENV_URL           - HF Spaces URL (Spaces mode).
                      One of LOCAL_IMAGE_NAME / ENV_URL must be set.

Optional env vars:
  API_BASE_URL      - default https://router.huggingface.co/v1
  MODEL_NAME        - default Qwen/Qwen2.5-72B-Instruct

Task ladder (3 tasks, restored in Session 7c).
  - outbreak_easy   seed=0  max_ticks=12
  - outbreak_medium seed=1  max_ticks=12
  - outbreak_hard   seed=2  max_ticks=12
  CrisisworldcortexEnvironment.reset() now accepts task_name/seed/
  max_ticks kwargs; the framework's ResetRequest already supports
  arbitrary kwargs via extra="allow", so the wire path needs no
  schema changes.

Score formula (Session 7a §7 + 7b §9.4 revision): see compute_score.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, List, Optional

from baselines.flat_agent import (
    build_system_prompt,
    parse_action,
    serialize_observation,
)
from cortex.llm_client import ChatMessage, LLMClient
from CrisisWorldCortex.models import (
    CrisisworldcortexAction,
    OuterActionPayload,
    PublicCommunication,
)
from CrisisWorldCortex.server.graders import terminal_bonus
from CrisisWorldCortex.server.simulator import WorldState

# ============================================================================
# Constants
# ============================================================================

BENCHMARK = "CrisisWorldCortex"
SUCCESS_THRESHOLD = 0.5
DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# Three-task ladder restored in Session 7c (env.reset(task_name=...) is now
# wired through). Difficulty progression: easy -> medium -> hard, with
# distinct seeds per task for cross-episode reproducibility.
TASK_CONFIGS: List[dict] = [
    {"task_name": "outbreak_easy", "seed": 0, "max_ticks": 12},
    {"task_name": "outbreak_medium", "seed": 1, "max_ticks": 12},
    {"task_name": "outbreak_hard", "seed": 2, "max_ticks": 12},
]

# Score-clamp bounds keep .3f formatting strictly inside (0, 1) so the
# validator's distribution check never sees a "0.000"/"1.000" round-down.
SCORE_LOWER_CLAMP = 1e-3
SCORE_UPPER_CLAMP = 1.0 - 1e-3


# ============================================================================
# Step record + line formatters
# ============================================================================


@dataclass(frozen=True)
class StepRecord:
    """One per-tick log entry. Frozen so it can't be mutated mid-render."""

    step: int
    action_str: str
    reward: float
    done: bool
    error: Optional[str]


def _format_start_line(task_name: str, env_name: str, model_name: str) -> str:
    return f"[START] task={task_name} env={env_name} model={model_name}"


def _format_step_line(record: StepRecord) -> str:
    error_val = record.error if record.error else "null"
    done_val = str(record.done).lower()
    return (
        f"[STEP] step={record.step} action={record.action_str} "
        f"reward={record.reward:.2f} done={done_val} error={error_val}"
    )


def _format_end_line(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> str:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    return (
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}"
    )


# ============================================================================
# Action -> compact string (for [STEP] line)
# ============================================================================


def action_to_str(payload: OuterActionPayload) -> str:
    """One-token-ish summary keyed by ``kind``. Quantity/amount/honesty
    are intentionally dropped to keep the [STEP] line short — the
    validator only needs to see WHICH action ran, not its parameters."""
    kind = payload.kind
    if kind == "deploy_resource":
        return f"deploy_resource:{payload.region}:{payload.resource_type}"
    if kind == "request_data":
        return f"request_data:{payload.region}:{payload.data_type}"
    if kind == "restrict_movement":
        return f"restrict_movement:{payload.region}:{payload.severity}"
    if kind == "escalate":
        return f"escalate:{payload.to_authority}"
    if kind == "reallocate_budget":
        return f"reallocate_budget:{payload.from_resource}:{payload.to_resource}"
    # no_op, public_communication: just the kind.
    return kind


# ============================================================================
# Score
# ============================================================================


def compute_score(rewards: List[float], terminal_bonus_value: float) -> float:
    """Compute episode score per design §14.3.

    Linear rescale of natural [-0.20, 1.20] range to [0, 1] before clamping.
    Diverges from triagesieve_env's ``rewards[-1]`` formula because triagesieve
    uses a terminal-only reward while ours accumulates per-tick r_outer per
    design §15.

    HACKATHON VALIDATOR COMPATIBILITY: if the validator's distribution-fit
    pipeline expects ``rewards[-1]``-shaped scores, this divergence may
    surface as anomalous score distributions. Revisit if validator complains;
    fallback is ``clamp(rewards[-1] if rewards else 1e-3, 1e-3, 1-1e-3)``.

    Empty-rewards case returns the lower clamp (1e-3) — a coarse failure
    signal. Session 14 (eval) will refine "env-failed-to-reset" vs
    "agent-did-nothing" distinctions.
    """
    if not rewards:
        return SCORE_LOWER_CLAMP
    raw = sum(rewards) / len(rewards) + terminal_bonus_value
    rescaled = (raw + 0.20) / 1.40
    return min(max(rescaled, SCORE_LOWER_CLAMP), SCORE_UPPER_CLAMP)


# ============================================================================
# Pure-function formatter (test path)
# ============================================================================


def format_episode_trace(
    task_name: str,
    model_name: str,
    steps: List[StepRecord],
    final_state: WorldState,
) -> str:
    """Render the full ``[START] / [STEP]xN / [END]`` block as a string.

    Used by tests to validate format-string shape on synthetic traces.
    Production (``main()``) doesn't call this — it streams the line
    helpers directly so per-tick output flushes in real time. Both paths
    share ``_format_*_line`` so the string format can't drift.

    Reads ``terminal_bonus(final_state)`` to incorporate the design-§14.3
    bonus into the score. Production cannot read latent state and passes
    0.0 instead; this is the test-only path that has access to a real
    WorldState (constructed via ``load_task`` in tests).
    """
    rewards = [s.reward for s in steps]
    bonus = terminal_bonus(final_state)
    score = compute_score(rewards, terminal_bonus_value=bonus)
    success = score >= SUCCESS_THRESHOLD

    lines: List[str] = [
        _format_start_line(task_name, BENCHMARK, model_name),
    ]
    for record in steps:
        lines.append(_format_step_line(record))
    lines.append(
        _format_end_line(
            success=success,
            steps=len(steps),
            score=score,
            rewards=rewards,
        )
    )
    return "\n".join(lines)


# ============================================================================
# Env construction
# ============================================================================


_DOCKER_READY_TIMEOUT_S = 120.0


def _make_env_from_docker(image_name: str) -> Any:
    """Spin up Docker container, return a sync wrapper.

    Mirrors triagesieve_env's manual ``LocalDockerProvider`` pattern
    rather than ``EnvClient.from_docker_image`` because the convenience
    constructor's default 30s ``wait_for_ready`` is too tight on Windows
    Docker Desktop after a cold image build (Session 7c smoke timed out
    at 30s; first-start commonly takes 45–90s here). 120s gives ample
    headroom without papering over a real hang.

    The ``SyncEnvClient`` returned by ``.sync()`` owns a persistent
    background event loop and routes ``connect()`` / ``reset()`` /
    ``step()`` through it. We MUST call ``connect()`` on the sync
    wrapper (not on the async client via ``asyncio.run``), or the
    one-shot loop ``asyncio.run`` creates closes immediately and
    leaves the connection tied to a dead loop — subsequent ``reset()``
    on the persistent loop then raises ``Event loop is closed``.
    """
    from openenv.core.containers.runtime.providers import LocalDockerProvider

    from CrisisWorldCortex import CrisisworldcortexEnv

    provider = LocalDockerProvider()
    base_url = provider.start_container(image_name)
    provider.wait_for_ready(base_url, timeout_s=_DOCKER_READY_TIMEOUT_S)
    async_client = CrisisworldcortexEnv(base_url=base_url, provider=provider)
    sync_env = async_client.sync()
    sync_env.connect()
    return sync_env


def _make_env_from_spaces(base_url: str) -> Any:
    """Connect to an already-running env at ``base_url`` (HF Spaces or
    any reachable OpenEnv server). Returns a sync wrapper.

    The constructor is sync (just stores ``base_url``); only ``.reset()``
    / ``.step()`` need event-loop machinery, which ``.sync()`` provides.
    """
    from CrisisWorldCortex import CrisisworldcortexEnv

    return CrisisworldcortexEnv(base_url=base_url).sync()


# ============================================================================
# Episode loop
# ============================================================================


# TODO(session-8): when B2 lands, refactor B1FlatAgent.run_episode to take
# a step_callback parameter so inference.py and B2's matched-compute tracer
# both consume per-tick output without duplicating loop control. The
# duplication below is ~25 lines of orchestration; parser/serializer/prompt
# are imported helpers, not duplicated.
def _run_episode(
    env: Any,
    llm: LLMClient,
    task_name: str,
    seed: int,
    model_name: str,
    max_ticks: int,
) -> dict:
    """Run one episode, streaming [START]/[STEP] lines, emit [END] at exit.

    Mirrors B1FlatAgent.run_episode's loop body but prints each step in
    real time with flush=True so a hung container doesn't lose records.
    Threads ``task_name`` / ``seed`` / ``max_ticks`` through env.reset()
    over the wire (Session 7c).
    """
    print(_format_start_line(task_name, BENCHMARK, model_name), flush=True)

    # Per-task counter reset — harness-driven per Session 7a §4.
    llm.reset_counters(caller_id_prefix="inference:")

    rewards: List[float] = []
    parse_failure_count = 0
    steps_taken = 0
    error_str: Optional[str] = None

    try:
        # Session 7c: pass task selection over the wire. The framework
        # serializes these as ResetRequest extra fields and the server
        # forwards to CrisisworldcortexEnvironment.reset(**kwargs).
        result = env.reset(
            task_name=task_name,
            seed=seed,
            max_ticks=max_ticks,
        )
        obs = _extract_obs(result)
    except Exception as exc:  # pragma: no cover - exercised manually
        print(
            f"[ERROR] env.reset() failed: {exc!r}",
            file=sys.stderr,
            flush=True,
        )
        # Coarse failure signal: empty rewards -> lower-clamp score.
        score = compute_score([], terminal_bonus_value=0.0)
        print(_format_end_line(False, 0, score, []), flush=True)
        return {
            "task": task_name,
            "steps_taken": 0,
            "score": score,
            "success": False,
            "rewards": [],
            "parse_failure_count": 0,
        }

    last_reward = 0.0
    system_prompt = build_system_prompt()

    for tick in range(1, max_ticks + 1):
        steps_taken = tick
        error_str = None

        user_prompt = serialize_observation(obs, last_reward=last_reward)
        caller_id = f"inference:t{tick}"

        try:
            response = llm.chat(
                caller_id=caller_id,
                messages=[
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=user_prompt),
                ],
            )
            raw = response.content
        except Exception as exc:  # pragma: no cover - exercised manually
            print(
                f"[WARN] inference: llm.chat failed at tick={tick}: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
            raw = ""
            error_str = "llm_call_failed"

        payload = parse_action(raw)
        if payload is None:
            parse_failure_count += 1
            snippet = (raw or "").strip().replace("\n", " ")[:80]
            print(
                f"[WARN] inference: parse_failure at tick={tick} "
                f"caller={caller_id!r} raw={snippet!r}",
                file=sys.stderr,
                flush=True,
            )
            # Synthetic V2-rejected stand-in (matches B1 §6 contract):
            # env returns accepted=False -> r_policy=0 lands as reward signal.
            payload = PublicCommunication(
                audience="general",
                message_class="informational",
                honesty=0.0,
            )
            if error_str is None:
                error_str = "parse_failure"

        try:
            result = env.step(CrisisworldcortexAction(action=payload))
            obs = _extract_obs(result)
            reward = _extract_reward(result, obs)
            done = _extract_done(result, obs)
        except Exception as exc:  # pragma: no cover - exercised manually
            print(
                f"[ERROR] env.step() failed at tick={tick}: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
            reward = 0.0
            done = True
            error_str = error_str or "env_step_failed"

        rewards.append(reward)
        last_reward = reward

        record = StepRecord(
            step=tick,
            action_str=action_to_str(payload),
            reward=reward,
            done=bool(done),
            error=error_str,
        )
        print(_format_step_line(record), flush=True)

        if done:
            break

    # Harness can't read state.terminal over the wire — pass 0.0. The
    # trainer (Session 14, reward_shaping.py) composes the real bonus
    # from server-side state, not from this stdout score.
    score = compute_score(rewards, terminal_bonus_value=0.0)
    success = score >= SUCCESS_THRESHOLD
    print(
        _format_end_line(success=success, steps=steps_taken, score=score, rewards=rewards),
        flush=True,
    )

    tokens = sum(llm.tokens_used_for(f"inference:t{i}") for i in range(1, steps_taken + 1))
    return {
        "task": task_name,
        "steps_taken": steps_taken,
        "score": score,
        "success": success,
        "rewards": rewards,
        "parse_failure_count": parse_failure_count,
        "tokens": tokens,
    }


def _extract_obs(result: Any) -> Any:
    """Normalise both ``StepResult`` (HTTP client) and bare observation
    (in-process env) shapes to the observation."""
    return getattr(result, "observation", result)


def _extract_reward(result: Any, obs: Any) -> float:
    """Reward field can live on ``StepResult.reward`` or ``obs.reward``."""
    r = getattr(result, "reward", None)
    if r is None:
        r = getattr(obs, "reward", None)
    return float(r) if r is not None else 0.0


def _extract_done(result: Any, obs: Any) -> bool:
    d = getattr(result, "done", None)
    if d is None:
        d = getattr(obs, "done", False)
    return bool(d)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Entry point for ``uv run python inference.py`` and the validator."""
    api_base_url = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL)
    hf_token = os.getenv("HF_TOKEN")
    local_image_name = os.getenv("LOCAL_IMAGE_NAME")
    env_url = os.getenv("ENV_URL")

    if not hf_token:
        raise SystemExit("ERROR: HF_TOKEN environment variable is not set.")

    if not local_image_name and not env_url:
        raise SystemExit(
            "ERROR: must set either LOCAL_IMAGE_NAME (Docker) or ENV_URL "
            "(HF Spaces). No default URL — set explicitly."
        )

    if local_image_name and env_url:
        print(
            "[INFO] both LOCAL_IMAGE_NAME and ENV_URL set; preferring Docker.",
            flush=True,
        )

    llm = LLMClient(
        api_base_url=api_base_url,
        api_key=hf_token,
        model=model_name,
    )

    results = []
    for cfg in TASK_CONFIGS:
        if local_image_name:
            print(f"[INFO] Using Docker image: {local_image_name}", flush=True)
            env = _make_env_from_docker(local_image_name)
        else:
            print(f"[INFO] Using env URL: {env_url}", flush=True)
            env = _make_env_from_spaces(env_url)

        try:
            result = _run_episode(
                env=env,
                llm=llm,
                task_name=cfg["task_name"],
                seed=cfg["seed"],
                model_name=model_name,
                max_ticks=cfg["max_ticks"],
            )
            results.append(result)
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # pragma: no cover
                    print(
                        f"[WARN] env.close() failed: {exc!r}",
                        file=sys.stderr,
                        flush=True,
                    )

    print("", flush=True)
    n = len(results)
    print(
        f"=== RESULTS SUMMARY ({n} task{'s' if n != 1 else ''}) ===",
        flush=True,
    )
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(
            f"  {r['task']}: score={r['score']:.3f} steps={r['steps_taken']} [{status}]",
            flush=True,
        )


if __name__ == "__main__":
    main()

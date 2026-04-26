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

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from baselines.flat_agent import B1FlatAgent, B1StepEvent
from cortex.llm_client import LLMClient
from CrisisWorldCortex.models import OuterActionPayload
from CrisisWorldCortex.server.graders import terminal_bonus
from CrisisWorldCortex.server.simulator import WorldState

AgentKind = Literal["b1", "b2", "b3"]
_AGENT_CHOICES: tuple = ("b1", "b2", "b3")

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


def _sync_if_available(env: Any) -> Any:
    """OpenEnv 0.2.2+ exposes .sync(); 0.2.1 reset/step are already sync."""
    sync = getattr(env, "sync", None)
    return sync() if callable(sync) else env


def _make_env_from_docker(image_name: str) -> Any:
    """Spin up Docker container, return a sync wrapper.

    Mirrors triagesieve_env's manual ``LocalDockerProvider`` pattern
    rather than ``EnvClient.from_docker_image`` because the convenience
    constructor's default 30s ``wait_for_ready`` is too tight on Windows
    Docker Desktop after a cold image build (Session 7c smoke timed out
    at 30s; first-start commonly takes 45–90s here). 120s gives ample
    headroom without papering over a real hang.

    OpenEnv 0.2.2+ returns an async client with a ``.sync()`` adapter.
    OpenEnv 0.2.1 exposes synchronous ``reset()`` / ``step()`` directly.
    We still call ``connect()`` because both API shapes expose it.
    """
    from openenv.core.containers.runtime.providers import LocalDockerProvider

    from CrisisWorldCortex import CrisisworldcortexEnv

    provider = LocalDockerProvider()
    base_url = provider.start_container(image_name)
    provider.wait_for_ready(base_url, timeout_s=_DOCKER_READY_TIMEOUT_S)
    async_client = CrisisworldcortexEnv(base_url=base_url, provider=provider)
    sync_env = _sync_if_available(async_client)
    sync_env.connect()
    return sync_env


def _make_env_from_spaces(base_url: str) -> Any:
    """Connect to an already-running env at ``base_url`` (HF Spaces or
    any reachable OpenEnv server). Returns a sync wrapper.

    OpenEnv version differences are handled by ``_sync_if_available``.
    """
    from CrisisWorldCortex import CrisisworldcortexEnv

    return _sync_if_available(CrisisworldcortexEnv(base_url=base_url))


# ============================================================================
# Episode loop — delegates to the selected agent's run_episode(step_callback=...)
# ============================================================================


class _SyncEnvAdapter:
    """Bridges the HTTP/sync env client (returns ``StepResult``) to
    B1FlatAgent's expected env shape (``reset() -> obs``, ``step(action)
    -> obs``).

    Pre-binds task-selection kwargs for the wire-level reset call. After
    each operation, copies ``result.reward`` and ``result.done`` from the
    StepResult wrapper onto the observation, since B1's loop reads them
    off ``obs`` directly.
    """

    def __init__(self, env: Any, *, reset_kwargs: Dict[str, Any]) -> None:
        self._env = env
        self._reset_kwargs = dict(reset_kwargs)

    def reset(self) -> Any:
        result = self._env.reset(**self._reset_kwargs)
        return self._normalize(result)

    def step(self, action: Any) -> Any:
        result = self._env.step(action)
        return self._normalize(result)

    @staticmethod
    def _normalize(result: Any) -> Any:
        # Some shapes: StepResult{observation, reward, done} (HTTP client)
        # or a bare observation (in-process). Try .observation; fall back
        # to result itself.
        obs = getattr(result, "observation", result)
        wrapper_reward = getattr(result, "reward", None)
        if wrapper_reward is not None:
            obs.reward = float(wrapper_reward)
        wrapper_done = getattr(result, "done", None)
        if wrapper_done:
            obs.done = True
        return obs


def _make_agent(kind: str, env: Any, llm: Any) -> Any:
    """Construct the B1/B2/B3 agent for ``kind``.

    All three agents share the ``(env, llm)`` constructor signature and
    expose ``run_episode(task, seed, max_ticks, *, step_callback)`` per
    Phase A Decision 54. Lazy imports for B2/B3 keep the cold-start cost
    of the default B1 path unchanged.
    """
    if kind == "b1":
        return B1FlatAgent(env=env, llm=llm)
    if kind == "b2":
        from baselines.flat_agent_matched_compute import B2MatchedComputeAgent

        return B2MatchedComputeAgent(env=env, llm=llm)
    if kind == "b3":
        from baselines.cortex_fixed_router import B3CortexFixedRouter

        return B3CortexFixedRouter(env=env, llm=llm)
    raise ValueError(f"unknown agent kind: {kind!r}; expected one of {_AGENT_CHOICES}")


def _build_argparser() -> argparse.ArgumentParser:
    """Argparse for inference.py CLI flags. Default --agent=b1 keeps the
    pre-Session-13 invocation working for the existing eval suite."""
    parser = argparse.ArgumentParser(
        prog="inference",
        description="CrisisWorldCortex inference harness (B1/B2/B3 dispatch).",
    )
    parser.add_argument(
        "--agent",
        choices=_AGENT_CHOICES,
        default="b1",
        help="Agent to run: b1 (flat), b2 (matched-compute), b3 (cortex+deterministic-router).",
    )
    return parser


def _run_episode(
    env: Any,
    llm: LLMClient,
    task_name: str,
    seed: int,
    model_name: str,
    max_ticks: int,
    agent_kind: str = "b1",
) -> dict:
    """Stream one episode end-to-end via ``<Agent>.run_episode``.

    The agent owns the per-tick LLM-call + parse + env.step loop; this
    harness owns the [START] / [STEP] / [END] stdout protocol via a
    callback. Net effect of the Session 8 refactor: ~80 LOC drop here.
    """
    print(_format_start_line(task_name, BENCHMARK, model_name), flush=True)

    rewards: List[float] = []
    parse_failure_count = 0

    def step_cb(ev: B1StepEvent) -> None:
        nonlocal parse_failure_count
        rewards.append(ev.reward)
        if ev.parse_failure:
            parse_failure_count += 1
        print(
            _format_step_line(
                StepRecord(
                    step=ev.tick,
                    action_str=action_to_str(ev.action),
                    reward=ev.reward,
                    done=ev.done,
                    error=ev.error,
                )
            ),
            flush=True,
        )

    adapter = _SyncEnvAdapter(
        env,
        reset_kwargs={"task_name": task_name, "seed": seed, "max_ticks": max_ticks},
    )
    agent = _make_agent(agent_kind, adapter, llm)

    try:
        traj = agent.run_episode(
            task=task_name,
            seed=seed,
            max_ticks=max_ticks,
            step_callback=step_cb,
        )
    except Exception as exc:  # pragma: no cover - exercised manually
        print(f"[ERROR] episode failed: {exc!r}", file=sys.stderr, flush=True)
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

    # Harness can't read state.terminal over the wire — pass 0.0. The
    # trainer (Session 14, reward_shaping.py) composes the real bonus
    # from server-side state, not from this stdout score.
    score = compute_score(rewards, terminal_bonus_value=0.0)
    success = score >= SUCCESS_THRESHOLD
    print(
        _format_end_line(
            success=success,
            steps=traj["steps_taken"],
            score=score,
            rewards=rewards,
        ),
        flush=True,
    )

    return {
        "task": task_name,
        "steps_taken": traj["steps_taken"],
        "score": score,
        "success": success,
        "rewards": rewards,
        "parse_failure_count": parse_failure_count,
        "tokens": traj.get("tokens_total", 0),
    }


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Entry point for ``uv run python inference.py`` and the validator."""
    args = _build_argparser().parse_args()
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
                agent_kind=args.agent,
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

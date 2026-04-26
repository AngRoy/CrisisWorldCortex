"""Collect SFT training trajectories for warm-starting B1 / Cortex policies.

Workstream B Phase 5c. Drives the deployed CrisisWorldCortex env with a
frontier teacher (default: Qwen 72B Instruct via HF Router/Novita) for
NUM_EPISODES * |TASKS| episodes, captures (observation, action) pairs
where the action parsed cleanly AND obs.reward >= MIN_REWARD_THRESHOLD
AND the env recorded accepted=True. Pushes the kept rows to an HF
dataset with train/eval splits.

The output dataset feeds Phase 5d (sft_warmstart.py), which teaches the
JSON action schema to a base model before GRPO refines strategy.

Local dry-run (no compute spend, just tests env reachability):
    DRY_RUN=1 uv run python training/scripts/collect_sft_data.py

Live run (~$1-2 HF Router credits, ~30-45 min):
    HF_TOKEN=hf_xxx OUTPUT_DATASET_REPO=Angshuman28/crisisworld-sft-trajectories \\
        uv run python training/scripts/collect_sft_data.py
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

# ============================================================================
# Configuration (env-var driven)
# ============================================================================


def _env(name: str, default: Optional[str] = None, *, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise SystemExit(f"[FATAL] env var {name} is required but unset")
    return value or ""


HF_TOKEN = _env("HF_TOKEN", required=True)
ENV_URL = _env("ENV_URL", "https://angshuman28-crisisworldcortex.hf.space")
MODEL_NAME = _env("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct:novita")
NUM_EPISODES = int(_env("NUM_EPISODES", "50"))
TASKS_CSV = _env("TASKS_CSV", "outbreak_easy,outbreak_medium,outbreak_hard")
MIN_REWARD_THRESHOLD = float(_env("MIN_REWARD_THRESHOLD", "0.5"))
OUTPUT_DATASET_REPO = _env("OUTPUT_DATASET_REPO", "Angshuman28/crisisworld-sft-trajectories")
EVAL_FRACTION = float(_env("EVAL_FRACTION", "0.2"))
EPISODE_TICKS = int(_env("EPISODE_TICKS", "12"))
SEED = int(_env("SEED", "42"))
DRY_RUN = _env("DRY_RUN", "0") not in ("0", "", "false", "False")
RUN_ID = _env("RUN_ID", "")  # optional suffix for OUTPUT_DATASET_REPO collision avoidance
MAX_COMPLETION_TOKENS = int(_env("MAX_COMPLETION_TOKENS", "256"))


def log(*args: object) -> None:
    print("[collect-sft]", *args, flush=True)


# ============================================================================
# Inlined helpers (duplicated from baselines/flat_agent.py per import-graph
# rule M-FR-13: training/* MUST NOT import baselines/*).
# Keep in sync with baselines/flat_agent.py if the helpers change.
# ============================================================================

_SYSTEM_PROMPT_BODY = textwrap.dedent(
    """
    You are an agent operating one outbreak-control simulator. You receive
    an observation each tick and must respond with EXACTLY ONE JSON object —
    no markdown fences, no prose around it, just the JSON.

    == ACTION TYPES (kind + required fields) ==

    1. {"kind": "no_op"}
    2. {"kind": "deploy_resource", "region": "<id>", "resource_type": "<type>", "quantity": <int>}
    3. {"kind": "request_data", "region": "<id>", "data_type": "case_survey" | "hospital_audit" | "compliance_check"}
    4. {"kind": "restrict_movement", "region": "<id>", "severity": "none" | "light" | "moderate" | "strict"}
    5. {"kind": "escalate", "to_authority": "regional" | "national"}
    6. {"kind": "reallocate_budget", "from_resource": "<type>", "to_resource": "<type>", "amount": <int>}

    Respond with ONLY the JSON action object. No explanation, no surrounding
    text, no markdown.
    """
).strip()


def _action_summary(action: Any) -> str:
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


def serialize_observation(obs: Any, last_reward: float = 0.0) -> str:
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


def parse_action_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON action dict from raw LLM output. Returns None on failure.

    We parse to dict here (not OuterActionPayload) because this script
    needs to capture the raw JSON string for the SFT dataset. The env
    will Pydantic-validate when we submit it.
    """
    if not raw_text or not raw_text.strip():
        return None
    text = raw_text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()
    try:
        candidate = json.loads(text)
        if isinstance(candidate, dict) and "kind" in candidate:
            return candidate
    except json.JSONDecodeError:
        pass
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
        if isinstance(candidate, dict) and "kind" in candidate:
            return candidate
    except json.JSONDecodeError:
        return None
    return None


# ============================================================================
# Pre-flight
# ============================================================================


def preflight_env_health(env_url: str) -> None:
    """Hit /health on the deployed env. Abort if not 200/healthy."""
    log(f"preflight: checking {env_url}/health")
    import urllib.request

    try:
        with urllib.request.urlopen(f"{env_url}/health", timeout=10) as resp:
            body = resp.read().decode("utf-8")
            if resp.status != 200 or "healthy" not in body.lower():
                raise SystemExit(
                    f"[FATAL] env {env_url} unhealthy: status={resp.status} body={body!r}. "
                    f"Run `openenv push` to rebuild the Space first."
                )
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(
            f"[FATAL] env {env_url} unreachable: {exc}. "
            f"Run `openenv push` to rebuild the Space first."
        ) from exc
    log("preflight: env healthy")


def _sync_if_available(env: Any) -> Any:
    """OpenEnv 0.2.2+ exposes .sync(); 0.2.1 reset/step are already sync."""
    sync = getattr(env, "sync", None)
    return sync() if callable(sync) else env


# ============================================================================
# Main
# ============================================================================


def call_teacher(client: Any, system_prompt: str, user_prompt: str) -> str:
    """One chat-completion round-trip via HF Router. Returns the completion text."""
    response = client.chat_completion(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=MAX_COMPLETION_TOKENS,
        temperature=0.1,  # low for SFT data quality
    )
    # huggingface_hub returns OpenAI-shaped objects.
    return response.choices[0].message.content or ""


def collect() -> int:
    log(f"MODEL_NAME={MODEL_NAME}")
    log(f"ENV_URL={ENV_URL}")
    log(f"NUM_EPISODES={NUM_EPISODES} TASKS={TASKS_CSV}")
    log(f"MIN_REWARD_THRESHOLD={MIN_REWARD_THRESHOLD} EVAL_FRACTION={EVAL_FRACTION}")
    log(f"OUTPUT_DATASET_REPO={OUTPUT_DATASET_REPO}{('-' + RUN_ID) if RUN_ID else ''}")

    preflight_env_health(ENV_URL)

    if DRY_RUN:
        log("DRY_RUN=1 — preflight only; not collecting or pushing")
        return 0

    # Lazy imports — keep the dry-run path fast.
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi, InferenceClient

    from CrisisWorldCortex.client import CrisisworldcortexEnv
    from CrisisWorldCortex.models import CrisisworldcortexAction

    tasks = tuple(t.strip() for t in TASKS_CSV.split(",") if t.strip())
    client = InferenceClient(token=HF_TOKEN)

    rows: List[Dict[str, Any]] = []
    parse_fail_count = 0
    rejected_count = 0
    low_reward_count = 0
    kept_count = 0

    for task in tasks:
        for ep in range(NUM_EPISODES):
            env = _sync_if_available(CrisisworldcortexEnv(base_url=ENV_URL))
            try:
                reset_result = env.reset(task_name=task, seed=ep, max_ticks=EPISODE_TICKS)
                obs = (
                    reset_result.observation
                    if hasattr(reset_result, "observation")
                    else reset_result
                )
            except Exception as exc:
                log(f"WARN env.reset failed task={task} ep={ep}: {exc}")
                env.close()
                continue
            try:
                last_reward = 0.0
                for tick in range(EPISODE_TICKS):
                    user_prompt = serialize_observation(obs, last_reward)
                    try:
                        completion = call_teacher(client, _SYSTEM_PROMPT_BODY, user_prompt)
                    except Exception as exc:
                        log(f"WARN teacher call failed task={task} ep={ep} tick={tick}: {exc}")
                        break
                    action_dict = parse_action_json(completion)
                    if action_dict is None:
                        parse_fail_count += 1
                        break
                    # Submit to env (Pydantic validates here).
                    try:
                        result = env.step(
                            CrisisworldcortexAction.model_validate({"action": action_dict})
                        )
                    except Exception as exc:
                        log(f"WARN env.step rejected task={task} ep={ep} tick={tick}: {exc}")
                        parse_fail_count += 1
                        break
                    next_obs = result.observation if hasattr(result, "observation") else result
                    reward = next_obs.reward if next_obs.reward is not None else 0.0
                    accepted = bool(
                        next_obs.recent_action_log and next_obs.recent_action_log[-1].accepted
                    )
                    if not accepted:
                        rejected_count += 1
                    elif reward < MIN_REWARD_THRESHOLD:
                        low_reward_count += 1
                    else:
                        rows.append(
                            {
                                "prompt": user_prompt,
                                "completion": json.dumps(action_dict, separators=(",", ":")),
                                "task": task,
                                "seed": ep,
                                "tick": tick,
                                "reward": float(reward),
                                "accepted": True,
                            }
                        )
                        kept_count += 1
                    last_reward = float(reward)
                    obs = next_obs
                    if next_obs.done:
                        break
            finally:
                env.close()
            if (ep + 1) % 5 == 0:
                log(
                    f"[{task}] {ep + 1}/{NUM_EPISODES} kept={kept_count} "
                    f"rejected={rejected_count} low_reward={low_reward_count} "
                    f"parse_fail={parse_fail_count}"
                )

    log(
        f"final tally: kept={kept_count} rejected={rejected_count} "
        f"low_reward={low_reward_count} parse_fail={parse_fail_count}"
    )

    if kept_count < 50:
        raise SystemExit(
            f"[FATAL] only {kept_count} kept rows; need >=50 for usable SFT. "
            f"Check teacher MODEL_NAME, MIN_REWARD_THRESHOLD, or env reachability."
        )

    # Per-task counts surface task balance.
    by_task: Dict[str, int] = {}
    for row in rows:
        by_task[row["task"]] = by_task.get(row["task"], 0) + 1
    for t, count in sorted(by_task.items()):
        log(f"  {t}: {count} rows ({count / kept_count:.0%})")
        if count < kept_count * 0.05:
            log(f"  WARN: {t} is <5% of dataset — task balance skewed")

    # Shuffle + split.
    rng = random.Random(SEED)
    rng.shuffle(rows)
    eval_size = max(1, int(len(rows) * EVAL_FRACTION))
    eval_rows = rows[:eval_size]
    train_rows = rows[eval_size:]
    log(f"split: train={len(train_rows)} eval={len(eval_rows)}")

    target_repo = f"{OUTPUT_DATASET_REPO}-{RUN_ID}" if RUN_ID else OUTPUT_DATASET_REPO

    def _to_ds(rs: List[Dict[str, Any]]) -> Any:
        return Dataset.from_list(rs)

    dsdict = DatasetDict({"train": _to_ds(train_rows), "eval": _to_ds(eval_rows)})
    api = HfApi()
    api.create_repo(target_repo, exist_ok=True, repo_type="dataset", private=False, token=HF_TOKEN)
    dsdict.push_to_hub(target_repo, token=HF_TOKEN)
    log(f"pushed https://huggingface.co/datasets/{target_repo}")
    return 0


def main() -> int:
    t0 = time.time()
    rc = collect()
    log(f"done in {time.time() - t0:.1f}s")
    return rc


if __name__ == "__main__":
    sys.exit(main())

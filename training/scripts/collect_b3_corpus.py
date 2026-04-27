"""Collect deterministic Cortex brain-choice corpus for router warmstart.

Workstream B Phase 6a. This script drives the deployed env with a
deterministic specialist-brain selector and a frozen teacher brain call,
then pushes rows shaped for router SFT:

    observation_text, deterministic_brain_choice, brain_action_json,
    env_reward, accepted, task, seed, tick

Why not import ``baselines.cortex_fixed_router.B3CortexFixedRouter``?
The repo's import-graph rule forbids ``training/*`` importing
``baselines/*``. Also, B3's existing deterministic router emits Council
metacognition actions, while this Workstream-B router is explicitly a
brain selector: {"brain": "epi" | "logistics" | "governance"}. The
heuristic below is the deterministic selector policy we distill in
Phase 6b before GRPO refines it.

Local preflight:
    DRY_RUN=1 uv run python training/scripts/collect_b3_corpus.py

Live run:
    HF_TOKEN=hf_xxx OUTPUT_DATASET_REPO=Angshuman28/crisisworld-b3-corpus \\
        uv run python training/scripts/collect_b3_corpus.py
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


def _env(name: str, default: Optional[str] = None, *, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise SystemExit(f"[FATAL] env var {name} is required but unset")
    return value or ""


DRY_RUN = _env("DRY_RUN", "0") not in ("0", "", "false", "False")
HF_TOKEN = _env("HF_TOKEN", required=not DRY_RUN)
ENV_URL = _env("ENV_URL", "https://angshuman28-crisisworldcortex.hf.space")
BRAIN_MODEL = _env("BRAIN_MODEL", "Qwen/Qwen2.5-72B-Instruct:novita")
OUTPUT_DATASET_REPO = _env("OUTPUT_DATASET_REPO", "Angshuman28/crisisworld-b3-corpus")
TASKS_CSV = _env("TASKS_CSV", "outbreak_easy,outbreak_medium,outbreak_hard")
NUM_EPISODES = int(_env("NUM_EPISODES", "5"))
EPISODE_TICKS = int(_env("EPISODE_TICKS", "12"))
SEED = int(_env("SEED", "42"))
MAX_COMPLETION_TOKENS = int(_env("MAX_COMPLETION_TOKENS", "192"))
MIN_ROWS = int(_env("MIN_ROWS", "150"))
RUN_ID = _env("RUN_ID", "")


def log(*args: object) -> None:
    print("[collect-b3-corpus]", *args, flush=True)


def preflight_env_health(env_url: str) -> None:
    import urllib.request

    log(f"preflight: checking {env_url}/health")
    with urllib.request.urlopen(f"{env_url}/health", timeout=10) as resp:
        body = resp.read().decode("utf-8")
        if resp.status != 200 or "healthy" not in body.lower():
            raise SystemExit(f"[FATAL] env unhealthy: status={resp.status} body={body!r}")
    log("preflight: env healthy")


def _sync_if_available(env: Any) -> Any:
    sync = getattr(env, "sync", None)
    return sync() if callable(sync) else env


def make_env() -> Any:
    from CrisisWorldCortex.client import CrisisworldcortexEnv

    return _sync_if_available(CrisisworldcortexEnv(base_url=ENV_URL))


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
    parts = [
        f"Tick {obs.tick} | Ticks remaining: {obs.ticks_remaining} | Last reward: {last_reward:.2f}",
        (
            "=== Resources ===\n"
            f"test_kits={obs.resources.test_kits} "
            f"hospital_beds_free={obs.resources.hospital_beds_free} "
            f"mobile_units={obs.resources.mobile_units} "
            f"vaccine_doses={obs.resources.vaccine_doses}"
        ),
    ]
    region_lines = ["=== Regions ==="]
    for region in obs.regions:
        region_lines.append(
            f"- {region.region}: cases_d_ago={region.reported_cases_d_ago} "
            f"hospital_load={region.hospital_load:.2f} "
            f"compliance_proxy={region.compliance_proxy:.2f}"
        )
    parts.append("\n".join(region_lines))
    if obs.active_restrictions:
        parts.append(
            "=== Active restrictions ===\n"
            + "\n".join(
                f"- {r.region}: severity={r.severity} ticks_remaining={r.ticks_remaining}"
                for r in obs.active_restrictions
            )
        )
    if obs.legal_constraints:
        parts.append(
            "=== Legal constraints ===\n"
            + "\n".join(
                f"- {lc.rule_id}: blocks {lc.blocked_action} (unlock via {lc.unlock_via})"
                for lc in obs.legal_constraints
            )
        )
    if obs.recent_action_log:
        parts.append(
            "=== Recent actions ===\n"
            + "\n".join(
                f"- tick={e.tick} {e.action.kind}{_action_summary(e.action)} accepted={e.accepted}"
                for e in obs.recent_action_log[-8:]
            )
        )
    return "\n\n".join(parts)


def deterministic_brain_choice(obs: Any) -> str:
    """Deterministic specialist selector distilled into the router SFT corpus.

    Governance owns legality/compliance choices, logistics owns scarcity and
    hospital-load triage, and epidemiology owns surveillance/spread diagnosis.
    The thresholds are intentionally simple and stable so Phase 6b learns a
    clean JSON brain-choice prior before GRPO optimizes reward online.
    """
    if obs.legal_constraints:
        return "governance"
    if any(r.severity in ("moderate", "strict") for r in obs.active_restrictions):
        return "governance"
    if any(region.compliance_proxy < 0.35 for region in obs.regions):
        return "governance"
    if obs.resources.hospital_beds_free <= 10 or obs.resources.mobile_units <= 1:
        return "logistics"
    if any(region.hospital_load >= 0.75 for region in obs.regions):
        return "logistics"
    return "epi"


def brain_system_prompt(brain: str) -> str:
    focus = {
        "epi": "epidemiology: surveillance, case growth, testing, and outbreak spread.",
        "logistics": "logistics: scarce resources, beds, mobile units, and allocation.",
        "governance": "governance: legal constraints, compliance, escalation, and restrictions.",
    }[brain]
    return textwrap.dedent(
        f"""
        You are the {focus}
        Choose exactly one CrisisWorld action as JSON. No prose, no markdown.

        Allowed actions:
        1. {{"kind": "no_op"}}
        2. {{"kind": "deploy_resource", "region": "<id>", "resource_type": "<type>", "quantity": <int>}}
        3. {{"kind": "request_data", "region": "<id>", "data_type": "case_survey" | "hospital_audit" | "compliance_check"}}
        4. {{"kind": "restrict_movement", "region": "<id>", "severity": "none" | "light" | "moderate" | "strict"}}
        5. {{"kind": "escalate", "to_authority": "regional" | "national"}}
        6. {{"kind": "reallocate_budget", "from_resource": "<type>", "to_resource": "<type>", "amount": <int>}}
        """
    ).strip()


def parse_action_json(raw_text: str) -> Optional[Dict[str, Any]]:
    text = raw_text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text).strip()
    candidates = [text]
    start = text.find("{")
    if start >= 0:
        depth = 0
        for index, char in enumerate(text[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break
    for candidate_text in candidates:
        try:
            candidate = json.loads(candidate_text)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "kind" in candidate:
            return candidate
    return None


def call_brain(client: Any, brain: str, observation_text: str) -> str:
    response = client.chat_completion(
        model=BRAIN_MODEL,
        messages=[
            {"role": "system", "content": brain_system_prompt(brain)},
            {"role": "user", "content": observation_text},
        ],
        max_tokens=MAX_COMPLETION_TOKENS,
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def collect() -> int:
    preflight_env_health(ENV_URL)
    tasks = tuple(t.strip() for t in TASKS_CSV.split(",") if t.strip())
    env = make_env()
    try:
        result = env.reset(task_name=tasks[0], seed=SEED, max_ticks=EPISODE_TICKS)
        obs = result.observation if hasattr(result, "observation") else result
        log(f"preflight: reset ok tick={obs.tick} task={tasks[0]}")
    finally:
        env.close()

    if DRY_RUN:
        log("DRY_RUN=1 - preflight only; not collecting or pushing")
        return 0

    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi, InferenceClient

    from CrisisWorldCortex.models import CrisisworldcortexAction, NoOp

    client = InferenceClient(token=HF_TOKEN)
    rng = random.Random(SEED)
    rows: List[Dict[str, Any]] = []
    parse_failures = 0

    for task in tasks:
        for episode in range(NUM_EPISODES):
            env = make_env()
            try:
                result = env.reset(task_name=task, seed=episode, max_ticks=EPISODE_TICKS)
                obs = result.observation if hasattr(result, "observation") else result
                last_reward = 0.0
                for tick in range(EPISODE_TICKS):
                    observation_text = serialize_observation(obs, last_reward)
                    brain = deterministic_brain_choice(obs)
                    completion = call_brain(client, brain, observation_text)
                    action_dict = parse_action_json(completion)
                    action_parse_success = action_dict is not None
                    if action_dict is None:
                        parse_failures += 1
                        wire_action = CrisisworldcortexAction(action=NoOp())
                        action_json = ""
                    else:
                        action_json = json.dumps(action_dict, separators=(",", ":"))
                        wire_action = CrisisworldcortexAction.model_validate(
                            {"action": action_dict}
                        )
                    step_result = env.step(wire_action)
                    next_obs = (
                        step_result.observation
                        if hasattr(step_result, "observation")
                        else step_result
                    )
                    reward = next_obs.reward if next_obs.reward is not None else 0.0
                    accepted = bool(
                        next_obs.recent_action_log and next_obs.recent_action_log[-1].accepted
                    )
                    rows.append(
                        {
                            "observation_text": observation_text,
                            "deterministic_brain_choice": brain,
                            "brain_action_json": action_json,
                            "env_reward": float(reward),
                            "accepted": bool(accepted and action_parse_success),
                            "action_parse_success": action_parse_success,
                            "task": task,
                            "seed": episode,
                            "tick": tick,
                        }
                    )
                    obs = next_obs
                    last_reward = float(reward)
                    if next_obs.done:
                        break
            finally:
                env.close()
            log(f"{task} episode={episode + 1}/{NUM_EPISODES} rows={len(rows)}")

    if len(rows) < MIN_ROWS:
        raise SystemExit(f"[FATAL] only {len(rows)} rows collected; need >= {MIN_ROWS}")

    rng.shuffle(rows)
    eval_size = max(1, int(len(rows) * 0.2))
    dsdict = DatasetDict(
        {
            "train": Dataset.from_list(rows[eval_size:]),
            "eval": Dataset.from_list(rows[:eval_size]),
        }
    )
    target_repo = f"{OUTPUT_DATASET_REPO}-{RUN_ID}" if RUN_ID else OUTPUT_DATASET_REPO
    api = HfApi()
    api.create_repo(target_repo, exist_ok=True, repo_type="dataset", private=False, token=HF_TOKEN)
    dsdict.push_to_hub(target_repo, token=HF_TOKEN)
    log(f"pushed https://huggingface.co/datasets/{target_repo}")
    log(f"rows={len(rows)} parse_failures={parse_failures}")
    return 0


if __name__ == "__main__":
    t0 = time.time()
    rc = collect()
    log(f"done in {time.time() - t0:.1f}s")
    sys.exit(rc)

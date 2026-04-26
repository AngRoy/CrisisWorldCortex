"""B1 flat-agent GRPO trainer (HF Jobs runnable).

Workstream B Phase 5b. Mirrors notebooks/train_b1_grpo.ipynb's logic
in a pure-Python script that ``hf jobs run a100-large ...`` can invoke
without Jupyter overhead.

Configuration is entirely env-var driven so the same script trains:
  - B1-Qwen   (MODEL_NAME=unsloth/Qwen3-7B-Instruct-bnb-4bit)
  - B1-Llama  (MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct)
without source edits.

Two-stage SFT-then-GRPO pipeline (Phase 5e):
    BASE_MODEL env var optionally overrides MODEL_NAME at the
    FastLanguageModel.from_pretrained call. Set BASE_MODEL to a
    Phase-5d SFT-warmstarted checkpoint (e.g. Angshuman28/qwen3-7b-sft-warmstart)
    to start GRPO from a model that already knows the JSON action schema.
    Defaults to MODEL_NAME so single-stage cold-start GRPO still works.

Llama-3.1-8B is HF-gated — script pre-flight checks model accessibility
and aborts with a clear error if the user has not accepted the license.
Per Phase-A M-FR-3, no silent fallback (fail-loud preserves the
multi-model diversity claim).

Usage on HF Jobs:
    hf jobs run --hardware a100-large --secret HF_TOKEN \\
        --env MODEL_NAME=unsloth/Qwen3-7B-Instruct-bnb-4bit \\
        --env HUB_REPO_ID=Angshuman28/crisisworld-b1-grpo-qwen3-7b \\
        ghcr.io/astral-sh/uv:latest \\
        bash -c "git clone https://huggingface.co/spaces/Angshuman28/CrisisWorldCortex /app && \\
                 cd /app && uv sync && uv run python training/scripts/train_b1_grpo.py"

Local dry-run (skip pip install if already in venv):
    MODEL_NAME=unsloth/Qwen3-1.7B HUB_REPO_ID=local/test \\
        MAX_TRAIN_STEPS=5 uv run python training/scripts/train_b1_grpo.py
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import textwrap
from typing import Any, Dict, Optional


def _env(name: str, default: Optional[str] = None, *, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise SystemExit(f"[FATAL] env var {name} is required but unset")
    return value or ""


# ============================================================================
# Configuration (env-var driven)
# ============================================================================

MODEL_NAME = _env("MODEL_NAME", "unsloth/Qwen3-7B-Instruct-bnb-4bit")
# BASE_MODEL: optional override pointing to an SFT-warmstarted checkpoint
# (e.g. "Angshuman28/qwen3-7b-sft-warmstart"). Falls back to MODEL_NAME so
# existing single-stage GRPO usage doesn't break. Per Phase-5e M-FR-25:
# MODEL_NAME stays the canonical model-family identifier (Qwen vs Llama)
# for telemetry/HF-Hub-commit purposes; BASE_MODEL is the actual checkpoint
# loaded into FastLanguageModel. They diverge after SFT warm-start.
BASE_MODEL = _env("BASE_MODEL", MODEL_NAME)
HF_TOKEN = _env("HF_TOKEN", required=True)
HUB_REPO_ID = _env("HUB_REPO_ID", required=True)
ENV_URL = _env("ENV_URL", "https://angshuman28-crisisworldcortex.hf.space")
OUTPUT_DIR = _env("OUTPUT_DIR", "/tmp/b1_grpo_lora")
TASKS_CSV = _env("TASKS_CSV", "outbreak_easy,outbreak_medium,outbreak_hard")
EPISODE_TICKS = int(_env("EPISODE_TICKS", "12"))
MAX_TRAIN_STEPS = int(_env("MAX_TRAIN_STEPS", "300"))
GROUP_SIZE = int(_env("GROUP_SIZE", "4"))
MAX_PROMPT_LEN = int(_env("MAX_PROMPT_LEN", "2048"))
MAX_COMPLETION_LEN = int(_env("MAX_COMPLETION_LEN", "512"))
LR = float(_env("LR", "5e-6"))
TEMPERATURE = float(_env("TEMPERATURE", "0.8"))
SEED = int(_env("SEED", "42"))
LORA_RANK = int(_env("LORA_RANK", "32"))
GPU_MEM_UTIL = float(_env("GPU_MEM_UTIL", "0.6"))
DATASET_SEEDS_PER_TASK = int(_env("DATASET_SEEDS_PER_TASK", "50"))


def log(*args: object) -> None:
    print("[b1-grpo]", *args, flush=True)


# ============================================================================
# Pre-flight: gated-model check (Llama-3.1-8B requires HF acceptance)
# ============================================================================


def preflight_model_access(model_name: str, token: str) -> None:
    """Raise SystemExit with a clear error if MODEL_NAME is gated and
    the user hasn't accepted the license.

    Per Phase-A M-FR-3: fail-loud, no silent fallback.
    """
    log(f"preflight: checking access to {model_name}")
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    try:
        info = HfApi().model_info(model_name, token=token)
        if getattr(info, "gated", False) and not getattr(info, "private", False):
            log(f"preflight: {model_name} is gated; access verified")
    except GatedRepoError as exc:
        raise SystemExit(
            f"[FATAL] {model_name} is gated and the provided HF_TOKEN does not have "
            f"access. Visit https://huggingface.co/{model_name} and accept the "
            f"license terms. Original error: {exc}"
        ) from exc
    except RepositoryNotFoundError as exc:
        raise SystemExit(f"[FATAL] {model_name} not found on HF Hub: {exc}") from exc
    log(f"preflight: {model_name} accessible")


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    log(f"MODEL_NAME={MODEL_NAME}")
    log(f"BASE_MODEL={BASE_MODEL}")
    log(f"HUB_REPO_ID={HUB_REPO_ID}")
    log(f"ENV_URL={ENV_URL}")
    log(f"MAX_TRAIN_STEPS={MAX_TRAIN_STEPS} GROUP_SIZE={GROUP_SIZE} LR={LR}")

    preflight_model_access(BASE_MODEL, HF_TOKEN)

    # Lazy imports — keeps preflight fast and avoids loading Unsloth/torch
    # on local machines that don't have GPU.
    from datasets import Dataset
    from pydantic import TypeAdapter, ValidationError
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel

    from CrisisWorldCortex.client import CrisisworldcortexEnv
    from CrisisWorldCortex.models import (
        CrisisworldcortexAction,
        CrisisworldcortexObservation,
        OuterActionPayload,
        PublicCommunication,
    )

    # ------------------------------------------------------------------
    # Inlined helpers — duplicated from baselines/flat_agent.py because
    # the import-graph rule (root CLAUDE.md) forbids training/* importing
    # baselines/*. Keep these in sync with the canonical helpers in
    # baselines/flat_agent.py if the prompt/parser/marker change.
    # ------------------------------------------------------------------

    _SYSTEM_PROMPT_BODY = textwrap.dedent(
        """
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

        Respond with ONLY the JSON action object. No explanation, no
        surrounding text, no markdown.
        """
    ).strip()

    def build_system_prompt() -> str:
        return _SYSTEM_PROMPT_BODY

    def _action_summary(action: OuterActionPayload) -> str:
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

    def serialize_observation(obs: CrisisworldcortexObservation, last_reward: float = 0.0) -> str:
        parts: list[str] = []
        parts.append(
            f"Tick {obs.tick} | Ticks remaining: {obs.ticks_remaining} | "
            f"Last reward: {last_reward:.2f}"
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

    _PAYLOAD_ADAPTER: TypeAdapter = TypeAdapter(OuterActionPayload)

    def parse_action(raw_text: str) -> Optional[OuterActionPayload]:
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

    def parse_failure_marker() -> PublicCommunication:
        return PublicCommunication(
            audience="general",
            message_class="informational",
            honesty=0.0,
        )

    # ---- Load model + LoRA ----
    log(f"loading model {BASE_MODEL} (LoRA rank={LORA_RANK})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_PROMPT_LEN + MAX_COMPLETION_LEN,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEM_UTIL,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_RANK * 2,
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    log("model + LoRA ready")

    # ---- Env client + dataset ----
    tasks = tuple(t.strip() for t in TASKS_CSV.split(",") if t.strip())
    log(f"tasks={tasks}")

    def make_env() -> Any:
        return CrisisworldcortexEnv(base_url=ENV_URL).sync()

    SYSTEM_PROMPT = build_system_prompt()

    def make_chat_prompt(obs: CrisisworldcortexObservation) -> str:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": serialize_observation(obs)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    log(f"building dataset ({DATASET_SEEDS_PER_TASK} seeds * {len(tasks)} tasks)")
    rng = random.Random(SEED)
    seed_pool = [{"task": t, "seed": s} for t in tasks for s in range(DATASET_SEEDS_PER_TASK)]
    rng.shuffle(seed_pool)

    prompts: list[str] = []
    meta: list[dict] = []
    for entry in seed_pool:
        env = make_env()
        try:
            result = env.reset(
                task_name=entry["task"],
                seed=entry["seed"],
                max_ticks=EPISODE_TICKS,
            )
            obs = result.observation if hasattr(result, "observation") else result
            prompts.append(make_chat_prompt(obs))
            meta.append(entry)
        finally:
            env.close()

    train_dataset = Dataset.from_dict(
        {
            "prompt": prompts,
            "task": [m["task"] for m in meta],
            "seed": [m["seed"] for m in meta],
        }
    )
    log(f"dataset: {len(train_dataset)} rows")

    # ---- Reward function (single-step GRPO) ----
    def crisisworld_reward(
        prompts: list[str],
        completions: list[str],
        task: list[str],
        seed: list[int],
        **_kwargs: object,
    ) -> list[float]:
        rewards: list[float] = []
        for completion, t, s in zip(completions, task, seed):
            env = make_env()
            try:
                env.reset(task_name=t, seed=int(s), max_ticks=EPISODE_TICKS)
                payload = parse_action(completion) or parse_failure_marker()
                try:
                    result = env.step(CrisisworldcortexAction(action=payload))
                    obs = result.observation if hasattr(result, "observation") else result
                    reward = obs.reward if obs.reward is not None else 0.0
                    rewards.append(float(reward))
                except Exception as exc:
                    log(f"WARN env.step failed task={t} seed={s}: {exc}")
                    rewards.append(-1.0)
            finally:
                env.close()
        return rewards

    # ---- GRPO training ----
    log("starting GRPOTrainer")
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=LR,
        per_device_train_batch_size=GROUP_SIZE,
        gradient_accumulation_steps=1,
        num_generations=GROUP_SIZE,
        max_prompt_length=MAX_PROMPT_LEN,
        max_completion_length=MAX_COMPLETION_LEN,
        max_steps=MAX_TRAIN_STEPS,
        save_steps=max(MAX_TRAIN_STEPS // 3, 1),
        logging_steps=max(MAX_TRAIN_STEPS // 60, 1),
        report_to="none",
        bf16=True,
        optim="adamw_8bit",
        temperature=TEMPERATURE,
        use_vllm=True,
        vllm_mode="colocate",
        seed=SEED,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[crisisworld_reward],
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()
    log("training done")

    # ---- Save + push ----
    log(f"saving LoRA adapter to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    log(f"pushing to https://huggingface.co/{HUB_REPO_ID}")
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(HUB_REPO_ID, exist_ok=True, repo_type="model", private=False, token=HF_TOKEN)
    api.upload_folder(
        folder_path=OUTPUT_DIR,
        repo_id=HUB_REPO_ID,
        repo_type="model",
        token=HF_TOKEN,
    )
    log("push complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())

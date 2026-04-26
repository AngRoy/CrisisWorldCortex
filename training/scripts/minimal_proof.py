# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "openenv-CrisisWorldCortex @ git+https://huggingface.co/spaces/Angshuman28/CrisisWorldCortex",
#   "accelerate>=1.13.0",
#   "peft>=0.19.0",
#   "torch>=2.11.0",
#   "transformers>=5.0",
#   "huggingface-hub>=1.0.0",
# ]
# ///
"""Minimal no-TRL proof that a model can receive an env-shaped policy update.

This is intentionally not a replacement for the full Workstream-B GRPO
pipeline. It is a small, dependency-light fallback for HF Jobs when TRL's
GRPOTrainer import path pulls mergekit into an unsatisfiable pydantic/openenv
resolver conflict.

The update is GRPO-like:
  1. reset the deployed env for one task/seed;
  2. sample GROUP_SIZE completions for the same observation prompt;
  3. score each completion by parsing it as an action and stepping the env;
  4. compute group-relative advantages;
  5. optimize completion log-probability weighted by those advantages.

HF Jobs:
    hf jobs uv run --hardware a10g-small --secret HF_TOKEN \\
        -e HUB_REPO_ID=Angshuman28/crisisworld-minimal-proof \\
        training/scripts/minimal_proof.py

Local preflight:
    DRY_RUN=1 HF_TOKEN=dummy uv run python training/scripts/minimal_proof.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time
from typing import Any, Dict, Optional


def _env(name: str, default: Optional[str] = None, *, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise SystemExit(f"[FATAL] env var {name} is required but unset")
    return value or ""


HF_TOKEN = _env("HF_TOKEN", required=True)
ENV_URL = _env("ENV_URL", "https://angshuman28-crisisworldcortex.hf.space")
MODEL_NAME = _env("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
HUB_REPO_ID = _env("HUB_REPO_ID", "")
OUTPUT_DIR = _env("OUTPUT_DIR", "/tmp/crisisworld_minimal_proof_lora")
TASK_NAME = _env("TASK_NAME", "outbreak_easy")
SEED = int(_env("SEED", "0"))
EPISODE_TICKS = int(_env("EPISODE_TICKS", "12"))
GROUP_SIZE = int(_env("GROUP_SIZE", "4"))
TRAIN_STEPS = int(_env("TRAIN_STEPS", "1"))
MAX_PROMPT_LEN = int(_env("MAX_PROMPT_LEN", "2048"))
MAX_NEW_TOKENS = int(_env("MAX_NEW_TOKENS", "128"))
LR = float(_env("LR", "5e-5"))
TEMPERATURE = float(_env("TEMPERATURE", "0.8"))
LORA_RANK = int(_env("LORA_RANK", "8"))
PUSH_TO_HUB = _env("PUSH_TO_HUB", "1") not in ("0", "", "false", "False")
DRY_RUN = _env("DRY_RUN", "0") not in ("0", "", "false", "False")


def log(*args: object) -> None:
    print("[minimal-proof]", *args, flush=True)


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an agent operating one outbreak-control simulator. You receive
    an observation each tick and must respond with EXACTLY ONE JSON object -
    no markdown fences, no prose around it, just the JSON.

    Allowed actions:
    1. {"kind": "no_op"}
    2. {"kind": "deploy_resource", "region": "<id>", "resource_type": "<type>", "quantity": <int>}
    3. {"kind": "request_data", "region": "<id>", "data_type": "case_survey" | "hospital_audit" | "compliance_check"}
    4. {"kind": "restrict_movement", "region": "<id>", "severity": "none" | "light" | "moderate" | "strict"}
    5. {"kind": "escalate", "to_authority": "regional" | "national"}
    6. {"kind": "reallocate_budget", "from_resource": "<type>", "to_resource": "<type>", "amount": <int>}
    """
).strip()


def preflight_env_health(env_url: str) -> None:
    import urllib.request

    log(f"preflight: checking {env_url}/health")
    with urllib.request.urlopen(f"{env_url}/health", timeout=10) as resp:
        body = resp.read().decode("utf-8")
        if resp.status != 200 or "healthy" not in body.lower():
            raise SystemExit(f"[FATAL] env unhealthy: status={resp.status} body={body!r}")
    log("preflight: env healthy")


def serialize_observation(obs: Any) -> str:
    parts = [
        f"Tick {obs.tick} | Ticks remaining: {obs.ticks_remaining}",
        (
            "Resources: "
            f"test_kits={obs.resources.test_kits} "
            f"hospital_beds_free={obs.resources.hospital_beds_free} "
            f"mobile_units={obs.resources.mobile_units} "
            f"vaccine_doses={obs.resources.vaccine_doses}"
        ),
    ]
    region_lines = ["Regions:"]
    for region in obs.regions:
        region_lines.append(
            f"- {region.region}: cases_d_ago={region.reported_cases_d_ago} "
            f"hospital_load={region.hospital_load:.2f} "
            f"compliance_proxy={region.compliance_proxy:.2f}"
        )
    parts.append("\n".join(region_lines))
    if obs.legal_constraints:
        parts.append(
            "Legal constraints:\n"
            + "\n".join(
                f"- {lc.rule_id}: blocks {lc.blocked_action}; unlock via {lc.unlock_via}"
                for lc in obs.legal_constraints
            )
        )
    return "\n\n".join(parts)


def extract_action_dict(raw_text: str) -> Optional[Dict[str, Any]]:
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


def parse_action(raw_text: str) -> Any:
    from pydantic import TypeAdapter, ValidationError

    from CrisisWorldCortex.models import OuterActionPayload

    data = extract_action_dict(raw_text)
    if data is None:
        return None
    try:
        return TypeAdapter(OuterActionPayload).validate_python(data)
    except ValidationError:
        return None


def _sync_if_available(env: Any) -> Any:
    """OpenEnv 0.2.2+ exposes .sync(); 0.2.1 reset/step are already sync."""
    sync = getattr(env, "sync", None)
    return sync() if callable(sync) else env


def make_env() -> Any:
    from CrisisWorldCortex.client import CrisisworldcortexEnv

    return _sync_if_available(CrisisworldcortexEnv(base_url=ENV_URL))


def reset_observation() -> Any:
    env = make_env()
    try:
        result = env.reset(task_name=TASK_NAME, seed=SEED, max_ticks=EPISODE_TICKS)
        return result.observation if hasattr(result, "observation") else result
    finally:
        env.close()


def score_completion(completion: str) -> float:
    from CrisisWorldCortex.models import CrisisworldcortexAction

    payload = parse_action(completion)
    if payload is None:
        return -1.0
    env = make_env()
    try:
        env.reset(task_name=TASK_NAME, seed=SEED, max_ticks=EPISODE_TICKS)
        result = env.step(CrisisworldcortexAction(action=payload))
        obs = result.observation if hasattr(result, "observation") else result
        reward = obs.reward if obs.reward is not None else 0.0
        return float(reward)
    except Exception as exc:
        log(f"WARN completion rejected: {exc}")
        return -1.0
    finally:
        env.close()


def build_prompt(tokenizer: Any, obs: Any) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": serialize_observation(obs)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> int:
    log(f"MODEL_NAME={MODEL_NAME}")
    log(f"ENV_URL={ENV_URL}")
    log(f"TASK_NAME={TASK_NAME} SEED={SEED} GROUP_SIZE={GROUP_SIZE} TRAIN_STEPS={TRAIN_STEPS}")
    if PUSH_TO_HUB and not HUB_REPO_ID:
        raise SystemExit("[FATAL] HUB_REPO_ID is required when PUSH_TO_HUB=1")

    preflight_env_health(ENV_URL)
    obs = reset_observation()
    log(f"preflight: env reset ok tick={obs.tick}")

    if DRY_RUN:
        log("DRY_RUN=1 - preflight only; not loading model or training")
        return 0

    import torch
    import torch.nn.functional as F
    from accelerate import Accelerator
    from huggingface_hub import HfApi
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    accelerator = Accelerator()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    log("loading tokenizer/model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        torch_dtype=dtype,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_RANK * 2,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model, optimizer = accelerator.prepare(model, optimizer)

    prompt = build_prompt(tokenizer, obs)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_LEN,
    )
    prompt_len = int(encoded["input_ids"].shape[1])
    encoded = {key: value.to(accelerator.device) for key, value in encoded.items()}

    for step in range(TRAIN_STEPS):
        model.eval()
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=TEMPERATURE,
                max_new_tokens=MAX_NEW_TOKENS,
                num_return_sequences=GROUP_SIZE,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completions = [
            tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
            for row in generated
        ]
        rewards = torch.tensor(
            [score_completion(completion) for completion in completions],
            dtype=torch.float32,
            device=accelerator.device,
        )
        advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)

        model.train()
        attention_mask = (generated != tokenizer.pad_token_id).long().to(accelerator.device)
        generated = generated.to(accelerator.device)
        outputs = model(input_ids=generated, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = generated[:, 1:]
        token_logprobs = F.log_softmax(logits.float(), dim=-1)
        selected = token_logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        completion_mask = torch.zeros_like(labels, dtype=torch.bool)
        completion_mask[:, max(prompt_len - 1, 0) :] = True
        completion_mask &= labels != tokenizer.pad_token_id
        completion_logprobs = (selected * completion_mask).sum(dim=1) / completion_mask.sum(
            dim=1
        ).clamp_min(1)
        loss = -(advantages.detach() * completion_logprobs).mean()

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        log(
            f"step={step + 1}/{TRAIN_STEPS} "
            f"rewards={[round(float(x), 3) for x in rewards.detach().cpu()]} "
            f"loss={float(loss.detach().cpu()):.4f}"
        )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        log(f"saving LoRA adapter to {OUTPUT_DIR}")
        unwrapped.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        if PUSH_TO_HUB:
            log(f"pushing to https://huggingface.co/{HUB_REPO_ID}")
            api = HfApi()
            api.create_repo(
                HUB_REPO_ID, exist_ok=True, repo_type="model", private=False, token=HF_TOKEN
            )
            api.upload_folder(
                folder_path=OUTPUT_DIR,
                repo_id=HUB_REPO_ID,
                repo_type="model",
                token=HF_TOKEN,
            )
    log("done")
    return 0


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    log(f"elapsed={time.time() - t0:.1f}s")
    sys.exit(rc)

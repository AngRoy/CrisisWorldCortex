"""SFT warmstart for the Cortex brain-selector router.

Workstream B Phase 6b. Trains a LoRA adapter on the Phase 6a B3 corpus so
the router starts GRPO already able to emit valid JSON:

    {"brain": "epi" | "logistics" | "governance"}

This intentionally uses a tiny raw Transformers + PEFT loop instead of
TRL's SFTTrainer. The objective is simple masked causal LM over the
assistant completion tokens, and avoiding TRL keeps the HF Jobs training
surface aligned with the no-TRL GRPO path in ``minimal_proof.py``.

Local preflight:
    DRY_RUN=1 HF_TOKEN=dummy uv run python training/scripts/sft_warmstart_router.py

Live run:
    HF_TOKEN=hf_xxx OUTPUT_REPO=Angshuman28/cortex-router-sft-warmstart \\
        uv run python training/scripts/sft_warmstart_router.py
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
MODEL_NAME = _env("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
B3_DATASET_REPO = _env("B3_DATASET_REPO", "Angshuman28/crisisworld-b3-corpus")
OUTPUT_REPO = _env("OUTPUT_REPO", "Angshuman28/cortex-router-sft-warmstart")
OUTPUT_DIR = _env("OUTPUT_DIR", "/tmp/cortex_router_sft_lora")
MAX_TRAIN_STEPS = int(_env("MAX_TRAIN_STEPS", "200"))
PER_DEVICE_BATCH = int(_env("PER_DEVICE_BATCH", "4"))
GRAD_ACCUM = int(_env("GRAD_ACCUM", "2"))
LR = float(_env("LR", "2e-5"))
LORA_RANK = int(_env("LORA_RANK", "16"))
LORA_ALPHA = int(_env("LORA_ALPHA", str(LORA_RANK * 2)))
LORA_DROPOUT = float(_env("LORA_DROPOUT", "0.05"))
MAX_SEQ_LEN = int(_env("MAX_SEQ_LEN", "1536"))
MAX_NEW_TOKENS = int(_env("MAX_NEW_TOKENS", "32"))
SEED = int(_env("SEED", "42"))
VALIDATION_ATTEMPTS = int(_env("VALIDATION_ATTEMPTS", "10"))
VALIDATION_MIN_RATE = float(_env("VALIDATION_MIN_RATE", "0.8"))
PUSH_TO_HUB = _env("PUSH_TO_HUB", "1") not in ("0", "", "false", "False")


def log(*args: object) -> None:
    print("[router-sft]", *args, flush=True)


ROUTER_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are the Cortex brain selector. Read one CrisisWorld observation and
    choose exactly one specialist brain to act next.

    Brain options:
    - epi: epidemiology, surveillance, case growth, and outbreak spread
    - logistics: scarce resources, hospital beds, mobile units, allocation
    - governance: legal constraints, compliance, escalation, restrictions

    Respond with exactly one JSON object and no prose:
    {"brain": "epi" | "logistics" | "governance"}
    """
).strip()

_BRAIN_ALIASES = {
    "epi": "epi",
    "epidemiology": "epi",
    "logistics": "logistics",
    "governance": "governance",
}


def parse_router_choice(raw_text: str) -> Optional[str]:
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
        if isinstance(candidate, dict):
            brain = _BRAIN_ALIASES.get(str(candidate.get("brain", "")).strip().lower())
            if brain is not None:
                return brain
    return None


def router_completion(brain: str) -> str:
    canonical = _BRAIN_ALIASES.get(brain)
    if canonical is None:
        raise ValueError(f"unknown brain label {brain!r}")
    return json.dumps({"brain": canonical}, separators=(",", ":"))


def render_prompt(tokenizer: Any, observation_text: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": observation_text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def preflight_dataset_access(dataset_repo: str, token: str) -> None:
    log(f"preflight: checking dataset {dataset_repo}")
    if DRY_RUN:
        log("DRY_RUN=1 - skipping Hub dataset access check")
        return
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        HfApi().dataset_info(dataset_repo, token=token or None)
    except RepositoryNotFoundError as exc:
        raise SystemExit(f"[FATAL] dataset {dataset_repo} not found: {exc}") from exc
    log("preflight: dataset accessible")


def make_collate(tokenizer: Any):
    def collate(features: List[Dict[str, List[int]]]) -> Dict[str, Any]:
        import torch

        max_len = max(len(item["input_ids"]) for item in features)
        input_ids, attention_mask, labels = [], [], []
        for item in features:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * pad)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def tokenize_rows(tokenizer: Any, rows: List[Dict[str, Any]]) -> List[Dict[str, List[int]]]:
    tokenized: List[Dict[str, List[int]]] = []
    eos = tokenizer.eos_token or ""
    for row in rows:
        prompt = render_prompt(tokenizer, row["observation_text"])
        completion = router_completion(row["deterministic_brain_choice"]) + eos
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(
            prompt + completion,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )["input_ids"]
        labels = list(full_ids)
        prompt_cutoff = min(len(prompt_ids), len(labels))
        labels[:prompt_cutoff] = [-100] * prompt_cutoff
        if all(label == -100 for label in labels):
            continue
        tokenized.append({"input_ids": full_ids, "labels": labels})
    if not tokenized:
        raise SystemExit("[FATAL] no tokenized training rows; check corpus schema")
    return tokenized


def validate_router_json(
    model: Any, tokenizer: Any, rows: List[Dict[str, Any]], device: Any
) -> float:
    import torch

    sample_rows = rows[: max(1, VALIDATION_ATTEMPTS)]
    ok = 0
    model.eval()
    for row in sample_rows:
        prompt = render_prompt(tokenizer, row["observation_text"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(
            device
        )
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        ok += parse_router_choice(text) is not None
    return ok / len(sample_rows)


def main() -> int:
    log(f"MODEL_NAME={MODEL_NAME}")
    log(f"B3_DATASET_REPO={B3_DATASET_REPO}")
    log(f"OUTPUT_REPO={OUTPUT_REPO}")
    log(f"MAX_TRAIN_STEPS={MAX_TRAIN_STEPS} LR={LR} LORA_RANK={LORA_RANK}")

    preflight_dataset_access(B3_DATASET_REPO, HF_TOKEN)
    if DRY_RUN:
        log("DRY_RUN=1 - preflight only; not loading dataset/model or training")
        return 0

    import torch
    from accelerate import Accelerator
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(SEED)
    torch.manual_seed(SEED)
    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM)

    log("loading dataset")
    dsdict = load_dataset(B3_DATASET_REPO, token=HF_TOKEN)
    train_split = dsdict["train"] if "train" in dsdict else dsdict[list(dsdict.keys())[0]]
    rows = [dict(row) for row in train_split]
    random.shuffle(rows)
    required = {"observation_text", "deterministic_brain_choice"}
    missing = required - set(train_split.column_names)
    if missing:
        raise SystemExit(f"[FATAL] dataset missing columns {sorted(missing)}")

    log("loading tokenizer/model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN, torch_dtype=dtype)
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    tokenized = tokenize_rows(tokenizer, rows)
    loader = DataLoader(
        tokenized,
        batch_size=PER_DEVICE_BATCH,
        shuffle=True,
        collate_fn=make_collate(tokenizer),
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    log(f"training rows={len(tokenized)}")
    step = 0
    while step < MAX_TRAIN_STEPS:
        for batch in loader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                step += 1
                if step % max(MAX_TRAIN_STEPS // 20, 1) == 0:
                    log(f"step={step}/{MAX_TRAIN_STEPS} loss={float(loss.detach().cpu()):.4f}")
                if step >= MAX_TRAIN_STEPS:
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        rate = validate_router_json(unwrapped, tokenizer, rows, accelerator.device)
        log(f"router JSON validation rate={rate:.0%}")
        if rate < VALIDATION_MIN_RATE:
            raise SystemExit(
                f"[FATAL] router JSON validation {rate:.0%} < {VALIDATION_MIN_RATE:.0%}"
            )
        log(f"saving LoRA adapter to {OUTPUT_DIR}")
        unwrapped.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        if PUSH_TO_HUB:
            log(f"pushing to https://huggingface.co/{OUTPUT_REPO}")
            api = HfApi()
            api.create_repo(
                OUTPUT_REPO, exist_ok=True, repo_type="model", private=False, token=HF_TOKEN
            )
            api.upload_folder(
                folder_path=OUTPUT_DIR,
                repo_id=OUTPUT_REPO,
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

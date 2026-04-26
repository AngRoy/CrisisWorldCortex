"""SFT warm-start trainer for B1 / Cortex base models.

Workstream B Phase 5d. TRL SFTTrainer over Unsloth-loaded base model
(Qwen3-7B or Llama-3.1-8B), 1-2 epochs over a Phase-5c-collected
trajectory dataset. Output is a LoRA adapter that teaches the JSON
action schema before downstream GRPO refines strategy.

Pipeline position:
    Base model (Qwen3-7B-Instruct or Llama-3.1-8B-Instruct)
        |
        v   THIS SCRIPT (~30 min on a100-large, ~$1.25)
        |
    SFT-warmstarted LoRA on HF Hub
        |
        v   Phase 5e: train_b1_grpo.py with BASE_MODEL=<this output>
    GRPO on env reward
        |
        v
    B1-trained / Cortex-router checkpoint

Usage on HF Jobs:
    hf jobs run --hardware a100-large --secret HF_TOKEN \\
        --env MODEL_NAME=unsloth/Qwen3-7B-Instruct-bnb-4bit \\
        --env SFT_DATASET_REPO=Angshuman28/crisisworld-sft-trajectories \\
        --env OUTPUT_REPO=Angshuman28/qwen3-7b-sft-warmstart \\
        ghcr.io/astral-sh/uv:latest \\
        bash -c "git clone https://huggingface.co/spaces/Angshuman28/CrisisWorldCortex /app && \\
                 cd /app && uv sync && uv run python training/scripts/sft_warmstart.py"

Local dry-run (skip GPU/dataset checks via DRY_RUN=1):
    DRY_RUN=1 OUTPUT_REPO=local/test uv run python training/scripts/sft_warmstart.py
"""

from __future__ import annotations

import os
import sys
import textwrap
import time
from typing import Optional


def _env(name: str, default: Optional[str] = None, *, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise SystemExit(f"[FATAL] env var {name} is required but unset")
    return value or ""


# ============================================================================
# Configuration (env-var driven)
# ============================================================================

HF_TOKEN = _env("HF_TOKEN", required=True)
MODEL_NAME = _env("MODEL_NAME", "unsloth/Qwen3-7B-Instruct-bnb-4bit")
SFT_DATASET_REPO = _env("SFT_DATASET_REPO", "Angshuman28/crisisworld-sft-trajectories")
OUTPUT_REPO = _env("OUTPUT_REPO", required=True)
OUTPUT_DIR = _env("OUTPUT_DIR", "/tmp/sft_warmstart_lora")
MAX_TRAIN_STEPS = int(_env("MAX_TRAIN_STEPS", "200"))
LR = float(_env("LR", "2e-5"))
LORA_RANK = int(_env("LORA_RANK", "32"))  # M-FR-19: matches GRPO downstream
NUM_EPOCHS = int(_env("NUM_EPOCHS", "2"))  # M-FR-20: cap by steps too
MAX_SEQ_LEN = int(_env("MAX_SEQ_LEN", "2560"))  # prompt 2048 + completion 512
PER_DEVICE_BATCH = int(_env("PER_DEVICE_BATCH", "4"))
GRAD_ACCUM = int(_env("GRAD_ACCUM", "2"))
SEED = int(_env("SEED", "42"))
GPU_MEM_UTIL = float(_env("GPU_MEM_UTIL", "0.6"))
DRY_RUN = _env("DRY_RUN", "0") not in ("0", "", "false", "False")


def log(*args: object) -> None:
    print("[sft-warmstart]", *args, flush=True)


_SYSTEM_PROMPT_BODY = textwrap.dedent(
    """
    You are an agent operating one outbreak-control simulator. You receive
    an observation each tick and must respond with EXACTLY ONE JSON object -
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


# ============================================================================
# Pre-flight: gated-model + dataset checks
# ============================================================================


def preflight_model_access(model_name: str, token: str) -> None:
    """Same fail-loud check as train_b1_grpo.py per Phase-A M-FR-3."""
    log(f"preflight: checking model access {model_name}")
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    try:
        info = HfApi().model_info(model_name, token=token)
        if getattr(info, "gated", False) and not getattr(info, "private", False):
            log(f"preflight: {model_name} is gated; access verified")
    except GatedRepoError as exc:
        raise SystemExit(
            f"[FATAL] {model_name} is gated and HF_TOKEN lacks access. "
            f"Visit https://huggingface.co/{model_name} and accept the license. "
            f"Original: {exc}"
        ) from exc
    except RepositoryNotFoundError as exc:
        raise SystemExit(f"[FATAL] {model_name} not found on HF Hub: {exc}") from exc
    log(f"preflight: {model_name} accessible")


def preflight_dataset_access(dataset_repo: str, token: str) -> None:
    """Verify the SFT dataset exists. Schema check happens after load."""
    log(f"preflight: checking dataset {dataset_repo}")
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        HfApi().dataset_info(dataset_repo, token=token)
    except RepositoryNotFoundError as exc:
        raise SystemExit(
            f"[FATAL] dataset {dataset_repo} not found. Run Phase-5c "
            f"(collect_sft_data.py) first. Original: {exc}"
        ) from exc
    log(f"preflight: dataset {dataset_repo} accessible")


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    log(f"MODEL_NAME={MODEL_NAME}")
    log(f"SFT_DATASET_REPO={SFT_DATASET_REPO}")
    log(f"OUTPUT_REPO={OUTPUT_REPO}")
    log(f"MAX_TRAIN_STEPS={MAX_TRAIN_STEPS} NUM_EPOCHS={NUM_EPOCHS} LR={LR}")
    log(f"LORA_RANK={LORA_RANK} MAX_SEQ_LEN={MAX_SEQ_LEN}")

    preflight_model_access(MODEL_NAME, HF_TOKEN)
    preflight_dataset_access(SFT_DATASET_REPO, HF_TOKEN)

    if DRY_RUN:
        log("DRY_RUN=1 — preflight only; not loading model or training")
        return 0

    # Lazy imports — keeps preflight fast and avoids loading Unsloth/torch
    # on local machines that don't have GPU.
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    # ---- Load dataset ----
    log(f"loading dataset {SFT_DATASET_REPO}")
    dsdict = load_dataset(SFT_DATASET_REPO, token=HF_TOKEN)
    if "train" not in dsdict:
        raise SystemExit(f"[FATAL] dataset {SFT_DATASET_REPO} missing 'train' split")
    train_ds = dsdict["train"]
    eval_ds = dsdict.get("eval")
    log(f"dataset: train={len(train_ds)} eval={len(eval_ds) if eval_ds else 0}")
    required_cols = {"prompt", "completion"}
    missing = required_cols - set(train_ds.column_names)
    if missing:
        raise SystemExit(
            f"[FATAL] dataset {SFT_DATASET_REPO} missing columns: {missing}. "
            f"Got {train_ds.column_names}. Re-run Phase 5c."
        )

    # ---- Load model + LoRA ----
    log(f"loading model {MODEL_NAME} (LoRA rank={LORA_RANK})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        fast_inference=False,  # SFT is forward-only at training time
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

    # ---- Format function: compose prompt + completion into trainable text ----
    eos = tokenizer.eos_token or "<|endoftext|>"

    def formatting_func(example: dict) -> str:
        # Phase 5c stores the raw serialized observation in "prompt".
        # Render it through the target tokenizer here so SFT matches the
        # prompt shape used by train_b1_grpo.py.
        rendered_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": _SYSTEM_PROMPT_BODY},
                {"role": "user", "content": example["prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return f"{rendered_prompt}{example['completion']}{eos}"

    # ---- Compute effective epoch budget ----
    effective_batch = PER_DEVICE_BATCH * GRAD_ACCUM
    steps_per_epoch = max(len(train_ds) // effective_batch, 1)
    epoch_step_budget = NUM_EPOCHS * steps_per_epoch
    final_max_steps = min(MAX_TRAIN_STEPS, epoch_step_budget)
    log(
        f"steps_per_epoch={steps_per_epoch} epoch_budget={epoch_step_budget} "
        f"final_max_steps={final_max_steps}"
    )

    # ---- SFT training ----
    log("starting SFTTrainer")
    sft_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=LR,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=final_max_steps,
        save_steps=max(final_max_steps // 3, 1),
        logging_steps=max(final_max_steps // 60, 1),
        report_to="none",
        bf16=True,
        optim="adamw_8bit",
        seed=SEED,
        max_length=MAX_SEQ_LEN,
        warmup_ratio=0.05,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_args,
        train_dataset=train_ds,
        formatting_func=formatting_func,
    )
    trainer.train()
    log("training done")

    # ---- Save + push ----
    log(f"saving LoRA adapter to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    log(f"pushing to https://huggingface.co/{OUTPUT_REPO}")
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(OUTPUT_REPO, exist_ok=True, repo_type="model", private=False, token=HF_TOKEN)
    api.upload_folder(
        folder_path=OUTPUT_DIR,
        repo_id=OUTPUT_REPO,
        repo_type="model",
        token=HF_TOKEN,
    )
    log("push complete")
    return 0


if __name__ == "__main__":
    t0 = time.time()
    try:
        rc = main()
    except SystemExit:
        raise
    log(f"done in {time.time() - t0:.1f}s")
    sys.exit(rc)

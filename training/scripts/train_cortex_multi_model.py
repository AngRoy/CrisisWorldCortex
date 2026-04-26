"""Multi-model Cortex GRPO training (Workstream B Phase 6).

Trains the Cortex router LM (Qwen3-1.5B-Instruct + LoRA) via GRPO
against episode reward, while three frozen brain LLMs (Qwen-7B epi,
Llama-8B logistics, Qwen-7B governance via shared weights) drive the
deliberation rollouts.

Per Phase-A docs/CORTEX_ARCHITECTURE.md and root CLAUDE.md "Frozen"
section: this is the multi-model training surface. Each Brain instance
holds its own LLMClient pointing to a different model — Session 11's
``Brain(__init__(llm_client))`` is multi-model-ready by design (audited
M-FR-12).

Memory budget on a100-large (80GB):
  - Qwen3-7B 4-bit (epi):                ~14 GB
  - Llama-3.1-8B 4-bit (logistics):      ~16 GB
  - Qwen3-7B 4-bit (governance, shared): ~ 0 GB additional
  - Qwen3-1.5B 4-bit + LoRA (router):    ~ 3 GB
  - vLLM rollout overhead (router only): ~10 GB
  - Total used:                         ~43 GB, ~37 GB headroom.

Per Phase-A M-FR-31: only the trainable router uses vLLM; frozen brains
use plain ``transformers.generate`` to avoid 4-way vLLM contention.

Trainer.train() is commented out below — the live run is user-gated
(~2 hours a100-large, ~$5). Skeleton ships with the full integration
graph; uncomment the train() line after a 5-step dry-run smoke-tests
the orchestration end-to-end.

Usage on HF Jobs:
    hf jobs run --hardware a100-large --secret HF_TOKEN \\
        --env HUB_REPO_ID=Angshuman28/crisisworld-cortex-router-llm \\
        ghcr.io/astral-sh/uv:latest \\
        bash -c "git clone https://huggingface.co/spaces/Angshuman28/CrisisWorldCortex /app && \\
                 cd /app && uv sync && uv run python training/scripts/train_cortex_multi_model.py"

Local DRY_RUN test (no GPU):
    DRY_RUN=1 HUB_REPO_ID=local/test \\
        uv run python training/scripts/train_cortex_multi_model.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, List, Optional


def _env(name: str, default: Optional[str] = None, *, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise SystemExit(f"[FATAL] env var {name} is required but unset")
    return value or ""


# ============================================================================
# Configuration (env-var driven)
# ============================================================================

HF_TOKEN = _env("HF_TOKEN", required=True)

# Brain model choices. Default: Qwen-7B for epi+governance (shared), Llama-8B
# for logistics. Each can be overridden to point at SFT-warmstarted checkpoints.
EPI_BRAIN_MODEL = _env("EPI_BRAIN_MODEL", "unsloth/Qwen3-7B-Instruct-bnb-4bit")
LOGISTICS_BRAIN_MODEL = _env("LOGISTICS_BRAIN_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
GOVERNANCE_BRAIN_MODEL = _env("GOVERNANCE_BRAIN_MODEL", EPI_BRAIN_MODEL)  # M-FR-27 default

# Router model (the only trainable surface).
ROUTER_MODEL = _env("ROUTER_MODEL", "unsloth/Qwen3-1.5B-Instruct-bnb-4bit")
ROUTER_BASE_MODEL = _env("ROUTER_BASE_MODEL", ROUTER_MODEL)  # Phase-5e style fallback

HUB_REPO_ID = _env("HUB_REPO_ID", required=True)
ENV_URL = _env("ENV_URL", "https://angshuman28-crisisworldcortex.hf.space")
OUTPUT_DIR = _env("OUTPUT_DIR", "/tmp/cortex_router_grpo_lora")

MAX_TRAIN_STEPS = int(_env("MAX_TRAIN_STEPS", "300"))
GROUP_SIZE = int(_env("GROUP_SIZE", "4"))
BRAIN_CALL_TIMEOUT_S = int(_env("BRAIN_CALL_TIMEOUT_S", "30"))  # Phase-A M-FR-9
LR = float(_env("LR", "5e-6"))
LORA_RANK = int(_env("LORA_RANK", "16"))  # M-FR-28: smaller for 1.5B
GPU_MEM_UTIL = float(_env("GPU_MEM_UTIL", "0.5"))  # Phase-A M-FR-10
MAX_PROMPT_LEN = int(_env("MAX_PROMPT_LEN", "512"))
MAX_COMPLETION_LEN = int(_env("MAX_COMPLETION_LEN", "256"))  # Phase-A M-FR-11
TASKS_CSV = _env("TASKS_CSV", "outbreak_easy,outbreak_medium,outbreak_hard")
EPISODE_TICKS = int(_env("EPISODE_TICKS", "12"))
SEED = int(_env("SEED", "42"))
DRY_RUN = _env("DRY_RUN", "0") not in ("0", "", "false", "False")
ALLOW_CORTEX_SKELETON = _env("ALLOW_CORTEX_SKELETON", "0") not in ("0", "", "false", "False")

# Memory budget hard floor: abort if less than this many GB free at script
# start. Conservative — the steady-state is ~43 GB used; this leaves a 30-GB
# margin for first-call activation peaks before the first vLLM kv-cache lock.
MIN_FREE_GPU_GB = float(_env("MIN_FREE_GPU_GB", "30"))


def log(*args: object) -> None:
    print("[cortex-multi-model]", *args, flush=True)


def _sync_if_available(env: Any) -> Any:
    """OpenEnv 0.2.2+ exposes .sync(); 0.2.1 reset/step are already sync."""
    sync = getattr(env, "sync", None)
    return sync() if callable(sync) else env


# ============================================================================
# Pre-flight
# ============================================================================


def preflight_model_access(model_name: str, token: str) -> None:
    """Same fail-loud check as Phase-5b/5d. Llama-3.1-8B may be gated."""
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


def check_memory_budget() -> None:
    """Pre-flight torch.cuda.mem_get_info() check.

    Hard floor: MIN_FREE_GPU_GB free. If less, abort before the first
    model-load OOM crash mid-loading.
    """
    try:
        import torch
    except ImportError:
        log("WARN torch not importable — skipping memory check")
        return
    if not torch.cuda.is_available():
        log("WARN CUDA not available — skipping memory check (DRY_RUN expected)")
        return
    free, total = torch.cuda.mem_get_info()
    free_gb = free / (1024**3)
    total_gb = total / (1024**3)
    log(f"GPU memory: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
    if free_gb < MIN_FREE_GPU_GB:
        raise RuntimeError(
            f"Insufficient GPU memory: {free_gb:.1f} GB free, need >= {MIN_FREE_GPU_GB} GB. "
            f"Reduce LORA_RANK / GPU_MEM_UTIL or pick a smaller GOVERNANCE_BRAIN_MODEL "
            f"(default shares with EPI). Aborting before model load."
        )


# ============================================================================
# Adapters: Unsloth-loaded model -> cortex._LLMClientLike
# ============================================================================


class _UnslothLLMAdapter:
    """Wrap an Unsloth-loaded (model, tokenizer) pair as a ``_LLMClientLike``.

    Matches ``cortex.subagents._base._LLMClientLike`` protocol so it can
    drop into ``Brain.__init__(llm_client=...)``. Frozen brains call this
    via plain ``transformers.generate`` (M-FR-31 — no vLLM for brains).

    Token accounting (``tokens_used_for``) returns 0 in this MVP; the
    GRPO reward signal is the env's ``obs.reward``, not the budget
    composition. If we later add token-budget-shaped reward, plug
    ``training.reward_shaping.shape_reward`` into the rollout.
    """

    def __init__(self, model: Any, tokenizer: Any, *, brain_label: str) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._brain_label = brain_label
        self._call_counts: dict[str, int] = {}

    def chat(
        self,
        caller_id: str,
        messages: List[Any],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Any:
        from cortex.llm_client import ChatResponse

        prompt = self._tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens or 256,
            "do_sample": (temperature or 0.0) > 0,
        }
        if (temperature or 0.0) > 0:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        prompt_tokens = inputs["input_ids"].shape[1]
        completion_tokens = out.shape[1] - prompt_tokens
        text = self._tokenizer.decode(out[0][prompt_tokens:], skip_special_tokens=True)
        self._call_counts[caller_id] = self._call_counts.get(caller_id, 0) + 1
        # cortex.llm_client.ChatResponse signature: text + tokens_in + tokens_out.
        return ChatResponse(text=text, tokens_in=prompt_tokens, tokens_out=completion_tokens)

    def tokens_used_for(self, caller_id: str) -> int:
        return self._call_counts.get(caller_id, 0)


# ============================================================================
# Adapter: trainable Unsloth router -> cortex.RoutingPolicy
# ============================================================================


class _TrainableRoutingPolicy:
    """Wrap the Unsloth-loaded router LM as a ``cortex.RoutingPolicy``.

    Input: ``MetacognitionState``. Output: ``RoutingAction``.

    The router emits structured JSON per the system prompt. On parse
    failure, returns ``stop_and_no_op`` per Phase-A M-FR-5 (close the
    tick gracefully; the negative reward gradient teaches the router
    to emit valid JSON).
    """

    SYSTEM_PROMPT = (
        "You are the Cortex router. You receive a metacognition state summary and emit "
        "ONE routing action as JSON. Allowed kinds: call_subagent (brain + subagent), "
        "request_challenge (challenger_brain + target_brain), switch_phase (new_phase), "
        "preserve_dissent (tag), emit_outer_action (action), stop_and_no_op. Hard caps: "
        "<=2 rounds/tick, <=1 cross-brain challenge/tick, <=1 critic per brain/tick, "
        "<=6000 tokens/tick. Output exactly one JSON object — no prose, no fences."
    )

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer

    @staticmethod
    def _state_to_prompt(state: Any) -> str:
        return (
            f"tick={getattr(state, 'tick', 0)} round={getattr(state, 'round', 1)} "
            f"phase={getattr(state, 'phase', 'divergence')}\n"
            f"agreement={getattr(state, 'inter_brain_agreement', 0.0):.2f} "
            f"avg_conf={getattr(state, 'average_confidence', 0.0):.2f} "
            f"evidence={getattr(state, 'average_evidence_support', 0.0):.2f}\n"
            f"novelty={getattr(state, 'novelty_yield_last_round', 0.0):.2f} "
            f"collapse={getattr(state, 'collapse_suspicion', 0.0):.2f} "
            f"budget_frac={getattr(state, 'budget_remaining_frac', 1.0):.2f} "
            f"urgency={getattr(state, 'urgency', 0.0):.2f}\n"
            f"preserved_dissent={getattr(state, 'preserved_dissent_count', 0)} "
            f"challenge_used={bool(getattr(state, 'challenge_used_this_tick', 0))}\n"
            f"Choose the next routing action."
        )

    def forward(self, state: Any) -> Any:
        import json

        import torch
        from pydantic import TypeAdapter, ValidationError

        from cortex.schemas import RoutingAction, StopAndNoOp

        prompt = self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self._state_to_prompt(state)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=MAX_COMPLETION_LEN, do_sample=False, temperature=0.0
            )
        text = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        try:
            data = json.loads(text)
            return TypeAdapter(RoutingAction).validate_python(data)
        except (json.JSONDecodeError, ValidationError):
            # Phase-A M-FR-5 graceful fallback.
            return StopAndNoOp()


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    log(f"EPI_BRAIN_MODEL={EPI_BRAIN_MODEL}")
    log(f"LOGISTICS_BRAIN_MODEL={LOGISTICS_BRAIN_MODEL}")
    log(f"GOVERNANCE_BRAIN_MODEL={GOVERNANCE_BRAIN_MODEL}")
    log(f"ROUTER_MODEL={ROUTER_MODEL} ROUTER_BASE_MODEL={ROUTER_BASE_MODEL}")
    log(f"HUB_REPO_ID={HUB_REPO_ID} ENV_URL={ENV_URL}")
    log(f"MAX_TRAIN_STEPS={MAX_TRAIN_STEPS} GROUP_SIZE={GROUP_SIZE} LR={LR}")
    log(f"LORA_RANK={LORA_RANK} GPU_MEM_UTIL={GPU_MEM_UTIL}")

    preflight_model_access(EPI_BRAIN_MODEL, HF_TOKEN)
    preflight_model_access(LOGISTICS_BRAIN_MODEL, HF_TOKEN)
    if GOVERNANCE_BRAIN_MODEL != EPI_BRAIN_MODEL:
        preflight_model_access(GOVERNANCE_BRAIN_MODEL, HF_TOKEN)
    preflight_model_access(ROUTER_BASE_MODEL, HF_TOKEN)

    if DRY_RUN:
        log("DRY_RUN=1 — preflight only; not loading models or training")
        return 0

    if not ALLOW_CORTEX_SKELETON:
        raise SystemExit(
            "[FATAL] train_cortex_multi_model.py is still a Phase-6 skeleton: "
            "its dataset is a placeholder and its GRPO reward path does not "
            "condition reward on the sampled router completion. Refusing to "
            "push an untrained adapter. Set ALLOW_CORTEX_SKELETON=1 only for "
            "manual construction debugging, not for submission training."
        )

    check_memory_budget()

    # Lazy imports — keeps DRY_RUN fast and avoids loading torch/Unsloth on
    # local machines that don't have GPU.
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel

    from cortex.brains import EpiBrain, GovernanceBrain, LogisticsBrain
    from cortex.council import Council
    from CrisisWorldCortex.client import CrisisworldcortexEnv

    # Phase 7 will import baselines.cortex_fixed_router inline at the warmup-data
    # step. Cannot import here at module scope: training/* MUST NOT import
    # baselines/* per the import-graph rule (enforced by tests/test_import_graph.py).

    # ---- Load 3 frozen brain LLMs (M-FR-31: no vLLM for brains) ----
    def _load_frozen(model_name: str, label: str) -> tuple[Any, Any]:
        log(f"loading frozen {label} brain: {model_name}")
        m, t = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=4096,
            load_in_4bit=True,
            fast_inference=False,  # plain transformers.generate per M-FR-31
            gpu_memory_utilization=GPU_MEM_UTIL,
        )
        FastLanguageModel.for_inference(m)
        return m, t

    epi_model, epi_tok = _load_frozen(EPI_BRAIN_MODEL, "epi")
    logistics_model, logistics_tok = _load_frozen(LOGISTICS_BRAIN_MODEL, "logistics")
    if GOVERNANCE_BRAIN_MODEL == EPI_BRAIN_MODEL:
        log("governance: sharing weights with epi (M-FR-27)")
        governance_model, governance_tok = epi_model, epi_tok
    else:
        governance_model, governance_tok = _load_frozen(GOVERNANCE_BRAIN_MODEL, "governance")

    # Build per-brain LLM client adapters.
    epi_client = _UnslothLLMAdapter(epi_model, epi_tok, brain_label="epi")
    logistics_client = _UnslothLLMAdapter(logistics_model, logistics_tok, brain_label="logistics")
    governance_client = _UnslothLLMAdapter(
        governance_model, governance_tok, brain_label="governance"
    )

    # Construct the 3 brains via Session-11 factory functions.
    brains = {
        "epidemiology": EpiBrain(llm_client=epi_client),
        "logistics": LogisticsBrain(llm_client=logistics_client),
        "governance": GovernanceBrain(llm_client=governance_client),
    }
    log(f"brains constructed: {list(brains.keys())}")

    # ---- Load the trainable router LLM (M-FR-31: only this one uses vLLM) ----
    log(f"loading trainable router: {ROUTER_BASE_MODEL} (LoRA r={LORA_RANK})")
    router_model, router_tok = FastLanguageModel.from_pretrained(
        model_name=ROUTER_BASE_MODEL,
        max_seq_length=MAX_PROMPT_LEN + MAX_COMPLETION_LEN,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEM_UTIL,
    )
    router_model = FastLanguageModel.get_peft_model(
        router_model,
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
    router_policy = _TrainableRoutingPolicy(router_model, router_tok)
    log("trainable router ready")

    # ---- Build prompt dataset from B3 deterministic-router trajectories ----
    # Phase 7 will populate this from real B3 rollouts; the skeleton emits
    # one placeholder row so GRPOTrainer construction succeeds.
    tasks = tuple(t.strip() for t in TASKS_CSV.split(",") if t.strip())
    log(f"tasks={tasks}")

    def make_env() -> Any:
        return _sync_if_available(CrisisworldcortexEnv(base_url=ENV_URL))

    train_dataset = Dataset.from_dict(
        {
            "prompt": ["placeholder until live B3 corpus collection"],
            "task": ["outbreak_easy"],
            "seed": [0],
        }
    )

    # ---- Reward function: full-episode rollout per (prompt, completion) ----
    def cortex_reward(
        prompts: list[str],
        completions: list[str],
        task: list[str],
        seed: list[int],
        **_kwargs: object,
    ) -> list[float]:
        rewards: list[float] = []
        for _completion, t, s in zip(completions, task, seed):
            env = None
            try:
                council = Council(brains=brains, routing_policy=router_policy)
                env = make_env()
                reset_result = env.reset(task_name=t, seed=int(s), max_ticks=EPISODE_TICKS)
                obs = (
                    reset_result.observation
                    if hasattr(reset_result, "observation")
                    else reset_result
                )
                cumulative = 0.0
                last_reward = 0.0
                for _ in range(EPISODE_TICKS):
                    action = council.step(obs, last_reward=last_reward)
                    result = env.step(action)
                    next_obs = result.observation if hasattr(result, "observation") else result
                    last_reward = next_obs.reward if next_obs.reward is not None else 0.0
                    cumulative += last_reward
                    obs = next_obs
                    if next_obs.done:
                        break
                rewards.append(float(cumulative))
            except Exception as exc:
                log(f"WARN rollout failed task={t} seed={s}: {exc}")
                rewards.append(-1.0)
            finally:
                if env is not None:
                    env.close()
        return rewards

    # ---- GRPO config + trainer ----
    log("constructing GRPOTrainer (router-only)")
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
        temperature=0.8,
        use_vllm=True,
        vllm_mode="colocate",
        seed=SEED,
    )
    trainer = GRPOTrainer(
        model=router_model,  # M-FR-23: router is the ONLY trainable surface
        processing_class=router_tok,
        reward_funcs=[cortex_reward],
        args=training_args,
        train_dataset=train_dataset,
    )
    log(f"trainer constructed: {type(trainer).__name__}")
    log("# Phase 6 ships skeleton — uncomment trainer.train() in Phase 7 after")
    log("# (a) B3 corpus has populated train_dataset, and (b) a 5-step dry-run")
    log("# verifies multi-model orchestration end-to-end (~$0.50).")
    # trainer.train()  # Phase 7 unblock: uncomment after dry-run verification.

    log(f"saving router LoRA to {OUTPUT_DIR}")
    router_model.save_pretrained(OUTPUT_DIR)
    router_tok.save_pretrained(OUTPUT_DIR)

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
    t0 = time.time()
    rc = main()
    log(f"done in {time.time() - t0:.1f}s")
    sys.exit(rc)

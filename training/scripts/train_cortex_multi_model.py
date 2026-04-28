"""Train the Cortex brain-selector router with no-TRL GRPO.

Workstream B Phase 6c. This is the trainable version of the project
thesis: a learned router chooses among frozen specialist brains
(epidemiology, logistics, governance), each frozen brain emits an env
action, and only the router LoRA receives policy-gradient updates.

Design choices:
  - Rollout structure: sample GROUP_SIZE router choices for the same
    observation/tick. This gives direct credit assignment for "which brain
    should act now" without full-episode confounding.
  - Env state for counterfactuals: replay the committed action prefix into
    fresh env sessions. The OpenEnv server does not expose cloning, and
    replay is the cleanest deterministic substitute.
  - Update cadence: one optimizer update per tick. Episodes are short
    (12 ticks), so per-tick updates produce more learning signal; low LR
    plus gradient clipping controls variance.
  - Invalid router output: reward 0.0 and no env call. This matches the
    deterministic floor instead of adding an arbitrary penalty that could
    swamp early structured-output learning.
  - Brain calls: local frozen model inference by default. It is cheaper and
    more reproducible than HF Inference Router calls; if memory is tight,
    switch to smaller model env vars or disable 4-bit only on A100-class GPUs.

HF Jobs:
    hf jobs run --hardware a100-large --secret HF_TOKEN \\
        --env HUB_REPO_ID=Angshuman28/cortex-router-trained \\
        ghcr.io/astral-sh/uv:latest \\
        bash -c "git clone https://huggingface.co/spaces/Angshuman28/CrisisWorldCortex /app && \\
                 cd /app && uv sync && uv run python training/scripts/train_cortex_multi_model.py"
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


HF_TOKEN = _env("HF_TOKEN", required=True)
ENV_URL = _env("ENV_URL", "https://angshuman28-crisisworldcortex.hf.space")

EPI_BRAIN_MODEL = _env("EPI_BRAIN_MODEL", "Qwen/Qwen2.5-3B-Instruct")
LOGISTICS_BRAIN_MODEL = _env("LOGISTICS_BRAIN_MODEL", "microsoft/Phi-3.5-mini-instruct")
GOVERNANCE_BRAIN_MODEL = _env("GOVERNANCE_BRAIN_MODEL", EPI_BRAIN_MODEL)

ROUTER_MODEL = _env("ROUTER_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ROUTER_WARMSTART_REPO = _env("ROUTER_WARMSTART_REPO", "Angshuman28/cortex-router-sft-warmstart")
HUB_REPO_ID = _env("HUB_REPO_ID", "Angshuman28/cortex-router-trained")
OUTPUT_DIR = _env("OUTPUT_DIR", "/tmp/cortex_router_grpo_lora")

TASKS_CSV = _env("TASKS_CSV", "outbreak_easy,outbreak_medium,outbreak_hard")
EPISODE_TICKS = int(_env("EPISODE_TICKS", "12"))
MAX_TRAIN_STEPS = int(_env("MAX_TRAIN_STEPS", "300"))
GROUP_SIZE = int(_env("GROUP_SIZE", "4"))
LR = float(_env("LR", "5e-6"))
TEMPERATURE = float(_env("TEMPERATURE", "0.8"))
ROUTER_MAX_PROMPT_LEN = int(_env("ROUTER_MAX_PROMPT_LEN", "2048"))
ROUTER_MAX_NEW_TOKENS = int(_env("ROUTER_MAX_NEW_TOKENS", "32"))
BRAIN_MAX_PROMPT_LEN = int(_env("BRAIN_MAX_PROMPT_LEN", "2048"))
BRAIN_MAX_NEW_TOKENS = int(_env("BRAIN_MAX_NEW_TOKENS", "192"))
LORA_RANK = int(_env("LORA_RANK", "16"))
LORA_ALPHA = int(_env("LORA_ALPHA", str(LORA_RANK * 2)))
LORA_DROPOUT = float(_env("LORA_DROPOUT", "0.05"))
GRAD_CLIP = float(_env("GRAD_CLIP", "1.0"))
SEED = int(_env("SEED", "42"))
SAVE_STEPS = int(_env("SAVE_STEPS", "20"))
LOG_STEPS = int(_env("LOG_STEPS", "5"))
LOAD_IN_4BIT = _env("LOAD_IN_4BIT", "1") not in ("0", "", "false", "False")
PUSH_TO_HUB = _env("PUSH_TO_HUB", "1") not in ("0", "", "false", "False")
DRY_RUN = _env("DRY_RUN", "0") not in ("0", "", "false", "False")

MIN_FREE_GPU_GB = float(_env("MIN_FREE_GPU_GB", "20"))

_BRAIN_CHOICES = ("epi", "logistics", "governance")


def log(*args: object) -> None:
    print("[cortex-grpo]", *args, flush=True)


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


def preflight_env_health(env_url: str) -> None:
    import urllib.request

    log(f"preflight: checking {env_url}/health")
    with urllib.request.urlopen(f"{env_url}/health", timeout=10) as resp:
        body = resp.read().decode("utf-8")
        if resp.status != 200 or "healthy" not in body.lower():
            raise SystemExit(f"[FATAL] env unhealthy: status={resp.status} body={body!r}")
    log("preflight: env healthy")


def preflight_model_access(model_name: str, token: str, *, repo_type: str = "model") -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    log(f"preflight: checking {repo_type} access {model_name}")
    try:
        HfApi().model_info(model_name, token=token)
    except GatedRepoError as exc:
        raise SystemExit(f"[FATAL] gated model access missing for {model_name}: {exc}") from exc
    except RepositoryNotFoundError as exc:
        raise SystemExit(f"[FATAL] model {model_name} not found: {exc}") from exc


def check_memory_budget() -> None:
    try:
        import torch
    except ImportError:
        log("WARN torch not importable - skipping memory check")
        return
    if not torch.cuda.is_available():
        log("WARN CUDA not available - local CPU smoke only; HF Jobs should use GPU")
        return
    free, total = torch.cuda.mem_get_info()
    free_gb = free / (1024**3)
    total_gb = total / (1024**3)
    log(f"GPU memory: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
    if free_gb < MIN_FREE_GPU_GB:
        raise RuntimeError(f"Insufficient GPU memory: {free_gb:.1f} GB free")


def _sync_if_available(env: Any) -> Any:
    sync = getattr(env, "sync", None)
    return sync() if callable(sync) else env


def make_env() -> Any:
    from CrisisWorldCortex.client import CrisisworldcortexEnv

    return _sync_if_available(CrisisworldcortexEnv(base_url=ENV_URL))


def normalize_step_result(result: Any) -> Any:
    return result.observation if hasattr(result, "observation") else result


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


def extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
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
            return candidate
    return None


def parse_router_choice(raw_text: str) -> Optional[str]:
    data = extract_json(raw_text)
    if data is None:
        return None
    brain = str(data.get("brain", "")).strip().lower()
    aliases = {
        "epidemiology": "epi",
        "epi": "epi",
        "logistics": "logistics",
        "governance": "governance",
    }
    return aliases.get(brain)


def parse_action(raw_text: str) -> Any:
    from pydantic import TypeAdapter, ValidationError

    from CrisisWorldCortex.models import OuterActionPayload

    data = extract_json(raw_text)
    if data is None or "kind" not in data:
        return None
    try:
        return TypeAdapter(OuterActionPayload).validate_python(data)
    except ValidationError:
        return None


class CandidateResult:
    def __init__(
        self,
        reward: float,
        brain: Optional[str],
        action: Any,
        completion: str,
        accepted: bool,
    ) -> None:
        self.reward = reward
        self.brain = brain
        self.action = action
        self.completion = completion
        self.accepted = accepted


class FrozenBrain:
    def __init__(self, brain: str, model: Any, tokenizer: Any) -> None:
        self.brain = brain
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def act(self, observation_text: str) -> Any:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": brain_system_prompt(self.brain)},
                {"role": "user", "content": observation_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=BRAIN_MAX_PROMPT_LEN
        ).to(device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=BRAIN_MAX_NEW_TOKENS,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        return parse_action(text)


def model_kwargs(torch: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"token": HF_TOKEN}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16
    if LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs["device_map"] = "auto"
    return kwargs


def load_tokenizer(name: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, token=HF_TOKEN)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(name: str, torch: Any) -> Any:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs(torch))
    if not LOAD_IN_4BIT and torch.cuda.is_available():
        model = model.to("cuda")
    return model


def load_router(torch: Any) -> tuple[Any, Any]:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    tokenizer = load_tokenizer(ROUTER_MODEL)
    model = load_causal_lm(ROUTER_MODEL, torch)
    if ROUTER_WARMSTART_REPO:
        log(f"loading router warmstart LoRA {ROUTER_WARMSTART_REPO}")
        model = PeftModel.from_pretrained(model, ROUTER_WARMSTART_REPO, is_trainable=True)
    else:
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
    model.train()
    return model, tokenizer


def load_brains(torch: Any) -> Dict[str, FrozenBrain]:
    log("loading frozen epi brain")
    epi_tok = load_tokenizer(EPI_BRAIN_MODEL)
    epi_model = load_causal_lm(EPI_BRAIN_MODEL, torch)
    log("loading frozen logistics brain")
    logistics_tok = load_tokenizer(LOGISTICS_BRAIN_MODEL)
    logistics_model = load_causal_lm(LOGISTICS_BRAIN_MODEL, torch)
    if GOVERNANCE_BRAIN_MODEL == EPI_BRAIN_MODEL:
        log("governance brain sharing epi model weights")
        governance_tok, governance_model = epi_tok, epi_model
    else:
        log("loading frozen governance brain")
        governance_tok = load_tokenizer(GOVERNANCE_BRAIN_MODEL)
        governance_model = load_causal_lm(GOVERNANCE_BRAIN_MODEL, torch)
    for model in {
        id(epi_model): epi_model,
        id(logistics_model): logistics_model,
        id(governance_model): governance_model,
    }.values():
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    return {
        "epi": FrozenBrain("epi", epi_model, epi_tok),
        "logistics": FrozenBrain("logistics", logistics_model, logistics_tok),
        "governance": FrozenBrain("governance", governance_model, governance_tok),
    }


def router_prompt(tokenizer: Any, observation_text: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": observation_text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def sample_router(
    router_model: Any, tokenizer: Any, observation_text: str
) -> tuple[Any, int, List[str]]:
    prompt = router_prompt(tokenizer, observation_text)
    device = next(router_model.parameters()).device
    encoded = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=ROUTER_MAX_PROMPT_LEN
    ).to(device)
    prompt_len = int(encoded["input_ids"].shape[1])
    router_model.eval()
    generated = router_model.generate(
        **encoded,
        do_sample=True,
        temperature=TEMPERATURE,
        max_new_tokens=ROUTER_MAX_NEW_TOKENS,
        num_return_sequences=GROUP_SIZE,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    completions = [
        tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip() for row in generated
    ]
    return generated, prompt_len, completions


def completion_logprobs(router_model: Any, tokenizer: Any, generated: Any, prompt_len: int) -> Any:
    import torch
    import torch.nn.functional as F

    router_model.train()
    device = next(router_model.parameters()).device
    generated = generated.to(device)
    attention_mask = (generated != tokenizer.pad_token_id).long().to(device)
    outputs = router_model(input_ids=generated, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = generated[:, 1:]
    token_logprobs = F.log_softmax(logits.float(), dim=-1)
    selected = token_logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    completion_mask = torch.zeros_like(labels, dtype=torch.bool)
    completion_mask[:, max(prompt_len - 1, 0) :] = True
    completion_mask &= labels != tokenizer.pad_token_id
    return (selected * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp_min(1)


def replay_prefix(env: Any, task: str, seed: int, prefix_actions: List[Any]) -> Any:
    result = env.reset(task_name=task, seed=seed, max_ticks=EPISODE_TICKS)
    obs = normalize_step_result(result)
    for action in prefix_actions:
        obs = normalize_step_result(env.step(action))
        if obs.done:
            break
    return obs


def score_candidate(
    *,
    env: Any,
    task: str,
    seed: int,
    prefix_actions: List[Any],
    observation_text: str,
    completion: str,
    brains: Dict[str, FrozenBrain],
) -> CandidateResult:
    """Score one router sample by resetting ``env``, replaying the prefix, then stepping the candidate.

    ``env`` is the same per-training-step client the caller will reuse
    for every candidate AND for committing the best action. We never hold
    a second ``EnvClient`` open in parallel: the deployed Space's
    WebSocket gateway drops one session of any same-caller pair, which
    surfaces as ``ConnectionClosedOK`` (close-code 1000) mid-call. See
    CORTEX_FIX_DIAGNOSIS.md. Cost is ``(GROUP_SIZE + 1)`` resets per tick
    rather than 1; this matches the one-client-at-a-time lifecycle in
    ``minimal_proof.py`` and ``collect_b3_corpus.py``.
    """
    from CrisisWorldCortex.models import CrisisworldcortexAction

    brain = parse_router_choice(completion)
    if brain not in _BRAIN_CHOICES:
        return CandidateResult(0.0, None, None, completion, False)
    action_payload = brains[brain].act(observation_text)
    if action_payload is None:
        return CandidateResult(0.0, brain, None, completion, False)
    try:
        replay_prefix(env, task, seed, prefix_actions)
        result = env.step(CrisisworldcortexAction(action=action_payload))
        obs = normalize_step_result(result)
        reward = obs.reward if obs.reward is not None else 0.0
        accepted = bool(obs.recent_action_log and obs.recent_action_log[-1].accepted)
        return CandidateResult(
            float(reward),
            brain,
            CrisisworldcortexAction(action=action_payload),
            completion,
            accepted,
        )
    except Exception as exc:
        log(f"WARN candidate failed brain={brain}: {exc}")
        return CandidateResult(0.0, brain, None, completion, False)


def train_step(
    *,
    router_model: Any,
    router_tokenizer: Any,
    optimizer: Any,
    task: str,
    seed: int,
    prefix_actions: List[Any],
    observation_text: str,
    brains: Dict[str, FrozenBrain],
    env: Any,
) -> tuple[float, List[CandidateResult]]:
    import torch

    generated, prompt_len, completions = sample_router(
        router_model, router_tokenizer, observation_text
    )
    results = [
        score_candidate(
            env=env,
            task=task,
            seed=seed,
            prefix_actions=prefix_actions,
            observation_text=observation_text,
            completion=completion,
            brains=brains,
        )
        for completion in completions
    ]
    rewards = torch.tensor(
        [result.reward for result in results],
        dtype=torch.float32,
        device=next(router_model.parameters()).device,
    )
    advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)
    logprobs = completion_logprobs(router_model, router_tokenizer, generated, prompt_len)
    loss = -(advantages.detach() * logprobs).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(router_model.parameters(), GRAD_CLIP)
    optimizer.step()
    return float(loss.detach().cpu()), results


def save_router(router_model: Any, router_tokenizer: Any, output_dir: str) -> None:
    log(f"saving router LoRA to {output_dir}")
    router_model.save_pretrained(output_dir)
    router_tokenizer.save_pretrained(output_dir)


def main() -> int:
    log(f"EPI_BRAIN_MODEL={EPI_BRAIN_MODEL}")
    log(f"LOGISTICS_BRAIN_MODEL={LOGISTICS_BRAIN_MODEL}")
    log(f"GOVERNANCE_BRAIN_MODEL={GOVERNANCE_BRAIN_MODEL}")
    log(f"ROUTER_MODEL={ROUTER_MODEL} ROUTER_WARMSTART_REPO={ROUTER_WARMSTART_REPO}")
    log(f"MAX_TRAIN_STEPS={MAX_TRAIN_STEPS} GROUP_SIZE={GROUP_SIZE} LR={LR}")

    preflight_env_health(ENV_URL)
    for model_name in {
        EPI_BRAIN_MODEL,
        LOGISTICS_BRAIN_MODEL,
        GOVERNANCE_BRAIN_MODEL,
        ROUTER_MODEL,
    }:
        preflight_model_access(model_name, HF_TOKEN)
    if ROUTER_WARMSTART_REPO:
        preflight_model_access(ROUTER_WARMSTART_REPO, HF_TOKEN)
    check_memory_budget()

    import torch
    from huggingface_hub import HfApi

    random.seed(SEED)
    torch.manual_seed(SEED)
    tasks = tuple(t.strip() for t in TASKS_CSV.split(",") if t.strip())
    if not tasks:
        raise SystemExit("[FATAL] TASKS_CSV produced no tasks")

    router_model, router_tokenizer = load_router(torch)
    brains = load_brains(torch)
    optimizer = torch.optim.AdamW((p for p in router_model.parameters() if p.requires_grad), lr=LR)
    rng = random.Random(SEED)

    update_step = 0
    recent_rewards: List[float] = []
    recent_parse_ok: List[float] = []
    while update_step < MAX_TRAIN_STEPS:
        task = rng.choice(tasks)
        seed = rng.randint(0, 10_000_000)
        # One env client per training step. Reused for candidate scoring
        # (reset + replay + step) and for committing the best action
        # (reset + replay + step). Never two concurrent EnvClient sessions
        # — see CORTEX_FIX_DIAGNOSIS.md.
        env = make_env()
        prefix_actions: List[Any] = []
        try:
            obs = normalize_step_result(
                env.reset(task_name=task, seed=seed, max_ticks=EPISODE_TICKS)
            )
            last_reward = 0.0
            for _tick in range(EPISODE_TICKS):
                observation_text = serialize_observation(obs, last_reward)
                loss, results = train_step(
                    router_model=router_model,
                    router_tokenizer=router_tokenizer,
                    optimizer=optimizer,
                    task=task,
                    seed=seed,
                    prefix_actions=prefix_actions,
                    observation_text=observation_text,
                    brains=brains,
                    env=env,
                )
                update_step += 1
                best = max(results, key=lambda result: result.reward)
                parse_rate = sum(result.brain is not None for result in results) / len(results)
                recent_rewards.append(sum(result.reward for result in results) / len(results))
                recent_parse_ok.append(parse_rate)
                # The policy update scores all sampled router choices from the
                # same state. For the next prefix, keep the best sampled action:
                # this turns each episode into a cheap on-policy beam of width
                # GROUP_SIZE while still applying exactly one env action per tick.
                if best.action is None:
                    from CrisisWorldCortex.models import CrisisworldcortexAction, NoOp

                    best_action = CrisisworldcortexAction(action=NoOp())
                else:
                    best_action = best.action
                # Commit best action on the same env client. Candidate
                # scoring left the env at an arbitrary post-step state, so
                # we reset + replay the prefix before stepping. (GROUP_SIZE
                # + 1) resets per tick total.
                replay_prefix(env, task, seed, prefix_actions)
                obs = normalize_step_result(env.step(best_action))
                prefix_actions.append(best_action)
                last_reward = obs.reward if obs.reward is not None else 0.0
                if update_step % LOG_STEPS == 0:
                    mean_reward = sum(recent_rewards[-LOG_STEPS:]) / min(
                        len(recent_rewards), LOG_STEPS
                    )
                    mean_parse = sum(recent_parse_ok[-LOG_STEPS:]) / min(
                        len(recent_parse_ok), LOG_STEPS
                    )
                    log(
                        f"step={update_step}/{MAX_TRAIN_STEPS} task={task} "
                        f"loss={loss:.4f} group_reward={mean_reward:.3f} "
                        f"parse_success={mean_parse:.0%} best_brain={best.brain}"
                    )
                if SAVE_STEPS > 0 and update_step % SAVE_STEPS == 0:
                    save_router(
                        router_model, router_tokenizer, f"{OUTPUT_DIR}/checkpoint-{update_step}"
                    )
                if obs.done or update_step >= MAX_TRAIN_STEPS:
                    break
        finally:
            env.close()

    save_router(router_model, router_tokenizer, OUTPUT_DIR)
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
    log("training complete")
    return 0


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    log(f"elapsed={time.time() - t0:.1f}s")
    sys.exit(rc)

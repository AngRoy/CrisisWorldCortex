"""B6 - Cortex with a trained LLM brain-selector router.

The Phase 3 router is trained to emit exactly one of:
``{"brain": "epi" | "logistics" | "governance"}``.
That is a direct specialist-selection policy, not the older B3
metacognition router shape. This agent keeps the B3 public surface
(``run_episode``) but swaps the deterministic all-brain Council pass for
a learned brain selector followed by one frozen specialist Brain pass.
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional, Protocol

from baselines.flat_agent import B1StepEvent, ErrorKind, StepCallback, parse_failure_marker
from cortex.brains import EpiBrain, GovernanceBrain, LogisticsBrain
from CrisisWorldCortex.models import (
    CrisisworldcortexAction,
    CrisisworldcortexObservation,
    NoOp,
)


class _EnvLike(Protocol):
    def reset(self) -> CrisisworldcortexObservation: ...

    def step(self, action: CrisisworldcortexAction) -> CrisisworldcortexObservation: ...


_BRAIN_ALIASES = {
    "epi": "epidemiology",
    "epidemiology": "epidemiology",
    "logistics": "logistics",
    "governance": "governance",
}


_ROUTER_SYSTEM_PROMPT = textwrap.dedent(
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


def _extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
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
    data = _extract_json(raw_text)
    if data is None:
        return None
    return _BRAIN_ALIASES.get(str(data.get("brain", "")).strip().lower())


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


def serialize_observation(obs: CrisisworldcortexObservation, last_reward: float = 0.0) -> str:
    parts: List[str] = [
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


class _LocalRouter:
    """Lazy local LoRA inference wrapper.

    Tests can instantiate B6 without downloading models. The actual
    Transformers/PEFT load happens on the first episode tick, which is the
    path used by the human's smoke test after a trained adapter exists.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        hf_token: Optional[str] = None,
        load_in_4bit: bool = True,
        max_prompt_len: int = 2048,
        max_new_tokens: int = 32,
    ) -> None:
        self.repo_id = repo_id
        self.base_model = base_model
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.load_in_4bit = load_in_4bit
        self.max_prompt_len = max_prompt_len
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: Dict[str, Any] = {"token": self.hf_token}
        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.bfloat16
        if self.load_in_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["device_map"] = "auto"
        base = AutoModelForCausalLM.from_pretrained(self.base_model, **kwargs)
        if not torch.cuda.is_available():
            base = base.to("cpu")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, token=self.hf_token)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._model = PeftModel.from_pretrained(base, self.repo_id)
        self._model.eval()
        self._tokenizer = tokenizer

    def select_brain(
        self, obs: CrisisworldcortexObservation, last_reward: float
    ) -> tuple[Optional[str], str]:
        import torch

        self._ensure_loaded()
        prompt = self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": serialize_observation(obs, last_reward)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        device = next(self._model.parameters()).device
        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_len,
        ).to(device)
        prompt_len = int(encoded["input_ids"].shape[1])
        with torch.no_grad():
            out = self._model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        raw = self._tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        return parse_router_choice(raw), raw


class B6CortexTrainedRouter:
    """Cortex specialist selector driven by a trained router LoRA."""

    CALLER_ID_PREFIX = "b6"

    def __init__(
        self,
        env: _EnvLike,
        llm: Any,
        *,
        router_repo: str,
        router_base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    ) -> None:
        self._env = env
        self._llm = llm
        self._router = _LocalRouter(router_repo, base_model=router_base_model)
        self._brains = {
            "epidemiology": EpiBrain(llm),
            "logistics": LogisticsBrain(llm),
            "governance": GovernanceBrain(llm),
        }

    def run_episode(
        self,
        task: str,
        seed: int,
        max_ticks: int = 12,
        *,
        step_callback: Optional[StepCallback] = None,
    ) -> Dict[str, Any]:
        if hasattr(self._llm, "reset_counters"):
            self._llm.reset_counters(caller_id_prefix=f"{self.CALLER_ID_PREFIX}:")
            self._llm.reset_counters(caller_id_prefix="cortex:")

        obs = self._env.reset()
        last_reward = 0.0
        rewards: List[float] = []
        action_history: List[Dict[str, Any]] = []
        steps_taken = 0
        parse_failure_count = 0

        for tick in range(1, max_ticks + 1):
            steps_taken = tick
            tick_error: Optional[ErrorKind] = None
            raw_router = ""
            parse_failure = False
            try:
                brain_id, raw_router = self._router.select_brain(obs, last_reward)
                if brain_id is None:
                    parse_failure = True
                    parse_failure_count += 1
                    tick_error = "parse_failure"
                    wire_action = CrisisworldcortexAction(action=parse_failure_marker())
                else:
                    recommendation = self._brains[brain_id].run_tick(obs, last_reward, tick)
                    wire_action = CrisisworldcortexAction(action=recommendation.top_action)
            except Exception as exc:
                print(
                    f"[WARN] b6: trained router/brain failed at tick={tick}: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
                tick_error = "llm_call_failed"
                wire_action = CrisisworldcortexAction(action=NoOp())

            obs = self._env.step(wire_action)
            current_reward = obs.reward if obs.reward is not None else 0.0
            rewards.append(current_reward)
            if step_callback is not None:
                step_callback(
                    B1StepEvent(
                        tick=tick,
                        action=wire_action.action,
                        reward=current_reward,
                        done=obs.done,
                        error=tick_error,
                        parse_failure=parse_failure,
                        raw_llm=raw_router,
                    )
                )
            action_history.append({"tick": tick, "kind": wire_action.action.kind, "accepted": True})
            if obs.done:
                break
            last_reward = current_reward

        return {
            "task": task,
            "seed": seed,
            "rewards": rewards,
            "action_history": action_history,
            "steps_taken": steps_taken,
            "parse_failure_count": parse_failure_count,
        }

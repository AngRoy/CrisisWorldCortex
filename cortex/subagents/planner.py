"""Planner LLM subagent.

Phase A docs/CORTEX_ARCHITECTURE.md §9 Decision 2: SYS = role + action
schema (B1's shape); USR = perception + WM BeliefState (full JSON if
provided per M-FR-4) + last_reward.
"""

from __future__ import annotations

from typing import ClassVar, List

from pydantic import TypeAdapter

from cortex.schemas import CandidatePlan, SubagentInput
from CrisisWorldCortex.models import NoOp

from ._base import _LLMSubagent, load_prompt

_PLAN_ADAPTER: TypeAdapter[CandidatePlan] = TypeAdapter(CandidatePlan)


class PlannerSubagent(_LLMSubagent):
    """LLM subagent that emits ``CandidatePlan`` for one brain per call."""

    _role_name: ClassVar[str] = "planner"
    _output_type: ClassVar[type] = CandidatePlan
    _system_prompt_filename: ClassVar[str] = "planner.txt"
    _SYSTEM_PROMPT_TEMPLATE: ClassVar[str] = load_prompt("planner.txt")
    _ADAPTER: ClassVar[TypeAdapter] = _PLAN_ADAPTER

    def _build_user_message(self, input: SubagentInput) -> str:
        sections: List[str] = []
        sections.append(f"# Perception\n{input.perception.model_dump_json(indent=2)}")
        if input.prior_belief is not None:
            sections.append(
                "# BeliefState (from this brain's WorldModeler)\n"
                f"{input.prior_belief.model_dump_json(indent=2)}"
            )
        sections.append(f"# Last tick reward: {input.last_reward}")
        sections.append(
            f"# Recent action log: {self._format_action_log(input.recent_action_log_excerpt)}"
        )
        return "\n\n".join(sections)

    @classmethod
    def empty_fallback(cls, brain: str, target_plan_id: str = "") -> CandidatePlan:
        # Phase A Decision 6: NoOp + confidence=0 means "no signal". The
        # Brain Executive's argmax(expected_value * confidence) picks any
        # non-empty plan over this one.
        return CandidatePlan(
            action_sketch="(empty: planner failed to produce a parseable plan)",
            expected_outer_action=NoOp(),
            expected_value=0.0,
            cost=0.0,
            assumptions=[],
            falsifiers=[],
            confidence=0.0,
        )

    def run(self, input: SubagentInput, step_idx: int) -> CandidatePlan:  # type: ignore[override]
        return super().run(input, step_idx)  # type: ignore[return-value]

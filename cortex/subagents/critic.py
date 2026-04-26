"""Critic LLM subagent.

Phase A docs/CORTEX_ARCHITECTURE.md §9 Decision 3: SYS = critic role;
USR = perception + target plan + WM belief (M-FR-5). Critic emits prose
``CriticReport`` only; never proposes alternative actions.
"""

from __future__ import annotations

from typing import ClassVar, List

from pydantic import TypeAdapter

from cortex.schemas import CriticReport, SubagentInput

from ._base import _LLMSubagent, load_prompt

_CRITIC_ADAPTER: TypeAdapter[CriticReport] = TypeAdapter(CriticReport)


class CriticSubagent(_LLMSubagent):
    """LLM subagent that emits ``CriticReport`` for one brain per call."""

    _role_name: ClassVar[str] = "critic"
    _output_type: ClassVar[type] = CriticReport
    _system_prompt_filename: ClassVar[str] = "critic.txt"
    _SYSTEM_PROMPT_TEMPLATE: ClassVar[str] = load_prompt("critic.txt")
    _ADAPTER: ClassVar[TypeAdapter] = _CRITIC_ADAPTER

    def _build_user_message(self, input: SubagentInput) -> str:
        sections: List[str] = []
        sections.append(f"# Perception\n{input.perception.model_dump_json(indent=2)}")
        # M-FR-5: target plan + WM belief, both as full JSON.
        target_json = self._select_target_plan(input)
        sections.append(f"# Target plan (id={input.target_plan_id})\n{target_json}")
        if input.prior_belief is not None:
            sections.append(f"# WM BeliefState\n{input.prior_belief.model_dump_json(indent=2)}")
        # Item B (Phase A review pass): cross-brain Critic. When the Council
        # routes a cross-brain challenge, it sets peer_perception to the
        # challenger's PerceptionReport so the Critic has both domain views.
        if input.peer_perception is not None:
            sections.append(
                f"# Peer perception (challenger {input.peer_perception.brain})\n"
                f"{input.peer_perception.model_dump_json(indent=2)}"
            )
        return "\n\n".join(sections)

    @staticmethod
    def _select_target_plan(input: SubagentInput) -> str:
        """Render the target plan's JSON body for the USR.

        Session 11's Brain Executive populates ``prior_plans`` from the
        Planner's outputs and sets ``target_plan_id`` to identify which
        plan the Critic should attack. Here we render the first plan
        (or a placeholder if none) — Session 11 wires up id-based
        lookup once plans carry ids.
        """
        if not input.prior_plans:
            return "(no target plan provided)"
        return input.prior_plans[0].model_dump_json(indent=2)

    @classmethod
    def empty_fallback(cls, brain: str, target_plan_id: str = "") -> CriticReport:
        # Phase A Decision 6: severity=0 + empty attacks signal "no
        # critique". Brain Executive ignores this Critic's vote weight.
        return CriticReport(
            brain=brain,
            target_plan_id=target_plan_id,
            attacks=[],
            missing_considerations=[],
            would_change_mind_if=[],
            severity=0.0,
        )

    def run(self, input: SubagentInput, step_idx: int) -> CriticReport:  # type: ignore[override]
        return super().run(input, step_idx)  # type: ignore[return-value]

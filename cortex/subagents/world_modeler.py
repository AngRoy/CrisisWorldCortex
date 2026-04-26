"""WorldModeler LLM subagent.

Phase A docs/CORTEX_ARCHITECTURE.md §9 Decision 1: SYS = role + schema;
USR = perception + last_reward + recent_action_log_excerpt (with prior
BeliefState in round 2 per Decision 62).
"""

from __future__ import annotations

from typing import ClassVar, List

from pydantic import TypeAdapter

from cortex.schemas import BeliefState, SubagentInput

from ._base import _LLMSubagent, load_prompt

_BELIEF_ADAPTER: TypeAdapter[BeliefState] = TypeAdapter(BeliefState)
"""Module-level constant per Phase A Decision 5 (encapsulation; avoid
circular imports through the package init)."""


class WorldModelerSubagent(_LLMSubagent):
    """LLM subagent that emits ``BeliefState`` for one brain per call."""

    _role_name: ClassVar[str] = "world_modeler"
    _output_type: ClassVar[type] = BeliefState
    _system_prompt_filename: ClassVar[str] = "world_modeler.txt"
    _SYSTEM_PROMPT_TEMPLATE: ClassVar[str] = load_prompt("world_modeler.txt")
    _ADAPTER: ClassVar[TypeAdapter] = _BELIEF_ADAPTER

    def _build_user_message(self, input: SubagentInput) -> str:
        sections: List[str] = []
        sections.append(f"# Perception\n{input.perception.model_dump_json(indent=2)}")
        if input.prior_belief is not None:
            sections.append(
                "# Prior BeliefState (round 1 result)\n"
                f"{input.prior_belief.model_dump_json(indent=2)}"
            )
        sections.append(f"# Last tick reward: {input.last_reward}")
        sections.append(
            f"# Recent action log: {self._format_action_log(input.recent_action_log_excerpt)}"
        )
        return "\n\n".join(sections)

    @classmethod
    def empty_fallback(cls, brain: str, target_plan_id: str = "") -> BeliefState:
        # Phase A Decision 6 + Decision 62: empty BeliefState as the
        # honest "no signal" state. uncertainty=1.0, no evidence -> r_proto = 0.
        return BeliefState(
            brain=brain,
            latent_estimates={},
            hypotheses=[],
            uncertainty=1.0,
            reducible_by_more_thought=0.0,
            evidence=[],
        )

    # Narrow run() return type for callers (refinement #1).
    def run(self, input: SubagentInput, step_idx: int) -> BeliefState:  # type: ignore[override]
        return super().run(input, step_idx)  # type: ignore[return-value]

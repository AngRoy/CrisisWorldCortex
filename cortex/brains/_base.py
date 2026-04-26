"""Brain class - composes Perception + Lens + 3 Subagents + Brain Executive.

Per cortex/CLAUDE.md: each brain has a deterministic Python Perception
+ Lens, three LLM subagents (router-callable), and a deterministic
Python Brain Executive. The Brain class wires these together.

Multi-model deployment: each Brain holds a SINGLE LLMClient instance
passed at construction. Different brains can use different models by
constructing each Brain with a different LLMClient (e.g., Qwen for epi,
Llama for logistics). NO module-level state, NO shared singletons.
"""

from __future__ import annotations

from typing import List, Literal

from cortex.lenses import lens_for
from cortex.schemas import (
    BeliefState,
    BrainLensedObservation,
    BrainRecommendation,
    CandidatePlan,
    CriticReport,
    PerceptionReport,
    SubagentInput,
)
from cortex.subagents import (
    CriticSubagent,
    PlannerSubagent,
    WorldModelerSubagent,
    perception_for,
)
from cortex.subagents._base import _LLMClientLike
from CrisisWorldCortex.models import CrisisworldcortexObservation

from ._executive import aggregate_brain_outputs

_BrainId = Literal["epidemiology", "logistics", "governance"]


class Brain:
    """Per-brain pipeline holder.

    Each Brain instance owns its own LLMClient. The orchestration layer
    (Session 12 Council, Workstream B trainers) instantiates one Brain
    per brain id, optionally with different LLMClients pointing to
    different models. The Brain class itself has NO module-level state
    and NO forced singleton.

    Args:
        brain_id: One of "epidemiology", "logistics", "governance".
        llm_client: This brain's LLM client. Subagents are constructed
            with the SAME client so token billing aggregates correctly.
        wm: WorldModeler subagent.
        planner: Planner subagent.
        critic: Critic subagent.
    """

    def __init__(
        self,
        brain_id: _BrainId,
        llm_client: _LLMClientLike,
        wm: WorldModelerSubagent,
        planner: PlannerSubagent,
        critic: CriticSubagent,
    ) -> None:
        self.brain_id = brain_id
        self.llm_client = llm_client
        self.wm = wm
        self.planner = planner
        self.critic = critic

    # ------------------------------------------------------------------
    # Deterministic Python pieces (no LLM)
    # ------------------------------------------------------------------

    def compute_perception(self, obs: CrisisworldcortexObservation) -> PerceptionReport:
        """Run this brain's Perception. Pure Python; no LLM."""
        return perception_for(self.brain_id, obs)

    def compute_lens(
        self, obs: CrisisworldcortexObservation, last_reward: float
    ) -> BrainLensedObservation:
        """Run this brain's Lens. Pure Python; no LLM."""
        return lens_for(self.brain_id, obs, last_reward)

    def aggregate(
        self,
        perception: PerceptionReport,
        beliefs: List[BeliefState],
        plans: List[CandidatePlan],
        critics: List[CriticReport],
        tokens_used: int = 0,
    ) -> BrainRecommendation:
        """Run this brain's Brain Executive. Pure Python; no LLM."""
        return aggregate_brain_outputs(
            brain_id=self.brain_id,
            perception=perception,
            beliefs=beliefs,
            plans=plans,
            critics=critics,
            tokens_used=tokens_used,
        )

    # ------------------------------------------------------------------
    # High-level convenience: round-1 single tick
    # ------------------------------------------------------------------

    def run_tick(
        self,
        obs: CrisisworldcortexObservation,
        last_reward: float,
        tick: int,
        round_: int = 1,
    ) -> BrainRecommendation:
        """Round-1 single-tick pipeline (Session 11 smoke).

        Round 2 is orchestrated by the Council Executive (Session 12)
        via the fine-grained methods (compute_perception, compute_lens,
        wm.run / planner.run / critic.run, aggregate). Calling this
        convenience method with ``round_!=1`` raises NotImplementedError
        to prevent accidental misuse before the Council exists.
        """
        if round_ != 1:
            raise NotImplementedError(
                f"Round {round_} orchestration is the Council Executive's "
                f"responsibility (Session 12). Use Brain.compute_perception/"
                f"compute_lens + WorldModelerSubagent.run/PlannerSubagent.run/"
                f"CriticSubagent.run + Brain.aggregate directly."
            )

        perception = self.compute_perception(obs)
        # Lens is computed for completeness; Session 11 doesn't yet plumb
        # it into SubagentInput (M-FR-4 step indices fixed). Session 12
        # Council will extend the SubagentInput contract to carry lens
        # output if subagents need it.
        _ = self.compute_lens(obs, last_reward)

        # WorldModeler (step_idx=0)
        wm_input = SubagentInput(
            brain=self.brain_id,
            role="world_modeler",
            tick=tick,
            round=round_,
            perception=perception,
            prior_belief=None,
            prior_plans=[],
            target_plan_id=None,
            last_reward=last_reward,
            recent_action_log_excerpt=list(obs.recent_action_log),
        )
        belief = self.wm.run(wm_input, step_idx=0)

        # Planner (step_idx=1)
        planner_input = SubagentInput(
            brain=self.brain_id,
            role="planner",
            tick=tick,
            round=round_,
            perception=perception,
            prior_belief=belief,
            prior_plans=[],
            target_plan_id=None,
            last_reward=last_reward,
            recent_action_log_excerpt=list(obs.recent_action_log),
        )
        plan = self.planner.run(planner_input, step_idx=1)

        # Critic (step_idx=2)
        critic_input = SubagentInput(
            brain=self.brain_id,
            role="critic",
            tick=tick,
            round=round_,
            perception=perception,
            prior_belief=belief,
            prior_plans=[plan],
            target_plan_id="plan-0",
            last_reward=last_reward,
            recent_action_log_excerpt=list(obs.recent_action_log),
        )
        critic = self.critic.run(critic_input, step_idx=2)

        # Tally tokens billed to this brain's caller_ids.
        caller_id_base = f"cortex:{self.brain_id}"
        tokens_used = sum(
            self.llm_client.tokens_used_for(f"{caller_id_base}:{role}:t{tick}:r{round_}:s{idx}")
            for role, idx in (
                ("world_modeler", 0),
                ("planner", 1),
                ("critic", 2),
            )
        )

        return self.aggregate(
            perception=perception,
            beliefs=[belief],
            plans=[plan],
            critics=[critic],
            tokens_used=tokens_used,
        )

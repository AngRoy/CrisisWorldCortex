"""Council Executive - 4-phase machine + hard caps + 5-step anti-hivemind protocol.

Per Phase A docs/CORTEX_ARCHITECTURE.md sections 3-5 + Decisions 22-31
+ Items A/B/C/F (Phase A review pass).

Council orchestrates:
  - Per-brain Perception at tick start (deterministic Python).
  - Round-1 fixed-order subagent calls (epi -> logistics -> governance,
    each WM -> Planner -> Critic). Decision 38 round-1 sequence.
  - Brain Executive aggregation per brain.
  - Router-loop for high-level decisions: emit / challenge / round-2 /
    preserve_dissent / extra subagent call. Each loop iteration validates
    the router's action against hard caps and overrides on violation.
  - Cross-brain Critic with peer perception (Item B / Decision 27).
  - Phase machine: Divergence -> Challenge -> Narrowing -> Convergence
    (Item F mapping).

Multi-model orchestration: Council holds ``Dict[brain_id, Brain]`` where
each Brain owns its own LLMClient. Workstream B's mixed-model deployment
(Qwen for epi/governance, Llama for logistics) is supported by passing
the right Brain instances at construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

from cortex.anti_hivemind import format_dissent_tag
from cortex.brains import Brain
from cortex.metacognition import compute_metacognition_state
from cortex.schemas import (
    BeliefState,
    BrainRecommendation,
    CandidatePlan,
    CriticReport,
    EpistemicPhase,
    MetacognitionState,
    PerceptionReport,
    RoutingAction,
    SubagentInput,
)
from CrisisWorldCortex.models import (
    CrisisworldcortexAction,
    CrisisworldcortexObservation,
    NoOp,
    OuterActionPayload,
)

DEFAULT_TICK_BUDGET = 6000  # Phase A section 5
DEFAULT_MAX_TICKS = 12
_MAX_ITERATIONS = 64  # router-loop safety net

_BRAIN_ORDER = ("epidemiology", "logistics", "governance")  # Decision 38 fixed order
_ROLE_ORDER = ("world_modeler", "planner", "critic")
_CROSS_BRAIN_CRITIC_STEP_IDX = 9  # M-FR-3
_ROUND2_STEP_OFFSET = 10  # round-2 step indices: 10/11/12 (avoid s9 collision)


class RoutingPolicy(Protocol):
    """Drop-in compatible across Session-12 _NaiveRouter, Session-13
    deterministic router, and Session-15 trainable router."""

    def forward(self, state: MetacognitionState) -> RoutingAction: ...


class _NaiveRouter:
    """Session 12 placeholder router. Always emits emit_outer_action.

    Session 13 replaces with the Decision-38 deterministic decision table.
    """

    def forward(self, state: MetacognitionState) -> RoutingAction:
        return RoutingAction(kind="emit_outer_action")


@dataclass
class _TickState:
    phase: EpistemicPhase = "Divergence"
    round: int = 1
    deliberation_rounds_used: int = 1
    cross_brain_challenges_used: int = 0
    critic_calls_per_brain: Dict[str, int] = field(default_factory=dict)
    tick_tokens_used: int = 0
    preserved_dissent: List[str] = field(default_factory=list)
    phase_trace: List[str] = field(default_factory=lambda: ["Divergence"])
    challenge_used_this_tick: bool = False


class Council:
    """Council Executive: orchestrates the 5-step anti-hivemind protocol."""

    def __init__(
        self,
        brains: Dict[str, Brain],
        routing_policy: Optional[RoutingPolicy] = None,
        tick_budget: int = DEFAULT_TICK_BUDGET,
        max_ticks: int = DEFAULT_MAX_TICKS,
    ) -> None:
        self.brains = brains
        self.routing_policy = routing_policy or _NaiveRouter()
        self.tick_budget = tick_budget
        self.max_ticks = max_ticks
        self.last_tick_state: Optional[_TickState] = None

    def step(
        self,
        observation: CrisisworldcortexObservation,
        last_reward: float = 0.0,
    ) -> CrisisworldcortexAction:
        """Run one tick of deliberation, return the wire-protocol action."""
        ts = _TickState()
        for bid in self.brains:
            ts.critic_calls_per_brain[bid] = 0

        # Step 1: Perception per brain (deterministic Python)
        perceptions: Dict[str, PerceptionReport] = {
            bid: brain.compute_perception(observation) for bid, brain in self.brains.items()
        }
        worst_region_infection = self._worst_region_infection(observation, last_reward)

        round_outputs: Dict[str, Dict[str, List]] = {
            bid: {"beliefs": [], "plans": [], "critics": []} for bid in self.brains
        }

        # Round 1: deterministic 9-call sequence (Decision 38)
        self._run_round(
            observation=observation,
            last_reward=last_reward,
            tick=observation.tick,
            round_=1,
            perceptions=perceptions,
            round_outputs=round_outputs,
            ts=ts,
            step_offset=0,
        )

        brain_recs = self._aggregate_all(perceptions, round_outputs)

        # Router loop
        for _ in range(_MAX_ITERATIONS):
            metacog = self._build_metacog(
                ts=ts,
                tick=observation.tick,
                ticks_remaining=observation.ticks_remaining,
                brain_recs=brain_recs,
                worst_region_infection=worst_region_infection,
            )
            raw_action = self.routing_policy.forward(metacog)
            action = self._enforce_caps(raw_action, ts)

            if action.kind == "emit_outer_action":
                if ts.phase_trace[-1] != "Convergence":
                    ts.phase_trace.append("Convergence")
                ts.phase = "Convergence"
                self.last_tick_state = ts
                final_action = action.outer_action or self._council_top(brain_recs)
                return CrisisworldcortexAction(action=final_action)

            if action.kind == "stop_and_no_op":
                if ts.phase_trace[-1] != "Convergence":
                    ts.phase_trace.append("Convergence")
                ts.phase = "Convergence"
                self.last_tick_state = ts
                return CrisisworldcortexAction(action=NoOp())

            if action.kind == "switch_phase":
                self._handle_switch_phase(
                    action,
                    ts,
                    perceptions,
                    round_outputs,
                    observation,
                    last_reward,
                )
                if (
                    ts.round == 2
                    and round_outputs[_BRAIN_ORDER[0]]["beliefs"]
                    and len(round_outputs[_BRAIN_ORDER[0]]["beliefs"]) >= 2
                ):
                    brain_recs = self._aggregate_all(perceptions, round_outputs)
                continue

            if action.kind == "preserve_dissent":
                self._handle_preserve_dissent(ts, brain_recs)
                continue

            if action.kind == "request_challenge":
                self._handle_cross_brain_challenge(
                    action,
                    ts,
                    perceptions,
                    round_outputs,
                    brain_recs,
                    observation,
                    last_reward,
                )
                brain_recs = self._aggregate_all(perceptions, round_outputs)
                continue

            if action.kind == "call_subagent":
                self._handle_extra_call_subagent(
                    action,
                    ts,
                    perceptions,
                    round_outputs,
                    observation,
                    last_reward,
                )
                continue

            break

        # Safety net
        if ts.phase_trace[-1] != "Convergence":
            ts.phase_trace.append("Convergence")
        ts.phase = "Convergence"
        self.last_tick_state = ts
        return CrisisworldcortexAction(action=self._council_top(brain_recs))

    def _run_round(
        self,
        *,
        observation: CrisisworldcortexObservation,
        last_reward: float,
        tick: int,
        round_: int,
        perceptions: Dict[str, PerceptionReport],
        round_outputs: Dict[str, Dict[str, List]],
        ts: _TickState,
        step_offset: int,
    ) -> None:
        """Run one deliberation round: 3 brains x 3 subagents = 9 LLM calls."""
        for bid in _BRAIN_ORDER:
            if bid not in self.brains:
                continue
            brain = self.brains[bid]

            prior_belief_for_round2 = (
                round_outputs[bid]["beliefs"][0]
                if round_ == 2 and round_outputs[bid]["beliefs"]
                else None
            )

            wm_input = self._make_subagent_input(
                bid,
                "world_modeler",
                tick,
                round_,
                perceptions[bid],
                prior_belief=prior_belief_for_round2,
                prior_plans=[],
                target_plan_id=None,
                last_reward=last_reward,
                obs=observation,
            )
            belief = brain.wm.run(wm_input, step_idx=step_offset + 0)
            assert isinstance(belief, BeliefState)
            round_outputs[bid]["beliefs"].append(belief)

            planner_input = self._make_subagent_input(
                bid,
                "planner",
                tick,
                round_,
                perceptions[bid],
                prior_belief=belief,
                prior_plans=[],
                target_plan_id=None,
                last_reward=last_reward,
                obs=observation,
            )
            plan = brain.planner.run(planner_input, step_idx=step_offset + 1)
            assert isinstance(plan, CandidatePlan)
            round_outputs[bid]["plans"].append(plan)

            critic_input = self._make_subagent_input(
                bid,
                "critic",
                tick,
                round_,
                perceptions[bid],
                prior_belief=belief,
                prior_plans=[plan],
                target_plan_id="plan-0",
                last_reward=last_reward,
                obs=observation,
            )
            critic = brain.critic.run(critic_input, step_idx=step_offset + 2)
            assert isinstance(critic, CriticReport)
            round_outputs[bid]["critics"].append(critic)
            ts.critic_calls_per_brain[bid] = ts.critic_calls_per_brain.get(bid, 0) + 1

            for role, idx in zip(_ROLE_ORDER, range(3)):
                caller_id = f"cortex:{bid}:{role}:t{tick}:r{round_}:s{step_offset + idx}"
                ts.tick_tokens_used += brain.llm_client.tokens_used_for(caller_id)

    def _handle_switch_phase(
        self,
        action: RoutingAction,
        ts: _TickState,
        perceptions: Dict[str, PerceptionReport],
        round_outputs: Dict[str, Dict[str, List]],
        observation: CrisisworldcortexObservation,
        last_reward: float,
    ) -> None:
        new_phase = action.new_phase
        if new_phase is None:
            return

        if new_phase == "Divergence":
            # Round 2 entry per Decision 61 (explicit only)
            ts.round = 2
            ts.deliberation_rounds_used = 2
            ts.phase = "Divergence"
            if ts.phase_trace[-1] != "Divergence":
                ts.phase_trace.append("Divergence")
            self._run_round(
                observation=observation,
                last_reward=last_reward,
                tick=observation.tick,
                round_=2,
                perceptions=perceptions,
                round_outputs=round_outputs,
                ts=ts,
                step_offset=_ROUND2_STEP_OFFSET,
            )
        elif new_phase == "Challenge":
            ts.phase = "Challenge"
            if ts.phase_trace[-1] != "Challenge":
                ts.phase_trace.append("Challenge")
        elif new_phase == "Narrowing":
            ts.phase = "Narrowing"
            if ts.phase_trace[-1] != "Narrowing":
                ts.phase_trace.append("Narrowing")
        elif new_phase == "Convergence":
            ts.phase = "Convergence"
            if ts.phase_trace[-1] != "Convergence":
                ts.phase_trace.append("Convergence")

    def _handle_cross_brain_challenge(
        self,
        action: RoutingAction,
        ts: _TickState,
        perceptions: Dict[str, PerceptionReport],
        round_outputs: Dict[str, Dict[str, List]],
        brain_recs: Dict[str, BrainRecommendation],
        observation: CrisisworldcortexObservation,
        last_reward: float,
    ) -> None:
        challenger_bid = action.brain
        target_bid = action.target_brain
        if challenger_bid is None or target_bid is None:
            if not brain_recs:
                return
            target_bid = max(brain_recs, key=lambda b: brain_recs[b].top_confidence)
            challenger_bid = min(brain_recs, key=lambda b: brain_recs[b].top_confidence)

        if challenger_bid not in self.brains or target_bid not in self.brains:
            return

        if ts.phase_trace[-1] != "Challenge":
            ts.phase_trace.append("Challenge")
        ts.phase = "Challenge"

        challenger = self.brains[challenger_bid]
        target_outputs = round_outputs[target_bid]
        if not target_outputs["plans"] or not target_outputs["beliefs"]:
            return
        target_plan = target_outputs["plans"][-1]
        target_belief = target_outputs["beliefs"][-1]

        critic_input = SubagentInput(
            brain=challenger_bid,  # type: ignore[arg-type]
            role="critic",
            tick=observation.tick,
            round=ts.round,
            perception=perceptions[target_bid],
            prior_belief=target_belief,
            prior_plans=[target_plan],
            target_plan_id="plan-0",
            last_reward=last_reward,
            recent_action_log_excerpt=list(observation.recent_action_log),
            peer_perception=perceptions[challenger_bid],
        )
        cross_critic = challenger.critic.run(critic_input, step_idx=_CROSS_BRAIN_CRITIC_STEP_IDX)
        round_outputs[target_bid]["critics"].append(cross_critic)
        ts.cross_brain_challenges_used += 1
        ts.challenge_used_this_tick = True

        caller_id = (
            f"cortex:{challenger_bid}:critic:t{observation.tick}"
            f":r{ts.round}:s{_CROSS_BRAIN_CRITIC_STEP_IDX}"
        )
        ts.tick_tokens_used += challenger.llm_client.tokens_used_for(caller_id)

        if ts.phase_trace[-1] != "Narrowing":
            ts.phase_trace.append("Narrowing")
        ts.phase = "Narrowing"

    def _handle_extra_call_subagent(
        self,
        action: RoutingAction,
        ts: _TickState,
        perceptions: Dict[str, PerceptionReport],
        round_outputs: Dict[str, Dict[str, List]],
        observation: CrisisworldcortexObservation,
        last_reward: float,
    ) -> None:
        """Router-emitted call_subagent beyond the deterministic round-1 9 calls.

        Cap-enforcement already happened in _enforce_caps; this just
        executes the call. Used by Session-13 trainable router that
        wants to re-call a specific subagent.
        """
        bid = action.brain
        role = action.subagent
        if bid is None or role is None or bid not in self.brains:
            return
        brain = self.brains[bid]
        outputs = round_outputs[bid]
        prior_belief = outputs["beliefs"][-1] if outputs["beliefs"] else None
        prior_plans = list(outputs["plans"])
        sub_input = self._make_subagent_input(
            bid,
            role,
            observation.tick,
            ts.round,
            perceptions[bid],
            prior_belief=prior_belief,
            prior_plans=prior_plans,
            target_plan_id="plan-0" if role == "critic" else None,
            last_reward=last_reward,
            obs=observation,
        )
        bonus_idx = 100 + len(outputs["beliefs"]) + len(outputs["plans"]) + len(outputs["critics"])
        if role == "world_modeler":
            outputs["beliefs"].append(brain.wm.run(sub_input, step_idx=bonus_idx))
        elif role == "planner":
            outputs["plans"].append(brain.planner.run(sub_input, step_idx=bonus_idx))
        elif role == "critic":
            outputs["critics"].append(brain.critic.run(sub_input, step_idx=bonus_idx))
            ts.critic_calls_per_brain[bid] = ts.critic_calls_per_brain.get(bid, 0) + 1
        caller_id = f"cortex:{bid}:{role}:t{observation.tick}:r{ts.round}:s{bonus_idx}"
        ts.tick_tokens_used += brain.llm_client.tokens_used_for(caller_id)

    def _handle_preserve_dissent(
        self, ts: _TickState, brain_recs: Dict[str, BrainRecommendation]
    ) -> None:
        if not brain_recs:
            return
        council_top = self._council_top(brain_recs)
        chosen_minority: Optional[str] = None
        for bid, rec in brain_recs.items():
            if rec.top_action.kind != council_top.kind:
                chosen_minority = bid
                break
        if chosen_minority is None:
            chosen_minority = min(brain_recs, key=lambda b: brain_recs[b].top_confidence)
        rec = brain_recs[chosen_minority]
        tag = format_dissent_tag(chosen_minority, rec.top_action.kind, rec.reasoning_summary)
        ts.preserved_dissent.append(tag)

    def _enforce_caps(self, action: RoutingAction, ts: _TickState) -> RoutingAction:
        # Budget check first (Phase A section 5)
        if ts.tick_tokens_used >= self.tick_budget and action.kind not in (
            "emit_outer_action",
            "stop_and_no_op",
        ):
            return RoutingAction(kind="emit_outer_action")

        # Deliberation rounds cap (Phase A section 4: 2 rounds max)
        if action.kind == "switch_phase" and action.new_phase == "Divergence":
            if ts.deliberation_rounds_used >= 2:
                return RoutingAction(kind="switch_phase", new_phase="Convergence")

        # Cross-brain challenge cap (1/tick total)
        if action.kind == "request_challenge":
            if ts.cross_brain_challenges_used >= 1:
                return RoutingAction(kind="stop_and_no_op")

        # Critic-per-brain cap (1/brain/tick)
        if action.kind == "call_subagent" and action.subagent == "critic":
            brain = action.brain or ""
            if ts.critic_calls_per_brain.get(brain, 0) >= 1:
                return RoutingAction(kind="stop_and_no_op")

        return action

    def _make_subagent_input(
        self,
        brain_id: str,
        role: str,
        tick: int,
        round_: int,
        perception: PerceptionReport,
        *,
        prior_belief: Optional[BeliefState],
        prior_plans: List[CandidatePlan],
        target_plan_id: Optional[str],
        last_reward: float,
        obs: CrisisworldcortexObservation,
    ) -> SubagentInput:
        return SubagentInput(
            brain=brain_id,  # type: ignore[arg-type]
            role=role,  # type: ignore[arg-type]
            tick=tick,
            round=round_,
            perception=perception,
            prior_belief=prior_belief,
            prior_plans=prior_plans,
            target_plan_id=target_plan_id,
            last_reward=last_reward,
            recent_action_log_excerpt=list(obs.recent_action_log),
        )

    def _aggregate_all(
        self,
        perceptions: Dict[str, PerceptionReport],
        round_outputs: Dict[str, Dict[str, List]],
    ) -> Dict[str, BrainRecommendation]:
        out: Dict[str, BrainRecommendation] = {}
        for bid, brain in self.brains.items():
            outputs = round_outputs[bid]
            out[bid] = brain.aggregate(
                perception=perceptions[bid],
                beliefs=outputs["beliefs"],
                plans=outputs["plans"],
                critics=outputs["critics"],
                tokens_used=0,
            )
        return out

    def _council_top(self, brain_recs: Dict[str, BrainRecommendation]) -> OuterActionPayload:
        """Decision 24-25: weighted vote, returns winning brain's top_action."""
        if not brain_recs:
            return NoOp()
        weighted = {
            bid: rec.top_confidence * max(1, len(rec.evidence)) for bid, rec in brain_recs.items()
        }
        chosen = max(brain_recs, key=lambda b: weighted[b])
        return brain_recs[chosen].top_action

    def _build_metacog(
        self,
        *,
        ts: _TickState,
        tick: int,
        ticks_remaining: int,
        brain_recs: Dict[str, BrainRecommendation],
        worst_region_infection: float,
    ) -> MetacognitionState:
        return compute_metacognition_state(
            tick=tick,
            round_=ts.round,
            phase=ts.phase,
            brain_recommendations=brain_recs,
            tick_tokens_used=ts.tick_tokens_used,
            tick_budget=self.tick_budget,
            ticks_remaining=ticks_remaining,
            max_ticks=self.max_ticks,
            worst_region_infection=worst_region_infection,
            preserved_dissent_count=len(ts.preserved_dissent),
            challenge_used_this_tick=ts.challenge_used_this_tick,
        )

    def _worst_region_infection(
        self, observation: CrisisworldcortexObservation, last_reward: float
    ) -> float:
        if "epidemiology" not in self.brains:
            return 0.0
        try:
            lensed = self.brains["epidemiology"].compute_lens(observation, last_reward)
        except Exception:
            return 0.0
        return float(lensed.derived_features.get("worst_region_infection", 0.0))

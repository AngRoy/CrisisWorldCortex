"""B3 - Cortex with a hand-coded (deterministic, non-learned) router.

Per baselines/CLAUDE.md + Phase A docs/CORTEX_ARCHITECTURE.md
Decisions 51-55 + the user's Session 13 proposal acceptance.

B3 composes ``Council(routing_policy=DeterministicRouter())`` directly
per Decision 53 and exposes the same ``run_episode`` surface as B1/B2
(``B1StepEvent`` per tick, optional ``step_callback``) per Decisions
54-55. Production B3 uses HTTP CrisisworldcortexEnv per
baselines/CLAUDE.md; tests use an in-process adapter.

Multi-model deployment (Workstream B): for mixed-LLM brains, construct
the Council externally and pass it via ``B3CortexFixedRouter.from_council``.
The default constructor uses one shared LLMClient across the 3 brains.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Protocol

from baselines.flat_agent import B1StepEvent, ErrorKind, StepCallback
from cortex.brains import EpiBrain, GovernanceBrain, LogisticsBrain
from cortex.council import Council
from cortex.routing_policy import DeterministicRouter
from CrisisWorldCortex.models import (
    CrisisworldcortexAction,
    CrisisworldcortexObservation,
    NoOp,
)


class _EnvLike(Protocol):
    """Minimal env interface B3 consumes; matches B1's _EnvLike."""

    def reset(self) -> CrisisworldcortexObservation: ...

    def step(self, action: CrisisworldcortexAction) -> CrisisworldcortexObservation: ...


class B3CortexFixedRouter:
    """Cortex with the deterministic (non-learned) router.

    Same construction shape as B1 ``(env, llm)``. Internally builds 3
    brains + Council + DeterministicRouter at __init__ so per-episode
    runs only need to call ``run_episode``.
    """

    CALLER_ID_PREFIX = "b3"

    def __init__(self, env: _EnvLike, llm: Any) -> None:
        self._env = env
        self._llm = llm
        self._council = Council(
            brains={
                "epidemiology": EpiBrain(llm),
                "logistics": LogisticsBrain(llm),
                "governance": GovernanceBrain(llm),
            },
            routing_policy=DeterministicRouter(),
        )

    @classmethod
    def from_council(cls, env: _EnvLike, llm: Any, council: Council) -> "B3CortexFixedRouter":
        """Construct B3 with a pre-built Council (multi-model deployment).

        For Workstream B's mixed-LLM brains (e.g. Qwen for epi, Llama
        for logistics), the caller builds the Council externally with
        per-brain LLMClients and passes it here. The ``llm`` arg stays
        on the instance for token-counter reset semantics; Council
        token billing aggregates per-brain via each Brain's own client.
        """
        instance = cls.__new__(cls)
        instance._env = env
        instance._llm = llm
        instance._council = council
        return instance

    def run_episode(
        self,
        task: str,
        seed: int,
        max_ticks: int = 12,
        *,
        step_callback: Optional[StepCallback] = None,
    ) -> Dict[str, Any]:
        """Run one episode. Returns a B1-shaped trajectory dict.

        Side effects: resets per-caller token counters at the start so
        episodes don't accumulate cross-episode totals. Mirrors B1's
        harness-driven reset per Session 7a section 4.
        """
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

            try:
                wire_action = self._council.step(obs, last_reward=last_reward)
            except Exception as exc:
                # Cortex-internal failure: treat as parse-failure-equivalent
                # so episode keeps running. Matches B1's llm_call_failed.
                print(
                    f"[WARN] b3: council.step failed at tick={tick}: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
                tick_error = "llm_call_failed"
                wire_action = CrisisworldcortexAction(action=NoOp())

            obs = self._env.step(wire_action)
            current_reward = obs.reward if obs.reward is not None else 0.0
            rewards.append(current_reward)

            event = B1StepEvent(
                tick=tick,
                action=wire_action.action,
                reward=current_reward,
                done=obs.done,
                error=tick_error,
                parse_failure=False,
                raw_llm="",
            )
            if step_callback is not None:
                step_callback(event)

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

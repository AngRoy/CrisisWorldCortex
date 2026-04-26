"""Rollout buffer for GRPO training.

Stores per-episode ``TrajectoryStep`` tuples, supports group sampling for
GRPO's relative-advantage computation, and exposes a clear/round-trip
contract for unit tests. Generic shape: works for B1 / B2 / Cortex
trajectories without coupling to any specific agent implementation.

Phase-2 scaffold (Workstream B). The buffer is intentionally minimal —
no batching, no tensor conversion, no on-disk persistence. Those land
in the actual GRPO trainer (``training/train_router.py``) when Session
15 implements it.

Allowed under ``training/CLAUDE.md`` import rules: ``models`` and stdlib
only. No ``cortex/*``, no ``server/*``, no ``baselines/*``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class TrajectoryStep:
    """One (obs, action, reward, log_prob, done) tuple from a rollout.

    ``obs`` and ``action`` are serialised to ``dict`` (via
    ``BaseModel.model_dump()`` at the call site) so the buffer never
    holds Pydantic objects directly — keeps GRPO trainer hot path free
    of validation overhead.

    ``log_prob`` is the policy's log-probability of the chosen action
    under the rollout-time temperature. ``None`` for non-stochastic
    baselines (e.g. B1 with temperature=0).
    """

    obs: dict
    action: dict
    reward: float
    log_prob: Optional[float]
    done: bool


@dataclass
class RolloutBuffer:
    """Per-episode rollout storage with GRPO group-sampling support.

    Episodes are keyed by an arbitrary ``episode_id`` string; the trainer
    is responsible for choosing IDs (typically ``f"{task}:{seed}:{run}"``).
    """

    _episodes: dict[str, list[TrajectoryStep]] = field(default_factory=dict)

    def add_step(self, episode_id: str, step: TrajectoryStep) -> None:
        """Append one step to the named episode (creates it if absent)."""
        self._episodes.setdefault(episode_id, []).append(step)

    def get_episode(self, episode_id: str) -> list[TrajectoryStep]:
        """Return the step list for ``episode_id`` (empty if unknown)."""
        return self._episodes.get(episode_id, [])

    def episode_ids(self) -> list[str]:
        """Return all episode IDs currently in the buffer."""
        return list(self._episodes.keys())

    def episode_return(self, episode_id: str) -> float:
        """Sum of rewards for the named episode."""
        return sum(s.reward for s in self.get_episode(episode_id))

    def sample_group(self, group_size: int, rng: Optional[random.Random] = None) -> list[str]:
        """Sample ``group_size`` episode IDs without replacement for GRPO.

        GRPO's relative-advantage step requires a group of trajectories
        from the same prompt; the trainer typically calls this once per
        update step. Returns episode IDs (not the full step lists) so
        the trainer can decide how to slice them into tensors.

        Raises ``ValueError`` if ``group_size`` exceeds the buffer size.
        """
        if group_size > len(self._episodes):
            raise ValueError(f"group_size={group_size} exceeds buffer size={len(self._episodes)}")
        rng = rng or random.Random()
        return rng.sample(list(self._episodes.keys()), group_size)

    def clear(self) -> None:
        """Drop all episodes. Called between GRPO update steps."""
        self._episodes.clear()

    def __len__(self) -> int:
        """Number of episodes currently stored."""
        return len(self._episodes)


__all__ = ["RolloutBuffer", "TrajectoryStep"]

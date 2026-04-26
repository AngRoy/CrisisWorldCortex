"""Training package: rollout buffer, reward shaping, eval metrics.

Phase-2 scaffold (Workstream B). The actual GRPO trainer
(``training/train_router.py``) lands in Session 15 of Workstream A.
"""

from .eval_metrics import (
    collapse_rate,
    consensus_calibration,
    dissent_value,
    novelty_yield,
)
from .reward_shaping import (
    DEFAULT_LAMBDA_BUDGET,
    DEFAULT_TICK_BUDGET,
    compose_episode_return,
    shape_reward,
)
from .rollout_buffer import RolloutBuffer, TrajectoryStep

__all__ = [
    "DEFAULT_LAMBDA_BUDGET",
    "DEFAULT_TICK_BUDGET",
    "RolloutBuffer",
    "TrajectoryStep",
    "collapse_rate",
    "compose_episode_return",
    "consensus_calibration",
    "dissent_value",
    "novelty_yield",
    "shape_reward",
]

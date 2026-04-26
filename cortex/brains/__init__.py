"""Cortex brains (Session 11+).

Public surface:
    - Brain: per-brain class wiring Perception + Lens + 3 LLM subagents + Brain Executive.
    - EpiBrain, LogisticsBrain, GovernanceBrain: factory functions.
    - aggregate_brain_outputs: Brain Executive aggregation function.

Each Brain holds its own LLMClient instance; the orchestration layer
(Council Executive in Session 12, Workstream B trainers) constructs one
Brain per brain id, optionally with different LLMClients pointing to
different models. NO module-level state.
"""

from ._base import Brain
from ._executive import aggregate_brain_outputs
from .epidemiology import EpiBrain
from .governance import GovernanceBrain
from .logistics import LogisticsBrain

__all__ = [
    "Brain",
    "EpiBrain",
    "GovernanceBrain",
    "LogisticsBrain",
    "aggregate_brain_outputs",
]

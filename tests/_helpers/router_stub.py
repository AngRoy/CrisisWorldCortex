"""Scripted RoutingPolicy test double.

Returns a pre-configured sequence of ``RoutingAction``s. When the
sequence is exhausted, returns ``stop_and_no_op`` so tests cannot
infinite-loop. Reusable by Session 12 (Council tests) and Session 13
(router tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from cortex.schemas import MetacognitionState, RoutingAction


@dataclass
class ScriptedRouter:
    """Returns scripted RoutingActions in order; default-tail = stop_and_no_op."""

    actions: List[RoutingAction]
    states_seen: List[MetacognitionState] = field(default_factory=list)
    call_count: int = 0

    def forward(self, state: MetacognitionState) -> RoutingAction:
        self.states_seen.append(state)
        self.call_count += 1
        if not self.actions:
            return RoutingAction(kind="stop_and_no_op")
        return self.actions.pop(0)

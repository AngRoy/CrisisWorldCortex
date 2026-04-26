"""Logistics brain factory."""

from __future__ import annotations

from cortex.subagents import CriticSubagent, PlannerSubagent, WorldModelerSubagent
from cortex.subagents._base import _LLMClientLike

from ._base import Brain


def LogisticsBrain(llm_client: _LLMClientLike) -> Brain:
    """Construct a Logistics Brain bound to ``llm_client``.

    Multi-model deployment: see EpiBrain. Pass a Llama-bound client
    here while the other brains use Qwen, etc.
    """
    return Brain(
        brain_id="logistics",
        llm_client=llm_client,
        wm=WorldModelerSubagent(llm_client),
        planner=PlannerSubagent(llm_client),
        critic=CriticSubagent(llm_client),
    )

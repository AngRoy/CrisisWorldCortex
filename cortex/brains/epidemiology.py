"""Epidemiology brain factory."""

from __future__ import annotations

from cortex.subagents import CriticSubagent, PlannerSubagent, WorldModelerSubagent
from cortex.subagents._base import _LLMClientLike

from ._base import Brain


def EpiBrain(llm_client: _LLMClientLike) -> Brain:
    """Construct an Epidemiology Brain bound to ``llm_client``.

    Multi-model deployment: pass a different ``llm_client`` per brain
    instance to use different models per brain (e.g., Qwen for epi,
    Llama for logistics). The 3 LLM subagents are constructed with the
    SAME client so token billing aggregates correctly.
    """
    return Brain(
        brain_id="epidemiology",
        llm_client=llm_client,
        wm=WorldModelerSubagent(llm_client),
        planner=PlannerSubagent(llm_client),
        critic=CriticSubagent(llm_client),
    )

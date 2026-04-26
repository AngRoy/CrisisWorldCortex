"""Abstract base for the 3 LLM subagents (WorldModeler, Planner, Critic).

Phase A docs/CORTEX_ARCHITECTURE.md Decisions 1-8 + 62 lock the role split,
prompt-loading mechanism, retry-with-history semantics, empty fallback
shape, caller-id format, and TypeAdapter validation pattern. This base
class implements the shared mechanics; concrete subclasses pin the
role name, output type, prompt path, TypeAdapter, and USR builder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, List, Optional, Protocol

from pydantic import BaseModel, TypeAdapter, ValidationError

from cortex.llm_client import ChatMessage, ChatResponse
from cortex.schemas import SubagentInput
from CrisisWorldCortex.models import ExecutedAction

# ============================================================================
# Module-level constants (loaded at import time)
# ============================================================================

PROMPTS_DIR: Path = Path(__file__).parent / "prompts"
"""Directory holding the per-role SYS prompt template files (Decision 4)."""

_RETRY_SNIPPET_MAX_CHARS: int = 200
"""Cap on the failed-response snippet included in the retry message
(Decision 8). 200 chars ~= 50 tokens; keeps retry overhead bounded."""

_BASE_RETRY_USER_TEMPLATE: str = (
    "Your previous response failed to parse as JSON. The response was:\n"
    "{snippet}\n\n"
    "Emit ONLY valid JSON matching the schema specified in the system prompt. "
    "No prose, no code fences."
)

_RECENT_ACTION_LOG_TAIL: int = 8
"""How many entries from ``recent_action_log_excerpt`` to render into the
USR summary. Matches the design-doc 8-deep history (M-FR-3)."""


# ============================================================================
# Duck-typed LLM client protocol (so tests can pass StubLLMClient)
# ============================================================================


class _LLMClientLike(Protocol):
    """Subset of ``cortex.llm_client.LLMClient`` that subagents call.

    Production: ``LLMClient``. Tests: ``tests._helpers.llm_stub.StubLLMClient``.
    """

    def chat(
        self,
        caller_id: str,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = ...,
        temperature: Optional[float] = ...,
    ) -> ChatResponse: ...


# ============================================================================
# Abstract base
# ============================================================================


class _LLMSubagent(ABC):
    """Shared run/retry/parse/empty-fallback skeleton for the 3 subagents.

    Subclasses override the class-level vars below and implement
    ``_build_user_message`` + ``empty_fallback``.
    """

    # --- Subclass class-level overrides -------------------------------------
    _role_name: ClassVar[str]  # one of: "world_modeler", "planner", "critic"
    _output_type: ClassVar[type]  # BeliefState / CandidatePlan / CriticReport
    _system_prompt_filename: ClassVar[str]  # e.g. "world_modeler.txt"
    _SYSTEM_PROMPT_TEMPLATE: ClassVar[str]  # populated by load_prompt() at module load
    _ADAPTER: ClassVar[TypeAdapter]  # populated at module load

    # --- Construction --------------------------------------------------------

    def __init__(self, llm_client: _LLMClientLike) -> None:
        self._llm = llm_client

    # --- Public surface ------------------------------------------------------

    def run(self, input: SubagentInput, step_idx: int) -> BaseModel:
        """Call the LLM (with 1 retry), parse, return typed output or empty fallback.

        Always returns a typed object - never ``None``. Decision 6: on
        any failure (parse, retry-parse, LLM call exception) returns the
        role-specific empty fallback.
        """
        # Defensive: subclass enforces role-input alignment so harnesses
        # don't accidentally route a Planner input through a Critic class.
        assert input.role == self._role_name, (
            f"SubagentInput.role={input.role!r} does not match "
            f"{type(self).__name__}._role_name={self._role_name!r}"
        )

        sys_content = self._SYSTEM_PROMPT_TEMPLATE.format(
            brain=input.brain,
            target_plan_id=input.target_plan_id or "",
        )
        usr_content = self._build_user_message(input)
        messages: List[ChatMessage] = [
            ChatMessage(role="system", content=sys_content),
            ChatMessage(role="user", content=usr_content),
        ]
        caller_id = self._caller_id(input, step_idx)

        # ---- Attempt 1 -----------------------------------------------------
        first_response = self._safe_chat(caller_id, messages)
        if first_response is None:
            return self._empty_fallback_for(input)
        parsed = self._try_parse(first_response.content)
        if parsed is not None:
            return parsed

        # ---- Attempt 2 (retry with chat-history continuation) --------------
        snippet = self._truncate_snippet(first_response.content)
        retry_messages: List[ChatMessage] = [
            *messages,
            ChatMessage(role="assistant", content=first_response.content),
            ChatMessage(
                role="user",
                content=_BASE_RETRY_USER_TEMPLATE.format(snippet=snippet),
            ),
        ]
        retry_response = self._safe_chat(caller_id, retry_messages)
        if retry_response is None:
            return self._empty_fallback_for(input)
        parsed_retry = self._try_parse(retry_response.content)
        if parsed_retry is not None:
            return parsed_retry

        # ---- Both attempts failed - empty fallback -------------------------
        return self._empty_fallback_for(input)

    # --- Subclass extension points ------------------------------------------

    @abstractmethod
    def _build_user_message(self, input: SubagentInput) -> str:
        """Render the role-specific USR message body."""

    @classmethod
    @abstractmethod
    def empty_fallback(cls, brain: str, target_plan_id: str = "") -> BaseModel:
        """Return the empty / no-signal output for this role.

        Phase A Decision 6: confidence/severity = 0 and empty evidence/attacks
        signal "no useful input from this subagent" to the Brain Executive.
        """

    # --- Internal helpers ---------------------------------------------------

    def _caller_id(self, input: SubagentInput, step_idx: int) -> str:
        # Phase A Decision 7: cortex:<brain>:<role>:t<tick>:r<round>:s<step_idx>
        return f"cortex:{input.brain}:{self._role_name}:t{input.tick}:r{input.round}:s{step_idx}"

    def _safe_chat(self, caller_id: str, messages: List[ChatMessage]) -> Optional[ChatResponse]:
        """Call LLM; on exception, return None so caller can empty-fallback."""
        try:
            return self._llm.chat(caller_id=caller_id, messages=messages)
        except Exception:
            # Decision 6: LLM call failure folds into the same empty-fallback path
            # as parse failure. Brain Executive sees a no-signal subagent.
            return None

    def _try_parse(self, content: str) -> Optional[BaseModel]:
        """Validate ``content`` as JSON via this role's TypeAdapter.

        Strips common markdown code fences before validating since some
        models wrap JSON in ```json ... ```.
        """
        cleaned = _strip_code_fences(content.strip())
        if not cleaned:
            return None
        try:
            return self._ADAPTER.validate_json(cleaned)
        except (ValidationError, ValueError):
            return None

    def _empty_fallback_for(self, input: SubagentInput) -> BaseModel:
        return type(self).empty_fallback(
            brain=input.brain,
            target_plan_id=input.target_plan_id or "",
        )

    @staticmethod
    def _truncate_snippet(content: str) -> str:
        if len(content) <= _RETRY_SNIPPET_MAX_CHARS:
            return content
        return content[:_RETRY_SNIPPET_MAX_CHARS] + "..."

    @staticmethod
    def _format_action_log(log: List[ExecutedAction]) -> str:
        """M-FR-3 - render recent_action_log_excerpt as a compact text summary.

        Format: ``"tick 4: deploy_resource accepted; tick 5: restrict_movement.strict rejected"``.
        Capped at the most recent 8 entries.
        """
        if not log:
            return "(empty)"
        items: List[str] = []
        for ea in log[-_RECENT_ACTION_LOG_TAIL:]:
            status = "accepted" if ea.accepted else "rejected"
            kind = ea.action.kind
            extra = ""
            if kind == "restrict_movement":
                extra = f".{getattr(ea.action, 'severity', '?')}"
            items.append(f"tick {ea.tick}: {kind}{extra} {status}")
        return "; ".join(items)


# ============================================================================
# Helpers (module-level)
# ============================================================================


def load_prompt(filename: str) -> str:
    """Load a SYS prompt template at module-load time (Decision 4)."""
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def _strip_code_fences(s: str) -> str:
    """Remove leading ``` / ```json fence and trailing ``` if present."""
    s = s.strip()
    if not s.startswith("```"):
        return s
    lines = s.split("\n")
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

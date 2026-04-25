"""Regression for the synthetic-rejection wire-validation bug (Session 7d).

Session 7c's manual smoke surfaced ``RuntimeError('Server error: Invalid
message (code: VALIDATION_ERROR)')`` when B1's parse-failure recovery
path submitted a synthetic ``PublicCommunication`` over the wire. The
trace pointed at ``apply_tick`` building an ``ExecutedAction`` whose
``action`` field rejected the runtime instance with Pydantic
``model_type`` even though the instance shape was valid.

Root cause: ``models.py`` was loaded twice in the container under two
``sys.modules`` entries. Different paths resolved to physically
different files (``/app/env/models.py`` vs ``/app/.venv/.../site-packages/
CrisisWorldCortex/models.py``), so each carried distinct class objects
for every variant of the discriminated union. The
``CrisisworldcortexAction`` registered by ``server.app`` was built from
one set; ``ExecutedAction`` (imported via ``server/simulator/
seir_model.py``) was built from the other. The instance produced by
deserialising the wire payload failed the ``isinstance`` check inside
``ExecutedAction``'s discriminated-union validator.

This file owns two regressions:

  - In-process round-trip sanity: synthetic ``PublicCommunication``
    constructed by ``baselines.flat_agent`` must round-trip through
    ``CrisisworldcortexAction`` and through ``apply_tick`` to land in
    ``recent_action_log`` with ``accepted=False``.
  - Structural import-consistency: every Python file under ``server/``
    must use the same import path for ``models`` so the container's
    runtime registers exactly one class identity per variant.
"""

from __future__ import annotations

import re
from pathlib import Path

from CrisisWorldCortex.models import CrisisworldcortexAction, PublicCommunication
from CrisisWorldCortex.server.simulator import apply_tick, load_task

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_DIR = REPO_ROOT / "server"


def test_synthetic_payload_round_trips_through_wire_wrapper() -> None:
    """The exact synthetic ``PublicCommunication`` B1 builds on parse
    failure must validate cleanly through ``CrisisworldcortexAction``.
    """
    payload = PublicCommunication(
        audience="general",
        message_class="informational",
        honesty=0.0,
    )
    action = CrisisworldcortexAction(action=payload)
    dumped = action.model_dump()
    rebuilt = CrisisworldcortexAction.model_validate(dumped)
    assert rebuilt.action.kind == "public_communication"
    assert rebuilt.action.honesty == 0.0


def test_synthetic_payload_lands_as_rejected_in_action_log() -> None:
    """Submitting the synthetic payload to ``apply_tick`` must record
    ``accepted=False`` (V2-illegal) without raising. This is the
    in-process equivalent of B1's recovery path.
    """
    state = load_task("outbreak_easy", episode_seed=0)
    payload = PublicCommunication(
        audience="general",
        message_class="informational",
        honesty=0.0,
    )
    new_state = apply_tick(state, payload)

    log = new_state.recent_action_log
    assert log, "apply_tick must record the action even when rejected"
    last = log[-1]
    assert last.action.kind == "public_communication"
    assert last.accepted is False, "PublicCommunication is V2-rejected per design §6.3 / §19"


_BARE_MODELS_IMPORT = re.compile(
    r"^\s*from\s+models\s+import",
    re.MULTILINE,
)
_RELATIVE_PARENT_MODELS_IMPORT = re.compile(
    r"^\s*from\s+\.\.+models\s+import",
    re.MULTILINE,
)


def test_server_files_use_canonical_models_import_path() -> None:
    """Every ``server/*.py`` file that imports wire types must use the
    canonical ``from CrisisWorldCortex.models import ...`` form.

    Two forbidden alternatives:

      - bare ``from models import ...`` resolves to ``/app/env/models.py``
        in the container (a different physical file from the wheel-installed
        ``site-packages/CrisisWorldCortex/models.py``), creating a second
        ``sys.modules`` entry with distinct class objects.
      - relative-parent ``from ..models import ...`` works in dev (where the
        package chain is ``CrisisWorldCortex.server.<file>``) but resolves
        to a non-existent parent in the container's ``server.<file>`` load
        and falls through to bare-name ``models`` — same trap.

    Pydantic's discriminated union built from one path's classes rejects
    instances constructed via the other path with a ``model_type`` error
    inside ``ExecutedAction(action=...)`` validation during ``apply_tick``.
    Pinning every server-internal models import to canonical resolves to
    a single ``sys.modules["CrisisWorldCortex.models"]`` entry both in
    dev (via the editable install) and in container (via the wheel).
    """
    bare_offenders: list[str] = []
    relative_offenders: list[str] = []
    for py_file in SERVER_DIR.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        text = py_file.read_text(encoding="utf-8")
        if _BARE_MODELS_IMPORT.search(text):
            bare_offenders.append(str(py_file.relative_to(REPO_ROOT)))
        if _RELATIVE_PARENT_MODELS_IMPORT.search(text):
            relative_offenders.append(str(py_file.relative_to(REPO_ROOT)))

    assert not bare_offenders, (
        f"server/*.py files using bare `from models import ...` will load a "
        f"different class identity than the canonical `from CrisisWorldCortex"
        f".models import ...`. Switch to canonical. Offenders: {bare_offenders}"
    )
    assert not relative_offenders, (
        f"server/*.py files using relative-parent `from ..models import ...` "
        f"work in dev but fall back to bare `models` in the container, "
        f"creating a second class identity. Switch to canonical "
        f"`from CrisisWorldCortex.models import ...`. Offenders: "
        f"{relative_offenders}"
    )

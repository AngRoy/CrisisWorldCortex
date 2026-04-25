"""Static enforcement of the cross-directory import graph (root CLAUDE.md).

Walks each subsystem's ``*.py`` files, parses them with ``ast``, and asserts
no top-level ``import`` / ``from … import`` statement names a forbidden
prefix. Relative imports (``from . import x``, ``from ..models``) stay
within the same package and cannot create cross-subsystem edges, so they
are skipped — only absolute module references are checked.

The rules below are derived directly from root ``CLAUDE.md`` →
"Import-graph rule (enforced)". Per-subsystem ``CLAUDE.md`` files restate
the same constraints. If the rules drift, this test must be updated in
lockstep.

Why AST instead of subprocess-import (per ``tests/CLAUDE.md``):
``tests/CLAUDE.md`` notes that ``test_import_graph.py`` "uses a fresh
subprocess import, not ``sys.modules`` monkey-patching — the latter
passes under contamination". AST parsing is even stricter than subprocess
import: it finds violations that lazy / conditional imports would hide
from a runtime check, and it does not require the subsystem to be
fully implemented before the test can run (training/, baselines/, demo/
are still stubs). When those subsystems acquire real runtime entry
points, a complementary subprocess-based test can be added.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SubsystemRule:
    """One rule: a directory + the absolute-module prefixes it cannot import.

    ``forbidden_prefixes`` matches at module-segment boundaries —
    ``server`` matches ``server`` and ``server.simulator`` but not
    ``serverless``. We also list both the bare path (``server``, used by
    ``server/`` internal code) and the canonical path
    (``CrisisWorldCortex.server``, used by tests / cross-boundary callers)
    because both resolve to the same modules at runtime.
    """

    name: str
    roots: Tuple[str, ...]
    forbidden_prefixes: Tuple[str, ...]


# Forbidden-edge table — one entry per top-level subsystem.
RULES: Tuple[SubsystemRule, ...] = (
    SubsystemRule(
        name="cortex",
        roots=("cortex",),
        forbidden_prefixes=(
            "server",
            "CrisisWorldCortex.server",
            "training",
            "CrisisWorldCortex.training",
            "baselines",
            "CrisisWorldCortex.baselines",
            "demo",
            "CrisisWorldCortex.demo",
            # cortex/CLAUDE.md: cortex hits LLMs, not the env HTTP client.
            "client",
            "CrisisWorldCortex.client",
        ),
    ),
    SubsystemRule(
        name="server",
        roots=("server",),
        forbidden_prefixes=(
            "cortex",
            "CrisisWorldCortex.cortex",
            "training",
            "CrisisWorldCortex.training",
            "baselines",
            "CrisisWorldCortex.baselines",
            "demo",
            "CrisisWorldCortex.demo",
            # server/CLAUDE.md: server is the env, never the HTTP client.
            "client",
            "CrisisWorldCortex.client",
        ),
    ),
    SubsystemRule(
        name="baselines",
        roots=("baselines",),
        forbidden_prefixes=(
            # baselines hit the env over HTTP via client.py — never reach in.
            "server",
            "CrisisWorldCortex.server",
            "training",
            "CrisisWorldCortex.training",
            "demo",
            "CrisisWorldCortex.demo",
        ),
    ),
    SubsystemRule(
        name="training",
        roots=("training",),
        forbidden_prefixes=(
            # training MAY import server.graders (reward-name constants).
            # All other server.* is forbidden, especially server.simulator.
            "server.simulator",
            "CrisisWorldCortex.server.simulator",
            "server.app",
            "CrisisWorldCortex.server.app",
            "server.CrisisWorldCortex_environment",
            "CrisisWorldCortex.server.CrisisWorldCortex_environment",
            "baselines",
            "CrisisWorldCortex.baselines",
            "demo",
            "CrisisWorldCortex.demo",
        ),
    ),
    SubsystemRule(
        name="demo",
        roots=("demo",),
        forbidden_prefixes=(
            "server",
            "CrisisWorldCortex.server",
            "training",
            "CrisisWorldCortex.training",
            "baselines",
            "CrisisWorldCortex.baselines",
            # demo imports cortex.schemas (types only). Other cortex/* is OK
            # except council and routing_policy, which would couple the
            # replay-only visualizer to the live agent.
            "cortex.council",
            "CrisisWorldCortex.cortex.council",
            "cortex.routing_policy",
            "CrisisWorldCortex.cortex.routing_policy",
        ),
    ),
)


# ============================================================================
# AST helpers
# ============================================================================


def _absolute_module_names_imported(source: str) -> List[str]:
    """Return the absolute module names referenced by import statements.

    Skips relative imports (``ImportFrom`` with ``level > 0``): those
    resolve inside the same package and cannot violate cross-subsystem
    rules. Returns module names exactly as written in source — caller
    matches them against forbidden prefixes.
    """
    tree = ast.parse(source)
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative — same-package, cannot cross subsystems
            if node.module is not None:
                names.append(node.module)
    return names


def _matches_forbidden_prefix(module: str, prefix: str) -> bool:
    """``prefix`` matches at segment boundary: ``server`` matches
    ``server`` and ``server.simulator`` but not ``serverless`` or
    ``server2``.
    """
    return module == prefix or module.startswith(prefix + ".")


def _scan_dir_for_violations(
    root_dir: Path,
    forbidden_prefixes: Tuple[str, ...],
) -> List[Tuple[Path, str, str]]:
    """Walk ``root_dir`` recursively. Return (file, imported_module,
    forbidden_prefix) for each violation."""
    violations: List[Tuple[Path, str, str]] = []
    for py_file in root_dir.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        source = py_file.read_text(encoding="utf-8")
        try:
            modules = _absolute_module_names_imported(source)
        except SyntaxError as e:  # pragma: no cover — surfaces parse errors
            raise AssertionError(f"failed to parse {py_file} for import-graph check: {e}") from e
        for module in modules:
            for prefix in forbidden_prefixes:
                if _matches_forbidden_prefix(module, prefix):
                    violations.append((py_file, module, prefix))
                    break  # one violation per import is enough
    return violations


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.parametrize("rule", RULES, ids=[r.name for r in RULES])
def test_subsystem_has_no_forbidden_imports(rule: SubsystemRule) -> None:
    """Per-subsystem AST scan: no forbidden cross-directory edge."""
    all_violations: List[Tuple[Path, str, str]] = []
    for root in rule.roots:
        root_dir = REPO_ROOT / root
        if not root_dir.is_dir():
            pytest.fail(f"subsystem root missing: {root_dir}")
        all_violations.extend(_scan_dir_for_violations(root_dir, rule.forbidden_prefixes))
    if all_violations:
        formatted = "\n".join(
            f"  {path.relative_to(REPO_ROOT)}: imports {module!r} (forbidden prefix {prefix!r})"
            for path, module, prefix in all_violations
        )
        pytest.fail(
            f"{rule.name}/ has {len(all_violations)} forbidden import(s):\n"
            f"{formatted}\n"
            f"Per root CLAUDE.md -> 'Import-graph rule (enforced)'."
        )


def test_models_is_a_leaf_module() -> None:
    """``models.py`` must not import any other internal subsystem.

    Allowed: stdlib, ``pydantic``, ``openenv.core.*``. Forbidden: any of
    the in-repo subsystem packages or the HTTP client.
    """
    models_file = REPO_ROOT / "models.py"
    assert models_file.is_file(), f"missing {models_file}"

    forbidden_internal_prefixes = (
        "server",
        "CrisisWorldCortex.server",
        "cortex",
        "CrisisWorldCortex.cortex",
        "training",
        "CrisisWorldCortex.training",
        "baselines",
        "CrisisWorldCortex.baselines",
        "demo",
        "CrisisWorldCortex.demo",
        "client",
        "CrisisWorldCortex.client",
    )
    modules = _absolute_module_names_imported(models_file.read_text(encoding="utf-8"))
    bad = [
        (m, p)
        for m in modules
        for p in forbidden_internal_prefixes
        if _matches_forbidden_prefix(m, p)
    ]
    assert not bad, f"models.py is a leaf - must not import internal subsystems. Found: {bad!r}"

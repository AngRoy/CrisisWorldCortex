"""Tests for inference harness bugs C6, C7, H14, H15.

C6/C7: import-graph violations (server.simulator, server.graders).
H14: score normalization range stale.
H15: done=False suppression in _normalize.
"""

import ast
import sys

from inference import compute_score


def test_inference_does_not_import_server_simulator_at_runtime() -> None:
    """C6: inference.py must not import from CrisisWorldCortex.server.simulator."""
    with open("inference.py") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "server.simulator" not in node.module, (
                f"inference.py imports from {node.module} — "
                f"violates import-graph rule (no server.simulator in inference)"
            )


def test_inference_does_not_import_server_graders_function() -> None:
    """C7: inference.py must not import grader functions from server.graders."""
    with open("inference.py") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if "server.graders" in node.module:
                imported_names = [alias.name for alias in node.names]
                assert "terminal_bonus" not in imported_names, (
                    f"inference.py imports terminal_bonus from {node.module} — "
                    f"violates import-graph rule"
                )


def test_compute_score_full_range_normalization() -> None:
    """H14: Score normalization should map [-1.20, 1.20] to [0, 1].

    With mean=-1.0 and bonus=-0.20, raw = -1.20 → rescaled should be ~0
    (lower clamp). The old formula (raw + 0.20)/1.40 would give a negative
    rescaled value, meaning clamp hides a range error."""
    # Best case: mean=1.0, bonus=+0.20 → raw=1.20 → should map near 1.0
    score_best = compute_score([1.0] * 5, terminal_bonus_value=0.20)
    assert 0.99 <= score_best <= 1.0

    # Worst case: mean=-1.0, bonus=-0.20 → raw=-1.20 → should map near 0.0
    score_worst = compute_score([-1.0] * 5, terminal_bonus_value=-0.20)
    assert score_worst <= 0.01

    # Mid case: mean=0.0, bonus=0.0 → raw=0.0 → should map to 0.5
    score_mid = compute_score([0.0] * 5, terminal_bonus_value=0.0)
    assert 0.45 <= score_mid <= 0.55


def test_normalize_done_false_not_suppressed() -> None:
    """H15: When StepResult.done is False, obs.done must be set to False,
    not left at whatever default the observation has."""
    # Import here to avoid triggering the server import at module level
    # We test the _normalize static method indirectly
    from unittest.mock import MagicMock

    from inference import _SyncEnvAdapter

    # Create a mock result where done=False explicitly
    mock_result = MagicMock()
    mock_result.observation = MagicMock()
    mock_result.observation.reward = 0.5
    mock_result.observation.done = True  # Pre-set to True
    mock_result.reward = 0.5
    mock_result.done = False  # StepResult says NOT done

    obs = _SyncEnvAdapter._normalize(mock_result)
    assert obs.done is False, (
        f"obs.done should be False when StepResult.done=False, got {obs.done}"
    )

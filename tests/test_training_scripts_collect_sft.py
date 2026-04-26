"""Static checks for training/scripts/collect_sft_data.py.

Phase 5c. No HF Router / no env / no GPU — these tests verify the script's
configuration surface, parse-action helper, observation serializer, and
preflight env-health logic without external dependencies.

Live data collection (~$1-2 HF Router credits, ~30-45 min) is a manual
post-V6 step, not pytest-driven.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / "training" / "scripts" / "collect_sft_data.py"


def _load_module():
    """Load training/scripts/collect_sft_data.py as a module without running collect()."""
    import os

    os.environ.setdefault("HF_TOKEN", "test_token_static_only")
    spec = importlib.util.spec_from_file_location("collect_sft_data_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_exists_and_loads() -> None:
    """File present and importable."""
    assert SCRIPT_PATH.exists(), f"missing {SCRIPT_PATH}"
    mod = _load_module()
    for attr in (
        "collect",
        "main",
        "parse_action_json",
        "serialize_observation",
        "preflight_env_health",
        "_SYSTEM_PROMPT_BODY",
    ):
        assert hasattr(mod, attr), f"missing {attr}"


def test_default_teacher_is_qwen72b_novita() -> None:
    """Default frontier teacher is Qwen 72B via Novita per the plan."""
    mod = _load_module()
    assert mod.MODEL_NAME == "Qwen/Qwen2.5-72B-Instruct:novita"


def test_default_min_reward_threshold_is_0p5() -> None:
    """M-FR-14: keep only rows with reward >= 0.5."""
    mod = _load_module()
    assert abs(mod.MIN_REWARD_THRESHOLD - 0.5) < 1e-9


def test_default_eval_fraction_is_0p2() -> None:
    """80/20 train/eval split."""
    mod = _load_module()
    assert abs(mod.EVAL_FRACTION - 0.2) < 1e-9


def test_parse_action_json_handles_bare_json() -> None:
    mod = _load_module()
    out = mod.parse_action_json('{"kind": "no_op"}')
    assert out == {"kind": "no_op"}


def test_parse_action_json_strips_markdown_fences() -> None:
    mod = _load_module()
    raw = (
        "```json\n"
        '{"kind": "deploy_resource", "region": "R1", "resource_type": "test_kits", '
        '"quantity": 100}\n'
        "```"
    )
    out = mod.parse_action_json(raw)
    assert out is not None
    assert out["kind"] == "deploy_resource"
    assert out["quantity"] == 100


def test_parse_action_json_extracts_brace_block_from_prose() -> None:
    mod = _load_module()
    raw = 'I think the best action is: {"kind": "no_op"}. This advances the tick.'
    out = mod.parse_action_json(raw)
    assert out == {"kind": "no_op"}


def test_parse_action_json_returns_none_on_invalid() -> None:
    mod = _load_module()
    assert mod.parse_action_json("") is None
    assert mod.parse_action_json("not json") is None
    assert mod.parse_action_json('{"missing": "kind"}') is None


def test_preflight_env_health_passes_on_healthy() -> None:
    """Mock urllib.request.urlopen returning a healthy /health response."""
    mod = _load_module()
    fake_resp = MagicMock(status=200)
    fake_resp.read.return_value = b'{"status":"healthy"}'
    fake_resp.__enter__ = lambda self: self
    fake_resp.__exit__ = lambda self, *a: None
    with patch("urllib.request.urlopen", return_value=fake_resp):
        mod.preflight_env_health("http://localhost:58300")  # should not raise


def test_preflight_env_health_aborts_on_unreachable() -> None:
    """Mock urlopen raising ConnectionRefusedError → SystemExit."""
    mod = _load_module()
    with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("dead")):
        with pytest.raises(SystemExit, match="unreachable"):
            mod.preflight_env_health("http://localhost:58300")


def test_preflight_env_health_aborts_on_unhealthy_body() -> None:
    """Mock urlopen returning 200 but body that doesn't say 'healthy'."""
    mod = _load_module()
    fake_resp = MagicMock(status=200)
    fake_resp.read.return_value = b'{"status":"BUILD_ERROR"}'
    fake_resp.__enter__ = lambda self: self
    fake_resp.__exit__ = lambda self, *a: None
    with patch("urllib.request.urlopen", return_value=fake_resp):
        with pytest.raises(SystemExit, match="unhealthy"):
            mod.preflight_env_health("http://localhost:58300")


def test_required_hf_token_raises_systemexit_when_missing() -> None:
    """HF_TOKEN is required at module load."""
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
    env["PYTHONPATH"] = str(SCRIPT_PATH.parent.parent.parent)
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "HF_TOKEN" in (result.stdout + result.stderr)

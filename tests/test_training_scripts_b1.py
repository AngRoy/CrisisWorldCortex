"""Static checks for training/scripts/train_b1_grpo.py.

Phase 5b. No GPU required — these tests verify the script's
configuration surface and pre-flight logic without loading Unsloth/torch.

Live training verification (M-FR-4 — 5-step Llama run, ~$0.50) is
gated behind V6 (HF Space rebuild) and is run manually post-validation
rather than in pytest.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / "training" / "scripts" / "train_b1_grpo.py"


def _load_module():
    """Load training/scripts/train_b1_grpo.py as a module without executing main()."""
    spec = importlib.util.spec_from_file_location("train_b1_grpo_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Set required env vars before module-level constants resolve.
    import os

    os.environ.setdefault("HF_TOKEN", "test_token_static_only")
    os.environ.setdefault("HUB_REPO_ID", "test/static_load")
    spec.loader.exec_module(module)
    return module


def test_script_exists_and_loads() -> None:
    """The script file is present and importable as a module."""
    assert SCRIPT_PATH.exists(), f"missing {SCRIPT_PATH}"
    mod = _load_module()
    assert hasattr(mod, "main")
    assert hasattr(mod, "preflight_model_access")


def test_default_model_name_is_qwen3_7b() -> None:
    """B1-Qwen default per M-FR-1A; B1-Llama via MODEL_NAME override."""
    mod = _load_module()
    assert mod.MODEL_NAME.startswith("unsloth/Qwen3-7B-Instruct") or mod.MODEL_NAME.startswith(
        "unsloth/Qwen3"
    )


def test_required_env_vars_raise_systemexit_when_missing() -> None:
    """HF_TOKEN and HUB_REPO_ID are required and raise SystemExit.

    Uses subprocess to get a clean env (in-process clearing collides
    with other tests' caches and the module's setdefault rescue).
    """
    import subprocess
    import sys

    env = {
        k: v for k, v in __import__("os").environ.items() if k not in ("HF_TOKEN", "HUB_REPO_ID")
    }
    env["PYTHONPATH"] = str(SCRIPT_PATH.parent.parent.parent)
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0, "expected non-zero exit when HF_TOKEN missing"
    assert "HF_TOKEN" in (result.stdout + result.stderr), (
        f"expected HF_TOKEN in failure output; got stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )


def _make_gated_error():
    """Construct a GatedRepoError without invoking its strict response= kwarg."""
    from huggingface_hub.utils import GatedRepoError

    class _FakeGated(GatedRepoError):
        def __init__(self, msg: str) -> None:
            Exception.__init__(self, msg)

    return _FakeGated("license required")


def _make_not_found_error():
    """Construct a RepositoryNotFoundError without invoking its strict response= kwarg."""
    from huggingface_hub.utils import RepositoryNotFoundError

    class _FakeNotFound(RepositoryNotFoundError):
        def __init__(self, msg: str) -> None:
            Exception.__init__(self, msg)

    return _FakeNotFound("no such repo")


def test_preflight_passes_for_accessible_model() -> None:
    """preflight_model_access exits 0 when HfApi.model_info returns successfully."""
    mod = _load_module()
    fake_info = MagicMock(gated=False, private=False)
    with patch("huggingface_hub.HfApi") as MockApi:
        MockApi.return_value.model_info.return_value = fake_info
        # Should not raise.
        mod.preflight_model_access("unsloth/Qwen3-7B-Instruct-bnb-4bit", "tok")


def test_preflight_aborts_on_gated_model_without_access() -> None:
    """preflight_model_access aborts with friendly error on GatedRepoError."""
    mod = _load_module()
    err = _make_gated_error()
    with patch("huggingface_hub.HfApi") as MockApi:
        MockApi.return_value.model_info.side_effect = err
        with pytest.raises(SystemExit, match="gated"):
            mod.preflight_model_access("meta-llama/Llama-3.1-8B-Instruct", "tok")


def test_preflight_aborts_on_repo_not_found() -> None:
    """preflight_model_access aborts cleanly on RepositoryNotFoundError."""
    mod = _load_module()
    err = _make_not_found_error()
    with patch("huggingface_hub.HfApi") as MockApi:
        MockApi.return_value.model_info.side_effect = err
        with pytest.raises(SystemExit, match="not found"):
            mod.preflight_model_access("does/not/exist", "tok")

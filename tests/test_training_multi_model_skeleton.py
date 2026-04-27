"""Static checks for real Cortex multi-brain router training script.

No GPU / no HF Hub access here. These tests verify the env-var surface,
memory guard, JSON parser, and no-TRL training helpers without loading
models or touching the deployed env.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / "training" / "scripts" / "train_cortex_multi_model.py"


def _load_module():
    import os

    os.environ.setdefault("HF_TOKEN", "test_token_static_only")
    spec = importlib.util.spec_from_file_location(
        "train_cortex_multi_model_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_exists_and_loads() -> None:
    assert SCRIPT_PATH.exists(), f"missing {SCRIPT_PATH}"
    mod = _load_module()
    for attr in (
        "main",
        "preflight_model_access",
        "check_memory_budget",
        "FrozenBrain",
        "CandidateResult",
        "parse_router_choice",
        "train_step",
    ):
        assert hasattr(mod, attr), f"missing {attr}"


def test_default_brain_models() -> None:
    mod = _load_module()
    assert mod.EPI_BRAIN_MODEL == "Qwen/Qwen2.5-3B-Instruct"
    assert mod.LOGISTICS_BRAIN_MODEL == "microsoft/Phi-3.5-mini-instruct"
    assert mod.GOVERNANCE_BRAIN_MODEL == mod.EPI_BRAIN_MODEL


def test_default_router_and_warmstart() -> None:
    mod = _load_module()
    assert mod.ROUTER_MODEL == "Qwen/Qwen2.5-1.5B-Instruct"
    assert mod.ROUTER_WARMSTART_REPO == "Angshuman28/cortex-router-sft-warmstart"
    assert mod.HUB_REPO_ID == "Angshuman28/cortex-router-trained"


def test_default_training_hyperparams() -> None:
    mod = _load_module()
    assert mod.LORA_RANK == 16
    assert mod.GROUP_SIZE == 4
    assert mod.MAX_TRAIN_STEPS == 300
    assert mod.GRAD_CLIP == 1.0
    assert mod.LOAD_IN_4BIT is True


def test_check_memory_budget_aborts_on_low_free_gpu() -> None:
    mod = _load_module()
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.mem_get_info.return_value = (10 * 1024**3, 80 * 1024**3)
    with patch.dict("sys.modules", {"torch": fake_torch}):
        with pytest.raises(RuntimeError, match="Insufficient GPU memory"):
            mod.check_memory_budget()


def test_check_memory_budget_passes_when_ample_free() -> None:
    mod = _load_module()
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.mem_get_info.return_value = (60 * 1024**3, 80 * 1024**3)
    with patch.dict("sys.modules", {"torch": fake_torch}):
        mod.check_memory_budget()


def test_check_memory_budget_skipped_when_no_cuda() -> None:
    mod = _load_module()
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": fake_torch}):
        mod.check_memory_budget()


def test_required_hf_token_raises_systemexit_when_missing() -> None:
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


def test_parse_router_choice_accepts_json_and_alias() -> None:
    mod = _load_module()
    assert mod.parse_router_choice('{"brain":"epi"}') == "epi"
    assert mod.parse_router_choice('```json\n{"brain":"epidemiology"}\n```') == "epi"
    assert mod.parse_router_choice('{"brain":"logistics"}') == "logistics"
    assert mod.parse_router_choice('{"brain":"governance"}') == "governance"


def test_parse_router_choice_rejects_bad_output() -> None:
    mod = _load_module()
    assert mod.parse_router_choice("not json") is None
    assert mod.parse_router_choice('{"brain":"finance"}') is None


def test_candidate_result_shape() -> None:
    mod = _load_module()
    candidate = mod.CandidateResult(1.0, "epi", object(), "raw", True)
    assert candidate.reward == 1.0
    assert candidate.brain == "epi"
    assert candidate.accepted is True

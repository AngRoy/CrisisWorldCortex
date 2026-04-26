"""Static checks for training/scripts/train_cortex_multi_model.py.

Phase 6 skeleton tests. No GPU / no HF Hub access — these tests verify
the script's configuration surface, preflight logic, memory budget guard,
and adapter signatures without loading Unsloth/torch or hitting the
network.

Live multi-model training (~2 hours a100-large, ~$5) is a manual Phase-7
step gated on user approval, not pytest-driven.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / "training" / "scripts" / "train_cortex_multi_model.py"


def _load_module():
    """Load the script as a module without running main()."""
    import os

    os.environ.setdefault("HF_TOKEN", "test_token_static_only")
    os.environ.setdefault("HUB_REPO_ID", "test/multi_model_static_load")
    spec = importlib.util.spec_from_file_location(
        "train_cortex_multi_model_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_exists_and_loads() -> None:
    """File present and importable."""
    assert SCRIPT_PATH.exists(), f"missing {SCRIPT_PATH}"
    mod = _load_module()
    for attr in (
        "main",
        "preflight_model_access",
        "check_memory_budget",
        "_UnslothLLMAdapter",
        "_TrainableRoutingPolicy",
    ):
        assert hasattr(mod, attr), f"missing {attr}"


def test_default_brain_models() -> None:
    """Defaults: Qwen-7B for epi+governance (shared), Llama-8B for logistics."""
    mod = _load_module()
    assert mod.EPI_BRAIN_MODEL.startswith("unsloth/Qwen3-7B-Instruct")
    assert mod.LOGISTICS_BRAIN_MODEL == "meta-llama/Llama-3.1-8B-Instruct"


def test_governance_shares_with_epi_by_default() -> None:
    """M-FR-27: governance brain shares weights with epi by default."""
    mod = _load_module()
    assert mod.GOVERNANCE_BRAIN_MODEL == mod.EPI_BRAIN_MODEL


def test_default_router_is_qwen_1p5b() -> None:
    """M-FR-28: small LLM router."""
    mod = _load_module()
    assert mod.ROUTER_MODEL.startswith("unsloth/Qwen3-1.5B")


def test_router_base_falls_back_to_router_model() -> None:
    """Phase-5e style fallback for SFT-warmstarted router checkpoints."""
    mod = _load_module()
    assert mod.ROUTER_BASE_MODEL == mod.ROUTER_MODEL


def test_default_lora_rank_is_16_for_router() -> None:
    """M-FR-28: router rank 16 (smaller than B1's 32 because 1.5B model)."""
    mod = _load_module()
    assert mod.LORA_RANK == 16


def test_default_gpu_mem_util_is_0p5() -> None:
    """Phase-A M-FR-10: tighter than B1's 0.6 to leave room for frozen brains."""
    mod = _load_module()
    assert abs(mod.GPU_MEM_UTIL - 0.5) < 1e-9


def test_default_brain_call_timeout() -> None:
    """Phase-A M-FR-9: 30s timeout per brain call."""
    mod = _load_module()
    assert mod.BRAIN_CALL_TIMEOUT_S == 30


def test_check_memory_budget_aborts_on_low_free_gpu() -> None:
    """Memory guard raises RuntimeError when < MIN_FREE_GPU_GB free."""
    mod = _load_module()
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    # Return (free_bytes, total_bytes) with only 10 GB free.
    fake_torch.cuda.mem_get_info.return_value = (10 * 1024**3, 80 * 1024**3)
    with patch.dict("sys.modules", {"torch": fake_torch}):
        with pytest.raises(RuntimeError, match="Insufficient GPU memory"):
            mod.check_memory_budget()


def test_check_memory_budget_passes_when_ample_free() -> None:
    """Memory guard does not raise when 60+ GB free."""
    mod = _load_module()
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.mem_get_info.return_value = (60 * 1024**3, 80 * 1024**3)
    with patch.dict("sys.modules", {"torch": fake_torch}):
        mod.check_memory_budget()  # should not raise


def test_check_memory_budget_skipped_when_no_cuda() -> None:
    """Memory guard skips gracefully when CUDA isn't available (DRY_RUN path)."""
    mod = _load_module()
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": fake_torch}):
        mod.check_memory_budget()  # should not raise


def test_required_env_vars_raise_systemexit_when_missing() -> None:
    """HF_TOKEN and HUB_REPO_ID required."""
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUB_REPO_ID")}
    env["PYTHONPATH"] = str(SCRIPT_PATH.parent.parent.parent)
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    out = result.stdout + result.stderr
    assert "HF_TOKEN" in out or "HUB_REPO_ID" in out


def test_unsloth_adapter_signature() -> None:
    """_UnslothLLMAdapter implements _LLMClientLike (chat + tokens_used_for)."""
    mod = _load_module()
    AdapterCls = mod._UnslothLLMAdapter
    fake_model = MagicMock()
    fake_tokenizer = MagicMock()
    adapter = AdapterCls(fake_model, fake_tokenizer, brain_label="test")
    assert callable(adapter.chat)
    assert callable(adapter.tokens_used_for)
    assert adapter.tokens_used_for("never_called") == 0


def test_trainable_routing_policy_signature() -> None:
    """_TrainableRoutingPolicy has the forward() shape required by Council."""
    mod = _load_module()
    PolicyCls = mod._TrainableRoutingPolicy
    fake_model = MagicMock()
    fake_tokenizer = MagicMock()
    policy = PolicyCls(fake_model, fake_tokenizer)
    assert callable(policy.forward)
    assert (
        "system_prompt" in PolicyCls.SYSTEM_PROMPT.lower()
        or "router" in PolicyCls.SYSTEM_PROMPT.lower()
    )

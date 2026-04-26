"""Static checks for training/scripts/sft_warmstart.py.

Phase 5d. No GPU / no HF Hub access — these tests verify config surface
and preflight logic without loading Unsloth/torch or hitting the network.

Live SFT training (~30 min A100, ~$1.25) is a manual post-Phase-5c step,
not pytest-driven.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / "training" / "scripts" / "sft_warmstart.py"


def _load_module():
    """Load training/scripts/sft_warmstart.py as a module without running main()."""
    import os

    os.environ.setdefault("HF_TOKEN", "test_token_static_only")
    os.environ.setdefault("OUTPUT_REPO", "test/sft_static_load")
    spec = importlib.util.spec_from_file_location("sft_warmstart_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_exists_and_loads() -> None:
    """File present and importable."""
    assert SCRIPT_PATH.exists(), f"missing {SCRIPT_PATH}"
    mod = _load_module()
    for attr in ("main", "preflight_model_access", "preflight_dataset_access"):
        assert hasattr(mod, attr), f"missing {attr}"


def test_default_model_is_qwen3_7b() -> None:
    """B1-Qwen warm-start default."""
    mod = _load_module()
    assert mod.MODEL_NAME.startswith("unsloth/Qwen3-7B-Instruct")


def test_default_dataset_repo_matches_phase_5c() -> None:
    """SFT trainer reads the dataset Phase 5c writes."""
    mod = _load_module()
    assert mod.SFT_DATASET_REPO == "Angshuman28/crisisworld-sft-trajectories"


def test_default_lr_is_2e_minus_5() -> None:
    """M-FR-22: SFT LR ~10x higher than GRPO's 5e-6."""
    mod = _load_module()
    assert abs(mod.LR - 2e-5) < 1e-9


def test_default_lora_rank_matches_grpo_downstream() -> None:
    """M-FR-19: LoRA rank 32, same as Phase 5b GRPO, for downstream compat."""
    mod = _load_module()
    assert mod.LORA_RANK == 32


def test_default_max_train_steps_is_200() -> None:
    """Per spec."""
    mod = _load_module()
    assert mod.MAX_TRAIN_STEPS == 200


def test_default_num_epochs_is_2() -> None:
    """M-FR-20: 2 epochs default with MAX_TRAIN_STEPS as cap."""
    mod = _load_module()
    assert mod.NUM_EPOCHS == 2


def test_required_env_vars_raise_systemexit_when_missing() -> None:
    """HF_TOKEN and OUTPUT_REPO required."""
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "OUTPUT_REPO")}
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
    # Either HF_TOKEN or OUTPUT_REPO surfaces first; both are valid fail signals.
    assert "HF_TOKEN" in out or "OUTPUT_REPO" in out


def test_preflight_dataset_aborts_on_missing() -> None:
    """preflight_dataset_access aborts cleanly if dataset doesn't exist."""
    mod = _load_module()
    from huggingface_hub.utils import RepositoryNotFoundError

    class _FakeNotFound(RepositoryNotFoundError):
        def __init__(self, msg: str) -> None:
            Exception.__init__(self, msg)

    err = _FakeNotFound("no such dataset")
    with patch("huggingface_hub.HfApi") as MockApi:
        MockApi.return_value.dataset_info.side_effect = err
        with pytest.raises(SystemExit, match="not found"):
            mod.preflight_dataset_access("does/not/exist", "tok")


def test_preflight_dataset_passes_when_exists() -> None:
    """preflight_dataset_access does not raise when dataset_info succeeds."""
    mod = _load_module()
    fake_info = MagicMock()
    with patch("huggingface_hub.HfApi") as MockApi:
        MockApi.return_value.dataset_info.return_value = fake_info
        mod.preflight_dataset_access("real/dataset", "tok")  # should not raise


def test_preflight_model_aborts_on_gated() -> None:
    """preflight_model_access aborts on GatedRepoError (Llama-3.1-8B path)."""
    mod = _load_module()
    from huggingface_hub.utils import GatedRepoError

    class _FakeGated(GatedRepoError):
        def __init__(self, msg: str) -> None:
            Exception.__init__(self, msg)

    err = _FakeGated("license required")
    with patch("huggingface_hub.HfApi") as MockApi:
        MockApi.return_value.model_info.side_effect = err
        with pytest.raises(SystemExit, match="gated"):
            mod.preflight_model_access("meta-llama/Llama-3.1-8B-Instruct", "tok")

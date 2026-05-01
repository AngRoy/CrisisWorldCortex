# Training Pipeline

## Launch path migration

On April 30 and May 1, three consecutive `hf jobs uv run` launches of
`training/scripts/train_cortex_multi_model.py` entered a RUNNING state but
produced no logs and no resource metrics before cancellation:

- `69f447e2d70108f37ace203d` on `l40sx1`
- `69f44f06d70108f37ace2058` on `l40sx1`
- `69f46d6cd2c8bd8662bd4499` on `h200`

Local diagnosis found that the Python trainer is not the silent-run cause. The
trainer prints flushed startup logs before env health checks, model preflights,
torch import, model loading, or any env-client interaction. Local diagnostics
reached those startup logs immediately, model-info preflights were sub-second,
and the serialized env lifecycle worked against the deployed Space.

The launch path is therefore migrated away from `hf jobs uv run` for Cortex GRPO
and toward explicit `hf jobs run` commands with bash entrypoints. The new pattern
prints a marker at every bootstrap stage, clones one Space repo checkout, installs
the package from that checkout, then runs Python with `python -u`. If a launch
stalls, the last marker identifies the failing stage.

### Bootstrap probe

Run this cheap probe before spending on a GPU training launch:

```bash
hf jobs run --flavor cpu-upgrade --secrets HF_TOKEN --timeout 5m \
  -e ENV_URL=https://angshuman28-crisisworldcortex.hf.space \
  ghcr.io/astral-sh/uv:python3.12-bookworm \
  bash -c "git clone --depth=1 https://huggingface.co/spaces/Angshuman28/CrisisWorldCortex /w && bash /w/training/scripts/bootstrap_probe.sh"
```

Expected result: logs should show `BOOTSTRAP STAGE 1` through
`BOOTSTRAP STAGE 7`.

### Cortex GRPO launch

After the bootstrap probe succeeds, launch Cortex GRPO through the explicit bash
entrypoint:

```bash
hf jobs run --flavor h200 --secrets HF_TOKEN --timeout 30m \
  -e PYTHONUNBUFFERED=1 \
  -e ENV_URL=https://angshuman28-crisisworldcortex.hf.space \
  -e EPI_BRAIN_MODEL=Qwen/Qwen2.5-3B-Instruct \
  -e LOGISTICS_BRAIN_MODEL=microsoft/Phi-3.5-mini-instruct \
  -e ROUTER_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
  -e ROUTER_WARMSTART_REPO=Angshuman28/cortex-router-sft-warmstart \
  -e HUB_REPO_ID=Angshuman28/cortex-router-dryrun \
  -e MAX_TRAIN_STEPS=2 \
  -e GROUP_SIZE=2 \
  -e PUSH_TO_HUB=1 \
  -e LOAD_IN_4BIT=1 \
  -e MIN_FREE_GPU_GB=15 \
  ghcr.io/astral-sh/uv:python3.12-bookworm \
  bash -c "git clone --depth=1 https://huggingface.co/spaces/Angshuman28/CrisisWorldCortex /w && bash /w/training/scripts/launch_cortex_grpo.sh"
```

This command deliberately avoids `hf jobs uv run`. The job command now has
observable stages before dependency installation and before trainer startup.

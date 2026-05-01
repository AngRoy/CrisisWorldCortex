# HF Jobs bootstrap diagnostics

`bootstrap_probe.sh` is a cheap sanity check for the HF Jobs runner before
launching expensive Cortex GRPO jobs. It does not install training dependencies
or load models. It only verifies that the container shell starts, logs stream,
basic environment variables are present, the filesystem exists, optional GPU
tools are visible, Python starts, and the deployed env health endpoint is
reachable.

Run this before spending on a full GPU launch:

```bash
hf jobs run --flavor cpu-upgrade --secrets HF_TOKEN --timeout 5m \
  -e ENV_URL=https://angshuman28-crisisworldcortex.hf.space \
  ghcr.io/astral-sh/uv:python3.12-bookworm \
  bash -c "git clone --depth=1 https://huggingface.co/spaces/Angshuman28/CrisisWorldCortex /w && bash /w/training/scripts/bootstrap_probe.sh"
```

Expected logs should include every marker from `BOOTSTRAP STAGE 1` through
`BOOTSTRAP STAGE 7`. If logs are empty or stop before stage 1, the failure is
in HF Jobs scheduling/container startup, not in the Cortex Python trainer.

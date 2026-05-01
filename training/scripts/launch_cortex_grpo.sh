#!/usr/bin/env bash
set -euxo pipefail
echo "[launch] STAGE 1: bootstrap entered at $(date -u +%FT%TZ)"
echo "[launch] STAGE 2: cloning Space repo"
git clone --depth=1 https://huggingface.co/spaces/Angshuman28/CrisisWorldCortex /workspace
cd /workspace
echo "[launch] STAGE 3: installing deps"
pip install --quiet --upgrade pip
pip install --quiet \
    "datasets" \
    "huggingface_hub" \
    "transformers>=5.0" \
    "peft" \
    "accelerate" \
    "bitsandbytes" \
    "requests" \
    "torch" \
    .
echo "[launch] STAGE 4: deps installed"
pip list | head -30
echo "[launch] STAGE 5: gpu visible"
nvidia-smi
echo "[launch] STAGE 6: launching trainer"
python -u training/scripts/train_cortex_multi_model.py
echo "[launch] STAGE 7: trainer returned $? at $(date -u +%FT%TZ)"

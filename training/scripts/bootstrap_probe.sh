#!/usr/bin/env bash
set -euxo pipefail
echo "BOOTSTRAP STAGE 1: shell entered at $(date -u +%FT%TZ)"
echo "BOOTSTRAP STAGE 2: env vars present"
env | sort | head -40
echo "BOOTSTRAP STAGE 3: filesystem"
ls /
echo "BOOTSTRAP STAGE 4: GPU visible"
nvidia-smi || echo "no nvidia-smi"
echo "BOOTSTRAP STAGE 5: python import test"
python -u -c "import sys; print(f'python {sys.version}')"
python -u -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()}')" || echo "torch not yet installed"
echo "BOOTSTRAP STAGE 6: network reachable"
curl -sf https://angshuman28-crisisworldcortex.hf.space/health
echo ""
echo "BOOTSTRAP STAGE 7: success at $(date -u +%FT%TZ)"

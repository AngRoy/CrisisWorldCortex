# Smoke Gate Runbook

Use this runbook for the Session 7d/lock-and-test smoke gate on Windows.

## Preferred Shells

Use Git Bash or WSL for the canonical command style:

```bash
docker run -d --rm -p 58300:8000 --name cwc-diag crisisworldcortex-env:7d-smoke
until curl -sf http://localhost:58300/health >/dev/null 2>&1; do sleep 2; done
ENV_URL=http://localhost:58300 HF_TOKEN="$HF_TOKEN" uv run python inference.py
docker logs cwc-diag
docker stop cwc-diag
```

## Windows `cmd.exe`

Do not write `set ENV_URL=http://localhost:58300 && uv run ...`.
`cmd.exe` includes the space before `&&` in the variable value, producing
bad values such as `58300 ` or URLs with a literal trailing space.

Use one assignment per line:

```bat
set ENV_URL=http://localhost:58300
set HF_TOKEN=<real-token>
uv run python inference.py
```

For Hugging Face Spaces, also clear Docker mode first:

```bat
set LOCAL_IMAGE_NAME=
set ENV_URL=https://angshuman28-crisisworldcortex.hf.space
set HF_TOKEN=<real-token>
uv run python inference.py
```

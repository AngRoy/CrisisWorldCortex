# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Crisisworldcortex Environment.

This module creates an HTTP server that exposes the CrisisworldcortexEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server import create_web_interface_app as create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

from pathlib import Path

from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles

# Wire types use canonical ``CrisisWorldCortex.models`` (Session 7d):
# the container's wheel install resolves this to one ``sys.modules`` entry,
# matching the deep server modules (graders/, simulator/) that already use
# canonical. The previous dual-fallback fired ``from models import ...``
# (bare) in container, which loaded ``/app/env/models.py`` as a separate
# ``sys.modules`` entry and produced two class identities per
# discriminated-union variant — breaking ``ExecutedAction(action=...)``
# validation in ``apply_tick``.
from CrisisWorldCortex.models import CrisisworldcortexAction, CrisisworldcortexObservation

from .CrisisWorldCortex_environment import CrisisworldcortexEnvironment

# Create the app with web interface and README integration
app = create_app(
    CrisisworldcortexEnvironment,
    CrisisworldcortexAction,
    CrisisworldcortexObservation,
    env_name="CrisisWorldCortex",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/cortex/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="cortex-css")
    app.mount("/cortex/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="cortex-js")

    @app.get("/cortex", include_in_schema=False)
    @app.get("/cortex/", include_in_schema=False)
    async def serve_cortex_frontend() -> FileResponse:
        """Serve the additive Cortex dashboard without replacing /web."""
        return FileResponse(str(FRONTEND_DIR / "index.html"))


def main() -> None:
    """
    Entry point for direct execution via uv run or python -m.

    Parses --host and --port from argv and runs the uvicorn server:
        uv run server
        uv run server --port 8001
        python -m server.app --host 0.0.0.0 --port 8000

    For production deployments, prefer uvicorn directly with multiple workers:
        uvicorn CrisisWorldCortex.server.app:app --workers 4
    """
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

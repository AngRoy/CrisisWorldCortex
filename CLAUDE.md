# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

OpenEnv environment package implementing **CrisisWorld** (regional outbreak-control simulation) and **Cortex** (multi-brain agent with a learned routing policy). Current `step()` is a scaffolded echo; real environment logic lands in a later phase and replaces it. Deployable to Hugging Face Spaces via the Docker SDK.

## Architecture — subsystem boundaries; one CLAUDE.md each

```
  server/               cortex/             baselines/          training/
  CrisisWorld env       Cortex agent        B1 / B2 / B3        GRPO loop +
  (SEIR + graders +     (brains +           flat & fixed-       rollout buffer
  FastAPI wiring)       router + Council)   router agents       + Colab nbs
      ▲                     │                    │                   │
      │  HTTP (client.py) ◀─┘                    │                   │
      └──────────── wire types (models.py) ◀─────┴───────────────────┘

  demo/       replay-only visualization of recorded traces
  scripts/    shell orchestrators (validate, run baselines, plot curves)
  tests/      pytest
```

Wire-protocol classes are template-locked: `CrisisworldcortexAction`, `CrisisworldcortexObservation`, `CrisisworldcortexEnvironment`, `CrisisworldcortexEnv`. Where the design doc says `CrisisWorldEnv`, the codebase uses `CrisisworldcortexEnvironment`.

## Frozen — changing any requires approval

- The four wire-protocol class names above. Do not rename.
- Flat package layout: `pyproject.toml:package-dir` maps `CrisisWorldCortex` to repo root. Do not move `models.py`, `client.py`, or root `__init__.py` off root.
- Dual-import fallback pattern (`try: from ..models / except: from models`) in every `server/` module that imports `models`.
- `pyproject.toml`: deps, entry points, `packages`, `package-dir`.
- `openenv.yaml` keys: `spec_version`, `runtime`, `app`, `port`.
- `server/app.py:create_app(...)` call signature.
- `server/Dockerfile` base image (`ghcr.io/meta-pytorch/openenv-base:latest`), two-stage `uv sync`, CMD.
- Top-level directory set: `server`, `cortex`, `baselines`, `training`, `demo`, `scripts`, `tests`. No new top-level dirs.
- `CRISISWORLD_CORTEX_SYSTEM_DESIGN.md` — outside this repo, read-only.
- `[V2]`-tagged features from the design doc (Communications Brain, Equity Brain, anonymized comparison, `public_communication` action, RLM operators, etc.) are out of MVP scope.

## Forbidden without approval

- Creating a root `Dockerfile` (planned; duplicates `server/Dockerfile` intentionally when added).
- Creating `inference.py` at repo root (planned).
- `.gitignore`-ing and `git rm --cached`-ing the 5 tracked `__pycache__/*.pyc` files from commit `f96cdd2`. Ask first.
- Adding dependencies to `pyproject.toml`.

## Import-graph rule (enforced)

- `cortex/` imports `models` and `cortex/*` only.
- `server/` imports `models`, `openenv.core.*`, `server/*` only. No `cortex/*`, no `training/*`, no `baselines/*`, no `demo/*`.
- `baselines/` imports `models`, `client`, `cortex/*`. Must not import `server/*` — baselines hit the env over HTTP like production.
- `training/` imports `models`, `client`, `cortex/*`, `server.graders` (reward-name constants only). Must not import `server.simulator/*`.
- `demo/` imports `cortex.schemas` (types only) and stdlib. Must not import `server/*`, `training/*`, `baselines/*`, `cortex.council`, `cortex.routing_policy`.
- `models.py` is a leaf — no internal imports.

The graph must be acyclic across directories. Enforced by `tests/test_import_graph.py` when it lands.

## Subsystem CLAUDE.md pointers — each public API is documented in exactly one file

- `server/CLAUDE.md` — CrisisWorld env, SEIR simulator, graders, OpenEnv compatibility constraints.
- `cortex/CLAUDE.md` — brains, subagents, Council Executive, routing policy, hard caps.
- `baselines/CLAUDE.md` — B1, B2 (matched-compute, spec-locked), B3.
- `training/CLAUDE.md` — GRPO trainer, rollout buffer, two Colab notebooks.
- `demo/CLAUDE.md` — trace replay, canned scenarios.
- `tests/CLAUDE.md` — per-subsystem test scope and coverage bar.

Do not restate subsystem APIs in this file or in other subsystem files.

## Import style (repo-wide)

- Wire package: `from CrisisWorldCortex import CrisisworldcortexAction, ...`.
- Dev / research directories (sibling-of-repo-root): bare-name — `import cortex`, `import baselines`, `import training`, `import demo`, `import scripts`.
- Server-internal: `from server.simulator import ...`, `from server.graders import ...`.
- **Cross-boundary into wire types**: when a dev / research directory (`cortex/`, `baselines/`, `training/`, `demo/`) crosses into the wire-protocol package, use `from CrisisWorldCortex.models import …` — **never** bare `from models import …`. Bare-name only applies to dev-directory siblings and to same-subpackage imports inside `server/`. The dual-path import creates distinct `sys.modules` entries and breaks Pydantic discriminator checks across import boundaries (verified by session 4's class-identity bug; `cortex/schemas.py:21` documents the canonicalised import).
- **Cross-boundary into wire types from deep server modules**: files inside `server/` more than one level deep (e.g., `server/simulator/seir_model.py`, `server/graders/outer_reward.py`) cannot use the `try: from ..models / except: from models` fallback — `..models` from a two-level-deep module resolves to a non-existent `CrisisWorldCortex.server.models`, the fallback fires, and bare `models` loads as a separate `sys.modules` entry. Use the absolute path: `from CrisisWorldCortex.models import …`. Session 4 (`cortex/schemas.py:21`) and Session 5a (`server/simulator/seir_model.py`, `server/simulator/tasks.py`) document this with inline comments. The dual-import fallback in `server/CrisisWorldCortex_environment.py` and `server/app.py` works only because they are one level deep (`..models` → `CrisisWorldCortex.models` directly).

## Commands (Git Bash; quote Windows paths)

```bash
uv sync                                                                         # install dependancies into venv 
uv run python -m pytest tests/ -v                                                         # run all tests
uv run python -m pytest tests/test_smoke_env.py::test_reset_returns_valid_observation -v  # single test
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000                      # dev server (auto-reload)
uv run server                                                                   # dev server via entry point
docker build -t CrisisWorldCortex-env:latest -f server/Dockerfile .             # build container
openenv push                                                                    # deploy to HF Spaces
openenv validate                                                                # OpenEnv spec check
```

Endpoints when the server runs: `POST /reset`, `POST /step`, `GET /state`, `GET /schema`, `WS /ws`, `/web` (needs to enabled via environment variable), `/docs`, `/health`.

## Deferred follow-ups (next session)

- **`openenv validate` — resolved (session 1 of implementation).** Now passes all 4 deployment modes. Fix: `main()` refactor + `except ImportError` widening in `server/app.py` (commit `2973968`).
- **Design-doc class-name annotation**: add a footnote in `CRISISWORLD_CORTEX_SYSTEM_DESIGN.md` §13 mapping `CrisisWorldEnv` / `server/env.py` → `CrisisworldcortexEnvironment` / `server/CrisisWorldCortex_environment.py`. Held until the `openenv validate` investigation completes (may affect naming decisions).

## Known local-only artifacts (ignored)

These appear in `git status` after every `uv sync` or test run but **must not be flagged as work to do**:

- `__pycache__/*.pyc` — Python bytecode caches.
- `openenv_CrisisWorldCortex.egg-info/*` — setuptools install metadata.

Both got into early commits before `.gitignore` was current. They regenerate locally on every dependency sync or pytest run, producing tracked diffs that **will not propagate when this branch is pushed** (the remote ignores them). Future sessions: ignore the diff; do not commit, do not flag.

# server/CLAUDE.md

CrisisWorld environment server — world state, simulator, graders, HTTP wiring.

## Belongs here

- FastAPI app factory call (`app.py`).
- Environment subclass (`CrisisWorldCortex_environment.py`, class `CrisisworldcortexEnvironment`).
- Simulator: SEIR dynamics, telemetry delay/noise, resources, policy legality, task configs (`server/simulator/`).
- Graders: outer reward, training reward (4 terms), eval-only metrics (`server/graders/`).

## Does not belong here

Cortex code (brains, subagents, router, metacognition), LLM clients, training loops, rollout buffers, baseline agents, demo visualizers.

## Import rule (binding)

**Inside `server/`, use package-relative imports for server internals**:
`from .simulator import ...` from one-level modules and
`from ..simulator import ...` from nested modules. This works when the
server is loaded as either `CrisisWorldCortex.server.*` (`uv run server`)
or top-level `server.*` (Docker / `uvicorn server.app:app`). Never use
`from CrisisWorldCortex.server...` for server-internal imports.

For `models`, use the canonical package path in every server module that needs it:

```python
from CrisisWorldCortex.models import CrisisworldcortexAction, CrisisworldcortexObservation
```

The server runs under multiple import contexts (`uv run server`,
`uvicorn server.app:app`, Docker `cd /app/env && uvicorn server.app:app`).
Canonical wire imports keep Pydantic model identity stable across those modes.

**Wire-type imports from deep modules (binding)**: two-or-more-levels-deep
files (`server/simulator/*`, `server/graders/*`) must still use
`from CrisisWorldCortex.models import …`; `..models` from those depths
resolves to a non-existent `CrisisWorldCortex.server.models`, and a bare
fallback loads a second `models` module.

## Allowed imports

`CrisisWorldCortex.models`, package-relative `server/simulator/*` and
`server/graders/*`, `openenv.core.*`, stdlib, FastAPI, Pydantic, numpy.

## Forbidden imports

`cortex/*`, `training/*`, `baselines/*`, `demo/*`. These would leak latent state to the agent or form a cross-directory cycle.

## Public APIs (owned here)

- `CrisisworldcortexEnvironment.reset() -> CrisisworldcortexObservation`
- `CrisisworldcortexEnvironment.step(action: CrisisworldcortexAction) -> CrisisworldcortexObservation`
- `CrisisworldcortexEnvironment.state -> State` (read-only; `State` has `episode_id: str`, `step_count: int`).
- `server.simulator.apply_tick(state, action) -> State` — advances one tick deterministically given seed.
- `server.simulator.make_observation(state) -> CrisisworldcortexObservation` — observed layer only.
- `server.simulator.load_task(name: Literal['outbreak_easy','outbreak_medium','outbreak_hard']) -> State` — initial state per task.
- `server.graders.outer_reward(state, action) -> float in [0.0, 1.0]`. **The only env-side reward signal.** Per the Q1 decision (root `CLAUDE.md`), `r_budget` is harness-tracked (not env-tracked) — composed in `training/reward_shaping.py` from `Trajectory.ticks[*].router_steps[*].tokens_spent`, never here.
- `server.graders.training_reward(trajectory) -> dict` with keys `{r_outer, r_proto, r_div_health}` (each `[0.0, 1.0]`) and `constraint_violations` (signed sum of §19 penalties; subtracted directly per §14.3 with γ=1.0). **Does NOT include `r_budget`** — that term is added downstream in `training/reward_shaping.py` from harness-counted token usage.
- `server.graders.eval_metrics(trajectory) -> dict` with keys `{collapse_rate, dissent_value, consensus_calibration, novelty_yield}`.

## Binding contracts

- Every grader scalar reward component lives in `[0.0, 1.0]`. The `constraint_violations` channel in `training_reward(...)` is the explicit exception (signed sum of §19 penalties; subtracted directly per §14.3 with γ=1.0).
- Every grader is non-constant across episodes; `test_reward_non_constancy.py` asserts this.
- `training_reward` (optimized) and `eval_metrics` (observed) live in separate trajectory-log columns. Never combine.
- `reset()` is idempotent; `step()` is deterministic given `state + action + seed`.
- **Latent-layer determinism (MVP)**: latent dynamics are fully deterministic given `(state, action)`. Per-tick RNG is plumbed through `apply_tick` for future stochastic-transition support but no current handler or SEIR step consumes it. Only `make_observation` consumes randomness (Gaussian noise on `reported_cases_d_ago` and `compliance_proxy`). Consequence: episode-level rollout variance comes from action-sequence variance and observation noise — **not** from latent stochasticity. Two episodes with identical action sequences produce identical latent trajectories regardless of `episode_seed`. Session 15 (GRPO trainer) must account for this when computing advantages: same-policy / same-seed groups will have zero latent variance, so advantage signal must come from policy stochasticity (LLM temperature, router sampling) or from intentionally varying seeds across the *observation* path.
- `State` is the single source of truth. It holds latent ground truth (never in observations) plus derived observed fields.
- Observations expose only the `observed` layer. Latent fields leave `server/` only as grader input.
- `SUPPORTS_CONCURRENT_SESSIONS = True` on `CrisisworldcortexEnvironment`. If env state becomes shared between sessions, set to False.

## OpenEnv compatibility constraints (frozen)

- `openenv.yaml:app = server.app:app`.
- `create_app(EnvClass, ActionClass, ObservationClass, env_name="CrisisWorldCortex", max_concurrent_envs=…)` — signature fixed; only `max_concurrent_envs` is tunable.
- `server/Dockerfile` base image `ghcr.io/meta-pytorch/openenv-base:latest`; two successive `uv sync` steps (cache-friendly layering) — keep both.
- Action variants legal at runtime: `deploy_resource`, `request_data`, `restrict_movement`, `escalate`, `reallocate_budget`, `no_op`. `public_communication` is declared in the schema (V2 forward-compat) and must be rejected at runtime with the `-0.1` well-formed-illegal penalty.

## Testing requirements

- `reset()` / `step()` shape smoke (already in `tests/test_smoke_env.py`).
- Each of the 6 MVP outer actions has a round-trip test through the wire protocol.
- Rejection test: `public_communication` action returns the rejection penalty.
- Reward shape smoke: every grader returns values in `[0.0, 1.0]`.
- Reward non-constancy: grader output varies across ≥ 2 synthetic episodes.
- Determinism: `apply_tick(same_state, same_action, same_seed)` returns identical next-state.

## Common failure modes

- Breaking the dual-import fallback: server fails to start in one of the three import contexts.
- Returning a grader value outside `[0.0, 1.0]`: disqualifies the submission.
- Importing from `cortex/*`: `test_import_graph.py` fails.
- Forgetting to thread `seed` through `step()`: training runs stop being reproducible.
- Removing `SUPPORTS_CONCURRENT_SESSIONS`: factory-mode WebSocket sessions silently serialize.

## Build / deploy

```bash
docker build -t CrisisWorldCortex-env:latest -f server/Dockerfile .   # from repo root
openenv push                                                          # HF Spaces
```

# baselines/CLAUDE.md

Flat baselines and the fixed-router Cortex. First-class scientific deliverable — these are what the pitch compares against.

## Belongs here

- `flat_agent.py` — B1: single LLM call per tick.
- `flat_agent_matched_compute.py` — B2: matched-compute self-revision. Exact filename and spec locked in design doc §20.1.1 (line 1211).
- `cortex_fixed_router.py` — B3: full Cortex with a hand-coded (non-learned) router.

Do not add B4 (no-protocol) or B5 (anonymization). Both are `[V2]`.

## Does not belong here

Novel agent architectures, training loops (→ `training/`), LLM-client implementation (reuse `cortex.llm_client`).

## Allowed imports

- `models`.
- `client` (the `CrisisworldcortexEnv` HTTP client — use it the same way production agents do).
- `cortex.llm_client` (shared OpenAI / HF-router wrapper).
- `cortex.council`, `cortex.routing_policy`, `cortex.schemas` (B3 only — B3 is Cortex with a fixed router).

## Forbidden imports

- `server/*` — binding. Baselines hit the env over HTTP, not in-process.
- `training/*`, `demo/*`.

## Binding contracts

- All baselines run under the **same per-tick token budget** as the learned Cortex. Matched-compute is the point; do not widen budgets asymmetrically.
- B2's self-revision policy is locked per design §20.1.1. If an implementation detail conflicts with that spec, the spec wins — ask before deviating.
- Each baseline produces a full trajectory per episode, logged in the same schema as Cortex trajectories (`cortex.schemas.Trajectory`).
- Baselines must use `CrisisworldcortexEnv(base_url=...)`. They must never instantiate `CrisisworldcortexEnvironment()` directly.

## Public APIs (owned here)

- `run_baseline(name: Literal['B1','B2','B3'], task: Literal['outbreak_easy','outbreak_medium','outbreak_hard'], seed: int) -> Trajectory`

## Testing requirements

- Each baseline runs one episode on `outbreak_easy` without errors (smoke).
- Matched-compute assertion: B1 / B2 / B3 per-episode token counts within ±10% on the same task.
- No-server-import test: grep each `baselines/*.py` for `import server` / `from server` — must be empty.

## Common failure modes

- Giving B2 more tokens than B1 — invalidates the pitch's matched-compute claim.
- B3 using a different LLM client than Cortex — same violation.
- In-process env import — breaks the "baselines hit the env the same way as production agents" invariant.
- Implementing B4 or B5 in MVP — out of scope.

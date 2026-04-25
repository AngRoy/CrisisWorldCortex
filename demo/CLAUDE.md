# demo/CLAUDE.md

Replay-only visualization. Live demos fail under judging pressure; ship canned scenarios.

## Belongs here

- `visualizer/trace_renderer.py` — renders a JSON trace as a "council in action" view.
- `visualizer/reward_curve_plot.py` — plots reward curves from training logs.
- `demo_scenarios/*.json` — pre-recorded trajectories for the pitch (e.g. `scenario_flat_fails.json`, `scenario_cortex_holds_dissent.json`).

## Does not belong here

Live agent execution (record offline, replay here). Training logic. Graders.

## Allowed imports

- `cortex.schemas` — typed parse of trace JSON. Types only, no logic.
- stdlib + plotting libs (matplotlib / plotly).

## Forbidden imports

- `server/*`, `training/*`, `baselines/*`.
- `cortex.council`, `cortex.routing_policy` — if you need to re-run the agent, do it offline and ship a new JSON.

## Binding contracts

- Every JSON scenario conforms to `cortex.schemas.Trajectory`.
- Rendering is deterministic: same JSON → same output, modulo timestamps.
- The pitch-demo scenario must showcase B2 overcommit/misallocate vs Cortex dissent-preservation (design §27).
- A pre-recorded demo video (MP4) lives alongside the JSON scenarios as the live-demo fallback.

## Public APIs (owned here)

- `render_trace(json_path: str, out_path: str) -> None`
- `plot_reward_curves(log_paths: list[str], out_path: str) -> None`

## Testing requirements

- Each committed JSON scenario parses into a `Trajectory` without error.
- `render_trace` produces a non-empty output file for each scenario.

## Common failure modes

- Live re-run during the demo — network/Colab flakiness kills the pitch. Replay only.
- Renderer depending on a `cortex.council` instance — import breaks when Cortex API shifts. Keep read-only on types.

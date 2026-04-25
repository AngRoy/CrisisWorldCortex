# tests/CLAUDE.md

Test surface per subsystem. Smoke bar first, boundary tests next, coverage last.

## Belongs here

- `conftest.py` — repo root on `sys.path` for bare-name imports.
- One test module per subsystem boundary (table below).

## Does not belong here

Helpers that mutate real graders, simulator state, or disk. Fixtures that hit the live HF Space — mock or `pytest.skip`.

## Run commands

```bash
uv run python -m pytest tests/ -v                                                               # all
uv run python -m pytest tests/test_smoke_env.py::test_reset_returns_valid_observation -v        # one
uv run python -m pytest --cov tests/                                                            # coverage
```

## Required tests — each maps to exactly one subsystem contract

| File | Scope | Asserts |
|---|---|---|
| `test_package_exports.py` | wire package | Root `__init__` re-exports `CrisisworldcortexAction/Observation/Env`. |
| `test_smoke_env.py` | `server/` env | `reset()` / `step()` return a valid `CrisisworldcortexObservation`. |
| `test_actions_round_trip.py` | `server/` env | 6 MVP outer actions round-trip; `public_communication` is rejected at runtime. |
| `test_reward_shape.py` | `server/graders/` | Every grader returns values in `[0.0, 1.0]`. |
| `test_reward_non_constancy.py` | `server/graders/` | Grader output varies across ≥ 2 synthetic episodes. |
| `test_anti_hivemind_protocol.py` | `cortex/` | 5 protocol steps fire in order; caps enforced (2 rounds, 1 cross-brain challenge, 1 Critic/brain/tick). |
| `test_collapse_detector.py` | `cortex/` | Metacognition flags when all brains recommend the same action. |
| `test_import_graph.py` | repo-wide | No `import server` under `cortex/**`; no `import cortex` under `server/**`; no `import server.simulator` under `training/**`. |
| `test_baselines_smoke.py` | `baselines/` | B1 / B2 / B3 each run one episode on `outbreak_easy`. |
| `test_training_smoke.py` | `training/` | `train_router.main()` runs one episode against a mocked env under 5 s. |

## Binding rules

- Every public API in a subsystem's CLAUDE.md has ≥ 1 test here.
- Coverage target: 80% per subsystem; 100% for `server/graders/` and `cortex/anti_hivemind.py`.
- No test may take > 10 s unless marked `@pytest.mark.slow` and gated behind `--runslow`.
- `test_import_graph.py` uses a fresh subprocess import, not `sys.modules` monkey-patching — the latter passes under contamination.

## Common failure modes

- Smoke test asserting on current-echo values — breaks when real env logic lands. Assert on shape, not value.
- Module-scope env instantiation in tests — slows collection and hides init errors until runtime.
- Tests that hit the HF Space without a skip guard — CI flakes on rate limits.

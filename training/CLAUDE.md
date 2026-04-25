# training/CLAUDE.md

GRPO training for the routing policy. Two Colab notebooks required for hackathon compliance.

## Belongs here

- `train_router.py` — manual PyTorch GRPO loop for the MLP router (Option B primary).
- `rollout_buffer.py` — trajectory collection + router-step serialization.
- `reward_shaping.py` — composes the 4-term training reward from `server.graders` outputs.
- `train_router_colab.ipynb` — Colab-runnable notebook for the MLP router. **Required.**
- `train_flat_agent_trl.ipynb` — TRL `GRPOTrainer` + Unsloth LoRA on a flat agent against `outbreak_easy`. **Required** for finale Unsloth/TRL compliance (design §E.3).
- `configs/grpo_config.yaml` — GRPO hyperparameters.
- `configs/tasks.yaml` — task-curriculum config (easy → medium → hard).

## Does not belong here

Baseline agents (→ `baselines/`). Plotting / reward curves (→ `demo/`, `scripts/`). SEIR dynamics (→ `server/simulator/`).

## Allowed imports

- `models`, `client`.
- `cortex.routing_policy`, `cortex.council`, `cortex.schemas`, `cortex.metacognition`.
- `server.graders` — **reward-name constants only** (e.g. the `training_reward` dict keys). Do not import `server.simulator`. Do not instantiate the env in-process.
- Torch, TRL, Unsloth (compliance notebook only).

## Forbidden imports

- `server.simulator/*` — training hits the env over HTTP like production.
- `baselines/*`, `demo/*`.

## Binding contracts

- **Training-data rows = router steps**, not ticks or rounds. One row per `RoutingAction` emission.
- **Training reward = exactly the 4 terms returned by `server.graders.training_reward`** (see `server/CLAUDE.md` for the dict schema). Never mix in eval-only metrics.
- Training episode length = 10–12 ticks. Eval episode length = 20 ticks (only if training is stable; otherwise eval also runs at 12).
- Temperature > 0 on LLM subagents during rollouts (exploration); temperature = 0 during eval (reproducibility).
- Pin the OpenEnv version in `pyproject.toml` before training runs — finale requires "latest release" at submission.

## Colab notebook contracts

- `train_router_colab.ipynb`: imports CrisisWorld as a local Python module (no Docker in Colab). Runs end-to-end on a fresh Colab T4.
- `train_flat_agent_trl.ipynb`: uses `trl.GRPOTrainer(environment_factory=CrisisworldcortexEnv, ...)` + `unsloth.FastLanguageModel` LoRA wrapper. Runs end-to-end on a fresh Colab T4. **This notebook's absence disqualifies the submission.**

## Public APIs (owned here)

- `train_router.main(config_path: str) -> None`
- `RolloutBuffer.add(router_step: RouterStep) -> None`
- `RolloutBuffer.sample(batch_size: int) -> list[RouterStep]`
- `shape_reward(trajectory: Trajectory) -> float` — weighted combination of the 4 training terms.

## Testing requirements

- `shape_reward` returns a scalar in `[0.0, 1.0]`.
- `RolloutBuffer` round-trips synthetic router steps without data loss.
- `train_router.py` runs 1 episode end-to-end against a mocked env in under 5 seconds (CI smoke).
- Both Colab notebooks execute to completion for ≥ a few hundred training steps on Colab T4 pre-onsite.

## Common failure modes

- Logging ticks as training rows — collapses router's action granularity; GRPO credit assignment breaks.
- Mixing eval metrics into the training reward — inflates the headline curve for reasons the paper can't defend.
- Widening training episodes past 12 ticks — rollouts stop fitting in the GRPO update window; wall-clock explodes.
- Training-reward dict keys drifting from `server.graders.training_reward` — shape-only tests miss this; trainer silently optimizes the wrong signal.
- Missing the TRL compliance notebook at submission — automatic finale failure.

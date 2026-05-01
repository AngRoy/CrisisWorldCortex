# Investigation: HF Jobs training pipeline failures

## TL;DR

This is not primarily an HF Jobs reliability problem. The hard blocker is a real dependency incompatibility triggered by this repo's `openenv-core[core]==0.2.3` pin plus TRL's `GRPOTrainer` import path requiring `mergekit`, whose available releases require older `pydantic` than `fastmcp>=3.0.0` allows.

There are also repo/script issues: the one-script `hf jobs uv run` pattern is brittle for this codebase, the SFT dataset formatting does not match the warmstart script's assumption, and `train_cortex_multi_model.py` currently constructs a trainer but does not call `trainer.train()`.

## Fixes applied after this investigation

These fixes are applied in the local repo:

- `pyproject.toml` and `uv.lock`: downgraded `openenv-core[core]` from `0.2.3` to `0.2.1`, the newest OpenEnv release that resolves with `mergekit`, and capped `fastmcp<3.0.0` plus `pydantic>=2.10.6,<2.11` so the checked-in environment stays mergekit-compatible.
- `inference.py`, `collect_sft_data.py`, `train_b1_grpo.py`, `train_cortex_multi_model.py`, and `minimal_proof.py`: added OpenEnv client compatibility for both API shapes. OpenEnv `0.2.2+` uses `.sync()`; OpenEnv `0.2.1` exposes synchronous `reset()` / `step()` directly.
- `training/scripts/sft_warmstart.py`: fixed SFT prompt formatting so raw observations from `collect_sft_data.py` are rendered through the target tokenizer chat template with the same system prompt shape used by GRPO.
- `training/scripts/train_cortex_multi_model.py`: now fails loudly by default instead of saving/pushing a placeholder untrained adapter. `ALLOW_CORTEX_SKELETON=1` is required for manual construction debugging.
- `training/scripts/minimal_proof.py`: added a no-TRL, no-mergekit proof-training fallback using `transformers`, `peft`, `accelerate`, and one GRPO-like group-relative policy update against the deployed env.
- `notebooks/colab_demo_unsloth_smollm.ipynb`: pinned the Colab install cell to the same OpenEnv/FastMCP/Pydantic-compatible dependency set, added `mergekit`, installed the cloned repo with `--no-deps`, and used the same OpenEnv client compatibility wrapper.

Verification after fixes:

```powershell
uv pip install --dry-run . mergekit
uv pip install --dry-run . "trl>=0.13" mergekit
uv pip install --dry-run . "transformers>=5.0" peft accelerate
uv run ruff check inference.py training/scripts/collect_sft_data.py training/scripts/minimal_proof.py training/scripts/sft_warmstart.py training/scripts/train_b1_grpo.py training/scripts/train_cortex_multi_model.py
uv run ruff format --check inference.py training/scripts/collect_sft_data.py training/scripts/minimal_proof.py training/scripts/sft_warmstart.py training/scripts/train_b1_grpo.py training/scripts/train_cortex_multi_model.py
$env:HF_TOKEN='dummy'; $env:DRY_RUN='1'; $env:PUSH_TO_HUB='0'; uv run python training/scripts/minimal_proof.py
$env:PRE_COMMIT_HOME = "$PWD\.pre-commit-cache"; uv run pytest tests/ -v
```

Observed:

- all resolver checks pass;
- `minimal_proof.py` dry-run reaches deployed Space `/health` and env reset successfully without loading a model or launching HF Jobs;
- Ruff passes;
- `274 passed` with one upstream `fastmcp`/`authlib` deprecation warning.

## Scope

Commands run locally only. I did not launch any HF Jobs and did not modify the deployed Space. I did read prior job logs with `hf jobs logs`.

## Root causes (ranked)

### 1. `mergekit` and `openenv-core==0.2.3` are unsatisfiable together

Evidence in this repo:

- `pyproject.toml` pins `openenv-core[core]==0.2.3`.
- `uv.lock` currently resolves `openenv-core==0.2.3`, `fastmcp==3.2.4`, and `huggingface-hub==1.11.0`.
- `client.py` subclasses `openenv.core.EnvClient`, so installing this repo as a package pulls OpenEnv into the HF Jobs environment.

Local reproduction:

```powershell
uv pip install --dry-run mergekit "git+https://huggingface.co/spaces/Angshuman28/CrisisWorldCortex"
```

Result:

```text
No solution found...
fastmcp>=3.0.0 depends on pydantic[email]>=2.11.7
openenv-core==0.2.3 depends on fastmcp>=3.0.0
all versions of mergekit ... require pydantic below 2.11
all versions of mergekit and openenv-core==0.2.3 are incompatible
```

The smoking-gun job log confirms the same failure:

```powershell
$env:PYTHONIOENCODING='utf-8'; hf jobs logs 69edf0f6d2c8bd8662bcfd4c
```

Result:

```text
all versions of mergekit and openenv-core==0.2.3 are incompatible
openenv-crisisworldcortex==0.1.0 depends on openenv-core[core]==0.2.3
requirements are unsatisfiable
```

Important nuance: this command resolves because `trl` does not force `mergekit` at solve time:

```powershell
uv pip install --dry-run "git+https://huggingface.co/spaces/Angshuman28/CrisisWorldCortex" "trl>=0.13"
```

It resolved to `trl==1.3.0` and `transformers==5.6.2`. The failure appears later because `from trl import GRPOConfig, GRPOTrainer` imports `trl.trainer.callbacks`, which imports `trl.mergekit_utils`, which imports `mergekit`.

Confirmed prior log:

```powershell
$env:PYTHONIOENCODING='utf-8'; hf jobs logs 69edef5ed70108f37ace013a
```

Result:

```text
ModuleNotFoundError: No module named 'mergekit'
RuntimeError: Failed to import trl.trainer.grpo_trainer...
```

So `openenv-core==0.2.3` does not block every TRL install, but it effectively blocks the current TRL `GRPOTrainer` path once the missing `mergekit` import is satisfied.

### 2. The `trl==0.12.2` fallback is not clean with this repo on Python 3.12

These resolve alone:

```powershell
uv pip install --dry-run trl==0.12.2 transformers==4.46.0
```

Result:

```text
Resolved...
tokenizers==0.20.3
transformers==4.46.0
trl==0.12.2
```

But with this repo package:

```powershell
uv pip install --dry-run "git+https://huggingface.co/spaces/Angshuman28/CrisisWorldCortex" trl==0.12.2 transformers==4.46.0
```

Result:

```text
openenv-crisisworldcortex==0.1.0 depends on huggingface-hub>=1.0.0
transformers==4.46.0 depends on huggingface-hub>=0.23.2,<1.0
requirements are unsatisfiable
```

Without the Transformers pin:

```powershell
uv pip install --dry-run "git+https://huggingface.co/spaces/Angshuman28/CrisisWorldCortex" trl==0.12.2
```

Result:

```text
Resolved...
transformers==4.12.2
tokenizers==0.10.3
trl==0.12.2
```

That explains the Python 3.12 failure mode: `tokenizers==0.10.3` has no suitable modern wheel and tries to build from Rust source in the container.

### 3. `hf jobs uv run` is the wrong long-term packaging pattern for these scripts

`hf jobs uv run` uploads one script and installs only the `--with` list. It does not use this repo's `pyproject.toml` unless the repo is explicitly added with:

```text
--with "git+https://huggingface.co/spaces/Angshuman28/CrisisWorldCortex"
```

That pattern made the package importable, but it also means:

- every missing transitive runtime import becomes a paid remote failure;
- the uploaded local script and the installed Space repo can drift;
- dependency state lives in shell command history instead of versioned code;
- the Space package brings the OpenEnv/FastMCP/pydantic constraint into the training job.

PEP 723 inline script metadata would improve reproducibility for `hf jobs uv run`, but it would not solve the core conflict if the script still depends on `openenv-CrisisWorldCortex @ git+...`.

The cleaner pattern for serious training is `hf jobs run` with either:

- a repo checkout plus `uv sync` from a tested lockfile; or
- a prebuilt Docker image with CUDA, Torch, Transformers, OpenEnv, and the local package already installed.

For the next hour, building and validating that image is probably too slow.

### 4. The async failure is fixed in the current local scripts, but it was real in an earlier job

Introspection:

```powershell
uv run python -c "import inspect; from CrisisWorldCortex.client import CrisisworldcortexEnv; print(inspect.iscoroutinefunction(CrisisworldcortexEnv.reset)); print(inspect.iscoroutinefunction(CrisisworldcortexEnv.step)); sync=CrisisworldcortexEnv(base_url='http://example.invalid').sync(); print(inspect.iscoroutinefunction(sync.reset)); print(type(sync).__name__)"
```

Result:

```text
reset coroutine: True
step coroutine: True
sync reset coroutine: False
sync type: SyncEnvClient
```

Current local script status after the fix:

- `training/scripts/collect_sft_data.py` uses `_sync_if_available(CrisisworldcortexEnv(...))` before `env.reset()` and `env.step()`.
- `training/scripts/train_b1_grpo.py` uses the same compatibility wrapper in `make_env()`.
- `training/scripts/train_cortex_multi_model.py` uses the same compatibility wrapper in `make_env()`.
- `training/scripts/sft_warmstart.py` does not touch the env.
- `training/scripts/eval_baselines.py` does not exist.

The prior async job log is still valid evidence for an older script/remote state:

```powershell
$env:PYTHONIOENCODING='utf-8'; hf jobs logs 69eddfbcd2c8bd8662bcfbcf
```

Result:

```text
AttributeError: 'coroutine' object has no attribute 'tick'
RuntimeWarning: coroutine 'EnvClient.reset' was never awaited
```

The current local fix is an OpenEnv-version compatibility wrapper, not `asyncio.run(...)`.

### 5. Some failures were launch/config errors, not dependency conflicts

Confirmed from logs:

```powershell
$env:PYTHONIOENCODING='utf-8'; hf jobs logs 69edf603d70108f37ace016b
```

Result:

```text
bash: line 1: 0.13: No such file or directory
```

That is shell redirection from unquoted `trl<0.13`.

Other confirmed logs:

```powershell
$env:PYTHONIOENCODING='utf-8'; hf jobs logs 69edeed2d2c8bd8662bcfd2c
```

Result:

```text
ModuleNotFoundError: No module named 'unsloth'
```

```powershell
$env:PYTHONIOENCODING='utf-8'; hf jobs logs 69edee30d70108f37ace012b
```

Result:

```text
[FATAL] Angshuman28/phi-3.5-mini-sft-warmstart not found on HF Hub
```

The `HUB_REPO_ID` and `EPI_BRAIN_MODEL` naming issues are real script-interface mistakes, but they are not the final dependency blocker.

## Reproduction

### Reproduce the hard conflict

```powershell
uv pip install --dry-run mergekit "git+https://huggingface.co/spaces/Angshuman28/CrisisWorldCortex"
```

Expected: unsatisfiable due `mergekit` versus `openenv-core==0.2.3 -> fastmcp>=3.0.0 -> pydantic>=2.11.7`.

### Show TRL resolves until mergekit is needed

```powershell
uv pip install --dry-run "git+https://huggingface.co/spaces/Angshuman28/CrisisWorldCortex" "trl>=0.13"
```

Expected: resolves. This does not prove `GRPOTrainer` can import; it only proves the declared dependencies solve.

### Show the `trl==0.12.2` trap

```powershell
uv pip install --dry-run "git+https://huggingface.co/spaces/Angshuman28/CrisisWorldCortex" trl==0.12.2
```

Expected: resolves to `transformers==4.12.2` and `tokenizers==0.10.3`, which is a bad Python 3.12 container target.

```powershell
uv pip install --dry-run "git+https://huggingface.co/spaces/Angshuman28/CrisisWorldCortex" trl==0.12.2 transformers==4.46.0
```

Expected: unsatisfiable because this repo requires `huggingface-hub>=1.0.0`, while `transformers==4.46.0` requires `<1.0`.

### Check OpenEnv release options

```powershell
python -m pip index versions openenv-core
```

Observed:

```text
Available versions: 0.2.3, 0.2.2, 0.2.1, 0.2.0, 0.1.1, 0.1.0
LATEST: 0.2.3
```

Dry-runs:

```powershell
uv pip install --dry-run "openenv-core[core]==0.2.2" mergekit
uv pip install --dry-run "openenv-core[core]==0.2.1" mergekit
uv pip install --dry-run "openenv-core[core]==0.2.0" mergekit
```

Observed:

- `0.2.2` fails the same way as `0.2.3` because it also depends on `fastmcp>=3.0.0`.
- `0.2.1` resolves with `fastmcp==2.9.2` and `pydantic==2.10.6`.
- `0.2.0` resolves.
- `0.1.x` resolves but warns that the package has no `core` extra.

There is no newer OpenEnv release to upgrade to. A downgrade to `0.2.1` might resolve `mergekit`, but it changes the env/client runtime under the submission package and should not be attempted casually in the last hour.

### Confirm current scripts import locally

```powershell
$env:HF_TOKEN='dummy'
$env:OUTPUT_REPO='local/test'
$env:HUB_REPO_ID='local/test'
uv run python -c "import importlib; mods=['training.scripts.collect_sft_data','training.scripts.sft_warmstart','training.scripts.train_b1_grpo','training.scripts.train_cortex_multi_model']; [print(m, 'OK') or importlib.import_module(m) for m in mods]"
```

Observed:

```text
training.scripts.collect_sft_data OK
training.scripts.sft_warmstart OK
training.scripts.train_b1_grpo OK
training.scripts.train_cortex_multi_model OK
```

These are import-only checks. They do not load TRL, Unsloth, Torch, models, datasets, or the remote env because those imports are lazy.

## Recommendations (ranked by effort x impact for the next hour)

### HIGH IMPACT, LOW EFFORT: ship the submission without more HF Jobs launches

Use the deployed Space, 274 green tests, architecture docs, and this investigation as the honest training-pipeline status. This is the only path with high confidence inside the remaining time.

### MEDIUM IMPACT, MEDIUM/HIGH EFFORT: write a no-TRL proof-of-training script

Dependency dry-run for a no-TRL path succeeds:

```powershell
uv pip install --dry-run "git+https://huggingface.co/spaces/Angshuman28/CrisisWorldCortex" "transformers>=5.0" peft accelerate
```

Observed:

```text
Resolved...
transformers==5.6.2
tokenizers==0.22.2
peft==0.19.1
accelerate==1.13.0
```

The smallest plausible proof is:

1. load a small instruct model with `transformers`;
2. add LoRA with `peft`;
3. reset the env for one task/seed with `.sync()`;
4. generate `GROUP_SIZE` completions for the same prompt;
5. parse actions and score each completion through the env;
6. compute group-normalized rewards;
7. backpropagate `-advantage * logprob(completion)` for one optimizer step;
8. save/upload the adapter.

This avoids TRL, Unsloth, and mergekit. It is dependency-feasible, but not low risk in under one hour because generation, logprob alignment, LoRA saving, and HF Jobs GPU behavior still need a real smoke test.

### LOW/MEDIUM IMPACT, HIGH RISK: downgrade `openenv-core`

`openenv-core==0.2.1` resolves with `mergekit`, but the current package and Space were built/tested on `0.2.3`. Downgrading the core env/client dependency this late can create new runtime incompatibilities. I would not do this before submission unless the only goal is a post-deadline experiment.

### LOW IMPACT: keep trying TRL pins in `hf jobs uv run`

This has already consumed many launches. The resolver evidence says the obvious paths are blocked:

- latest TRL GRPO imports `mergekit`;
- `mergekit` conflicts with `openenv-core==0.2.3`;
- `trl==0.12.2` falls to `transformers==4.12.2` unless pinned;
- pinning `transformers==4.46.0` conflicts with this repo's `huggingface-hub>=1.0.0`.

## Specific code issues found

1. `train_cortex_multi_model.py` does not actually train.

   It builds `GRPOTrainer`, logs that Phase 6 is a skeleton, and leaves `trainer.train()` commented out. It then saves and pushes the router adapter. Even if dependencies resolve, this script will not produce a trained Cortex router checkpoint.

2. `train_cortex_multi_model.py` uses a placeholder dataset.

   The dataset prompt is `"placeholder until live B3 corpus collection"`. That is enough to construct a trainer, not enough to train the intended router policy.

3. SFT collection and warmstart formatting disagree.

   `collect_sft_data.py` stores `"prompt": user_prompt`, where `user_prompt` is only the serialized observation. `sft_warmstart.py` says the prompt already contains the chat-template-rendered system and user turns, then appends the completion directly. That means the warmstart data does not match the prompt format used by `train_b1_grpo.py`, which applies `tokenizer.apply_chat_template(...)`.

4. Current async handling is okay in local scripts.

   `collect_sft_data.py`, `train_b1_grpo.py`, and `train_cortex_multi_model.py` all use `.sync()` before `reset()` and `step()`. `sft_warmstart.py` does not call the env. The earlier coroutine failure came from an older script state or a non-sync env client path.

5. Env var naming is easy to mislaunch.

   `sft_warmstart.py` wants `OUTPUT_REPO`; `train_b1_grpo.py` and `train_cortex_multi_model.py` want `HUB_REPO_ID`; Cortex wants `EPI_BRAIN_MODEL`, not `EPI_MODEL`. The scripts fail loudly, but the launch commands need to match each script exactly.

## Answers to the explicit questions

### Is the pyproject dependency set compatible with HF Jobs?

Compatible for inference/env usage, yes. Compatible with current TRL `GRPOTrainer`, effectively no.

The exact bad edge is `openenv-core[core]==0.2.3 -> fastmcp>=3.0.0 -> pydantic>=2.11.7`, which cannot coexist with any available `mergekit` release. Since TRL's GRPO import path currently requires `mergekit`, this repo's current dep set blocks that training stack.

There is no newer `openenv-core` than `0.2.3`. `0.2.2` still fails. `0.2.1` resolves with `mergekit`, but downgrading the env runtime is a risky architectural change.

### Is script-vs-repo-install the right pattern?

For quick single-file scripts, it can work. For this training pipeline, it is the wrong long-term pattern.

PEP 723 inline metadata would make each `hf jobs uv run` script more reproducible, but it does not remove the need to install the repo package. A Docker or `hf jobs run` repo-checkout pattern is cleaner because it can use one tested dependency environment and avoid rediscovering missing imports remotely.

### Are TRL GRPO scripts the only blocker?

No. TRL GRPO is the main dependency blocker for `train_b1_grpo.py` and `train_cortex_multi_model.py`, but `sft_warmstart.py` also depends on TRL and Unsloth, and the Cortex script is currently a skeleton that does not train.

A raw Transformers/PEFT/Accelerate proof-of-training path is dependency-feasible and avoids mergekit. It is still engineering work, not a one-command fix.

### Is async patched in all four scripts?

Yes for the current local scripts:

- `collect_sft_data.py`: uses `.sync()`.
- `sft_warmstart.py`: no env calls.
- `train_b1_grpo.py`: uses `.sync()`.
- `train_cortex_multi_model.py`: uses `.sync()`.
- `eval_baselines.py`: not present.

### Minimum viable submission path

After the local fixes, there are two viable paths:

1. If one final GPU launch is acceptable, use `training/scripts/minimal_proof.py` as the proof-training path because it avoids TRL and mergekit entirely.
2. If no more launches are acceptable, ship documentation and the completed/deployed Workstream A.

Option (a), resolving the dep conflict, now has a local implementation: downgrade `openenv-core` to `0.2.1` and add compatibility wrappers for both OpenEnv client APIs. This passed local resolver checks and 274 tests, but it still needs the updated package state to be present wherever HF Jobs installs the repo.

Option (b), dropping TRL, is implemented as `training/scripts/minimal_proof.py`. Its dry-run verified Space health and reset. The live model-load/update/upload path has not been launched from this investigation.

Option (c) remains honest and defensible: deployed env, green tests, architecture, training plan, and a clear dependency diagnosis explaining why the original HF Jobs training did not complete.

## May 1 silent-run incident

After the dependency and env-client fixes, three `hf jobs uv run` launches of
`training/scripts/train_cortex_multi_model.py` entered RUNNING state with zero
logs and zero resource metrics before cancellation:

- `69f447e2d70108f37ace203d` on `l40sx1`
- `69f44f06d70108f37ace2058` on `l40sx1`
- `69f46d6cd2c8bd8662bd4499` on `h200`

Follow-up diagnosis found that the Python trainer was not the silent-run cause.
The script's first `main()` actions are flushed startup logs, and local
diagnostics reached them immediately. Model access preflights were sub-second,
and the serialized env-client lifecycle worked against the deployed Space. The
absence of both logs and resource metrics points to the `hf jobs uv run`
bootstrap/container layer failing before the Python process was reached.

The Cortex GRPO launch path is migrated to explicit `hf jobs run` entrypoints:
`training/scripts/bootstrap_probe.sh` for a cheap runner/logging probe and
`training/scripts/launch_cortex_grpo.sh` for the real training launch. The new
path prints stage markers, clones the Space repo once, installs dependencies via
`pip install .`, and runs the trainer with `python -u`, avoiding the opaque
`hf jobs uv run` wrapper for this multi-model job.

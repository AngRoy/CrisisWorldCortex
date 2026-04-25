# cortex/CLAUDE.md

Cortex = the multi-brain agent. Client-side only. Never imports `server/*`.

## Belongs here

- 3 brains — Epidemiology, Logistics, Governance (`cortex/brains/`).
- Per-brain subagents — Perception, World Modeler, Planner, Critic, Brain Executive (`cortex/subagents/`).
- Council Executive + 4-phase state machine (`cortex/council.py`).
- Routing policy: MLP head over `MetacognitionState` (MVP primary); LoRA stretch (`cortex/routing_policy.py`).
- Metacognition signals (`cortex/metacognition.py`).
- Anti-hivemind protocol + collapse detectors (`cortex/anti_hivemind.py`).
- Brain-specific observation lenses (`cortex/lenses.py`).
- LLM client wrapper (`cortex/llm_client.py`).
- Cortex-internal schemas (`cortex/schemas.py`).

## Does not belong here

SEIR dynamics, telemetry noise, reward math (→ `server/`). GRPO loop, rollout buffer (→ `training/`). Baseline agents (→ `baselines/`).

## Allowed imports

`models` (wire types), `cortex/*`, torch, OpenAI SDK, stdlib.

## Forbidden imports

`server/*` (binding — no reaching into SEIR constants, latent state, or graders; training hits the env over HTTP like production). `training/*`, `baselines/*`, `demo/*`.

---

## Enforcement rules (all binding)

### Role split

- **Perception is pure Python. Do not introduce an LLM call.**
- **Brain Executive is pure Python. Do not introduce an LLM call.**
- **World Modeler, Planner, Critic are LLM calls. Do not rewrite in Python.**
- **Router-callable subagent set is exactly `{WorldModeler, Planner, Critic}`.** The routing policy must not invoke Perception or Brain Executive.

### Per-tick hard caps (enforce in Council Executive)

- **At most 2 deliberation rounds per tick.** Force convergence after round 2.
- **At most 1 cross-brain challenge per tick total** — not per brain, total across brains.
- **At most 1 Critic call per brain per tick.** A brain's Critic must not run twice in one tick.
- **Read hard token budget per tick from the task config at tick start.** When depleted, the router may only emit `emit_outer_action` or `stop_and_no_op`.
- **Worst-case LLM calls per tick = 19** (9 round-1 + 9 round-2 + 1 cross-brain challenge). Any design exceeding 19 is a bug.

### Execution order

- Perception runs once per brain at tick start. Not router-callable.
- Brain Executive runs once per brain at round end. Not router-callable.
- Only World Modeler / Planner / Critic are invoked by the router, via `call_subagent`.

### Anti-hivemind protocol — exactly 5 steps per deliberation round (MVP; no anonymization)

1. **Private first pass.** Each brain produces its `BrainRecommendation` without seeing peers.
2. **Typed evidence disclosure.** Each recommendation must cite `EvidenceCitation` objects and include a `falsifier` field. Uncited claims zero out `r_proto` for that brain.
3. **Targeted challenge.** Metacognition selects ≤ 1 cross-brain challenge per tick. Structured objection, not free chat.
4. **Dissent preservation.** Minority recommendations ride the emitted action as risk flags and are re-checked next tick. Do not discard.
5. **Constitutional decision.** Council Executive selects: act / request more data / another round (within 2-round cap) / escalate / `no_op`.

Step 4′ (anonymized comparison) is `[V2]`. Do not implement in MVP.

### Router action space — exactly these 6 kinds

- `call_subagent(brain, subagent ∈ {WorldModeler, Planner, Critic})`
- `request_challenge(challenger_brain, target_brain)`
- `switch_phase(new_phase)` — Divergence → Challenge → Narrowing → Convergence. Cannot skip forward.
- `preserve_dissent(tag)`
- `emit_outer_action(OuterAction)` — closes the tick.
- `stop_and_no_op` — closes the tick.

`recurse_in(...)` is `[V2]`. Do not add a 7th router action.

### Phase machine invariants

- Phases may not skip forward (Divergence cannot jump to Convergence).
- Phases may re-enter backward on explicit dissent-triggered flag.
- The first `emit_outer_action` or `stop_and_no_op` step closes the tick, even mid-round.

### Logging contract

- **Training-data row = one router step**, not a tick or round: `(episode_id, tick, round, step_idx, RoutingAction, MetacognitionState, tokens_spent, subagent_report?)`.
- Do not log ticks or rounds as training rows.

### Temperature

- Eval: temperature 0 on all LLM subagents for reproducibility.
- Training rollouts: temperature > 0 on LLM subagents so the router sees exploration.

---

## Public APIs (owned here)

- `Council.step(observation: CrisisworldcortexObservation) -> OuterAction`
- `RoutingPolicy.forward(state: MetacognitionState) -> RoutingAction`
- `cortex.schemas` types: `BrainRecommendation`, `BeliefState`, `CandidatePlan`, `MetacognitionState`, `RoutingAction`, `EvidenceCitation`, `RouterStep`, `Trajectory`.
- `cortex.llm_client` (Session 7+): OpenAI / HF-router wrapper with **per-caller token-counting middleware**. Exposes `tokens_used_for(caller_id) -> int` so harnesses (`inference.py`, `baselines/*`, `training/train_router.py`) can compose `r_budget` per the Q1 decision (root `CLAUDE.md`). The env never sees these counts; the wire protocol carries no `tokens_spent_*` fields.

## Testing requirements

- Protocol-invariant test: after any `Council.step`, the 5 protocol steps ran in order or a short-circuit reason is logged.
- Cap tests: synthetic signals inviting > 2 rounds / > 1 cross-brain challenge / > 1 Critic-per-brain must fail closed, not silently exceed.
- No-LLM-call tests: Perception and Brain Executive must not call the LLM client — mock the client, assert zero invocations.
- Import-graph test: grep `cortex/**.py` for `import server` / `from server` — must return empty.
- Determinism (eval-mode): same observation + same policy checkpoint → identical `OuterAction`.

## Common failure modes

- Not clamping rounds at 2 — silently inflates tick cost, breaks matched-compute baseline comparisons.
- Router calling Perception or Brain Executive — violates role split, pollutes training data.
- Discarding minority recommendations — deletes the dissent-preservation signal the pitch depends on.
- Skipping typed evidence disclosure — grader zeros `r_proto`.
- Sharing a single observation lens across brains — collapses evidence-diversity.
- Adding an LLM call inside Perception or Brain Executive — breaks the "3 LLM calls per brain per round" cost guarantee.

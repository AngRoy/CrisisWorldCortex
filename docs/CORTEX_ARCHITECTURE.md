# Cortex Architecture (Phase A — locked contract for Sessions 9–13)

**Status:** locked. Sessions 9–13 implement against this document. Per-session
proposals reference sections by number ("per Phase A §3.2…"); decisions
recorded here are not re-litigated in implementation sessions.

**Source-of-truth dependencies (binding):**
- `CRISISWORLD_CORTEX_SYSTEM_DESIGN.md` v2.1 — the project design intent,
  particularly §7 (Cortex), §8 (anti-hivemind), §9 (phase state machine),
  §10 (worked example), §11.2 (Cortex schemas), §13 (file structure),
  §14–§19 (rewards), §22 (policy parameterization).
- `cortex/CLAUDE.md` — binding role split, hard caps, anti-hivemind 5-step
  protocol, 6-kind router action space, phase-machine invariants, logging
  contract, temperature rules.
- `cortex/schemas.py` — already-implemented Pydantic types; this doc reuses
  them verbatim and notes the small set of additions Sessions 9–13 must
  introduce.
- `baselines/CLAUDE.md` — for B3 (cortex_fixed_router) integration in
  Session 13.

When this doc and a `CLAUDE.md` disagree, the `CLAUDE.md` wins (it's the
binding contract); flag the divergence to the human reviewer before
implementing.

---

## §1 — Complete data flow

A single CrisisWorld tick triggers exactly one full Cortex pass that
produces exactly one `OuterAction`. The pass walks five layers, with
state propagating in a strict order:

```
                          ┌────────────────────────────────────────┐
  CrisisworldcortexEnv ──►│ 0. Observation arrives                 │
  (HTTP / SyncEnvClient)  │    CrisisworldcortexObservation v[…]   │
                          └────────────────┬───────────────────────┘
                                           ▼
                          ┌────────────────────────────────────────┐
                          │ 1. Lenses (per-brain, deterministic)   │
                          │    cortex.lenses.lens_for(brain, obs)  │
                          │    → BrainLensedObservation × 3        │
                          └────────────────┬───────────────────────┘
                                           ▼
                          ┌────────────────────────────────────────┐
                          │ 2. Perception (per-brain, deterministic│
                          │    Python — NOT router-callable)       │
                          │    → PerceptionReport × 3              │
                          └────────────────┬───────────────────────┘
                                           ▼
   ┌── routing‐policy step ─►┌────────────────────────────────────┐
   │                         │ 3. LLM Subagents (router-callable):│
   │                         │      WorldModeler / Planner / Critic│
   │                         │    Each call returns a typed        │
   │                         │    SubagentReport (BeliefState |   │
   │                         │    CandidatePlan | CriticReport).   │
   │                         └────────────────┬───────────────────┘
   │                                          ▼
   │                         ┌────────────────────────────────────┐
   │                         │ 4. Brain Executive (per-brain,     │
   │                         │    deterministic Python — NOT      │
   │                         │    router-callable)                │
   │                         │    → BrainRecommendation × 3       │
   │                         └────────────────┬───────────────────┘
   │                                          ▼
   │                         ┌────────────────────────────────────┐
   │                         │ 5. Metacognition signals           │
   │                         │    → MetacognitionState            │
   │                         └────────────────┬───────────────────┘
   │                                          ▼
   │                         ┌────────────────────────────────────┐
   │     RoutingPolicy ◄─────│ 6. Council Phase Machine           │
   │     (deterministic for  │    Divergence → Challenge →        │
   │      Session 13;        │    Narrowing → Convergence         │
   │      trainable from     │    enforces hard caps and emits    │
   │      Session 15)        │    one RoutingAction per step      │
   └─────────────────────────└────────────────┬───────────────────┘
                                              ▼
                          ┌────────────────────────────────────────┐
                          │ 7. CouncilDecision → OuterAction       │
                          │    submitted to env via .step(action)  │
                          └────────────────────────────────────────┘
```

**Data flow contract:**

- The env-side observation flows through `CrisisworldcortexEnv.step()` →
  `Council.step(observation)`. The Council never imports `server/*` —
  baselines/CLAUDE.md and cortex/CLAUDE.md both forbid this.
- Per design §7.2, **Perception** is *deterministic Python* and runs
  once per brain at tick start. **Brain Executive** is *deterministic
  Python* and runs once per brain at round end. Only **WorldModeler /
  Planner / Critic** are router-callable LLM subagents.
- The router (`RoutingPolicy.forward`) is the only place an LLM is
  invoked. Each `RoutingAction` of kind `call_subagent` triggers exactly
  one LLM call and produces one `SubagentReport`.
- **Tokens flow through `cortex.llm_client.LLMClient`**. Caller-IDs follow
  the `cortex:<brain>:<subagent>:t<tick>:r<round>:s<step>` format so per-tick
  budget tracking matches the harness pattern from Session 7a.
- **State that persists across rounds within a tick**: the latest
  `BrainRecommendation` per brain, the latest `BeliefState` per brain
  (so round 2 can read round 1's beliefs), the `MetacognitionState`
  signals updated each round, the deliberation-rounds-used counter,
  the cross-brain-challenges-used counter (capped at 1), the per-brain
  Critic-calls-used counter (capped at 1).
- **State that persists across ticks**: the env-side `recent_action_log`
  (already on the wire); Cortex itself is **stateless across ticks** in
  the MVP — design §7.2 V2 note flags long-term memory as `[V2]` only.

---

## §2 — Pydantic schemas at every boundary

Most schema work is already done in `cortex/schemas.py` (Session 4); this
section locks the small set of additions Sessions 9–13 need and pins the
already-existing shapes against any drift.

### Already locked in `cortex/schemas.py` (do not change)

- `EvidenceCitation` — `source ∈ {telemetry, resource, policy, action_log,
  belief, memory}`, `ref: str`, `excerpt: str`.
- `PerceptionReport` — `brain, salient_signals, anomalies, confidence,
  evidence`. Output of the deterministic Python Perception subagent.
- `BeliefState` — `brain, latent_estimates, hypotheses, uncertainty,
  reducible_by_more_thought, evidence`. WorldModeler output.
- `CandidatePlan` — `action_sketch, expected_outer_action, expected_value,
  cost, assumptions, falsifiers, confidence`. Planner output.
- `CriticReport` — `brain, target_plan_id, attacks, missing_considerations,
  would_change_mind_if, severity`. Critic output.
- `BrainRecommendation` — `brain, top_action, top_confidence,
  minority_actions, reasoning_summary, evidence, falsifier, uncertainty,
  tokens_used, anonymous_id (V2 slot)`. Brain Executive output.
- `CouncilDecision` — `action, rationale, preserved_dissent, phase_trace,
  rounds_used, tokens_used`. Council Executive output.
- `MetacognitionState` — `tick, round, phase, inter_brain_agreement,
  average_confidence, average_evidence_support, novelty_yield_last_round,
  collapse_suspicion, budget_remaining_frac, urgency,
  preserved_dissent_count, challenge_used_this_tick`.
- `RoutingAction` — `kind ∈ {call_subagent, request_challenge, switch_phase,
  preserve_dissent, emit_outer_action, stop_and_no_op}` plus per-kind
  optional fields.

### Additions Sessions 9–13 must introduce (locked here)

#### A1. `BrainLensedObservation` (Session 10)

```
class BrainLensedObservation(BaseModel):
    brain: Literal["epidemiology", "logistics", "governance"]
    raw_obs: CrisisworldcortexObservation       # full obs reference (not copy)
    salient_field_ids: List[str]                # e.g. ["regions[*].hospital_load"]
    derived_features: Dict[str, float]          # brain-specific scalar features
    last_reward: float                          # plumbed from B1's pattern
```

**Rationale:** Lenses don't strip fields from the observation (the agent
might need them later) — they project a *salience map* alongside it.
`derived_features` lets each brain pre-compute domain-specific scalars
once and pass them to all three of its LLM subagents without re-reading
the raw obs.

#### A2. `SubagentInput` (Session 9)

```
class SubagentInput(BaseModel):
    brain: Literal["epidemiology", "logistics", "governance"]
    role: Literal["world_modeler", "planner", "critic"]
    tick: int
    round: int
    perception: PerceptionReport
    prior_belief: Optional[BeliefState] = None       # round 2+ only
    prior_plans: List[CandidatePlan] = []            # passed to Critic; empty for WM/Planner
    target_plan_id: Optional[str] = None             # required for Critic
    last_reward: float
    recent_action_log_excerpt: List[ExecutedAction]
```

**Rationale:** Typed inputs make subagent prompts deterministic and
testable. `prior_belief` is `None` on round 1 (nothing to revise yet).
`prior_plans` is empty for WorldModeler and Planner; populated for
Critic so it can attack a specific plan.

#### A3. `RouterStep` (Session 13 — already named in cortex/CLAUDE.md exports)

```
class RouterStep(BaseModel):
    episode_id: str
    tick: int
    round: int
    step_idx: int
    routing_action: RoutingAction
    metacognition_state: MetacognitionState
    tokens_spent: int
    subagent_report: Optional[SubagentReport] = None  # populated when routing_action.kind == "call_subagent"
    policy_kind: Literal["trainable", "deterministic_fallback"] = "trainable"  # which policy emitted this step
```

**Rationale:** Per cortex/CLAUDE.md "Logging contract", the training-data
row IS one router step. This schema makes it easy to dump a full
trajectory as `List[RouterStep]` without bespoke serialisation. The
`policy_kind` field (OQ-3 resolution) tags which policy emitted the
step: `"trainable"` for the Session-15 learned router, `"deterministic_fallback"`
for the Session-13 router used both as B3 baseline AND as the crash
fallback when the trainable router raises mid-tick (see §7 metacognition
layer). GRPO advantage computation filters to `policy_kind == "trainable"`
only — fallback steps are off-policy and excluded from the gradient.
Default is `"trainable"` so B3 round-tripping does not need to set the
field; B3 is a baseline, not training data, so the default value is
inert for B3 trajectories.

#### A4. `Trajectory` (Session 13 — referenced by training/* and inference.py)

```
class Trajectory(BaseModel):
    episode_id: str
    task: Literal["outbreak_easy", "outbreak_medium", "outbreak_hard"]
    seed: int
    router_steps: List[RouterStep]              # flat list, one per router step
    council_decisions_per_tick: List[CouncilDecision]
    rewards_per_tick: List[float]               # obs.reward each tick
    terminal_kind: Literal["none", "success", "failure", "timeout"]
```

These four additions are the **only** new schema types Sessions 9–13 need.
Everything else is already typed in `cortex/schemas.py`.

---

## §3 — Council phase machine

The Council Executive owns four phases (§9 of the design doc) and
exactly one transition arrow from each (with a bidirectional arrow for
dissent-triggered backtrack). Sessions 12 implements the machine;
Sessions 9–11 implement the phase-internal work that the machine drives.

### Phase definitions

| Phase | Purpose | Phase-end criterion |
|---|---|---|
| **Divergence** | Each brain produces an independent `BrainRecommendation` (anti-hivemind step 1). Subagent calls fire here. | Inter-brain diversity sufficient OR ≥ ⅓ of per-tick budget spent OR all 3 brains have produced a recommendation |
| **Challenge** | At most one cross-brain Critic call (the single allowed challenge per tick — cortex/CLAUDE.md cap). | Challenge resolved (challenger received a counter-recommendation or accepted the original) OR challenge cap exceeded OR diminishing novelty |
| **Narrowing** | Council ranks `top_action` candidates from each brain; minority recommendations get tagged for `preserved_dissent`. | Top candidate stable across last router step OR ⅔ of budget spent OR `urgency` flag set |
| **Convergence** | Emit the chosen `OuterAction` and finalise `CouncilDecision`. | First `emit_outer_action` or `stop_and_no_op` step closes the tick — even mid-round. |

### State machine (start → end)

`Council.step(observation)` initialises:

1. `phase = Divergence`
2. `round = 1`
3. `deliberation_rounds_used = 0`
4. `cross_brain_challenges_used = 0`
5. `critic_calls_per_brain = {epi: 0, logistics: 0, governance: 0}`
6. `tick_tokens_used = 0`
7. `lenses = {brain: lens_for(brain, observation) for brain in BRAINS}`
8. `perceptions = {brain: run_perception(brain, lenses[brain]) for brain in BRAINS}` (Python, 0 tokens)

Then the deliberation loop:

```
while True:
    metacog_state = compute_metacognition_signals(...)
    routing_action = routing_policy.forward(metacog_state)
    apply(routing_action)            # may invoke subagent, change phase, etc.
    log_router_step(routing_action, metacog_state, ...)
    if routing_action.kind in {"emit_outer_action", "stop_and_no_op"}:
        break
    if deliberation_rounds_used == 2 and phase == Convergence:
        force_emit()
        break
    if budget_exhausted():
        force_emit_or_no_op()
        break
```

### Transition rules (binding)

- **Forward only**: Divergence → Challenge → Narrowing → Convergence. No
  forward-skipping (cortex/CLAUDE.md "Phase machine invariants").
- **Backward re-entry on dissent**: if `preserve_dissent` was emitted *and*
  `metacog_state.preserved_dissent_count >= 2`, the router may emit
  `switch_phase(Challenge)` to re-open challenge. This is the only
  allowed backward transition. The `>= 2` threshold (Item A resolution)
  reflects "two preserved dissents = real minority views surfaced; a
  single one is noise" — re-opening challenge on a single preserved
  dissent would burn the cross-brain-challenge cap on what may be a
  one-off objection.
- **Round-2 entry trigger** (Item C resolution): the router enters
  round 2 ONLY by emitting an explicit `switch_phase(Divergence)`
  routing action. Council Executive forbids implicit round increment
  (e.g. silently bumping `round = 2` after Convergence-without-emit).
  Rationale: every round boundary is an explicit policy decision,
  which keeps the Session-15 router's training signal clean — the
  trainable router learns "when to spend a second round" as a
  first-class action choice rather than as an emergent side-effect.
- **Round increment**: every time `phase` returns to Divergence (or
  re-enters Challenge from Convergence/Narrowing via a backtrack), the
  Council Executive checks `deliberation_rounds_used`. If it's already 2,
  it overrides the router's `switch_phase(...)` and forces a
  `switch_phase(Convergence)`. Never silently skip — log the override as
  a `RouterStep` with kind `switch_phase` and a `phase_trace` entry
  noting "forced by 2-round cap".

### Termination conditions (any one terminates the tick)

1. Router emits `emit_outer_action` — submit and advance.
2. Router emits `stop_and_no_op` — submit a `NoOp()` and advance.
3. Round 2 + Convergence reached → force `emit_outer_action` with the
   current top-ranked candidate.
4. Per-tick budget depleted (`tick_tokens_used ≥ TICK_BUDGET`) → router
   may only emit `emit_outer_action` or `stop_and_no_op`. If it emits
   anything else, the Council Executive **overrides** to
   `emit_outer_action` (with the current best candidate or NoOp marker
   if no candidate exists yet).
5. Brain failure (`SubagentReport` not received within timeout, see §7) →
   the failing brain's recommendation is treated as `top_action = NoOp()`
   with `top_confidence = 0.0`; the round continues with the other two.

### Phase-to-protocol mapping (Item F resolution)

The 5-step anti-hivemind protocol (cortex/CLAUDE.md, §8.1 of design doc)
maps onto the 4-phase machine as follows. This pin lets
`tests/test_cortex_protocol_invariants.py` assert the exact protocol
step exercised in each phase.

| Anti-hivemind step | Phase |
|---|---|
| 1. Independent reasoning (private first pass) | Divergence |
| 2. Cross-brain critique (typed evidence disclosure + targeted challenge entry) | Challenge phase entry |
| 3. Counter-recommendation (challenge resolved or accepted) | Challenge phase end |
| 4. Anonymized comparison | **Deferred V2 per Decision 56** — not implemented in MVP |
| 5. Final aggregation with preserved dissent | Narrowing + Convergence |

The Session-12 protocol-invariant test asserts that for any `Council.step`
trace, every phase-entry `RouterStep` is paired with exactly the protocol
step listed above (or a logged short-circuit reason — e.g. budget
exhausted, brain failure, override). Step 4 is asserted as
**not-fired** in MVP; if it ever fires, the test treats that as a
forward-leak from a V2 branch into MVP and fails closed.

---

## §4 — Hard cap accounting and preserved-dissent tracking

### Counters (held in Council Executive instance)

| Counter | Cap | Where decremented / incremented |
|---|---|---|
| `deliberation_rounds_used` | ≤ 2 | Incremented each time `phase` returns to Divergence (i.e. round boundary); checked before the router's next `switch_phase`. |
| `cross_brain_challenges_used` | ≤ 1 | Incremented when `RoutingAction.kind == "request_challenge"` is applied; future `request_challenge` actions are blocked by the Council Executive (override → `switch_phase(Narrowing)`). |
| `critic_calls_per_brain[brain]` | ≤ 1 each | Incremented when `call_subagent(brain, "critic")` fires; subsequent Critic calls on the same brain are blocked. |
| `tick_tokens_used` | < `TICK_BUDGET` (default 6000) | See §5. |

### Invariants (asserted at end of `Council.step`)

- `deliberation_rounds_used ≤ 2` — ALWAYS. If violated, raise
  `AssertionError` (NOT a logged warning — this is an integrity break).
- `cross_brain_challenges_used ≤ 1` — ALWAYS.
- `for brain in BRAINS: critic_calls_per_brain[brain] ≤ 1` — ALWAYS.
- `total LLM calls in a tick ≤ 19` — checked end-of-tick. Calls are
  3 brains × 3 LLM subagents × 2 rounds = 18 + 1 cross-brain challenge.

### Preserved-dissent contract

- **What gets recorded:** when `RoutingAction.kind == "preserve_dissent"`
  fires, the Council appends a string tag to a tick-local
  `preserved_dissent: List[str]`. The tag format is
  `"<brain>.<minority_action_kind>:<short_rationale>"`, e.g.
  `"governance.escalate:precautionary_until_R3_confirms"`. Max 80 chars
  per tag (truncate longer rationales — keep the prefix readable).
- **When recorded:** *during* deliberation, by the router. Multiple tags
  can accumulate within one tick.
- **Who reads it:**
  - The Council Executive embeds the list verbatim in
    `CouncilDecision.preserved_dissent`.
  - The training reward composer reads `len(preserved_dissent)` to
    feed the `r_div_health` term (per design §18). This is **eval signal
    that IS in the training reward** — preserved dissent rewards
    diversity behaviorally.
  - The eval-only `dissent_value` metric (design §20.1, score not
    optimized but logged) reads the contents — it scores whether a
    minority warning that was preserved later proved correct (next-tick
    re-check).

### Cap-violation behavior

If the router somehow asks for an action that would violate a cap, the
Council Executive **silently overrides** the action AND logs an
`OverrideEvent` to the trajectory. Specifically:

- `request_challenge` after the challenge cap → drop to
  `switch_phase(Narrowing)` with override note.
- `call_subagent(brain, "critic")` after the Critic cap → drop to
  `call_subagent(brain, "world_modeler")` with override note (give the
  brain another belief-revision pass instead).
- `switch_phase(Convergence)` skipping over Challenge in round 1 → drop
  to `switch_phase(Challenge)` first, with override note.

The override path exists to keep training stable: a learned router that
emits a forbidden action gets the closest-legal action it would have
gotten anyway, so the policy gradient is not corrupted by raw failures.

---

## §5 — Token budget propagation

### Initial state per tick

- `TICK_BUDGET` is read **once at tick start** from
  `state.task_config.cognition_budget_per_tick`. Default = 6000 (locked
  at every TaskConfig per `server/simulator/tasks.py`).
- `Council` exposes `TICK_BUDGET` as a class attribute that can be
  overridden via a constructor kwarg — same shape as B2's
  `tick_budget` parameter (Session 8). Session 16 (training-eval
  re-calibration) will use the override.

### Propagation pattern

- Tokens flow **bottom-up** through `cortex.llm_client.LLMClient`. Each
  subagent call uses a unique caller-id following:
  `cortex:<brain>:<role>:t<tick>:r<round>:s<step_idx>`. The LLMClient
  middleware accumulates per-caller tokens.
- Per-call cost is read from `ChatResponse.prompt_tokens +
  completion_tokens` and accumulated into the Council's
  `tick_tokens_used: int`. This mirrors B2's pattern (Session 8 §4).
- **No reclaim**: each brain implicitly gets ⅓ of the per-tick budget,
  but there's no enforcement at the brain level. The router decides
  call ordering; if Epidemiology consumes more than ⅓, Logistics simply
  has fewer rounds available. This is the matched-compute story —
  Cortex's flexibility comes from the router, not from per-brain quotas.

### Behavior at cap

The router's space contracts at the cap:

- `tick_tokens_used + estimated_next_call_cost ≥ TICK_BUDGET`: the
  Council Executive **rewrites** any `call_subagent` or
  `request_challenge` to `emit_outer_action` (with the current best
  candidate) or `stop_and_no_op` (if no candidate yet).
- `estimated_next_call_cost` uses the same simple-moving-average pattern
  as B2 (window=3, reset at tick start to `_INITIAL_CALL_COST_ESTIMATE = 600`).

### Matched-compute defense

Cortex's per-tick LLM-token consumption is what B2's `tick_budget`
matches against (per design §20.1.1 line 1201:
*"Token budget per tick: identical to Cortex's per-tick token budget in
that task config. Measured by actual tokens consumed, not theoretical
caps."*). Session 16 (eval calibration) will:

1. Run Cortex on each task with the trained router.
2. Measure the median per-tick `tick_tokens_used` across episodes.
3. Update B2's `tick_budget` to match.
4. Re-run B2 baselines for the ablation chart.

---

## §6 — Routing policy interface

### Locked signature

```
class RoutingPolicy(Protocol):
    def forward(self, state: MetacognitionState) -> RoutingAction: ...
```

Sessions 13 (deterministic) and 15 (trainable) ship two implementations
with this exact signature.

### `MetacognitionState` carries (already locked in `cortex/schemas.py`)

- **Tick context**: `tick`, `round` (1 or 2), `phase`.
- **Per-tick deliberation history (aggregate signals only)**:
  `inter_brain_agreement`, `average_confidence`,
  `average_evidence_support`, `novelty_yield_last_round`,
  `collapse_suspicion`.
- **Resource state**: `budget_remaining_frac`, `urgency`.
- **Hard-cap state**: `preserved_dissent_count`,
  `challenge_used_this_tick`.

What it deliberately does **not** carry:
- The full observation (the lensed observation never reaches the router).
- The brain recommendations themselves (those go into
  `inter_brain_agreement` aggregation).
- Raw subagent reports (collapsed into `average_evidence_support`).

This collapse is intentional — design §22 line 1257 says *"Featurize
MetacognitionState into a fixed-length vector (~20–40 dims)"*. A router
trained on this vector with Option B (small MLP head) is the MVP target;
Option A (LoRA on a small instruct model) is the stretch.

### `RoutingAction` (locked in `cortex/schemas.py`)

```
RoutingAction.kind ∈ {
    "call_subagent",         # → brain + subagent ∈ {WM, Planner, Critic}
    "request_challenge",     # → target_brain + challenger_brain (the brain whose Critic will run)
    "switch_phase",          # → new_phase (cannot skip forward)
    "preserve_dissent",      # → tag (string)
    "emit_outer_action",     # → outer_action: OuterActionPayload
    "stop_and_no_op",        # → emit NoOp, terminate tick
}
```

Sessions 13's deterministic router uses a fixed decision table over
`(phase, round, agreement, budget)`. Sessions 15's trainable router is
GRPO over the same input/output shape — identical interface, swappable
implementation.

### Determinism contract (eval-mode)

Per cortex/CLAUDE.md: *"Determinism (eval-mode): same observation +
same policy checkpoint → identical OuterAction."* This requires:

- LLM temperature = 0 in eval (already locked in cortex/CLAUDE.md).
- All RNG seeds threaded explicitly (not pulled from `random.random()`).
- Sessions 13's deterministic router has no internal randomness —
  ties broken by deterministic ordering of brains.
- Sessions 15's trainable router uses argmax in eval mode (sampling is
  for training rollouts only).

---

## §7 — Failure modes per layer and fallback behavior

### Subagent (LLM) layer

| Failure | Detect | Fallback | Logged | Reward signal sees |
|---|---|---|---|---|
| LLM call raises (auth, network, rate limit) | `try/except` in `LLMClient.chat`-equivalent wrapper | Empty `SubagentReport` with `confidence=0.0`, `evidence=[]`. Brain Executive treats this as "no useful input from this subagent". | `[WARN] cortex: <caller_id> llm.chat failed: <exc>` to stderr. Trajectory `RouterStep.subagent_report` set to `None` and `tokens_spent=0`. | `r_proto` zero for that brain (no evidence cited) — already aligned with the protocol-integrity reward. |
| LLM output fails to parse (typed JSON malformed) | Pydantic `ValidationError` on `BeliefState.model_validate(json)` etc. | One retry with a "your previous response failed to parse" prefix; if that also fails, return empty `SubagentReport` as above. | `[WARN] cortex: <caller_id> parse_failure raw=<snippet>`. Counter `parse_failure_count` increments in trajectory. | Same as above — empty report → no evidence → `r_proto = 0` for the brain. |
| Subagent output references nonexistent region or invalid action variant | `OuterActionPayload` discriminated-union validation | Treat as parse failure (above). | Same as above. | Same as above. |

### Brain layer (Brain Executive)

| Failure | Detect | Fallback | Logged | Reward signal sees |
|---|---|---|---|---|
| All 3 subagents returned empty reports | `aggregate_brain_outputs(...)` sees no plans | `BrainRecommendation(top_action=NoOp(), top_confidence=0.0, ..., uncertainty=1.0)` | `[WARN] cortex: brain=<X> all-subagent-empty fallback to NoOp` | The brain's `top_action` is NoOp; if that wins the council vote, env runs a NoOp tick. |
| Pydantic constructor of `BrainRecommendation` raises | (defensive — should not happen if subagent fallbacks are correct) | Same as above. | `[ERROR] cortex: brain=<X> recommendation construction failed: <exc>` | Same as above. |

### Council layer

| Failure | Detect | Fallback | Logged | Reward signal sees |
|---|---|---|---|---|
| Router emits a cap-violating action | Council's cap counters + override logic (§4) | Closest-legal action, recorded as override. | `[INFO] cortex: router action overridden: <kind> -> <kind>` | Training-data row records the OVERRIDDEN action so the policy learns the legal substitute. |
| Per-tick budget exhausted before any candidate exists | `tick_tokens_used ≥ TICK_BUDGET` and `current_top_candidate is None` | Submit synthetic V2-rejected `parse_failure_marker()` (reuse the shared B1/B2 helper). | `[ERROR] cortex: budget exhausted with no candidate; submitting parse_failure_marker` | r_policy = 0 lands on this tick (matches B1/B2 contract). |
| Wire-level `env.step()` raises | `try/except` in `Council.step` outer loop | Emit a `CouncilDecision` with `action=NoOp()`, `phase_trace=["env_step_failed"]`. Episode continues with the env in an unknown state. | `[ERROR] cortex: env.step failed at tick=<N>: <exc>` | obs.reward = 0 for this tick (per inference.py's session 7d handling). |

### Metacognition layer

| Failure | Detect | Fallback | Logged | Reward signal sees |
|---|---|---|---|---|
| `MetacognitionState` field outside [0, 1] (e.g. `inter_brain_agreement = 1.2`) | Pydantic validation in the constructor | Clamp to [0, 1] silently; this is a metric-computation bug worth fixing but should not crash the tick. | `[WARN] cortex: metacog state field <name>=<value> out of range; clamped` | Router sees the clamped value; training learns from clamped state. |
| Routing policy raises (Sessions 15 trainable router crashes mid-tick) | `try/except` in `Council.deliberate()` | Fall back to Sessions 13's deterministic router for the rest of the tick. | `[ERROR] cortex: trainable router crashed: <exc>; using deterministic fallback for tick <N>` | The deterministic router emits a sane action. |

---

## §8 — Test strategy across sessions 9–13

Each session ships its own RED tests; integration smoke gates fire after
Sessions 11, 12, 13.

### Session 9 — Subagents

- **Per-role tests** (`tests/test_cortex_subagents.py`):
  - `test_world_modeler_emits_belief_state` — stub LLMClient returns
    valid `BeliefState` JSON; subagent returns parsed `BeliefState`.
  - `test_planner_emits_candidate_plan` — likewise for `CandidatePlan`.
  - `test_critic_emits_critic_report` — likewise for `CriticReport`.
  - `test_subagent_parse_failure_returns_empty_report` — stub returns
    garbage; subagent returns empty `BeliefState/CandidatePlan/
    CriticReport` with confidence 0.
  - `test_perception_runs_without_llm_call` — stub LLMClient with
    `chat` raising; Perception completes (Python only).
- **Schema tests** (`tests/test_cortex_schemas.py` already exists for
  the existing types; extend for `SubagentInput`).

### Session 10 — Lenses

- **Lens transformation tests** (`tests/test_cortex_lenses.py`):
  - `test_epi_lens_emphasizes_telemetry` — derived features include
    `r_effective_estimate`, `worst_region_infection`.
  - `test_logistics_lens_emphasizes_resources` — derived features
    include `total_inventory`, `hospital_load_max`.
  - `test_governance_lens_emphasizes_legal` — derived features include
    `escalation_unlocked_strict`, `legal_constraints_count`.
  - `test_lens_does_not_strip_raw_obs` — `BrainLensedObservation.raw_obs`
    is the same observation object (or model-equal); not a stripped
    subset.

### Session 11 — Brains (single-brain end-to-end)

- **Brain coordination tests** (`tests/test_cortex_brains.py`):
  - `test_brain_executive_packages_subagent_outputs` — given fixture
    `(perception, belief, plan, critic)`, Brain Executive emits a
    correctly-shaped `BrainRecommendation`.
  - `test_brain_executive_zero_llm_calls` — mock LLMClient,
    instantiate brain, run executive; assert zero LLM calls (Python
    only).
  - `test_brain_handles_empty_subagent_outputs` — fallback path covered.
- **Single-brain end-to-end smoke** (`tests/test_cortex_brain_smoke.py`):
  - One brain processes one observation, returns a `BrainRecommendation`.
  - Total LLM calls = exactly 3 (WM + Planner + Critic).
  - **Integration smoke gate after Session 11**: a single brain runs
    end-to-end on a real observation. Council still doesn't exist yet.

### Session 12 — Council Executive + phase machine

- **Phase machine tests** (`tests/test_cortex_council.py`):
  - `test_council_runs_5_protocol_steps_in_order` — assert the 5 anti-
    hivemind steps fire in order with stub brains.
  - `test_council_caps_deliberation_rounds_at_2` — synthetic signals
    inviting 3+ rounds force convergence at round 2.
  - `test_council_caps_cross_brain_challenges_at_1` — second
    `request_challenge` is overridden.
  - `test_council_caps_critic_calls_per_brain_at_1` — second
    `call_subagent(brain, "critic")` is overridden.
  - `test_council_preserves_dissent_via_routing_action` —
    `preserve_dissent` tag accumulates in `CouncilDecision`.
  - `test_council_emits_outer_action_terminates_tick` — `emit_outer_action`
    closes the tick mid-round.
  - `test_council_token_budget_exhaustion_forces_emit` — when
    `tick_tokens_used ≥ TICK_BUDGET`, only emit / no_op are allowed.
- **Integration smoke gate after Session 12**: Council runs one tick
  end-to-end with stub brains, produces a valid `OuterAction`.

### Session 13 — Metacognition + Routing + B3 (cortex_fixed_router)

- **Metacognition tests** (`tests/test_cortex_metacognition.py`):
  - All `MetacognitionState` fields computable from
    `BrainRecommendation`s + tick context.
  - `inter_brain_agreement` formula tested against hand-crafted cases.
  - `collapse_suspicion` flags when all 3 brains return identical
    `top_action`.
- **Deterministic router tests** (`tests/test_cortex_routing_policy.py`):
  - Decision table fires the expected `RoutingAction` per
    `(phase, round, agreement, budget)`.
  - Determinism: same `MetacognitionState` → same `RoutingAction`,
    bytewise.
- **B3 baseline tests** (`tests/test_baseline_b3.py`):
  - `B3CortexFixedRouter` runs one full episode on `outbreak_easy`.
  - Token totals are within ±10% of B1's per cortex/CLAUDE.md
    (matched-compute test).
- **Integration smoke gate after Session 13**: B3 runs end-to-end via
  `inference.py --agent b3` (forward-compat with the agent_cls hook
  added in Session 8).

### Cumulative test count

- Session 9: ≈ 8 new tests
- Session 10: ≈ 5 new tests
- Session 11: ≈ 6 new tests + smoke
- Session 12: ≈ 10 new tests + smoke
- Session 13: ≈ 12 new tests + B3 + smoke
- **Total: ≈ 41 new tests** on top of the current 143.

---

## §9 — Pre-approved modeling decisions

Decisions are grouped by layer. Each entry: **decision** / **rationale** /
**alternative considered & why rejected**.

### Subagents (Session 9)

1. **WorldModeler prompt structure: SYS = role + schema; USR = perception + last_reward + recent_action_log_excerpt.** / Mirrors B1's structure for matched-compute defensibility. / Considered including raw `CrisisworldcortexObservation` JSON; rejected because perception already extracted the salient signals — re-including raw obs doubles tokens.

2. **Planner prompt structure: SYS = role + action schema (B1's); USR = perception + WM's `BeliefState` JSON + last_reward.** / Planner needs the action schema to emit valid `OuterActionPayload`. The schema reuse from B1 keeps matched-compute clean. / Considered a fully different action vocabulary; rejected — `OuterActionPayload` is the wire contract.

3. **Critic prompt structure: SYS = critic role; USR = perception + target plan + WM belief.** / Critic emits prose `CriticReport` fields; no action schema needed (saves ~400 tokens vs Planner's prompt). / Considered including all sibling brains' plans; rejected — that's a cross-brain challenge (Phase 2), not the per-brain Critic.

4. **All subagent SYS prompts loaded from `cortex/subagents/prompts/<role>.txt` at module load.** / Decouples prompt iteration from code changes; fixture-friendly. / Considered inline f-strings in code; rejected — multi-paragraph prompts in code are unreadable.

5. **Subagent output validated via Pydantic `TypeAdapter[<role-specific-type>]`.** / Already the pattern in `parse_action` (B1); reusable. / Considered hand-rolled JSON parsing; rejected — Pydantic gives discriminated-union safety free.

6. **Empty `SubagentReport` on parse / call failure: `confidence=0.0, evidence=[]`** for any of the three subagent types (with role-specific other fields zeroed/empty). / Honest "no signal" state; downstream r_proto naturally penalizes. / Considered raising the exception; rejected — would break the per-brain isolation contract. **(D-FR-3 reinforcement)** A single subagent's empty fallback does NOT trigger `parse_failure_marker` — that fires only on full-brain emptiness (Brain Executive sees `top_confidence == 0` from all 3 subagents); see Decision 42.

7. **Subagent caller_id format: `cortex:<brain>:<role>:t<tick>:r<round>:s<step_idx>`.** / Mirrors B2's `b2:t<tick>:p<n>:<role>` pattern. / Considered a flat `cortex:<step_idx>` form; rejected — losing the per-brain breakdown loses critical training analytics.

8. **Subagent retry on parse failure: 1 retry max, then fallback to empty.** / Bounded compute, deterministic outcome. / Considered no retry; rejected — first-attempt parse failures are common with smaller models, retry recovery is cheap.

### Lenses (Session 10)

9. **Three lenses: epidemiology, logistics, governance. Each is a function `lens_for(brain, obs) -> BrainLensedObservation`.** / Matches the 3 brains. / **(Item E resolution)** V2 brains (Communications, Equity) deferred entirely per Decision 56 — no MVP stub lens functions. `lens_for(brain, obs)` raises `KeyError` on `brain ∈ {'communications', 'equity'}`. Rationale: stubs returning the raw obs would silently let a V2-leaked code path "work", masking the boundary; raising is the loud-fail behaviour V2 deferral contracts demand.

10. **Epidemiology derived features: `r_effective_estimate, worst_region_infection, transmission_rate_trend`.** / These are exactly the signals an epi-savvy human would compute from telemetry. / Considered raw telemetry only (no derived features); rejected — forces every WM call to redo the same computations.

11. **Logistics derived features: `total_inventory, hospital_load_max, deployment_feasibility_per_region`.** / Operational salience. / Same rationale.

12. **Governance derived features: `escalation_unlocked_strict, legal_constraints_count, restrictions_active_count`.** / Legal/policy salience. / Same.

13. **Lens does NOT strip the raw `CrisisworldcortexObservation` from the lensed object.** / Subagents may need to look at fields the lens didn't emphasize. / Considered stripping for "true" lens isolation; rejected — over-aggressive, would force lens to anticipate every subagent need.

14. **`derived_features` is `Dict[str, float]` (flat dict).** / Easy to log, easy to train on. / Considered a typed schema per brain; rejected — lens features evolve; flat dict trades type safety for iteration speed.

### Brain Executive (Session 11)

15. **Decision rule: argmax over `CandidatePlan.expected_value × confidence`.** / Reasonable heuristic; matches design §7.2. / Considered a weighted vote among multiple plans; rejected — single brain should commit to one top action and surface alternatives as `minority_actions`.

16. **`top_confidence` = the chosen plan's `confidence` × the WM's `(1 - uncertainty)`.** / Multiplicative — both must be high. / Considered just plan confidence; rejected — high-plan-confidence-with-high-WM-uncertainty is a known failure mode the council should distrust.

17. **`minority_actions` = all `CandidatePlan.expected_outer_action` from this round NOT chosen as top.** / Preserves dissent at brain level too. / Considered top-2 only; rejected — design §7.2 says the Brain Exec carries minority forward; arbitrary truncation loses signal.

18. **Brain Executive runs ONCE per brain at round end; not router-callable.** / Per cortex/CLAUDE.md binding. / N/A — locked.

19. **Reasoning summary: a 1–2 sentence string from the Planner's `action_sketch`.** / Fits the 400-char `BrainRecommendation.reasoning_summary` cap. / Considered an LLM call to produce a summary; rejected — Brain Executive must be Python-only per cortex/CLAUDE.md.

20. **`evidence` field on `BrainRecommendation` = union of all `EvidenceCitation` lists from `BeliefState`, `CandidatePlan`, `CriticReport`.** / Ensures the council sees the brain's full evidence chain. / Considered Critic only; rejected — claims with no upstream evidence get zeroed `r_proto`.

21. **Brain identifier strings: `"epidemiology"`, `"logistics"`, `"governance"` (lowercase, full word).** / Readable and grep-friendly. / Considered abbreviations (epi, log, gov); rejected — log-grep collisions.

### Council Executive (Session 12)

22. **Phase initialisation: `Divergence` always.** / Per design §9. / N/A.

23. **Initial `MetacognitionState`: all aggregate fields = 0.0 (or 1.0 for `budget_remaining_frac`).** / Honest pre-deliberation state. / Considered prior-tick carry-over; rejected — Cortex is stateless across ticks (no memory in MVP).

24. **Aggregation rule for the 3 `BrainRecommendation`s: weighted vote on `top_action.kind` first, then on parameters within the winning kind.** / Two-level vote keeps semantically-similar actions together. / Considered exact-match-only; rejected — would force Cortex to flip-flop on minor parameter differences.

25. **Vote weight per brain: `top_confidence × evidence_count`.** / Rewards confident, well-evidenced brains. / Considered uniform weights; rejected — would erase brain-level differences. **(D-FR-2 resolution)** Do NOT multiply by `(1 - uncertainty)` here — Decision 16 already bakes `(1 - uncertainty)` into `top_confidence` via the multiplicative `confidence × (1 - uncertainty)` rule. Adding it again at the council layer would double-count uncertainty and over-penalise the most-uncertain brain. Cross-reference Decision 16 when reviewing this in future sessions.

26. **Tie-breaking: deterministic ordering `epidemiology > logistics > governance`.** / Required for eval determinism. / Considered random; rejected — breaks the eval-mode determinism contract.

27. **Cross-brain challenge selection: when `request_challenge` fires, the *challenger_brain*'s Critic runs against the *target_brain*'s top plan.** / Matches design §10 worked example. / Considered pairwise challenge (both brains' Critics); rejected — exceeds the 1-challenge cap. **(Item B resolution)** Cross-brain Critic USR field includes BOTH the target's `PerceptionReport` AND the challenger's `PerceptionReport` (≈ 200 extra tokens / cross-brain challenge — affordable within `TICK_BUDGET=6000`). Rationale: a logistics Critic challenging an epi plan needs the epi-lens view of the data (otherwise the critique is uninformed) plus its own logistics-lens view (otherwise the challenge has no domain leverage). Single-perception Critic was considered and rejected — would force the Critic to "guess" what the target brain saw.

28. **Phase advancement triggers** (Session 13's deterministic router; Session 15 learns these):
    - Divergence → Challenge: `inter_brain_agreement < 0.4` AND `round 1 complete`.
    - Challenge → Narrowing: challenge resolved OR cap hit.
    - Narrowing → Convergence: top candidate stable across last router step OR `urgency > 0.7` OR `budget_remaining_frac < 0.2`.
    / These thresholds are the deterministic-router defaults; trainable router learns its own. / Other thresholds considered; locked here for Session 13's reproducibility. **(D-FR-1 resolution)** Conservative-only: `< 0.4` is the *single* threshold for `request_challenge` firing. The aggressive variant — also firing on `agreement ∈ [0.4, 0.7] AND round == 1` — was rejected. Reasoning: the challenge cap is 1/tick, so aggressive triggering buys *earlier* challenges, not *more* challenges. Challenging moderate disagreement (the [0.4, 0.7] band) misallocates the cap; high agreement is consensus and must not be challenged at all. Aligns with the anti-hivemind story — challenge real dissent, not moderate disagreement.

29. **Forced convergence on round 2 cap: emit current top-ranked action; if no candidate, emit `parse_failure_marker()`.** / Matches B2's pattern. / Considered emitting `NoOp()` directly; rejected — `parse_failure_marker` correctly fires the r_policy=0 penalty.

30. **`CouncilDecision.phase_trace`: list of strings, one per phase entered, e.g. `["Divergence", "Challenge", "Narrowing", "Convergence"]`.** / Cheap to log; useful for debugging. / Considered timestamped events; rejected — too verbose for trajectory storage.

31. **`CouncilDecision.rationale`: a 1–3 sentence string explaining the chosen action, generated deterministically from the winning brain's `reasoning_summary`.** / No extra LLM cost. / Considered an LLM-generated summary; rejected — would add a per-tick LLM call to the council layer.

### Metacognition (Session 13)

32. **`inter_brain_agreement` formula: 1.0 if all 3 `top_action.kind` match; 0.5 if 2/3 match; 0.0 otherwise.** / Three buckets capture the meaningful states. / Considered continuous parameter-overlap score; rejected — too noisy for the small router.

33. **`average_confidence` formula: `mean(brain_rec.top_confidence for brain_rec in [...])`** across the 3 brains. / Standard. / N/A.

34. **`average_evidence_support` formula: `mean(len(brain_rec.evidence) / max(1, claims_count(brain_rec)))` across the 3 brains** where `claims_count = len(reasoning_summary.split('.')) - 1`. / Approximate but computable from the recommendation alone. / Considered LLM-rated evidence support; rejected — adds LLM call.

35. **`collapse_suspicion`: 1.0 if all 3 brains' `top_action` is bytewise-equal AND all 3 evidence lists are length 0; else 0.0.** / Catches the "all said NoOp with no reasoning" failure. / Considered a graded score; rejected — this is the binary signal eval cares about.

36. **`urgency`: `1.0 - ticks_remaining / max_ticks` clipped + `worst_region_infection_estimate × 0.5`, clamped to [0, 1].** / Both time pressure and crisis severity. / Considered ticks-only; rejected — task-difficulty differences vanish. **(D-FR-4 resolution)** Multiplier confirmed at `0.5`. The alternative — `× 1.0` — was rejected: a 1.0 weight would force premature convergence on `outbreak_hard` precisely when anti-hivemind reasoning matters most. Hard tasks need *more* deliberation, not faster convergence. The 0.5 weight lets infection severity nudge urgency without dominating it.

37. **`novelty_yield_last_round`: only computed in round 2; round 1 always returns 0.0.** / Eval-only signal per design §7.4.3 line 404. / N/A.

### Routing policy (Session 13 deterministic, Session 15 trainable)

38. **Session 13 deterministic router decision table** (binding for B3 baseline):
    - Round 1, Divergence, no recommendations yet: `call_subagent(epi, world_modeler)`, then `(epi, planner)`, then `(epi, critic)`, then logistics, then governance — fixed brain order.
    - End of round 1, agreement < 0.4: `switch_phase(Challenge)` → `request_challenge(challenger, target)` where **`challenger = brain with min(top_confidence)`** and **`target = brain with max(top_confidence)`**, ties broken by Decision 26's deterministic ordering (`epidemiology > logistics > governance`).
    - **All-equal-confidences edge case**: when `min(top_confidence) == max(top_confidence)` across all three brains, do NOT fire `request_challenge` at all. Skip the Challenge phase and `switch_phase(Narrowing)` directly. Cross-brain challenge requires productive asymmetry; identical confidences provide none — Decision-26 tie-break would otherwise collapse to `challenger == target == epidemiology`, violating the cross-brain contract. The 1-challenge cap is preserved (unspent), and the council still has the dissent-preservation channel for surfacing minority recommendations.
    - End of round 2 OR agreement ≥ 0.7: `switch_phase(Convergence)` → `emit_outer_action(council_top)`.
    - Budget < 20% remaining: `emit_outer_action` immediately.
    / Reproduces design §10 worked example. / **(Decision-38 inconsistency fix)** The earlier draft hardcoded `challenger=logistics, target=epidemiology` without justification. The dynamic pair lets the *most-uncertain* brain push back on the *most-confident*, which is where anti-hivemind correction has highest expected value: a low-confidence brain challenging a high-confidence one is exactly the configuration where dissent could productively flip the council's decision. Hardcoded pair was rejected because (a) it presupposes which domain is "always right to be challenged", which the design doc does not assert, and (b) it defeats the point of the metacognition signal feeding the router. The all-equal-confidences edge case is handled by short-circuiting to Narrowing rather than firing a same-brain self-challenge.

39. **Session 15 trainable router: Option B (small MLP head over 24-dim featurized state, discrete output over top-50 most-common (kind, brain, subagent) tuples).** / Per design §22; Colab-compatible. / Option A (LoRA) is the stretch.

40. **Featurization: `MetacognitionState` → `np.ndarray[float, (24,)]`.** / Fixed dim across all phases. / Considered phase-conditional feature sets; rejected — breaks the trainable-router's argmax over a fixed action space.

41. **Routing policy returns the action with highest log-prob (eval) or sampled (training).** / Standard. / N/A.

### Failure / fallback (cross-cutting)

42. **`parse_failure_marker()` reuse from `baselines.flat_agent`** when the council has no candidate at budget exhaustion. / Cross-baseline rejection contract; already public API. / N/A.

43. **All `[WARN]` / `[ERROR]` lines go to stderr with the `cortex:` prefix.** / Consistent with B1/B2 pattern. / N/A.

44. **Trajectory storage: in-memory `List[RouterStep]` per episode; trainer persists at episode end.** / Bounded memory (worst-case 19 calls/tick × 12 ticks = 228 router steps). / Considered streaming to disk; rejected — adds I/O without need.

### Testing (cross-cutting)

45. **All Cortex tests use mock `LLMClient`** with scripted responses; no real-network calls in unit tests. / Same pattern as Sessions 7a/8 stub LLMs. / N/A.

46. **Integration smoke (Sessions 11 / 12 / 13) uses in-process env adapter** (same as B1/B2 tests). / Production B3 hits HTTP per baselines/CLAUDE.md; smoke can short-circuit. / N/A.

47. **`tests/test_cortex_protocol_invariants.py`** (Session 12): asserts post-`Council.step` that the 5 anti-hivemind steps fired in order, OR a short-circuit reason is logged. / Per cortex/CLAUDE.md "Testing requirements". / N/A.

48. **`tests/test_cortex_no_llm_call_in_python_layers.py`** (Session 11/12): mock `LLMClient`, assert zero invocations during Perception or Brain Executive runs. / Per cortex/CLAUDE.md binding. / N/A.

### Tooling (cross-cutting)

49. **All caller_ids are strings (no enum types).** / Matches B1/B2 pattern; pickle-safe. / N/A.

50. **Cortex never imports from `server/*` or `baselines/*`.** / cortex/CLAUDE.md binding; enforced by `tests/test_import_graph.py`. / N/A.

51. **B3's class name: `B3CortexFixedRouter`.** / Mirrors `B1FlatAgent`, `B2MatchedComputeAgent`. / N/A.

52. **B3's filename: `baselines/cortex_fixed_router.py`.** / Per `baselines/CLAUDE.md` line 9. / N/A.

53. **B3 composes `Council(routing_policy=DeterministicRouter())` directly; doesn't subclass.** / Same composition pattern as B1/B2. / N/A.

54. **Council exposes `step_callback` parameter matching B1/B2's signature.** / inference.py and harnesses can stream `B1StepEvent`s for B3 too. / N/A.

55. **`B1StepEvent.action` for B3 = the council's emitted `OuterActionPayload`.** / Same shape as B1/B2. / N/A.

56. **Anonymized comparison (anti-hivemind step 4′ from design §8.1) deferred to V2.** / Already locked in cortex/CLAUDE.md. / N/A.

57. **`recurse_in` (RLM operator) deferred to V2.** / Already locked. / N/A.

58. **Memory subagent (long-term episodic memory) deferred to V2 with stub return.** / Per design §7.2. / N/A.

59. **Communications and Equity brains deferred to V2.** / Per design §7.1. / N/A.

60. **All MetacognitionState fields use `float ∈ [0, 1]` except `tick`, `round`, `preserved_dissent_count`, `challenge_used_this_tick`.** / Already locked in `cortex/schemas.py`. / N/A.

### Phase A review-pass additions (Sessions 9–13 binding)

These decisions land during the Phase A review pass (April 2026); each
is locked to the same standard as Decisions 1–60.

61. **Round-2 entry mechanism** (Item C): the router enters round 2 ONLY by emitting an explicit `switch_phase(Divergence)` routing action. Council Executive forbids implicit round increment. / Every round boundary becomes a first-class policy decision; the Session-15 trainable router learns "when to spend a second round" as an explicit action choice rather than an emergent side-effect — cleaner gradient signal. / Considered implicit round-bump on Convergence-without-emit; rejected — would couple round increment to phase state and obscure the router's training signal. (Cross-reference §3 transition rules.)

62. **Round-2 `prior_belief` encoding when round 1 produced nothing** (Item D): when round 1 produced no useful `BeliefState` (parse failure, empty subagent fallback, or LLM-call exception), round-2 `SubagentInput.prior_belief` is an EMPTY `BeliefState(brain=<X>, latent_estimates={}, hypotheses=[], uncertainty=1.0, reducible_by_more_thought=False, evidence=[])`, NOT `None`. / Different prompt signals: `None` tells the WorldModeler "round 1 has not happened, no history to revise"; an empty `BeliefState` tells it "round 1 happened, produced nothing — start clean but acknowledge the failed pass". Conflating the two would mask the failure mode in the trajectory log. / Considered using `None` for both; rejected — collapses two distinct epistemic states into one.

63. **`PerceptionReport.salient_signals` cap** (OQ-2 resolution): `salient_signals` is `List[str]` with **at most 5 entries** per report. Pydantic validator enforces. / Cap prevents the LLM (in V2 lens-extension experiments) from emitting a paragraph as a "signal"; tests assert content not schema. / Considered a per-brain enum; rejected — over-rigid for early MVP iteration where lens features are still being tuned.

64. **Training-rollout temperature** (OQ-1 resolution): deferred to Session 15. Phase A does not pin a numeric value. `cortex.llm_client.LLMClient.chat(..., temperature: float = 0.0)` already exposes the parameter; Session 15 trainer reads its rollout temperature from the GRPO config and passes it through. / Right session for this decision is Session 15, where the trainer will run early experiments and pick a value based on observed exploration. / Considered pinning `0.7` (common GRPO default); rejected — premature; locks a hyperparameter before any training data exists.

65. **`RouterStep.policy_kind` GRPO advantage filter** (OQ-3 resolution): when the Session-15 trainable router crashes mid-tick and Council falls back to the deterministic router (per §7 metacognition layer), the deterministic-router steps that close the tick are tagged `policy_kind="deterministic_fallback"`. GRPO advantage computation filters to `policy_kind == "trainable"` ONLY — fallback steps are off-policy with respect to the gradient. B3 baseline runs are unaffected: every B3 RouterStep gets the default `policy_kind="trainable"`, but B3 trajectories are not training data so the field is inert for B3. / Cleanest split: keep all router steps on the trajectory (so trainer can still measure failure rate, fallback-tick reward, etc.) but exclude fallback steps from the policy gradient. / Considered dropping fallback steps entirely from the trajectory; rejected — loses observability into how often the trainable router crashes and what the deterministic fallback does instead. (Cross-reference §2 A3 schema and §7 metacognition layer.)

### Decisions resolved during Phase A review pass (April 2026)

All four D-FR items below were resolved in the review pass and folded
into the relevant decision entries above. Listed here for trace-
ability; the binding text now lives on the referenced decisions.

- **D-FR-1 → resolved.** Conservative threshold only (`inter_brain_agreement < 0.4`). Aggressive variant rejected. → Decision 28 updated.
- **D-FR-2 → resolved.** Vote weight stays `top_confidence × evidence_count`; do NOT add `(1 - uncertainty)` factor (already in Decision 16's `top_confidence`). → Decision 25 updated.
- **D-FR-3 → resolved.** `parse_failure_marker` fires only on full-brain emptiness (status quo of Decisions 6 + 42). Single-subagent failures use 1-retry + empty-report fallback. → Decision 6 updated.
- **D-FR-4 → resolved.** `urgency` keeps `worst_region_infection × 0.5`. The 1.0 weight was rejected — would force premature convergence on `outbreak_hard`. → Decision 36 updated.

### Open questions resolved during Phase A review pass

- **OQ-1 → resolved.** Training-rollout temperature deferred to Session 15. → Decision 64 added.
- **OQ-2 → resolved.** `salient_signals` is free-form `List[str]` capped at 5. → Decision 63 added.
- **OQ-3 → resolved.** Trainable-router fallback steps logged with `policy_kind` field on `RouterStep`; GRPO filter excludes fallback. → §2 A3 + Decision 65 added.

### Additional pins resolved during Phase A review pass

- **Item A → resolved.** Backward re-entry threshold pinned at `preserved_dissent_count >= 2`. → §3 transition rules updated.
- **Item B → resolved.** Cross-brain Critic USR includes BOTH target's and challenger's perception (~200 extra tokens / challenge). → Decision 27 updated.
- **Item C → resolved.** Round-2 entry: only via explicit `switch_phase(Divergence)` from router. → §3 transition rules + Decision 61 added.
- **Item D → resolved.** Round-2 `prior_belief` when round 1 failed: empty `BeliefState`, NOT `None`. → Decision 62 added.
- **Item E → resolved.** No V2-brain stub lens functions; `lens_for(brain, obs)` raises `KeyError` on V2 brain ids. → Decision 9 updated.
- **Item F → resolved.** Phase-to-protocol mapping pinned (steps 1, 2, 3, 5 → MVP phases; step 4 → V2-deferred). → §3 phase-to-protocol mapping subsection added.

### Inconsistency resolved during Phase A review pass

- **Decision 38 inconsistency → resolved.** Hardcoded `request_challenge(challenger=logistics, target=epidemiology)` replaced with dynamic pair: `challenger = brain with min(top_confidence)`, `target = brain with max(top_confidence)`, ties broken by Decision 26's deterministic ordering. → Decision 38 updated.

---

## §10 — Implementation sequencing and dependencies

### DAG (read top-to-bottom, branches indicate parallelisable work)

```
Session 9 (Subagents)
    │
    ├─► WorldModeler / Planner / Critic schemas + prompts + tests
    │
Session 10 (Lenses)
    │  (depends on Session 9 only for SubagentInput.perception field shape)
    │
    ├─► Per-brain lens functions + BrainLensedObservation tests
    │
Session 11 (Brains)
    │  (depends on Sessions 9 + 10)
    │
    ├─► Brain Executive Python aggregation
    ├─► Single-brain end-to-end test
    │  ── INTEGRATION SMOKE GATE 1 (after Session 11) ──
    │
Session 12 (Council Executive + phase machine)
    │  (depends on Session 11)
    │
    ├─► Phase machine (4 phases)
    ├─► Hard-cap counters + override logic
    ├─► Preserved-dissent recording
    ├─► CouncilDecision aggregation
    │  ── INTEGRATION SMOKE GATE 2 (after Session 12) ──
    │
Session 13 (Metacognition + Routing + B3)
    │  (depends on Session 12)
    │
    ├─► Metacognition signal computation
    ├─► Deterministic RoutingPolicy (decision table)
    ├─► B3CortexFixedRouter baseline
    │  ── INTEGRATION SMOKE GATE 3 (after Session 13) ──
    │
Session 15 (out of Phase A scope) (Trainable router)
    │
    ├─► Featurization → MLP head
    ├─► GRPO loop on stored trajectories
    ── (drop-in replacement for Session 13's router via the locked Protocol) ──
```

### Per-session inputs / outputs

| Session | Inputs (what must exist) | Outputs (what unlocks the next session) |
|---|---|---|
| 9 | `cortex/schemas.py`, `cortex/llm_client.py` (already in repo) | `cortex/subagents/{world_modeler,planner,critic}.py` files; `tests/test_cortex_subagents.py` green |
| 10 | Session 9 outputs; `CrisisworldcortexObservation` schema (locked) | `cortex/lenses.py` with `lens_for(brain, obs) -> BrainLensedObservation`; lens tests green |
| 11 | Sessions 9 + 10 | `cortex/brains/{epidemiology,logistics,governance}.py` files; brain Python aggregation in `cortex/brains/executive.py`; single-brain smoke green |
| 12 | Session 11 | `cortex/council.py` with `Council.step(observation) -> CouncilDecision`; phase machine tests green; Council smoke green |
| 13 | Session 12 | `cortex/metacognition.py` (signal computation); `cortex/routing_policy.py` (DeterministicRouter only — trainable router is Session 15); `baselines/cortex_fixed_router.py` (B3); B3 smoke green |

### Integration smoke gates

- **Gate 1 (Session 11 end):** a single brain takes a real observation
  on `outbreak_easy`, makes 3 LLM calls (mocked), returns a valid
  `BrainRecommendation`. No council yet. Pass = unblock Session 12.
- **Gate 2 (Session 12 end):** Council instantiated with 3 brains and
  the deterministic-router stub (placeholder until Session 13). Runs
  one tick, emits one `CouncilDecision`, returns a valid `OuterAction`.
  Pass = unblock Session 13.
- **Gate 3 (Session 13 end):** B3CortexFixedRouter runs a full episode
  on `outbreak_easy` via in-process env (smoke) and via Docker (manual
  smoke per Session 7c pattern). LLM-call count per tick is in [9, 19].
  Token total per episode is within ±10% of B1's. Pass = ship Phase A.

### Dependency on already-shipped infrastructure

- `cortex.llm_client.LLMClient` (Session 7a): caller-id token tracking.
- `baselines.flat_agent.parse_failure_marker` (Session 8): rejection
  marker reuse.
- `B1StepEvent / StepCallback` (Session 8): callback contract.
- `tests/test_import_graph.py`: structural enforcement; will start
  flagging cortex/* if anything imports server/*.
- `inference.py` (Session 7b/8): `agent_cls` parameter is the forward-
  compat hook for B3 + future Cortex-with-trainable-router.

---

## Final report

### Word count

≈ 8,275 words after the Phase A review pass — about 275 over the
original 8,000 soft ceiling. (Initial draft: ≈ 6,529 words; review
pass added ≈ 1,750 words of resolution prose, including per-resolution
"why rejected" rationale to keep §9's decision-format consistent.)
The overage is accepted as the cost of folding all 13 resolutions
inline rather than as a separate addendum file — single source of
truth for Sessions 9–13.

### Sections completed (10/10)

§1 Data flow / §2 Schemas / §3 Phase machine (now includes phase-to-
protocol mapping) / §4 Hard caps + dissent / §5 Token budget /
§6 Routing policy / §7 Failure modes / §8 Test strategy / §9 Pre-
approved decisions (65 entries — 60 initial + 5 review-pass additions) /
§10 Sequencing.

### Decisions auto-resolved (rationales recorded inline in §9)

65 modeling decisions, grouped by layer:
- 8 subagent decisions
- 6 lens decisions
- 7 brain executive decisions
- 10 council executive decisions
- 6 metacognition decisions
- 4 routing policy decisions
- 3 failure / fallback decisions
- 4 testing decisions
- 12 tooling / V2-deferral decisions
- **5 review-pass additions (61–65)**: round-2 entry mechanism,
  round-2 `prior_belief` encoding, `salient_signals` cap, training-
  rollout temperature deferral, `RouterStep.policy_kind` GRPO filter.

### Phase A review pass — resolutions (April 2026)

All 4 D-FR items, 3 OQ items, 6 additional pins (Items A–F), and 1
Decision-38 inconsistency resolved. Each is folded into the relevant
inline decision; the §9 "resolved during Phase A review pass"
subsection lists the cross-references for traceability.

| ID | Resolution (1-line) | Lands in |
|---|---|---|
| D-FR-1 | Conservative-only `< 0.4`; aggressive variant rejected. | Decision 28 |
| D-FR-2 | Vote weight stays `top_confidence × evidence_count`; no extra `(1 - uncertainty)` factor (Decision 16 already bakes it in). | Decision 25 |
| D-FR-3 | `parse_failure_marker` fires only on full-brain emptiness; single-subagent failures use empty-report fallback. | Decision 6 |
| D-FR-4 | `urgency` keeps `worst_region_infection × 0.5`; 1.0 weight rejected. | Decision 36 |
| OQ-1 | Training-rollout temperature deferred to Session 15. | Decision 64 (new) |
| OQ-2 | `salient_signals` is free-form `List[str]` capped at 5. | Decision 63 (new) |
| OQ-3 | `policy_kind` field on `RouterStep`; GRPO filter excludes fallback. | §2 A3 + Decision 65 (new) |
| Item A | Backward re-entry pinned at `preserved_dissent_count >= 2`. | §3 transition rules |
| Item B | Cross-brain Critic USR includes target + challenger perception (~200 extra tokens). | Decision 27 |
| Item C | Round-2 entry: only via explicit `switch_phase(Divergence)`. | §3 + Decision 61 (new) |
| Item D | Round-2 `prior_belief` when round 1 failed: empty `BeliefState`, NOT `None`. | Decision 62 (new) |
| Item E | No V2-brain stub lens; `lens_for` raises `KeyError` on V2 brain ids. | Decision 9 |
| Item F | Phase-to-protocol mapping pinned (steps 1, 2, 3, 5 → MVP phases; 4 → V2-deferred). | §3 phase-to-protocol mapping subsection |
| Decision 38 inconsistency | Hardcoded `(challenger=logistics, target=epidemiology)` → dynamic `(min top_confidence, max top_confidence)`, ties via Decision 26. | Decision 38 |

### Standing-by status

**Decisions locked. Session 9 unblocked.** Phase A is complete; no
remaining flagged items, no remaining open questions. Sessions 9–13
implement against this document. Future revisions to Phase A
contracts require explicit ✅ in a new review pass.

### What is NOT in this document

- No Python code (per the no-code constraint).
- No Session 14+ details (training reward composition, eval harness,
  HF Spaces deploy). Phase A intentionally bounds at Session 13.

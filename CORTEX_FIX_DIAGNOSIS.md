# Cortex GRPO env-interaction failure — diagnosis

## Hypothesis: confirmed

The Cortex script fails because **two `EnvClient` instances hold open
WebSocket sessions to the same HF Space simultaneously** — `env`
(`make_env()` per outer training iteration) and `candidate_env`
(`make_env()` once before the while loop, kept alive across all
candidates). The 1000/OK error is `websockets.ConnectionClosedOK` from
the gateway closing one session while the other remains live; close-code
1000 is the normal-closure code returned on both sides of the dead one.

`EnvClient.__init__` does not open a socket; `connect()` is called
lazily on the first `_send_and_receive`. Both clients open on the first
tick of the first episode, and from then on we hold two concurrent
sessions throughout training.

## What the working scripts do

- **`minimal_proof.py:score_completion`** — `make_env()` → `reset` →
  `step` → `env.close()` per scoring call, in a `try/finally`. One
  client at a time.
- **`collect_b3_corpus.py:collect`** — `env = make_env()` per episode,
  reset and step inside, `env.close()` in `finally`. One client at a
  time.
- **`inference.py:main`** — one env client per task, closed before the
  next task starts.

None of them ever hold two clients open concurrently.

## Fix

Drop `candidate_env`. Reuse the per-episode `env` for both candidate
scoring (`reset` + replay prefix + `step`) and committing the best
action (`reset` + replay prefix + `step`). One WebSocket session per
training step; `(GROUP_SIZE + 1)` resets per tick.

"""env.step() must populate obs.reward = outer_reward(state, action).

Two binding regressions for the Session 7a env-step wiring (design §15
+ root CLAUDE.md "Latent-layer determinism (MVP)" note):

1. ``test_step_does_not_compose_terminal_bonus`` — at the terminal
   tick, ``obs.reward`` must equal ``outer_reward(state, action)``
   exactly. The +/-0.20 ``terminal_bonus`` is composed downstream by
   the trainer (design §14.3), never by env.step().

2. ``test_step_reward_in_range`` — across a 12-tick rollout, every
   ``obs.reward`` returned by ``env.step`` is inside ``[0.0, 1.0]``.
"""

from CrisisWorldCortex.models import CrisisworldcortexAction, NoOp
from CrisisWorldCortex.server.CrisisWorldCortex_environment import (
    CrisisworldcortexEnvironment,
)
from CrisisWorldCortex.server.graders import outer_reward, terminal_bonus


def _force_failure_terminal(env: CrisisworldcortexEnvironment) -> None:
    """Push every region into a catastrophic-I state so the next
    apply_tick's ``_advance_terminal_state`` sets terminal='failure'.

    Design §6.4: >=3 regions with I > 0.30 triggers failure terminal.
    Mutating ``env._world_state`` directly is fine here — Pydantic
    validate-on-assignment is off, and the next ``apply_tick`` runs
    SEIR (which barely moves an already-saturated state) and then
    detects the catastrophic count.
    """
    assert env._world_state is not None
    for region in env._world_state.regions:
        region.S, region.E, region.I, region.R = 0.0, 0.0, 0.95, 0.05


def test_step_does_not_compose_terminal_bonus() -> None:
    """At the terminal tick, ``obs.reward`` is per-tick r_outer ONLY.

    Verifies design §14.3: ``episode_return = Sum_t r_outer + terminal_bonus``
    — the bonus is summed by the trainer, never bundled into the
    env-emitted per-tick scalar.
    """
    env = CrisisworldcortexEnvironment()
    env.reset()
    _force_failure_terminal(env)

    action = CrisisworldcortexAction(action=NoOp())
    obs = env.step(action)

    expected = outer_reward(env._world_state, action.action)
    assert obs.reward is not None, "env.step must set obs.reward"
    assert obs.reward == expected, (
        f"obs.reward={obs.reward!r} should equal outer_reward(state, action)="
        f"{expected!r} (no terminal_bonus shift). State.terminal="
        f"{env._world_state.terminal!r}"
    )

    # Sanity: we did reach a failure terminal, and the bonus would have
    # been -0.20. If the env had bundled it into obs.reward, the value
    # would be at most expected - 0.20 (and likely negative, since
    # expected is small under high-I).
    assert env._world_state.terminal == "failure", (
        "test setup invariant: catastrophic state must yield failure"
    )
    assert terminal_bonus(env._world_state) == -0.20
    assert obs.reward >= 0.0, (
        "obs.reward must stay in [0,1] even at failure terminal — "
        "any negative value indicates the bonus leaked in"
    )


def test_step_reward_in_range() -> None:
    """Every obs.reward across a full 12-tick episode is in [0.0, 1.0]."""
    env = CrisisworldcortexEnvironment()
    env.reset()

    rewards: list[float] = []
    for _ in range(12):
        obs = env.step(CrisisworldcortexAction(action=NoOp()))
        assert obs.reward is not None, "env.step must set obs.reward each tick"
        rewards.append(obs.reward)

    for i, r in enumerate(rewards, start=1):
        assert 0.0 <= r <= 1.0, f"tick {i}: obs.reward={r!r} out of [0.0, 1.0]"

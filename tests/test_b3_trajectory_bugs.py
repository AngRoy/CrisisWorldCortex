"""Tests for B3 baseline trajectory bugs H8, H9.

H8: B3 run_episode must return tokens_total.
H9: B3 must not hardcode accepted=True; must read from obs.
"""

from baselines.cortex_fixed_router import B3CortexFixedRouter


def test_b3_trajectory_includes_tokens_total() -> None:
    """H8: B3 run_episode return dict must include tokens_total key."""
    # We don't need to run a real episode — just check the return dict shape.
    # Use the in-process adapter from test_baseline_b3.
    from CrisisWorldCortex.server.simulator import load_task, apply_tick, make_observation
    from CrisisWorldCortex.models import CrisisworldcortexAction, NoOp
    from tests._helpers.llm_stub import StubLLMClient

    class _InProcessEnvAdapter:
        """Minimal sync adapter for testing without HTTP."""
        def __init__(self, task: str, seed: int, max_ticks: int):
            self._state = load_task(task, episode_seed=seed)
            self._state.max_ticks = max_ticks
        def reset(self, **kw):
            return make_observation(self._state)
        def step(self, action):
            self._state = apply_tick(self._state, action.action)
            return make_observation(self._state)

    stub = StubLLMClient(
        scripted_responses=['{"kind": "no_op"}'] * 50,
        prompt_tokens_per_call=10,
        completion_tokens_per_call=5,
    )
    agent = B3CortexFixedRouter(env=_InProcessEnvAdapter("outbreak_easy", 0, 3), llm=stub)
    result = agent.run_episode(task="outbreak_easy", seed=0, max_ticks=3)

    assert "tokens_total" in result, (
        f"B3 trajectory missing tokens_total key. Keys: {list(result.keys())}"
    )


def test_b3_accepted_reflects_actual_env_response() -> None:
    """H9: action_history 'accepted' must not be hardcoded True."""
    # We verify by checking the code structure — action_history should
    # reference obs.recent_action_log, not hardcode True.
    import ast
    with open("baselines/cortex_fixed_router.py") as f:
        source = f.read()

    # Check that '"accepted": True' literal is not present in any
    # action_history.append call
    assert '"accepted": True' not in source, (
        "B3 cortex_fixed_router.py hardcodes 'accepted': True in action_history"
    )

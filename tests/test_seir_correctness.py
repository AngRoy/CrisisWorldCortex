"""SEIR correctness: mass conservation and resource transfer accuracy.

H1: test_kits must transfer removed I mass to R (mass conservation).
H2: reallocate_budget must not destroy resources via int() truncation.
"""

from CrisisWorldCortex.models import DeployResource, NoOp, ReallocateBudget
from CrisisWorldCortex.server.simulator import apply_tick, load_task


def _seir_sum(region) -> float:
    return region.S + region.E + region.I + region.R


def test_test_kits_preserves_seir_mass() -> None:
    """H1: After deploying test_kits, S+E+I+R must still equal 1.0."""
    state = load_task("outbreak_easy", episode_seed=0)
    # Deploy a large batch of test_kits to region R1
    action = DeployResource(region="R1", resource_type="test_kits", quantity=500)
    state = apply_tick(state, action)
    # Run a second tick with NoOp so the pending effect is applied
    state = apply_tick(state, NoOp())

    for region in state.regions:
        total = _seir_sum(region)
        assert abs(total - 1.0) < 1e-9, (
            f"Region {region.name}: S+E+I+R={total}, expected 1.0 "
            f"(S={region.S}, E={region.E}, I={region.I}, R={region.R})"
        )


def test_test_kits_transfers_i_to_r() -> None:
    """H1: test_kits should move infected to recovered, not just delete them."""
    state = load_task("outbreak_easy", episode_seed=0)
    action = DeployResource(region="R1", resource_type="test_kits", quantity=500)
    state = apply_tick(state, action)

    # Snapshot R before pending effects fire
    r1 = next(r for r in state.regions if r.region == "R1")
    r_before = r1.R

    state = apply_tick(state, NoOp())

    r1_after = next(r for r in state.regions if r.region == "R1")
    # R should have increased (I mass transferred to R)
    assert r1_after.R >= r_before, (
        f"R1.R should increase after test_kits effect: "
        f"before={r_before}, after={r1_after.R}"
    )


def test_reallocate_budget_amount_1_transfers_at_least_1() -> None:
    """H2: reallocate_budget with amount=1 must not destroy the unit."""
    state = load_task("outbreak_easy", episode_seed=0)
    initial_mobile = state.resources.mobile_units
    action = ReallocateBudget(
        from_resource="test_kits", to_resource="mobile_units", amount=1
    )
    state = apply_tick(state, action)
    assert state.resources.mobile_units >= initial_mobile + 1, (
        f"mobile_units should gain at least 1 unit: "
        f"before={initial_mobile}, after={state.resources.mobile_units}"
    )


def test_reallocate_budget_small_amounts_not_destroyed() -> None:
    """H2: int(amount * 0.95) truncates to 0 for amount=1; round() does not."""
    state = load_task("outbreak_easy", episode_seed=0)
    initial_mobile = state.resources.mobile_units
    total_transferred = 0
    # Transfer 5 units one at a time
    for _ in range(5):
        state_copy = state.model_copy(deep=True)
        action = ReallocateBudget(
            from_resource="test_kits", to_resource="mobile_units", amount=1
        )
        state = apply_tick(state, action)
    gained = state.resources.mobile_units - initial_mobile
    # With round(1*0.95)=1 per transfer, we should gain 5.
    # With int(1*0.95)=0, we'd gain 0.
    assert gained >= 5, (
        f"5 x reallocate(amount=1) should transfer at least 5 units, "
        f"got {gained}"
    )

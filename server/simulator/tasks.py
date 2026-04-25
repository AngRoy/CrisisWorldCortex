# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task configurations and ``load_task`` factory for CrisisWorld (design §6.5).

Three MVP tasks: ``outbreak_easy``, ``outbreak_medium``, ``outbreak_hard``.
Difficulty gradient comes from R_0, telemetry delay/noise, resource
inventory, legal constraints, and superspreader events — not from per-task
budget (Q2: cognition_budget_per_tick = 6000 across all tasks).
"""

from __future__ import annotations

from typing import Dict, List, Literal

# Wire-protocol imports use the absolute path (canonicalized per root CLAUDE.md;
# see seir_model.py's import block for the rationale — two-levels-deep modules
# can't use the dual-import fallback without hitting the dual-sys.modules trap).
from CrisisWorldCortex.models import LegalConstraint, ResourceInventory

from .seir_model import (
    ChainBeta,
    RegionLatentState,
    SuperSpreaderEvent,
    TaskConfig,
    WorldState,
)

# ============================================================================
# TASK_CONFIGS — locked per design proposal §1b
# ============================================================================

TASK_CONFIGS: Dict[str, TaskConfig] = {
    "outbreak_easy": TaskConfig(
        name="outbreak_easy",
        region_count=4,
        max_ticks=12,
        base_R0=1.5,
        default_cross_beta=0.01,
        chain_betas=[],
        telemetry_delay_ticks=1,
        telemetry_noise_stddev_cases=0.02,
        telemetry_noise_stddev_compliance=0.05,
        cognition_budget_per_tick=6000,
        initial_resources=ResourceInventory(
            test_kits=1000,
            hospital_beds_free=500,
            mobile_units=20,
            vaccine_doses=2000,
        ),
        initial_compliance=0.95,
        initial_seir_hot=(0.95, 0.02, 0.03, 0.0),
        initial_seir_quiet=(0.999, 0.0, 0.001, 0.0),
        hot_regions=["R1"],
        quiet_regions=["R2", "R3", "R4"],
        superspreader_schedule=[],
        legal_constraints=[],
    ),
    "outbreak_medium": TaskConfig(
        name="outbreak_medium",
        region_count=4,
        max_ticks=12,
        base_R0=2.0,
        default_cross_beta=0.05,
        chain_betas=[],
        telemetry_delay_ticks=2,
        telemetry_noise_stddev_cases=0.10,
        telemetry_noise_stddev_compliance=0.10,
        cognition_budget_per_tick=6000,
        initial_resources=ResourceInventory(
            test_kits=500,
            hospital_beds_free=300,
            mobile_units=10,
            vaccine_doses=800,
        ),
        initial_compliance=0.85,
        initial_seir_hot=(0.92, 0.03, 0.05, 0.0),
        initial_seir_quiet=(0.999, 0.0, 0.001, 0.0),
        hot_regions=["R1", "R2", "R3"],
        quiet_regions=["R4"],
        superspreader_schedule=[],
        legal_constraints=[],
    ),
    "outbreak_hard": TaskConfig(
        name="outbreak_hard",
        region_count=5,
        max_ticks=12,
        base_R0=2.5,
        default_cross_beta=0.03,
        chain_betas=[
            ChainBeta(from_region="R1", to_region="R2", beta=0.10),
            ChainBeta(from_region="R2", to_region="R3", beta=0.10),
            ChainBeta(from_region="R3", to_region="R4", beta=0.10),
            ChainBeta(from_region="R4", to_region="R5", beta=0.10),
        ],
        telemetry_delay_ticks=3,
        telemetry_noise_stddev_cases=0.20,
        telemetry_noise_stddev_compliance=0.15,
        cognition_budget_per_tick=6000,
        initial_resources=ResourceInventory(
            test_kits=200,
            hospital_beds_free=150,
            mobile_units=5,
            vaccine_doses=400,
        ),
        initial_compliance=0.75,
        initial_seir_hot=(0.93, 0.03, 0.04, 0.0),
        initial_seir_quiet=None,
        hot_regions=["R1", "R2", "R3", "R4", "R5"],
        quiet_regions=[],
        # The superspreader event is HIDDEN by design (§6.5). It perturbs latent
        # state but does not reliably surface through telemetry — the spike's
        # contribution to I (~+0.05) is below the noise floor (stddev=0.20 in
        # fractional units, = ±200 cases on population 1000). Detection by an
        # agent requires inferring from secondary signals: cascade amplification
        # through cross-region β (R3 → R4 → R5 chain with β=0.10), hospital_load
        # creep on R3, compliance-proxy degradation under sustained restrictions.
        #
        # Indexing convention (verified by session-5a→5b calibration check):
        #   - apply_tick called when state.tick=N injects the spike before the
        #     SEIR step, then the SEIR step amplifies the spiked I, and the
        #     post-step I value lands in region.history_I[N+1] after state.tick
        #     advances to N+1.
        #   - make_observation at state.tick=T with delay=D reads
        #     region.history_I[max(0, T-D)].
        #   - So fires_at_tick=7 with delay=3 means the spike's first observable
        #     trace is at observation tick 11 (= 8 + 3), not tick 9. The
        #     surfaces_at_tick field below records the design's nominal-spec
        #     value; the actual observable trace lags it by 2 ticks under the
        #     current implementation's history_I indexing.
        #
        # Future grader/eval-metric code should read state.regions directly
        # (not via make_observation) to detect the spike for ground-truth
        # purposes. Reward signal in outer_reward.py (session 6) reflects
        # cascade outcomes, not direct spike observation.
        superspreader_schedule=[
            SuperSpreaderEvent(
                region="R3",
                fires_at_tick=7,
                surfaces_at_tick=9,
                magnitude_I=0.05,
            ),
        ],
        legal_constraints=[
            LegalConstraint(
                rule_id="L1",
                blocked_action="restrict_movement.strict",
                unlock_via="escalate",
            ),
        ],
    ),
}


def load_task(
    name: Literal["outbreak_easy", "outbreak_medium", "outbreak_hard"],
    episode_seed: int = 0,
    max_ticks: int = 12,
) -> WorldState:
    """Construct the initial WorldState for a named task.

    Per Q3: ``max_ticks`` defaults to 12 (training); pass 20 for eval.
    """
    if name not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {name!r}. Valid: {sorted(TASK_CONFIGS.keys())}")

    config = TASK_CONFIGS[name].model_copy()
    config.max_ticks = max_ticks

    regions: List[RegionLatentState] = []
    for region_id in config.hot_regions:
        S, E, I, R = config.initial_seir_hot
        regions.append(
            RegionLatentState(
                region=region_id,
                S=S,
                E=E,
                I=I,
                R=R,
                true_compliance=config.initial_compliance,
                history_I=[I],
                pending_effects=[],
                noise_reduction_ticks=0,
            )
        )
    if config.initial_seir_quiet is not None:
        for region_id in config.quiet_regions:
            S, E, I, R = config.initial_seir_quiet
            regions.append(
                RegionLatentState(
                    region=region_id,
                    S=S,
                    E=E,
                    I=I,
                    R=R,
                    true_compliance=config.initial_compliance,
                    history_I=[I],
                    pending_effects=[],
                    noise_reduction_ticks=0,
                )
            )

    return WorldState(
        task_name=name,
        task_config=config,
        episode_seed=episode_seed,
        tick=0,
        max_ticks=max_ticks,
        regions=regions,
        resources=config.initial_resources.model_copy(),
        restrictions={},
        legal_constraints=list(config.legal_constraints),
        escalation_level=0,
        escalation_unlocked_strict=False,
        superspreader_schedule=list(config.superspreader_schedule),
        recent_action_log=[],
        consecutive_safe_ticks=0,
        terminal="none",
    )

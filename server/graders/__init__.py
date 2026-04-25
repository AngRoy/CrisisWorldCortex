# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public re-exports for the CrisisWorld graders package.

Per ``server/CLAUDE.md``: this package owns ``outer_reward`` (the only
env-side reward signal), with ``training_reward`` and ``eval_metrics`` to
land in later sessions. ``terminal_bonus`` is exposed alongside
``outer_reward`` because the trainer composes them per design §14.3
(``episode_return = Σ_t r_outer + terminal_bonus``).
"""

from .outer_reward import outer_reward, terminal_bonus

__all__ = [
    "outer_reward",
    "terminal_bonus",
]

"""The one fixed Triple command pair used by main Figures 4--8.

Use the accepted center-node solver with the requested 30,000-iteration budget.
The prior 10,000-iteration cache records remain distinct. No connected
Triple grid, long-branch sweep, or observation figure is requested here.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .triple_long_iteration import _cached_node, _caps, _configs, SEED_OFFSET
from .cache import digest_array


def ensure_main_triple_endpoint(ctx):
    key = "main_triple_endpoint"
    if key in ctx.memo:
        return ctx.memo[key]
    cap, checkpoints = _caps(ctx)
    conventional, fe = _configs(ctx, cap)
    seed = int(ctx.config.random_seed + SEED_OFFSET)
    initial = np.random.default_rng(seed).uniform(-np.pi, np.pi, len(ctx.positions_m))
    index, node = _cached_node(ctx, (1, 1), initial=initial, seed=seed,
        conventional_config=conventional, fe_config=fe,
        conventional_cap=cap, checkpoints=checkpoints)
    bank = dict(objects={index: node}, initial_phase=initial, seed=seed,
        conventional_cap=cap, fe_cap=sum(fe.smooth_stage_maxiters),
        checkpoints=checkpoints, nodes=((1, 1),))
    ctx.tables["main_triple_endpoint_protocol"] = pd.DataFrame([dict(
        target_configuration="fixed accepted Triple center", node_count=1,
        conventional_cap=cap, fe_cap=sum(fe.smooth_stage_maxiters), seed=seed,
        fe_stage_maxiters=str(tuple(fe.smooth_stage_maxiters)),
        conventional_checkpoints=str(checkpoints),
        initial_phase_content_id=digest_array(initial),
        triple_long_branch_test="removed", evidence_status="SMOKE")])
    ctx.memo[key] = bank
    return bank

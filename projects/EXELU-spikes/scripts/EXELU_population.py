"""
Pooled multi-unit activity per region, and the cross-session summary.

**What pooling is.** Every spike from every `mua` cluster in a region is merged into one
train, then treated as a single virtual unit. Cortex becomes unit -1, hippocampus unit -2.
Because `SpikeStore` returns pooled trains like any other unit, the block PSTH, the responder
test and the transition-triggered analysis all run on them unchanged.

**What pooling costs, stated because it must go in the report.**

* A pooled train has **no refractory period**, so every contamination diagnostic is
  meaningless for it and none is computed.
* The pool is **dominated by its highest-rate members**. It is closer to "population rate near
  these contacts" than to "the average neuron". A large pooled modulation is therefore
  consistent with a few strongly driven units rather than with broad engagement, and the two
  cannot be distinguished from the pooled trace alone — which is why the individual clusters
  are plotted too.
* Whether `good` clusters join the pool is a real choice, exposed as
  `params.pool_include_good`. Excluding them matches the conventional meaning of "MUA";
  including them gives a truer population rate. Default is to exclude.
* No spike-count filter is applied to pool membership: a 50-spike cluster contributes 50
  spikes, and excluding it would be arbitrary.

**The magnitude that gets saved.** Four numbers, because no single one is adequate:
`delta_hz` (interpretable, scale-dependent), `pct_change` (scale-free, unstable at low rate),
`mod_index` (bounded, well-behaved), and `peak_z` (how far the response stands out of that
region's own trial-to-trial variability).
"""

from __future__ import annotations

# Importing EXELU_paths puts <framework>/scripts on sys.path, so the shared
# analysis modules below resolve wherever this file is imported from.
import EXELU_paths  # noqa: F401

from pathlib import Path

import numpy as np
import pandas as pd

import EXELU_paths as paths
import EXELU_records as records
import EXELU_regions as regions
from EXELU_config import Params
from psth import block_psth
from responders import trial_rates
from spiketrains import SpikeStore
from statstools import paired_wilcoxon, sign_flip_permutation
from EXELU_triggers import Trigger

POOL_IDS = {regions.CORTEX: -1, regions.HIPPOCAMPUS: -2}
CROSS_SESSION_FILE = "pooled_mua_regulation.csv"


def register_pools(store: SpikeStore, units: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Create one pooled virtual unit per region. Returns a unit table describing them."""
    labels = ["mua", "good"] if params.pool_include_good else ["mua"]
    members = units[units.label.isin(labels)]

    rows = []
    for region, pool_id in POOL_IDS.items():
        in_region = members[members.region == region]
        store.register_pool(pool_id, in_region.unit.astype(int).tolist())
        rows.append({
            "unit": pool_id,
            "label": "mua_pooled",
            "region": region,
            "n_clusters": len(in_region),
            "n_spikes": int(in_region.n_spikes.sum()),
            "depth_um": float(in_region.depth_um.median()) if len(in_region) else np.nan,
            "fr_hz": float(in_region.fr_hz.sum()),
            "contam_pct": np.nan,
            "cell_type": "n/a (pooled)",
        })
    return pd.DataFrame(rows)


def regulation_table(
    session, store: SpikeStore, pooled_units: pd.DataFrame,
    triggers: dict[str, Trigger], params: Params, fs: float,
) -> pd.DataFrame:
    """Up/down-regulation magnitude of each pooled region, per condition.

    The window is exactly [-psth_pre_s, +psth_post_s] around block onset.
    """
    rows = []
    for condition, trigger in triggers.items():
        for _, pool in pooled_units.iterrows():
            spikes = store.spikes(int(pool.unit))
            profile = block_psth(
                spikes, trigger.block_onsets,
                params.psth_pre_s, params.psth_post_s, params.psth_bin_s, fs,
            )
            pre, post = trial_rates(
                spikes, trigger.block_onsets, params.psth_pre_s, params.psth_post_s, fs
            )

            during = profile["centres_s"] >= 0
            z_during = profile["z"][during]
            fr_pre, fr_post = float(pre.mean()), float(post.mean())

            rows.append({
                "session": session.id,
                "animal_id": records.animal_id(session),
                "genotype": records.genotype(session),
                "region": pool.region,
                "condition": condition,
                "n_clusters": int(pool.n_clusters),
                "n_trials": len(pre),
                "fr_pre_hz": fr_pre,
                "fr_stim_hz": fr_post,
                "delta_hz": fr_post - fr_pre,
                "pct_change": (100.0 * (fr_post - fr_pre) / fr_pre) if fr_pre > 0 else np.nan,
                "mod_index": ((fr_post - fr_pre) / (fr_post + fr_pre)) if (fr_post + fr_pre) > 0 else np.nan,
                "peak_z": float(np.max(z_during)),
                "trough_z": float(np.min(z_during)),
                "mean_z": float(np.mean(z_during)),
                "baseline_sigma_hz": profile["sigma_hz"],
                "p_wilcoxon": paired_wilcoxon(post, pre),
                "p_permutation": sign_flip_permutation(post, pre, params.responder_n_permutations),
            })

    table = pd.DataFrame(rows)
    table["direction"] = np.where(table.delta_hz > 0, "up", "down")
    return table


def append_across_sessions(
    table: pd.DataFrame, project: Path | None = None, filename: str = CROSS_SESSION_FILE
) -> Path:
    """Upsert this session's rows into the cross-session table.

    Keyed on (session, region, condition), so re-running a session *replaces* its rows rather
    than duplicating them. That is what makes the notebook safe to re-run and what lets the
    future batch script simply loop.
    """
    path = paths.cross_session_dir(project) / filename
    key = ["session", "region", "condition"]

    if path.is_file():
        existing = pd.read_csv(path)
        incoming = set(map(tuple, table[key].values.tolist()))
        keep = ~existing[key].apply(tuple, axis=1).isin(incoming)
        combined = pd.concat([existing[keep], table], ignore_index=True)
    else:
        combined = table

    combined.sort_values(key).reset_index(drop=True).to_csv(path, index=False)
    return path

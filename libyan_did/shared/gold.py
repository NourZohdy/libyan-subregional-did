"""Shared store for human same/different (merge) decisions.

One CSV at ``outputs/gold/merge_decisions.csv``, one row per decided actor pair,
keyed by ``(actor_a, actor_b)`` where each side is ``<region>/<playlist>#<global_actor>``.

Both writers go through here so their schema can never drift:
  • the per-playlist **Validate clustering** window (within-playlist gray-zone pairs)
  • the corpus-wide **Speaker Validation** app (within + cross-playlist pairs)
"""

from __future__ import annotations

import pandas as pd

from libyan_did.shared import config

MERGE_CSV = config.GOLD_DIR / "merge_decisions.csv"
MERGE_DEC_COLUMNS = [
    "actor_a", "actor_b", "similarity", "kind", "cross_region",
    "decision", "resolved_region", "notes", "validator", "timestamp",
]


def load_merge_decisions() -> dict:
    """``{(actor_a, actor_b): row_dict}`` for every decided pair (empty if none yet)."""
    if MERGE_CSV.exists():
        m = pd.read_csv(MERGE_CSV)
        return {(str(r["actor_a"]), str(r["actor_b"])): r.to_dict() for _, r in m.iterrows()}
    return {}


def save_merge_decisions(decs: dict) -> None:
    MERGE_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(decs.values())).reindex(columns=MERGE_DEC_COLUMNS).to_csv(
        MERGE_CSV, index=False)

"""Param-hash state-locking controller for resumable incremental runs.

Hashes the segmentation parameter signature into a state file. If the params change,
the run must hard-reset (re-segment everything); otherwise it soft-updates, processing
only ``file_id``s not already recorded. Pure-Python — unit-testable here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def params_hash(params: dict) -> str:
    """Deterministic hash of a parameter dict (order-independent)."""
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_state(state_path: str | Path) -> dict:
    p = Path(state_path)
    if not p.exists():
        return {"params_hash": None, "processed_file_ids": []}
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_state(state_path: str | Path, state: dict) -> None:
    p = Path(state_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)


def plan_run(state_path: str | Path, params: dict, all_file_ids: list[str]) -> dict:
    """Decide what to process.

    Returns a dict:
        mode            : "reset" | "incremental"
        to_process      : list of file_ids to (re)segment
        new_state       : state dict to persist AFTER processing succeeds
    """
    new_hash = params_hash(params)
    state = load_state(state_path)
    old_hash = state.get("params_hash")
    processed = set(state.get("processed_file_ids", []))

    if old_hash != new_hash:
        # Params changed (or first run) -> hard reset: everything is stale.
        return {
            "mode": "reset",
            "to_process": list(all_file_ids),
            "new_state": {"params_hash": new_hash, "processed_file_ids": list(all_file_ids)},
        }

    # Soft incremental: only unseen file_ids.
    to_process = [fid for fid in all_file_ids if fid not in processed]
    merged_ids = sorted(processed.union(all_file_ids))
    return {
        "mode": "incremental",
        "to_process": to_process,
        "new_state": {"params_hash": new_hash, "processed_file_ids": merged_ids},
    }

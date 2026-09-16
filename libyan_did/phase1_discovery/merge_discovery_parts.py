"""Merge per-region discovery part-files into the single review registry.

Each discovery agent writes discovery/parts/<region>.psv. This merges them with the
existing discovery/candidates.psv, de-duplicating by canonical channel_id (falling back
to url). Rows the human already decided (status approved/rejected in candidates.psv) win
over fresh pending duplicates, so a re-merge never clobbers your review.

Writes discovery/candidates.psv (a one-time .premerge backup is kept).

Usage (from libyan_did_pipeline/):
    python scripts/merge_discovery_parts.py
"""

from __future__ import annotations

import sys
from pathlib import Path


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import pandas as pd  # noqa: E402

from libyan_did.shared import config
from libyan_did.shared import registry  # noqa: E402

DISC = config.DISCOVERY_DIR
PARTS = DISC / "parts"
MAIN = DISC / "candidates.psv"
COLS = registry.COLS


def main() -> None:
    frames: list[pd.DataFrame] = []
    if MAIN.exists():
        frames.append(registry.read(MAIN))
    parts = sorted(PARTS.glob("*.psv")) if PARTS.exists() else []
    for p in parts:
        n = len(registry.read(p))
        print(f"[part] {p.name}: {n} rows")
        frames.append(registry.read(p))
    if not frames:
        raise SystemExit("no candidates.psv and no parts/*.psv to merge")

    allrows = pd.concat(frames, ignore_index=True)
    allrows = allrows[allrows["url"].str.strip() != ""]
    allrows["__key"] = allrows.apply(
        lambda r: (r["channel_id"].strip() or r["url"].strip()), axis=1)
    # decided rows (approved/rejected) win over fresh pending duplicates
    allrows["__pri"] = allrows["status"].isin(["approved", "rejected"]).astype(int)
    allrows = allrows.sort_values("__pri", ascending=False, kind="stable")
    before = len(allrows)
    merged = (allrows.drop_duplicates("__key", keep="first")
              .drop(columns=["__key", "__pri"]).reset_index(drop=True).fillna(""))

    registry.write(merged, MAIN, backup=DISC / "candidates.premerge.psv")

    print(f"\n[merge] {before} rows in -> {len(merged)} unique -> {MAIN}")
    print("  by region:")
    for reg, c in merged["region"].value_counts().items():
        print(f"    {reg:>13}: {c}")
    print("  by status:")
    for stt, c in merged["status"].value_counts().items():
        print(f"    {stt:>13}: {c}")


if __name__ == "__main__":
    main()

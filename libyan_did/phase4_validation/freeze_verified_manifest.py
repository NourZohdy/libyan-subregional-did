"""Freeze human validation onto the corpus → outputs/manifests/verified_manifest.csv.

The CLI twin of the validation app's "Export verified manifest" button: both call
src.verified_manifest.build_verified_manifest, so the frozen corpus is identical
whether you click in Streamlit or run this after a validation session.

Usage:
    python scripts/freeze_verified_manifest.py
"""

from __future__ import annotations

import sys
from pathlib import Path


import pandas as pd  # noqa: E402

from libyan_did.shared import config
from libyan_did.shared.verified_manifest import build_verified_manifest  # noqa: E402

KEEP_DECISIONS = ("verified", "wrong_region")
VERIF_CSV = config.GOLD_DIR / "speaker_verifications.csv"
MERGE_CSV = config.GOLD_DIR / "merge_decisions.csv"
VERIFIED_MANIFEST = config.MANIFEST_DIR / "verified_manifest.csv"


def _load_verifs() -> dict:
    if not VERIF_CSV.exists():
        return {}
    v = pd.read_csv(VERIF_CSV)
    return {str(r["actor_uid"]): r.to_dict() for _, r in v.iterrows()}


def _load_merge_decisions() -> dict:
    if not MERGE_CSV.exists():
        return {}
    m = pd.read_csv(MERGE_CSV)
    return {(str(r["actor_a"]), str(r["actor_b"])): r.to_dict() for _, r in m.iterrows()}


def main() -> None:
    if not config.CORPUS_MANIFEST.exists():
        raise SystemExit("corpus_manifest.csv not found — run scripts/combine_corpus.py first")

    df = pd.read_csv(config.CORPUS_MANIFEST, dtype={"file_id": str})  # keep numeric-looking ids as str
    out = build_verified_manifest(
        df, _load_verifs(), _load_merge_decisions(),
        regions=config.REGIONS, keep_decisions=KEEP_DECISIONS,
        min_segments=config.DUST_MIN_SEGMENTS, min_seconds=config.DUST_MIN_SECONDS)
    VERIFIED_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(VERIFIED_MANIFEST, index=False)

    kept = out[out["verified_keep"]]
    _true = {"true", "1", "1.0", "yes"}
    _ex = kept["is_excluded"].astype(str).str.strip().str.lower().isin(_true)
    eff = kept[~_ex]  # post-gate segments — the hours Chapter-3 tables/splits report
    hrs = kept["duration"].sum() / 3600.0
    eff_hrs = eff["duration"].sum() / 3600.0
    print(f"verified_manifest -> {VERIFIED_MANIFEST}")
    print(f"  rows                 : {len(out):,}")
    print(f"  verified_keep        : {len(kept):,} segs | {hrs:.2f} h (kept speakers, incl. gate-excluded segs)")
    print(f"  EFFECTIVE (post-gate): {len(eff):,} segs | {eff_hrs:.2f} h  <- use this everywhere (tables/splits)")
    print(f"  final speakers       : {kept['speaker_final'].nunique():,}")
    print(f"  excluded by floor    : "
          f"{int((out['exclusion_rule'] == 'insufficient_evidence').sum()):,} rows")
    print("  per-region hours (kept / effective):")
    for reg, g in kept.groupby("final_dialect"):
        g_eff = g[~g["is_excluded"].astype(str).str.strip().str.lower().isin(_true)]
        print(f"    {reg:14s}: {g['duration'].sum()/3600.0:6.2f} h / {g_eff['duration'].sum()/3600.0:6.2f} h eff | "
              f"{g['speaker_final'].nunique():4d} speakers")


if __name__ == "__main__":
    main()

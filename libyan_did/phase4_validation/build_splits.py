"""Step 3 (separate): balanced, speaker-disjoint train/dev/test from the VERIFIED
corpus. Run after a validation session has been frozen with
scripts/freeze_verified_manifest.py.

Splits are built on the post-validation picture:
  * only ``verified_keep`` segments that are not ``is_excluded`` (bad-audio/music
    segments are dropped even when their speaker was kept),
  * the grouping key is ``speaker_final`` — the human-merged identity, so two
    actor_uids the validator merged as one person never split across train/test,
  * regions are ``final_dialect`` — the validator's ``wrong_region`` fixes and
    cross-region merge resolutions, not the raw pipeline region.

Speakers are assigned atomically and balanced per region by duration (or
segments). An optional per-speaker cap down-samples anchor hosts first.

Usage:
    python scripts/build_splits.py
    python scripts/build_splits.py --cap 400 --balance duration
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


import pandas as pd  # noqa: E402

from libyan_did.shared import config
from libyan_did.shared import combine, diagnostics, splits  # noqa: E402

VERIFIED_MANIFEST = config.MANIFEST_DIR / "verified_manifest.csv"
SPLITS_MANIFEST = config.MANIFEST_DIR / "splits_manifest.csv"
SPLITS_MANIFEST_V2 = config.MANIFEST_DIR / "splits_manifest_v2.csv"
ACTOR_COL, REGION_COL = "speaker_final", "final_dialect"
_SHARED_COLS = ["source", "duration", "final_dialect", "speaker_final", "file_id"]
_TRUE = {"true", "1", "1.0", "yes"}


def _libyan_kept() -> pd.DataFrame:
    """Verified Libyan kept rows (post-validation identities)."""
    vm = pd.read_csv(VERIFIED_MANIFEST, dtype={"file_id": str})  # keep numeric-looking ids as str
    vm["is_excluded"] = vm["is_excluded"].astype(str).str.strip().str.lower().isin(_TRUE)
    return vm[vm["verified_keep"].astype(bool) & (~vm["is_excluded"])].copy()


def _control_kept() -> pd.DataFrame:
    """Non-Libyan control kept rows, column-aligned to the Libyan shared schema.

    Reads the per-resource control manifests via combine.combine_resources() (region ->
    final_dialect = the class CODE, speaker_global -> speaker_final = corpus-wide merged id).
    Returns an empty (correctly-columned) frame if the controls have not been re-derived yet.
    """
    corpus = combine.combine_resources()
    kept = corpus[(~corpus["is_excluded"]) & corpus["tier"].isin(config.KEEP_TIERS)]
    kept = kept[kept["macro_class"] == "Non-Libyan"]
    kept = kept.rename(columns={"region": "final_dialect", "speaker_global": "speaker_final"})
    return kept


def _build_speaker_disjoint(args) -> None:
    kept = _libyan_kept()
    out = splits.build_balanced_splits(
        kept, actor_col=ACTOR_COL, region_col=REGION_COL, balance=args.balance, cap=args.cap)
    splits.verify_no_speaker_overlap(out, actor_col=ACTOR_COL)
    out.to_csv(SPLITS_MANIFEST, index=False)

    print(diagnostics.split_summary(out).to_string(index=False))
    tot_h = out["duration"].sum() / 3600.0
    print(f"\nTOTAL: {len(out):,} segments | {tot_h:.2f} h | {out[ACTOR_COL].nunique()} speakers")
    cap_note = f"cap={args.cap}" if args.cap else "no cap"
    print(f"[{args.balance}-balanced, {cap_note}]  splits_manifest -> {SPLITS_MANIFEST}")


def _build_channel_disjoint(args) -> None:
    lib = _libyan_kept()[_SHARED_COLS]
    ctl = _control_kept()
    ctl = ctl[[c for c in _SHARED_COLS if c in ctl.columns]]
    if ctl.empty:
        print("[channel-disjoint] WARNING: no Non-Libyan control rows found — controls not "
              "re-derived yet (run phase3_corpus.run_resources on the non_libyan root). "
              "Building a Libyan-only v2 split; re-run once controls exist for the full 21 classes.")
    combined = pd.concat([lib, ctl], ignore_index=True)

    out = splits.build_channel_disjoint_splits(combined, source_col="source")
    splits.verify_no_channel_overlap(out, source_col="source")
    out.to_csv(SPLITS_MANIFEST_V2, index=False)

    print(diagnostics.split_summary(out).to_string(index=False))
    n_classes = out["final_dialect"].nunique()
    tot_h = out["duration"].sum() / 3600.0
    print(f"\nTOTAL: {len(out):,} segments | {tot_h:.2f} h | {n_classes} classes")
    print(f"[channel-disjoint, coverage-first]  splits_manifest_v2 -> {SPLITS_MANIFEST_V2}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["speaker-disjoint", "channel-disjoint"],
                    default="speaker-disjoint")
    ap.add_argument("--balance", choices=["duration", "segments"], default=config.SPLIT_BALANCE_BY)
    ap.add_argument("--cap", type=int, default=config.PER_SPEAKER_SEGMENT_CAP,
                    help="max segments kept per speaker before splitting (anchor cap; "
                         "omit for no cap)")
    args = ap.parse_args()

    if not VERIFIED_MANIFEST.exists():
        raise SystemExit("verified_manifest.csv not found — run "
                         "python -m libyan_did.phase4_validation.freeze_verified_manifest first")

    if args.mode == "channel-disjoint":
        _build_channel_disjoint(args)
    else:
        _build_speaker_disjoint(args)


if __name__ == "__main__":
    main()

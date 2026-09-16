"""Per-channel metadata table for the released corpus (camera-ready).

Answers the reviewer/AC asks that need channel-level counts:
  - channels by region (P1-3: the channel-disjoint result needs a stated denominator)
  - the source platform mix, which the paper currently mis-states as YouTube-only

Platform is INFERRED from the harvest path/source naming, because the discovery
registry (phase1_discovery/data/candidates.psv) only ever held YouTube URLs -- the
short-form sources were acquired outside it. TikTok accounts are named `tiktok_<handle>`;
everything else is assumed YouTube.
# ponytail: string-match on the harvest path is the only platform signal that exists in
# the manifests. If a platform column is ever written at harvest time, read that instead.

Run:
    python -m libyan_did.phase6_reporting.make_channel_metadata
"""

from __future__ import annotations

import pandas as pd

from libyan_did.shared import config

SPLITS_MANIFEST = config.MANIFEST_DIR / "splits_manifest.csv"
CHANNELS_CSV = config.METADATA_DIR / "channels.csv"


def infer_platform(row_text: pd.Series) -> pd.Series:
    """TikTok accounts are named `tiktok_<handle>`; YouTube is the residual."""
    return pd.Series(
        ["TikTok" if "tiktok" in t else "YouTube" for t in row_text.str.lower()],
        index=row_text.index,
    )


def build(sp: pd.DataFrame) -> pd.DataFrame:
    # Match on the channel name, not the per-clip filename: clip titles are free text and
    # can mention a platform the clip did not come from.
    sp = sp.copy()
    sp["platform"] = infer_platform(sp["playlist"].fillna("") + " " + sp["source"].fillna(""))

    ch = sp.groupby("playlist").agg(
        platform=("platform", "first"),
        region=("final_dialect", lambda s: s.value_counts().idxmax()),
        hours=("duration", lambda s: round(s.sum() / 3600, 2)),
        segments=("duration", "size"),
        speakers=("speaker_final", "nunique"),
        splits=("split", lambda s: "|".join(sorted(s.unique()))),
    )
    return ch.sort_values(["region", "hours"], ascending=[True, False]).reset_index()


def main() -> None:
    if not SPLITS_MANIFEST.exists():
        raise SystemExit(f"{SPLITS_MANIFEST} not found -- run build_splits.py first")

    ch = build(pd.read_csv(SPLITS_MANIFEST, low_memory=False))
    CHANNELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    ch.to_csv(CHANNELS_CSV, index=False, encoding="utf-8")

    print(f"wrote {CHANNELS_CSV}  ({len(ch)} channels)")
    print("\nchannels x platform:")
    print(ch.pivot_table(index="region", columns="platform", values="playlist",
                         aggfunc="count", fill_value=0))
    print("\nhours x platform:")
    print(ch.pivot_table(index="region", columns="platform", values="hours",
                         aggfunc="sum", fill_value=0).round(1))


def _selfcheck() -> None:
    sp = pd.DataFrame({
        "playlist": ["tiktok_abc", "tiktok_abc", "Drama Show", "Drama Show"],
        "source": ["tiktok_abc", "tiktok_abc", "Drama Show", "Drama Show"],
        "final_dialect": ["Fezzan", "Fezzan", "Tripolitania", "Cyrenaica"],
        "duration": [3600.0, 1800.0, 7200.0, 3600.0],
        "speaker_final": ["s1", "s2", "s3", "s3"],
        "split": ["test", "test", "train", "train"],
    })
    ch = build(sp).set_index("playlist")
    assert ch.loc["tiktok_abc", "platform"] == "TikTok"
    assert ch.loc["Drama Show", "platform"] == "YouTube"
    assert ch.loc["tiktok_abc", "hours"] == 1.5
    assert ch.loc["Drama Show", "speakers"] == 1          # same speaker, two rows
    assert ch.loc["Drama Show", "region"] == "Tripolitania"  # dominant by segment count
    print("selfcheck ok")


if __name__ == "__main__":
    import sys
    _selfcheck() if "--selfcheck" in sys.argv else main()

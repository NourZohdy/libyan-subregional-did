"""Descriptive control speaker-count upper bounds (V2-plan T4 / D10, FR-013).

Counts distinct per-playlist cluster identities per control class. These are AUTOMATIC
UPPER BOUNDS — no cross-playlist linking, no human validation — so they are descriptive
only and MUST NOT be used in splits or labels. Written to outputs/tables/.

Run:
    python -m libyan_did.phase3_corpus.control_speaker_counts
"""

from __future__ import annotations

import pandas as pd

from libyan_did.shared import combine, config

OUT = config.TABLES_DIR / "control_speaker_counts.csv"


def main() -> None:
    corpus = combine.combine_resources()
    kept = corpus[(~corpus["is_excluded"]) & corpus["tier"].isin(config.KEEP_TIERS)]
    kept = kept[kept["macro_class"] == "Non-Libyan"]

    rows = []
    for cls, g in kept.groupby("sub_cat"):
        rows.append({
            "class": cls,
            "est_speakers": int(g["actor_uid"].nunique()),
            "note": "automatic upper bound",
        })
    df = pd.DataFrame(rows).sort_values("class").reset_index(drop=True)

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    print(f"\ncontrol_speaker_counts -> {OUT}")


if __name__ == "__main__":
    main()

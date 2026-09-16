"""Step 2 (combine): merge all per-resource manifests into corpus_manifest.csv +
corpus_metadata.json, then write corpus-level diagnostic tables/figures.

Run after run_resources.py and before build_splits.py.

Usage:
    python scripts/combine_corpus.py
"""

from __future__ import annotations

import sys
from pathlib import Path


from libyan_did.shared import config
from libyan_did.shared import combine, diagnostics  # noqa: E402


def main() -> None:
    config.ensure_dirs()
    corpus = combine.combine_resources()
    kept = corpus[(~corpus["is_excluded"]) & (corpus["tier"].isin(config.KEEP_TIERS))].copy()

    print(f"corpus_manifest -> {config.CORPUS_MANIFEST}")
    print(f"  {corpus['playlist'].nunique()} playlists / {kept['region'].nunique()} regions")
    spk_col = "speaker_global" if "speaker_global" in kept.columns else "actor_uid"
    print(f"  kept {len(kept)} segments / {kept[spk_col].nunique()} speakers / "
          f"{round(kept['duration'].sum() / 3600.0, 2)} h")
    for _, row in (kept.groupby('region')
                   .agg(speakers=(spk_col, 'nunique'),
                        segments=('duration', 'size'),
                        hours=('duration', lambda s: round(s.sum() / 3600.0, 2)))
                   .reset_index().iterrows()):
        print(f"    - {row['region']}: {row['speakers']} spk, {row['segments']} seg, {row['hours']} h")

    tables = diagnostics.write_tables(kept)
    figs = diagnostics.write_figures(kept)

    # Temporal coverage (release statement + figure): aggregated into corpus_metadata.json.
    import json
    temporal = json.loads(config.CORPUS_METADATA.read_text(encoding="utf-8")).get("temporal", {})
    if temporal.get("date_min"):
        print(f"  temporal coverage: videos uploaded {temporal['date_min']} → {temporal['date_max']} "
              f"({temporal['n_dated']}/{temporal['n_total']} episodes dated)")
        fig = diagnostics.plot_upload_coverage(temporal)
        if fig:
            figs.append(fig)
    elif temporal.get("n_total"):
        print(f"  temporal coverage: 0/{temporal['n_total']} episodes carry an upload_date — "
              "run `python -m libyan_did.phase2_harvest.backfill_ids --root <root> --backfill`")

    print(f"  tables: {list(tables)}; figures: {[p.name for p in figs]}")
    print(f"corpus_metadata -> {config.CORPUS_METADATA}")
    print("Next: python scripts/build_splits.py")


if __name__ == "__main__":
    main()

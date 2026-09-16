"""Per-actor triage scoring: dialect (HuBERT) + LID (VoxLingua107) + cohesion +
music aggregates -> outputs/metadata/actor_scores.csv, the risk ranking the
validation app reviews in descending order.

Run AFTER run_resources.py / --recluster-only (actor IDs must be final).
Incremental: actors already in actor_scores.csv are kept unless --force.

Usage:
    python scripts/score_actors.py
    python scripts/score_actors.py --region Fezzan --force
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import pandas as pd


from libyan_did.shared import config
from libyan_did.shared import actor_triage  # noqa: E402

SCORES_CSV = config.METADATA_DIR / "actor_scores.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=None, help="canonical region filter")
    ap.add_argument("--only", default=None, help="substring filter on <region>/<playlist>")
    ap.add_argument("--limit", type=int, default=None, help="first N resources")
    ap.add_argument("--force", action="store_true", help="re-score already-scored actors")
    args = ap.parse_args()

    finals = sorted(config.CORPUS_DIR.glob("*/*/final_manifest.csv"))
    if args.region:
        finals = [f for f in finals if f.parent.parent.name.lower() == args.region.lower()]
    if args.only:
        needle = args.only.lower()
        finals = [f for f in finals
                  if needle in f"{f.parent.parent.name}/{f.parent.name}".lower()]
    if args.limit:
        finals = finals[: args.limit]

    prev = pd.DataFrame(columns=actor_triage.SCORE_COLUMNS)
    if SCORES_CSV.exists() and not args.force:
        prev = pd.read_csv(SCORES_CSV)
    done_resources = (set(zip(prev["region"], prev["playlist"])) if len(prev) else set())

    n = len(finals)
    print(f"Resources to score: {n} ({len(done_resources)} already in {SCORES_CSV.name})")
    all_rows: list[dict] = []
    failed = 0
    t0 = time.time()
    for i, fpath in enumerate(finals, 1):
        region = fpath.parent.parent.name
        playlist = fpath.parent.name
        if (region, playlist) in done_resources:
            continue
        emb = fpath.parent / "embeddings.npy"
        if not emb.exists():
            print(f"[{i}/{n}] {region}/{playlist}: no embeddings.npy — skipped")
            continue
        try:
            rows = actor_triage.score_resource_actors(fpath, emb, region, playlist)
            all_rows.extend(rows)
            if rows:
                worst = max(r["risk_score"] for r in rows)
                print(f"[{i}/{n}] {region}/{playlist}: {len(rows)} actors (max risk {worst:.2f})")
            # checkpoint every resource so an interrupt loses nothing
            out = pd.concat([prev, pd.DataFrame(all_rows)], ignore_index=True)
            config.METADATA_DIR.mkdir(parents=True, exist_ok=True)
            out.to_csv(SCORES_CSV, index=False)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[{i}/{n}] {region}/{playlist}: !! FAILED {exc}")
            traceback.print_exc()

    dt = (time.time() - t0) / 60.0
    total = len(prev) + len(all_rows)
    print(f"\nDone in {dt:.1f} min — {total} actors scored ({failed} resources failed)")
    print(f"-> {SCORES_CSV}")
    print("Next: python scripts/combine_corpus.py")


if __name__ == "__main__":
    main()

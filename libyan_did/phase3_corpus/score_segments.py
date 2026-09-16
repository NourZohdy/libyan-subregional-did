"""Music-scoring pass: write ``music_prob`` (PANNs CNN14) into every resource's
master_segments.csv. Run AFTER run_resources.py created the per-resource manifests
and BEFORE re-clustering (the music content gate reads this column).

Incremental: resources whose column is already fully populated are skipped.

Usage:
    python scripts/score_segments.py --root "D:\\Research\\DID_Set\\libyan"
    python scripts/score_segments.py --root "..." --only "Hakim" --force
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path


from libyan_did.shared import config
from libyan_did.shared import harvest, resource, score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=config.LOCAL_AUDIO_ROOT,
                    help="dialect parent, e.g. D:\\Research\\DID_Set\\libyan")
    ap.add_argument("--only", default=None, help="substring filter on <dialect>/<playlist>")
    ap.add_argument("--region", default=None, help="canonical region filter")
    ap.add_argument("--limit", type=int, default=None, help="first N resources")
    ap.add_argument("--force", action="store_true", help="re-score even if already scored")
    args = ap.parse_args()
    if not args.root:
        ap.error("provide --root (or set config.LOCAL_AUDIO_ROOT)")

    resources = harvest.enumerate_resources(args.root)
    if args.region:
        resources = [r for r in resources if r["region"].lower() == args.region.lower()]
    if args.only:
        needle = args.only.lower()
        resources = [r for r in resources
                     if needle in r["rel_path"].lower() or needle in r["playlist"].lower()]
    if args.limit:
        resources = resources[: args.limit]

    n = len(resources)
    print(f"Resources to score: {n}")
    scored = skipped = failed = 0
    t0 = time.time()
    for i, r in enumerate(resources, 1):
        p = resource.resource_paths(r["region"], r["playlist"])
        if not p["segments"].exists():
            print(f"[{i}/{n}] {r['rel_path']}: no master_segments.csv — run run_resources.py first")
            continue
        try:
            res = score.score_resource_music(p["segments"], force=args.force)
            if res["status"] == "scored":
                scored += 1
                print(f"[{i}/{n}] {r['rel_path']}: scored {res['n']} segments "
                      f"({res.get('n_musicish', 0)} music-ish)")
            else:
                skipped += 1
                print(f"[{i}/{n}] {r['rel_path']}: {res['status']}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[{i}/{n}] {r['rel_path']}: !! FAILED {exc}")
            traceback.print_exc()

    dt = (time.time() - t0) / 60.0
    print(f"\nDone in {dt:.1f} min — scored {scored}, skipped {skipped}, failed {failed}")
    print("Next: python scripts/run_resources.py --root ... --recluster-only")


if __name__ == "__main__":
    main()

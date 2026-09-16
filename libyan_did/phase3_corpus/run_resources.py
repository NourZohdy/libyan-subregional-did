"""Step 1 (per resource): reuse sibling RTTMs, build each playlist's manifest +
embeddings, cluster within the playlist, mirror clips to clips/<region>/<playlist>/,
and write per-resource metadata. Incremental: unchanged playlists are skipped.

Usage:
    python scripts/run_resources.py --root "D:\\Research\\DID_Set\\libyan"
    python scripts/run_resources.py --root "..." --only "Hakim"          # substring filter
    python scripts/run_resources.py --root "..." --region Cyrenaica      # one dialect
    python scripts/run_resources.py --root "..." --limit 5               # first N resources
    python scripts/run_resources.py --root "..." --force                 # ignore state, rebuild

No HuggingFace token is needed when every WAV has a sibling .rttm (the DID_Set case).
Pass --hf-token only to diarize episodes that are missing one.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path


from libyan_did.shared import config
from libyan_did.shared import harvest, resource  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=config.LOCAL_AUDIO_ROOT,
                    help="dialect parent, e.g. D:\\Research\\DID_Set\\libyan")
    ap.add_argument("--only", default=None,
                    help="process only resources whose '<dialect>/<playlist>' path or "
                         "playlist name contains this substring (case-insensitive)")
    ap.add_argument("--region", default=None,
                    help="canonical region filter (Tripolitania / Cyrenaica / Fezzan)")
    ap.add_argument("--limit", type=int, default=None, help="cap to the first N resources")
    ap.add_argument("--force", action="store_true", help="ignore the per-resource state lock")
    ap.add_argument("--recluster-only", action="store_true",
                    help="re-run clustering + clip export on cached manifest/embeddings "
                         "(no audio decode, no re-embedding); use after changing "
                         "clustering thresholds or logic")
    ap.add_argument("--hf-token", default=None,
                    help="only needed for episodes that lack a sibling .rttm")
    args = ap.parse_args()
    if not args.root:
        ap.error("provide --root (or set config.LOCAL_AUDIO_ROOT)")

    config.ensure_dirs()
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
    n_ep = sum(len(r["episodes"]) for r in resources)
    print(f"Resources to consider: {n}  ({n_ep} episodes)")

    proc = skip = empty = fail = 0
    failures: list[tuple[str, str]] = []
    t0 = time.time()
    for i, r in enumerate(resources, 1):
        print(f"[{i}/{n}] {r['region']}/{r['playlist']} ({len(r['episodes'])} ep)")
        try:
            if args.recluster_only:
                res = resource.recluster_resource(r)
                proc += res.get("status") == "reclustered"
                skip += res.get("status") == "missing_inputs"
            else:
                res = resource.process_resource(r, hf_token=args.hf_token, force=args.force)
                proc += res.get("status") == "processed"
                skip += res.get("status") == "skipped"
                empty += res.get("status") == "empty"
        except Exception as exc:  # noqa: BLE001
            fail += 1
            failures.append((r["rel_path"], repr(exc)))
            print(f"   !! FAILED: {exc}")
            traceback.print_exc()

    dt = (time.time() - t0) / 60.0
    print(f"\nDone {n} resources in {dt:.1f} min — "
          f"processed {proc}, skipped {skip}, empty {empty}, failed {fail}")
    if failures:
        print("Failures:")
        for rel, err in failures:
            print(f"  - {rel}: {err}")
    print("Next: python scripts/combine_corpus.py")


if __name__ == "__main__":
    main()

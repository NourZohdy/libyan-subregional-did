"""Smoke test for the upgraded pipeline: music scoring + recluster on 3 playlists.

Picks: the 'Hakim' vlog (known single speaker), the largest-actor Tripolitania
playlist, and the largest-actor Fezzan playlist. Prints before/after actor counts,
exclusion reasons, and merge-candidate counts.

Usage:
    python scripts/smoke_test.py --root "D:\\Research\\DID_Set\\libyan"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd


from libyan_did.shared import config
from libyan_did.shared import harvest, resource, score  # noqa: E402


def pick_smoke_resources(resources: list[dict]) -> list[dict]:
    chosen = []
    by_key = {}
    for r in resources:
        p = resource.resource_paths(r["region"], r["playlist"])
        meta = {}
        if p["metadata"].exists():
            meta = json.loads(p["metadata"].read_text(encoding="utf-8"))
        by_key[r["rel_path"]] = (r, int(meta.get("n_actors", 0)))

    hakim = next((r for r, _ in by_key.values() if "hakim" in r["rel_path"].lower()), None)
    if hakim:
        chosen.append(hakim)
    for region in ("Tripolitania", "Fezzan"):
        cands = [(r, n) for r, n in by_key.values() if r["region"] == region]
        if cands:
            chosen.append(max(cands, key=lambda t: t[1])[0])
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=config.LOCAL_AUDIO_ROOT, required=False)
    args = ap.parse_args()

    resources = harvest.enumerate_resources(args.root)
    chosen = pick_smoke_resources(resources)
    print(f"Smoke resources: {[r['rel_path'] for r in chosen]}\n")

    for r in chosen:
        p = resource.resource_paths(r["region"], r["playlist"])
        old_meta = (json.loads(p["metadata"].read_text(encoding="utf-8"))
                    if p["metadata"].exists() else {})
        print(f"=== {r['rel_path']} ===")
        print(f"  before: {old_meta.get('n_actors', '?')} actors, "
              f"{old_meta.get('n_segments_kept', '?')}/{old_meta.get('n_segments_total', '?')} kept")

        t0 = time.time()
        sres = score.score_resource_music(p["segments"])
        print(f"  music scoring: {sres['status']} n={sres['n']} "
              f"musicish={sres.get('n_musicish', 0)} ({(time.time()-t0)/60:.1f} min)")

        t0 = time.time()
        meta = resource.recluster_resource(r)
        print(f"  recluster: {meta['status']} ({(time.time()-t0)/60:.1f} min)")

        final = pd.read_csv(p["final"])
        excl = final[final["is_excluded"].astype(str).str.lower().isin({"true", "1"})]
        reasons = excl["exclude_reason"].fillna("").replace("", "unknown").value_counts().to_dict()
        ncand = len(pd.read_csv(p["merge_candidates"])) if p["merge_candidates"].exists() else 0
        print(f"  after: {meta.get('n_actors', '?')} actors, "
              f"{meta.get('n_segments_kept', '?')}/{meta.get('n_segments_total', '?')} kept, "
              f"{ncand} merge candidate(s)")
        print(f"  exclude reasons: {reasons}\n")


if __name__ == "__main__":
    main()

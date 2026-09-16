"""Calibration report for the actor-merge / speaker-linking similarity bands.

Computes, from the CACHED per-resource embeddings (no audio, no GPU):
  * WITHIN-playlist actor-centroid pairwise cosine similarities
  * CROSS-playlist (same region / cross region) actor-centroid similarities
and prints the percentile table + how many pairs each candidate band captures.

Usage:
    python scripts/calibrate_merge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


from libyan_did.shared import config
from libyan_did.shared import link_actors  # noqa: E402


def pct_table(name: str, sims: np.ndarray) -> None:
    if sims.size == 0:
        print(f"{name}: no pairs")
        return
    qs = [50, 75, 90, 95, 97.5, 99, 99.5, 99.9]
    vals = np.percentile(sims, qs)
    print(f"{name}  (n={sims.size:,})")
    print("   " + "  ".join(f"p{q}={v:.3f}" for q, v in zip(qs, vals)))
    for lo, hi in [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
                   (0.70, 0.75), (0.75, 1.01)]:
        n = int(((sims >= lo) & (sims < hi)).sum())
        if n:
            print(f"   [{lo:.2f},{hi:.2f}): {n:,} pairs")


def main() -> None:
    actors = link_actors.collect_actor_centroids()
    print(f"actors: {len(actors)} across {actors['playlist'].nunique()} playlists\n")
    C = np.stack(actors["centroid"].to_list())
    sims = C @ C.T
    iu, ju = np.triu_indices(len(actors), k=1)
    s = sims[iu, ju]

    pl = actors["playlist"].to_numpy()
    rg = actors["region"].to_numpy()
    same_pl = pl[iu] == pl[ju]
    same_rg = rg[iu] == rg[ju]

    pct_table("WITHIN-playlist pairs", s[same_pl])
    print()
    pct_table("CROSS-playlist, same region", s[~same_pl & same_rg])
    print()
    pct_table("CROSS-region pairs", s[~same_rg])
    print()
    print(f"current bands: ACTOR auto>={config.ACTOR_MERGE_AUTO_SIM} "
          f"review>={config.ACTOR_MERGE_REVIEW_SIM} | "
          f"LINK auto>={config.LINK_AUTO_SIM} review>={config.LINK_REVIEW_SIM}")
    for name, mask, auto, review in [
        ("within-playlist", same_pl, config.ACTOR_MERGE_AUTO_SIM, config.ACTOR_MERGE_REVIEW_SIM),
        ("cross-playlist same-region", ~same_pl & same_rg, config.LINK_AUTO_SIM, config.LINK_REVIEW_SIM),
        ("cross-region (always review)", ~same_rg, None, config.LINK_CROSS_REGION_REVIEW_SIM),
    ]:
        v = s[mask]
        n_auto = int((v >= auto).sum()) if auto else 0
        n_rev = int(((v >= review) & (v < (auto or 1.01))).sum())
        print(f"  {name}: auto-merge {n_auto:,} pairs, human review {n_rev:,} pairs")


if __name__ == "__main__":
    main()

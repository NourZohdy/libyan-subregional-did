"""Cross-playlist speaker linking: assign corpus-wide canonical speaker IDs.

Run AFTER run_resources.py (or --recluster-only) and BEFORE combine_corpus.py.

Usage:
    python scripts/link_actors.py
"""

from __future__ import annotations

import sys
from pathlib import Path


from libyan_did.shared import link_actors  # noqa: E402


def main() -> None:
    res = link_actors.run_linking()
    print(f"actors:                {res['n_actors']}")
    print(f"canonical speakers:    {res['n_speakers_global']}")
    print(f"multi-actor speakers:  {res['n_multi_actor_speakers']} (auto-linked across playlists)")
    print(f"review candidates:     {res['n_candidates']} "
          f"({res['n_cross_region_candidates']} cross-region)")
    print(f"-> {res['links_path']}")
    print(f"-> {res['candidates_path']}")
    print("Next: python scripts/score_actors.py  (then combine_corpus.py)")


if __name__ == "__main__":
    main()

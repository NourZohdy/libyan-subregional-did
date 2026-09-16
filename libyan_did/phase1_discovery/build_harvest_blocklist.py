"""Build the already-harvested blocklist for the discovery agent.

Scans the source tree and lists every show folder + the channel names found in its `.json`
sidecars, so the discovery agent (discovery/DISCOVERY_AGENT.md) never re-proposes a
channel/show already in the corpus. Many legacy sidecars carry `channel_or_author: "unknown"`
(lost in an old migration), so the show-folder names are the more complete signal — both are
emitted.

Writes discovery/harvested_blocklist.txt (pipe-delimited).

Usage (from libyan_did_pipeline/):
    python scripts/build_harvest_blocklist.py --root "D:\\Research\\DID_Set\\libyan"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from libyan_did.shared import config
from libyan_did.shared import harvest  # noqa: E402

DEFAULT_OUT = config.DISCOVERY_DIR / "harvested_blocklist.txt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=config.LOCAL_AUDIO_ROOT,
                    help="source-tree root (e.g. D:\\Research\\DID_Set\\libyan)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    if not args.root:
        raise SystemExit("no --root and config.LOCAL_AUDIO_ROOT is None; pass --root")

    resources = harvest.enumerate_resources(args.root)
    rows: list[tuple[str, str, int, str]] = []
    channels_all: set[str] = set()
    for r in resources:
        channels: set[str] = set()
        for e in r["episodes"]:
            jp = e.get("json_path")
            if not jp:
                continue
            try:
                ch = json.loads(Path(jp).read_text(encoding="utf-8")).get("channel_or_author")
            except Exception:  # noqa: BLE001
                ch = None
            if ch and ch != "unknown":
                channels.add(ch)
                channels_all.add(ch)
        rows.append((r["region"], r["playlist"], len(r["episodes"]),
                     " ; ".join(sorted(channels)) or "unknown"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Already-harvested sources — discovery must NEVER propose these (match on show name or channel).",
        "# region | show_folder | n_episodes | channel_or_author",
    ]
    for region, show, n, chans in sorted(rows):
        lines.append(f"{region} | {show} | {n} | {chans}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[blocklist] {len(rows)} shows, {len(channels_all)} distinct named channels "
          f"-> {out}")
    per_region: dict[str, int] = {}
    for region, _show, _n, _c in rows:
        per_region[region] = per_region.get(region, 0) + 1
    for reg, c in sorted(per_region.items()):
        print(f"    {reg:>13}: {c} shows")


if __name__ == "__main__":
    main()

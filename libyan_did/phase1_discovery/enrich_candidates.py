"""Backfill cheap metadata onto the review registry: video_count + playlist_count.

The reviewer wants, for every candidate, *how many videos* the channel has and *how many
playlists (shows)* it runs — a big playlist count is the tell-tale of a multi-show TV channel
whose shows you pick individually in the Playlist browser. This walks
``discovery/candidates.psv`` and fills those two columns with ``yt-dlp`` metadata only
(NO audio download), concurrently and resumably.

  * channel rows -> video_count (from the /videos tab) + playlist_count (from /playlists)
  * playlist rows -> video_count (the playlist's own size); playlist_count left blank

Already-filled rows are skipped (re-run anytime); ``--refresh`` recomputes everything.
Counts are bounded (e.g. ``"500+"``) so a giant national channel can't stall the run.

Usage (from libyan_did_pipeline/):
    python scripts/enrich_candidates.py
    python scripts/enrich_candidates.py --workers 8 --refresh
    python scripts/enrich_candidates.py --status pending      # only un-reviewed rows
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import pandas as pd  # noqa: E402

from libyan_did.shared import config
from libyan_did.shared import registry  # noqa: E402
from libyan_did.shared import ytmeta  # noqa: E402

MAIN = config.DISCOVERY_DIR / "candidates.psv"


def _blank(v: str) -> bool:
    return str(v).strip().lower() in ("", "na", "nan", "none")


def enrich_row(row: dict) -> tuple[str, str]:
    """Return (video_count, playlist_count) for one registry row."""
    url = (row.get("url") or "").strip()
    cid = (row.get("channel_id") or "").strip()
    utype = (row.get("url_type") or "").strip().lower()
    if utype == "playlist" or ("list=" in url and utype != "channel"):
        return ytmeta.playlist_video_count(url), ""
    return (ytmeta.channel_video_count(cid, url),
            ytmeta.channel_playlist_count(cid, url))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", default=str(MAIN))
    ap.add_argument("--workers", type=int, default=6, help="concurrent yt-dlp metadata calls")
    ap.add_argument("--refresh", action="store_true", help="recompute even already-filled rows")
    ap.add_argument("--status", default=None, help="only rows with this status (e.g. pending)")
    args = ap.parse_args()

    path = Path(args.registry)
    if not path.exists():
        raise SystemExit(f"no registry at {path}")
    df = registry.read(path)
    n = len(df)
    print(f"[enrich] {n} rows in {path.name}; columns: {list(df.columns)}")

    todo: list[int] = []
    for i in range(n):
        if args.status and df.at[i, "status"].strip().lower() != args.status.lower():
            continue
        is_pl = (df.at[i, "url_type"].strip().lower() == "playlist")
        needs = _blank(df.at[i, "video_count"]) or (not is_pl and _blank(df.at[i, "playlist_count"]))
        if args.refresh or needs:
            todo.append(i)
    print(f"[enrich] {len(todo)} row(s) to fetch (workers={args.workers}"
          f"{', refresh' if args.refresh else ''}). Metadata only — no audio.")
    if not todo:
        print("[enrich] nothing to do."); return

    bak = path.with_suffix(".psv.prenrich.bak")

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(enrich_row, df.iloc[i].to_dict()): i for i in todo}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                vc, pc = fut.result()
            except Exception as exc:  # noqa: BLE001
                vc, pc = "", ""
                print(f"  ! row {i} failed: {type(exc).__name__}")
            if vc:
                df.at[i, "video_count"] = vc
            if pc:
                df.at[i, "playlist_count"] = pc
            done += 1
            src = df.at[i, "source"][:32]
            print(f"  [{done}/{len(todo)}] {src:<32} vids={vc or '-':>5} lists={pc or '-':>4}")
            if done % 15 == 0:  # periodic checkpoint so a crash keeps partial progress
                registry.write(df, path, backup=bak)

    registry.write(df, path, backup=bak)
    filled_v = int((~df["video_count"].map(_blank)).sum())
    filled_p = int((~df["playlist_count"].map(_blank)).sum())
    print(f"\n[enrich] wrote {path}")
    print(f"[enrich] video_count filled: {filled_v}/{n}   playlist_count filled: {filled_p}/{n}")
    multishow = df[df["playlist_count"].map(lambda x: x.strip().rstrip('+').isdigit()
                                            and int(x.strip().rstrip('+')) >= 8)]
    if len(multishow):
        print(f"[enrich] {len(multishow)} channel(s) with >=8 playlists "
              f"(open these in the Playlist browser to pick their shows):")
        for _, r in multishow.sort_values("playlist_count", key=lambda s: s.str.rstrip('+').astype(int),
                                          ascending=False).head(15).iterrows():
            print(f"    {r['playlist_count']:>4} lists | {r['region']:<13} | {r['source'][:40]}")


if __name__ == "__main__":
    main()

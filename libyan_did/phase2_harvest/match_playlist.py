"""Match locally harvested episodes to their YouTube ids via a channel/playlist URL.

The sidecars for legacy harvests lost `youtube_id`, but the WAV *filenames* kept the real
video titles and the audio itself gives an exact duration. So: enumerate the remote
channel/playlist (metadata only, no download) and pair each local file with the one remote
video whose runtime matches.

Duration is the decider, not the title: observed deltas on confirmed matches are all under
1 second (YouTube reports whole seconds, our WAVs are frame-exact), while a re-upload of
the same episode was 218 s off. Titles only break ties -- some uploads are translated into
English, so a title mismatch is not evidence against a match.
# ponytail: appends to recovered_ids.csv rather than owning its own store; one file is the
# provenance record for every recovery route.

Run:
    python -m libyan_did.phase2_harvest.match_playlist --local "على_بياض" --url "https://youtube.com/@X/videos"
    python -m libyan_did.phase2_harvest.match_playlist --dry-run --local ... --url ...
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import sys
from pathlib import Path

from libyan_did.shared import config

EPISODES_CSV = config.METADATA_DIR / "episodes_to_find.csv"
RECOVERED_CSV = config.METADATA_DIR / "recovered_ids.csv"
TIGHT_SEC = 1.0          # a true match is sub-second; anything looser is a different cut
FIELDS = ["playlist", "file", "title_from_filename", "duration_sec", "channel_url",
          "playlist_url", "youtube_id", "remote_title", "remote_duration_sec",
          "duration_delta_sec", "confidence", "note"]


def fetch(url: str) -> list[dict]:
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True,
                           "extract_flat": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    return [{"id": e["id"], "dur": float(e.get("duration") or 0), "title": e.get("title", "")}
            for e in (info.get("entries") or []) if e]


def _norm(s: str) -> str:
    s = str(s).replace("｜", "|").replace("_", " ")
    return " ".join(re.sub(r"[^\w؀-ۿ ]", " ", s).split())


def match(locals_: list[dict], remote: list[dict]) -> list[dict]:
    """One remote video per local file; ties broken by title similarity."""
    used: set[str] = set()
    out = []
    for row in locals_:
        loc = float(row["duration_sec"])
        cands = [r for r in remote if abs(r["dur"] - loc) <= TIGHT_SEC and r["id"] not in used]
        if len(cands) > 1:
            lt = _norm(row["title_from_filename"])
            cands.sort(key=lambda r: -difflib.SequenceMatcher(None, lt, _norm(r["title"])).ratio())
        hit = cands[0] if cands else None
        if hit:
            used.add(hit["id"])
        out.append({**row, "hit": hit,
                    "delta": round(abs(hit["dur"] - loc), 1) if hit else None,
                    "ambiguous": len(cands) > 1})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True, help="playlist name as it appears on disk")
    ap.add_argument("--url", required=True, help="channel /videos or playlist URL")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    eps = [r for r in csv.DictReader(EPISODES_CSV.open(encoding="utf-8-sig"))
           if r["playlist"] == args.local]
    if not eps:
        raise SystemExit(f"no local episodes named {args.local!r} in {EPISODES_CSV}")

    remote = fetch(args.url)
    print(f"{args.local}: {len(eps)} local vs {len(remote)} remote")

    results = match(eps, remote)
    hits = [r for r in results if r["hit"]]
    for r in results:
        h = r["hit"]
        flag = "OK " if h and not r["ambiguous"] else ("TIE" if h else "-- ")
        print(f"  {flag} {float(r['duration_sec'])/60:6.1f}min {str(r['title_from_filename'])[:38]:40s}"
              + (f" -> {h['id']} d={r['delta']}s {h['title'][:34]}" if h else "  NO MATCH"))
    print(f"\nmatched {len(hits)}/{len(results)}")

    if args.dry_run or not hits:
        return

    is_playlist = "list=" in args.url
    existing = list(csv.DictReader(RECOVERED_CSV.open(encoding="utf-8-sig"))) \
        if RECOVERED_CSV.exists() else []
    seen = {r["file"] for r in existing}
    for r in results:
        if r["file"] in seen:
            continue
        h = r["hit"]
        existing.append({
            "playlist": r["playlist"], "file": r["file"],
            "title_from_filename": r["title_from_filename"], "duration_sec": r["duration_sec"],
            "channel_url": "" if is_playlist else args.url,
            "playlist_url": args.url if is_playlist else "",
            "youtube_id": h["id"] if h else "",
            "remote_title": h["title"] if h else "",
            "remote_duration_sec": str(int(h["dur"])) if h else "",
            "duration_delta_sec": str(r["delta"]) if h else "",
            "confidence": "high" if h else "none",
            "note": ("tie broken by title; " if r["ambiguous"] else "") +
                    f"matched from {'playlist' if is_playlist else 'channel'} (+/-{TIGHT_SEC}s)"
            if h else "no remote video within tolerance",
        })
    with RECOVERED_CSV.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(existing)
    print(f"wrote {RECOVERED_CSV} ({len(existing)} rows)")


def _selfcheck() -> None:
    loc = [{"playlist": "p", "file": "a.wav", "title_from_filename": "الحلقة الاولى", "duration_sec": "600.4"},
           {"playlist": "p", "file": "b.wav", "title_from_filename": "الحلقة الثانية", "duration_sec": "600.6"}]
    rem = [{"id": "aaaaaaaaaaa", "dur": 600.0, "title": "الحلقة الاولى"},
           {"id": "bbbbbbbbbbb", "dur": 601.0, "title": "الحلقة الثانية"},
           {"id": "ccccccccccc", "dur": 9999.0, "title": "unrelated"}]
    out = match(loc, rem)
    assert out[0]["hit"]["id"] == "aaaaaaaaaaa", out[0]
    assert out[1]["hit"]["id"] == "bbbbbbbbbbb", out[1]   # not reused, tie broken by title
    far = match([{"playlist": "p", "file": "c.wav", "title_from_filename": "x", "duration_sec": "60"}], rem)
    assert far[0]["hit"] is None, "a 540s gap must not match"
    print("selfcheck ok")


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()

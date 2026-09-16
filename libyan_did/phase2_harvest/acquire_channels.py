"""Harvest approved YouTube sources into the source tree (the "harvest skill" driver).

Reads a registry CSV of confirmed sources and downloads + diarizes each into
``<root>/<dialect_folder>/<show>/`` as .wav + .rttm + .json triplets, ready for the
incremental ingest (``scripts/run_resources.py``). De-dups by youtube_id + audio hash;
resumable. See ``src/acquire.py`` for the engine.

Registry CSV columns (header row): at least ``url, region, source``. Optional:
``url_type, show, status, format_type, est_videos, discovery_confidence, evidence, notes``.
Only rows with ``status`` == ``approved`` are harvested unless ``--all`` is given (rows with
no ``status`` column are all harvested).

Usage (diarization env, from libyan_did_pipeline/):
    python scripts/acquire_channels.py --registry sources_registry.csv --root "D:\\Research\\DID_Set\\libyan"
    python scripts/acquire_channels.py --registry test.csv --root "D:\\..." --limit-per-source 3 --dry-run
    python scripts/acquire_channels.py --registry test.csv --root "D:\\..." --region Fezzan --no-diarize
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from pathlib import Path


# Arabic show names print to the console; force UTF-8 so Windows cp1252 can't crash the run.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001 — older Pythons / non-reconfigurable streams
    pass

from libyan_did.shared import config
from libyan_did.shared import acquire  # noqa: E402


def _sniff_delimiter(header: str) -> str:
    """The discovery sheet is pipe-delimited (Arabic titles contain commas); fall back to tab/comma."""
    if "|" in header:
        return "|"
    if "\t" in header:
        return "\t"
    return ","


def load_registry(path: Path, *, only_approved: bool, region: str | None) -> list[dict]:
    raw = Path(path).read_text(encoding="utf-8-sig")
    first = next((ln for ln in raw.splitlines() if ln.strip()), "")
    if not first:
        raise SystemExit(f"empty registry: {path}")
    reader = csv.DictReader(io.StringIO(raw), delimiter=_sniff_delimiter(first))
    cols = [(c or "").strip() for c in (reader.fieldnames or [])]
    if not {"url", "region"} <= set(cols):
        raise SystemExit(f"registry must have at least 'url' and 'region' columns; got {cols}")
    has_status = "status" in cols

    def g(row: dict, key: str) -> str:  # tolerate surrounding spaces from a ' | '-formatted sheet
        for k, v in row.items():
            if (k or "").strip() == key:
                return (v or "").strip()
        return ""

    kept: list[dict] = []
    for r in reader:
        url = g(r, "url")
        if not url or url.startswith("#"):
            continue
        if region and g(r, "region") != region:
            continue
        if only_approved and has_status and g(r, "status").lower() != "approved":
            continue
        kept.append({
            "url": url,
            "region": g(r, "region"),
            "source": g(r, "source") or g(r, "show"),
            "show": g(r, "show") or g(r, "source"),
            "lang_flag": g(r, "lang_flag") or "arabic",
            "url_type": g(r, "url_type"),
            "channel_id": g(r, "channel_id"),
        })
    # A channel is harvested WHOLE only if you did NOT cherry-pick its playlists in the
    # Playlist browser. If any approved playlist shares this channel's channel_id, harvest
    # those picked shows instead of the whole channel (no double-harvest; respects the picks).
    picked_channels = {row["channel_id"] for row in kept
                       if row["url_type"] == "playlist" and row["channel_id"]}
    kept = [row for row in kept
            if not (row["url_type"] == "channel" and row["channel_id"] in picked_channels)]
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", required=True, help="path to the sources registry CSV")
    ap.add_argument("--root", default=config.LOCAL_AUDIO_ROOT,
                    help="source-tree root (e.g. D:\\Research\\DID_Set\\libyan)")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                    help="HuggingFace token for pyannote (default: env HF_TOKEN)")
    ap.add_argument("--limit-per-source", type=int, default=None, help="cap videos per source")
    ap.add_argument("--max-duration-min", type=float, default=None,
                    help="skip videos longer than this many minutes")
    ap.add_argument("--region", default=None, help="harvest only this canonical region")
    ap.add_argument("--all", action="store_true", help="ignore the status column; harvest every row")
    ap.add_argument("--no-diarize", action="store_true", help="download + provenance only, no RTTM")
    ap.add_argument("--dry-run", action="store_true", help="enumerate only; download nothing")
    args = ap.parse_args()

    if not args.root:
        raise SystemExit("no --root and config.LOCAL_AUDIO_ROOT is None; pass --root")
    diarize_audio = not args.no_diarize
    if diarize_audio and not args.dry_run and not args.hf_token:
        raise SystemExit("diarization needs an HF token: pass --hf-token or set env HF_TOKEN "
                         "(or use --no-diarize for a download-only smoke test)")

    rows = load_registry(Path(args.registry), only_approved=not args.all, region=args.region)
    print(f"[registry] {len(rows)} source(s) to harvest from {args.registry}")
    if not rows:
        raise SystemExit("nothing to harvest (check --all / --region / the status column)")

    max_dur_s = args.max_duration_min * 60 if args.max_duration_min else None
    summaries = acquire.harvest_registry(
        rows, args.root, hf_token=args.hf_token, limit=args.limit_per_source,
        max_duration_s=max_dur_s, diarize_audio=diarize_audio, dry_run=args.dry_run)

    print("\n=== harvest summary ===")
    tot = {k: 0 for k in ("enumerated", "downloaded", "dup_id", "dup_audio", "exists", "too_long", "failed")}
    for s in summaries:
        print(f"  {s['region']:>13} / {s['show'][:34]:<34} "
              f"dl={s['downloaded']} dupID={s['dup_id']} dupAudio={s['dup_audio']} "
              f"exists={s['exists']} long={s['too_long']} fail={s['failed']} (of {s['enumerated']})")
        for k in tot:
            tot[k] += s[k]
    print(f"  {'TOTAL':>13}   downloaded={tot['downloaded']}  dup_id={tot['dup_id']}  "
          f"dup_audio={tot['dup_audio']}  exists={tot['exists']}  too_long={tot['too_long']}  "
          f"failed={tot['failed']}")
    if not args.dry_run:
        print("\nNext: python scripts/run_resources.py --root \"%s\"  (incremental ingest)" % args.root)


if __name__ == "__main__":
    main()

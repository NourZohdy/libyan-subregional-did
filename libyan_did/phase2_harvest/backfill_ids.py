"""Recover the YouTube **video id** + **upload date** for legacy-harvested audio.

A public, ethically-released YouTube corpus must reference each clip by its **video id**
(you publish ids + timestamps + labels, not raw audio). Early harvests stored neither the
``youtube_id`` nor the ``upload_date`` in the ``.json`` sidecar; newer harvests
(``acquire.write_sidecar``) store both. This tool closes that gap **without re-downloading
audio**:

  1. **Coverage report (default, no network):** scan every sidecar under ``--root`` and report
     how many files already have a real ``youtube_id`` / ``upload_date``, per playlist — so you
     see exactly how much legacy data is affected.
  2. **Back-fill (``--backfill``):** for playlists whose URL we can resolve (sidecar
     ``source_url`` → else the discovery registry / playlist_state by show name), re-enumerate
     the remote playlist (metadata only) and match each local file to a remote video by
     **normalized title + duration**, then write the recovered ``youtube_id`` / ``upload_date``
     / ``playlist_id`` / ``channel_id`` into the existing sidecar. Local files with no remote
     match are listed as a **re-download set** (``backfill_report.csv``); audio is never touched.

Usage (from the repo root):
    python -m libyan_did.phase2_harvest.backfill_ids --root "D:\\Research\\DID_Set\\libyan"
    python -m libyan_did.phase2_harvest.backfill_ids --root "..." --backfill --dry-run
    python -m libyan_did.phase2_harvest.backfill_ids --root "..." --backfill --only "Katiba"
    python -m libyan_did.phase2_harvest.backfill_ids --selfcheck      # offline matcher test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from libyan_did.shared import acquire
from libyan_did.shared import config
from libyan_did.shared import harvest
from libyan_did.shared import pstate
from libyan_did.shared import registry
from libyan_did.phase1_discovery import playlist_tools

YTDLP = "yt-dlp"


def _valid_id(v) -> bool:
    s = str(v or "").strip().lower()
    return bool(s) and s != "unknown"


def _valid_date(v) -> bool:
    s = str(v or "").strip()
    return len(s) == 8 and s.isdigit()


def _read_json(p) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


# --------------------------------------------------------------------------- #
# 1) Coverage (no network)
# --------------------------------------------------------------------------- #
def scan_coverage(root) -> list[dict]:
    """Per-playlist id/date coverage from local sidecars only. No network."""
    try:
        resources = harvest.enumerate_resources(root)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"cannot read root {root}: {exc}")
    rows = []
    for r in resources:
        n = n_id = n_date = 0
        for e in r["episodes"]:
            j = _read_json(e["json_path"]) if e.get("json_path") else {}
            n += 1
            n_id += _valid_id(j.get("youtube_id"))
            n_date += _valid_date(j.get("upload_date"))
        rows.append({"region": r["region"], "playlist": r["playlist"], "files": n,
                     "have_id": n_id, "have_date": n_date,
                     "missing_id": n - n_id, "missing_date": n - n_date})
    return rows


# --------------------------------------------------------------------------- #
# 2) Remote enumeration + the pure matcher
# --------------------------------------------------------------------------- #
def _video_upload_date(video_id: str, timeout: int = 60) -> str:
    """One light metadata call for a single video's upload_date (flat omits it)."""
    cmd = [YTDLP, f"https://www.youtube.com/watch?v={video_id}", "--skip-download",
           "--no-warnings", "--print", "%(upload_date)s"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=timeout).stdout.strip()
    except Exception:  # noqa: BLE001 — a slow/dead video must not stall the batch
        return ""
    return out if _valid_date(out) else ""


def match_local_to_remote(local_eps: list[dict], remote: list[dict], *,
                          dur_tol: float = 2.0) -> list[dict]:
    """Match each local episode to a remote video by normalized title + duration. PURE.

    ``local_eps``: ``[{basename, video_title, duration_sec}]`` (from sidecars).
    ``remote``  : ``[{id, title, duration}]`` (from ``acquire.enumerate_videos``).

    Title match is the primary key (normalized via ``playlist_tools.normalize_name``);
    duration disambiguates collisions and rescues a renamed video (duration-unique fallback).
    Returns one record per local ep: ``{basename, matched_id, remote_title, ambiguous}``.
    """
    by_title: dict[str, list[dict]] = {}
    for v in remote:
        by_title.setdefault(playlist_tools.normalize_name(v.get("title", "")), []).append(v)

    out = []
    for e in local_eps:
        key = playlist_tools.normalize_name(e.get("video_title", ""))
        dur = e.get("duration_sec")
        cands = list(by_title.get(key, [])) if key else []
        if dur is not None:
            close = [v for v in cands if v.get("duration") is not None
                     and abs(float(v["duration"]) - float(dur)) <= dur_tol]
            if close:
                cands = close
            elif not cands:  # title changed on YouTube -> last resort: a duration-UNIQUE remote
                dmatch = [v for v in remote if v.get("duration") is not None
                          and abs(float(v["duration"]) - float(dur)) <= dur_tol]
                cands = dmatch
        rec = {"basename": e.get("basename"), "matched_id": None,
               "remote_title": "", "ambiguous": False}
        if len(cands) == 1:
            rec["matched_id"] = cands[0].get("id")
            rec["remote_title"] = cands[0].get("title", "")
        elif len(cands) > 1:
            rec["ambiguous"] = True
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# URL resolution (sidecar -> registry/pstate by show name)
# --------------------------------------------------------------------------- #
def _name_url_map() -> dict[str, str]:
    """normalized show/playlist name -> playlist URL, from candidates.psv + playlist_state.csv."""
    out: dict[str, str] = {}
    try:
        reg = registry.read(config.DISCOVERY_DIR / "candidates.psv")
        for _, r in reg[reg["url_type"].str.strip() == "playlist"].iterrows():
            nm = playlist_tools.normalize_name(r.get("source", ""))
            if nm and r.get("url"):
                out.setdefault(nm, r["url"].strip())
    except Exception:  # noqa: BLE001
        pass
    try:
        ps = pstate.read()
        for _, r in ps.iterrows():
            nm = playlist_tools.normalize_name(r.get("title", ""))
            if nm and r.get("url"):
                out.setdefault(nm, str(r["url"]).strip())
    except Exception:  # noqa: BLE001
        pass
    return out


def _resolve_url(resource: dict, name_map: dict[str, str]) -> str:
    """Best-effort playlist URL: sidecar source_url first, then a name match in the registry."""
    for e in resource["episodes"]:
        j = _read_json(e["json_path"]) if e.get("json_path") else {}
        su = str(j.get("source_url") or "").strip()
        if su and "list=" in su:
            return su
    for cand in (resource["playlist"], ):
        url = name_map.get(playlist_tools.normalize_name(cand))
        if url:
            return url
    return ""


# --------------------------------------------------------------------------- #
# 2b) Back-fill one resource
# --------------------------------------------------------------------------- #
def backfill_resource(resource: dict, url: str, *, dry_run: bool) -> dict:
    """Recover ids/dates for one playlist's local files. Writes sidecars unless dry_run."""
    region, playlist = resource["region"], resource["playlist"]
    pid = acquire._playlist_id_from_url(url)
    try:
        remote = acquire.enumerate_videos(url)
    except Exception as exc:  # noqa: BLE001
        return {"region": region, "playlist": playlist, "url": url, "error": str(exc)[:140],
                "matched": 0, "written": 0, "unmatched": [], "ambiguous": 0}

    local_eps, sidecar_of, chan = [], {}, ""
    for e in resource["episodes"]:
        j = _read_json(e["json_path"]) if e.get("json_path") else {}
        chan = chan or str(j.get("channel_id") or "").strip()
        local_eps.append({"basename": e["basename"], "video_title": j.get("video_title", ""),
                          "duration_sec": j.get("duration_sec")})
        sidecar_of[e["basename"]] = (e.get("json_path"), j)

    matches = match_local_to_remote(local_eps, remote)
    need_date = {m["matched_id"] for m in matches if m["matched_id"]}
    dates: dict[str, str] = {}
    if need_date and not dry_run:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for vid, d in zip(need_date, pool.map(_video_upload_date, need_date)):
                dates[vid] = d

    written, unmatched, ambiguous = 0, [], 0
    for m in matches:
        if m["ambiguous"]:
            ambiguous += 1
        if not m["matched_id"]:
            unmatched.append(m["basename"])
            continue
        jp, j = sidecar_of[m["basename"]]
        if not jp:
            continue
        changed = False
        if not _valid_id(j.get("youtube_id")):
            j["youtube_id"] = m["matched_id"]; changed = True
        if not _valid_date(j.get("upload_date")) and dates.get(m["matched_id"]):
            j["upload_date"] = dates[m["matched_id"]]; changed = True
        if pid and not str(j.get("playlist_id") or "").strip():
            j["playlist_id"] = pid; changed = True
        if not str(j.get("source_url") or "").strip():
            j["source_url"] = url; changed = True
        if chan and not str(j.get("channel_id") or "").strip():
            j["channel_id"] = chan; changed = True
        if changed and not dry_run:
            Path(jp).write_text(json.dumps(j, ensure_ascii=False, indent=4), encoding="utf-8")
        written += int(changed)

    return {"region": region, "playlist": playlist, "url": url, "remote": len(remote),
            "matched": sum(1 for m in matches if m["matched_id"]), "written": written,
            "unmatched": unmatched, "ambiguous": ambiguous}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_coverage(rows: list[dict]) -> None:
    tot = {k: sum(r[k] for r in rows) for k in ("files", "have_id", "have_date")}
    print(f"\n=== ID / upload-date coverage ({len(rows)} playlists) ===")
    for r in sorted(rows, key=lambda x: x["missing_id"], reverse=True):
        flag = "  <-- needs id back-fill" if r["missing_id"] else ""
        print(f"  {r['region']:>13} / {r['playlist'][:34]:<34} "
              f"files={r['files']:>3} id={r['have_id']:>3} date={r['have_date']:>3}{flag}")
    n = tot["files"] or 1
    print(f"  TOTAL files={tot['files']}  with id={tot['have_id']} ({100*tot['have_id']/n:.0f}%)  "
          f"with date={tot['have_date']} ({100*tot['have_date']/n:.0f}%)")


def _selfcheck() -> None:
    """Offline matcher test (no network): title match, duration disambiguation, rename rescue."""
    remote = [{"id": "aaa", "title": "حلقة ١ نص نص", "duration": 600},
              {"id": "bbb", "title": "حلقة ٢ نص نص", "duration": 605},
              {"id": "ccc", "title": "برومو", "duration": 30}]
    local = [
        {"basename": "e1", "video_title": "حلقة ١ نص نص", "duration_sec": 600},   # exact title+dur
        {"basename": "e2", "video_title": "RENAMED على يوتيوب", "duration_sec": 605},  # dur-unique rescue -> bbb
        {"basename": "e3", "video_title": "مفقود", "duration_sec": 9999},          # no match
    ]
    res = {m["basename"]: m for m in match_local_to_remote(local, remote)}
    assert res["e1"]["matched_id"] == "aaa", res["e1"]
    assert res["e2"]["matched_id"] == "bbb", res["e2"]   # title changed, duration 605 unique -> matched
    assert res["e3"]["matched_id"] is None, res["e3"]
    # ambiguous: two remotes share a normalized title and the duration can't split them
    amb = match_local_to_remote(
        [{"basename": "x", "video_title": "نفس", "duration_sec": 100}],
        [{"id": "1", "title": "نفس", "duration": 100}, {"id": "2", "title": "نفس", "duration": 100}])
    assert amb[0]["ambiguous"] and amb[0]["matched_id"] is None, amb
    print("OK backfill self-check: title match · duration rescue · no-match · ambiguity.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=config.LOCAL_AUDIO_ROOT, help="corpus root to scan")
    ap.add_argument("--backfill", action="store_true",
                    help="actually recover ids/dates (default: coverage report only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --backfill: match + report, but write nothing")
    ap.add_argument("--only", default=None, help="restrict to playlists whose name contains this")
    ap.add_argument("--selfcheck", action="store_true", help="run the offline matcher test and exit")
    args = ap.parse_args()

    if args.selfcheck:
        _selfcheck()
        return
    if not args.root:
        raise SystemExit("provide --root (or set config.LOCAL_AUDIO_ROOT)")

    rows = scan_coverage(args.root)
    _print_coverage(rows)
    if not args.backfill:
        print("\n(run with --backfill to recover the missing ids/dates; --dry-run to preview)")
        return

    resources = harvest.enumerate_resources(args.root)
    if args.only:
        needle = args.only.lower()
        resources = [r for r in resources if needle in r["playlist"].lower()]
    name_map = _name_url_map()

    print(f"\n=== back-fill {'(dry-run) ' if args.dry_run else ''}{len(resources)} playlist(s) ===")
    reports, no_url, redownload = [], [], []
    for r in resources:
        url = _resolve_url(r, name_map)
        if not url:
            no_url.append(f"{r['region']}/{r['playlist']}")
            continue
        rep = backfill_resource(r, url, dry_run=args.dry_run)
        reports.append(rep)
        for b in rep["unmatched"]:
            redownload.append({"region": rep["region"], "playlist": rep["playlist"], "basename": b})
        tag = f"err={rep['error']}" if rep.get("error") else (
            f"matched {rep['matched']}/{rep['remote']} · wrote {rep['written']}"
            f" · unmatched {len(rep['unmatched'])}" + (f" · ambiguous {rep['ambiguous']}" if rep['ambiguous'] else ""))
        print(f"  {rep['region']:>13} / {rep['playlist'][:30]:<30} {tag}")

    tot_w = sum(r["written"] for r in reports)
    print(f"\n  wrote ids/dates into {tot_w} sidecar(s) across {len(reports)} playlist(s)"
          + (" [dry-run: nothing written]" if args.dry_run else ""))
    if no_url:
        print(f"  {len(no_url)} playlist(s) had no resolvable URL (add a playlist row to "
              f"candidates.psv or harvest them via the app so a source_url is stored):")
        for nm in no_url[:20]:
            print(f"     - {nm}")
    if redownload:
        out = config.DISCOVERY_DIR / "backfill_report.csv"
        import csv
        with open(out, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["region", "playlist", "basename"])
            w.writeheader(); w.writerows(redownload)
        print(f"  {len(redownload)} unmatched local file(s) -> re-download set written to {out}")


if __name__ == "__main__":
    main()

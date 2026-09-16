"""Harvest YouTube channels / playlists straight INTO the source tree the pipeline ingests.

Unlike ``src/harvest.py``'s legacy ``harvest()`` (flat SHA1 WAVs in ``outputs/raw_audio/``,
no diarization, no provenance), this writes the exact on-disk contract that
``harvest.enumerate_resources`` expects, so the existing incremental engine
(``scripts/run_resources.py``) picks up each new show with no downstream change::

    <root>/<dialect_folder>/<show>/<basename>.wav     16 kHz mono PCM
                                   <basename>.rttm    sibling pyannote diarization
                                   <basename>.json    provenance sidecar

Key properties:
  * **Region -> on-disk folder** via ``config.REGION_TO_DIALECT_FOLDER`` (tripoli/benghazi/
    southern) so new shows land beside the first harvest.
  * **De-duplication** (the documented PIPELINE.md TODO): a video already present anywhere in
    the tree is skipped by ``youtube_id`` *before* download; an exact re-upload is caught by an
    audio SHA1 *after* download and discarded — so the same video never mints twin actors.
  * **Resumable**: existing basenames and known ``youtube_id``s are skipped; a crashed run is
    safe to re-run. A seen-manifest persists ids/hashes across runs.
  * **Provenance**: writes the full sidecar schema observed on disk (a superset of what
    ``resource.build_resource_metadata`` reads), populated from yt-dlp metadata.

Requires yt-dlp + ffmpeg on PATH; diarization needs an HF token for pyannote.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import shutil
from pathlib import Path

from libyan_did.shared import config
from libyan_did.shared import diarize, harvest

# Characters Windows forbids in a path component; Arabic + spaces are kept.
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WS = re.compile(r"\s+")
# De-dup cache + audio-hash store, scoped to the source tree it harvests (lives with the data).
SEEN_NAME = ".harvest_seen.json"


# --------------------------------------------------------------------------- #
# Naming helpers
# --------------------------------------------------------------------------- #
def slug_folder(name: str) -> str:
    """A safe show-FOLDER name. Keeps Arabic and spaces; strips illegal chars."""
    s = _ILLEGAL.sub(" ", name or "").strip()
    s = _WS.sub(" ", s)
    return s[:120] or "show"


def slug_tag(name: str) -> str:
    """A short ASCII tag for the file-name prefix (e.g. the source/show name)."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", (name or "")).strip("_")
    return s[:40] or "src"


def dialect_folder_for(region: str) -> str:
    """Canonical region -> on-disk dialect folder (mirrors the first harvest)."""
    return config.REGION_TO_DIALECT_FOLDER.get(region, region.strip().lower())


def macro_for(region: str) -> str:
    return "non-libyan" if region == "Non-Libyan" else "libyan"


def next_index(show_dir: Path, prefix: str) -> int:
    """Next free NNN for ``<prefix>_NNN.wav`` in a show folder (continues a series)."""
    hi = 0
    for wav in show_dir.glob(f"{prefix}_*.wav"):
        m = re.search(rf"{re.escape(prefix)}_(\d+)\.wav$", wav.name)
        if m:
            hi = max(hi, int(m.group(1)))
    return hi + 1


# --------------------------------------------------------------------------- #
# yt-dlp enumeration + download
# --------------------------------------------------------------------------- #
def enumerate_videos(url: str, limit: int | None = None) -> list[dict]:
    """Flat-list a playlist/channel/video as ``[{id, url, title, duration}]`` (no download)."""
    import yt_dlp  # lazy

    opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist", "skip_download": True}
    if limit:
        opts["playlistend"] = int(limit)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = info.get("entries")
    if not entries:  # a single video URL flat-extracts to itself
        entries = [info] if info.get("id") else []
    out: list[dict] = []
    for e in entries:
        if not e:
            continue
        vid = e.get("id")
        vurl = e.get("url") or e.get("webpage_url") or (
            f"https://www.youtube.com/watch?v={vid}" if vid else None)
        if not vurl:
            continue
        out.append({"id": vid, "url": vurl, "title": e.get("title"), "duration": e.get("duration")})
    return out


def download_audio(video_url: str, dest_wav: Path, *, sample_rate: int, channels: int) -> dict | None:
    """Download one video to ``dest_wav`` (16 kHz mono PCM). Returns yt-dlp info or None."""
    import yt_dlp  # lazy

    tmpl = str(dest_wav.with_suffix("")) + ".%(ext)s"
    opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": tmpl,
        "noplaylist": True,
        "retries": 5,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "postprocessor_args": ["-ar", str(sample_rate), "-ac", str(channels)],
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
    except Exception as exc:  # noqa: BLE001 — one bad video must not kill the batch
        print(f"      ! download failed ({type(exc).__name__}): {str(exc)[:120]}")
        return None
    return info if dest_wav.exists() else None


# --------------------------------------------------------------------------- #
# De-dup bookkeeping
# --------------------------------------------------------------------------- #
def audio_sha1(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def load_seen(root: Path) -> tuple[set[str], dict[str, str]]:
    """Known youtube_ids + audio hashes: the persisted manifest UNION existing sidecars."""
    ids: set[str] = set()
    hashes: dict[str, str] = {}
    manifest = Path(root) / SEEN_NAME
    if manifest.exists():
        try:
            d = json.loads(manifest.read_text(encoding="utf-8"))
            ids |= set(d.get("youtube_ids", []))
            hashes.update(d.get("audio_sha1", {}))
        except Exception:  # noqa: BLE001
            pass
    for jf in Path(root).rglob("*.json"):
        if jf.name == SEEN_NAME:
            continue
        try:
            yid = json.loads(jf.read_text(encoding="utf-8")).get("youtube_id")
            if yid and yid != "unknown":
                ids.add(yid)
        except Exception:  # noqa: BLE001
            continue
    return ids, hashes


def save_seen(root: Path, ids: set[str], hashes: dict[str, str]) -> None:
    manifest = Path(root) / SEEN_NAME
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"youtube_ids": sorted(ids), "audio_sha1": hashes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _playlist_id_from_url(url: str) -> str:
    """Extract the ``list=<id>`` playlist id from a source URL (``""`` if none)."""
    if not url or "list=" not in url:
        return ""
    return url.split("list=", 1)[1].split("&", 1)[0].strip()


def write_sidecar(json_path: Path, *, basename: str, info: dict, show_name: str, source: str,
                  dialect_folder: str, macro_class: str, sample_rate: int, channels: int,
                  lang_flag: str = "arabic", source_url: str = "") -> None:
    fields = {
        "file_id": basename,
        "youtube_id": info.get("id", "unknown"),
        "video_title": info.get("title", basename),
        "playlist_name": show_name,
        "duration_sec": int(info.get("duration") or 0),
        "upload_date": info.get("upload_date", "unknown"),
        "source_name": source,
        # Provenance: a durable disk -> source link so Phase 1 can do exact remote-vs-local
        # completion checks (channel_id from the video, playlist_id/source_url from the
        # harvested registry row). Older sidecars lack these — they stay name-matched.
        "channel_id": info.get("channel_id") or "",
        "playlist_id": _playlist_id_from_url(source_url),
        "source_url": source_url,
        "macro_class": macro_class,
        "sub_dialect": dialect_folder,
        "lang_flag": lang_flag,  # arabic | code-switch:<lang> | minority:<lang> (from discovery)
        "domain": "youtube",
        "channel_or_author": info.get("channel") or info.get("uploader") or "unknown",
        "sample_rate": sample_rate,
        "channels": channels,
        "notes": f"harvested by acquire.py on {_dt.date.today().isoformat()}",
    }
    json_path.write_text(json.dumps(fields, ensure_ascii=False, indent=4), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def harvest_source(entry: dict, root: Path, *, pipeline, seen_ids: set[str],
                   seen_hashes: dict[str, str], limit: int | None, max_duration_s: float | None,
                   diarize_audio: bool, dry_run: bool) -> dict:
    """Harvest one registry row (channel/playlist/video) into ``root``."""
    url = entry["url"]
    region = entry["region"]
    source = entry.get("source") or "src"
    lang_flag = entry.get("lang_flag") or "arabic"
    dialect = dialect_folder_for(region)
    macro = macro_for(region)
    show_name = slug_folder(entry.get("show") or source)
    show_dir = root / dialect / show_name
    prefix = f"{'LIB' if macro == 'libyan' else 'NON'}_{dialect}_{slug_tag(source)}"

    print(f"\n=== {region} / {show_name}  <-  {url}")
    try:
        videos = enumerate_videos(url, limit=limit)
    except Exception as exc:  # noqa: BLE001 — a dead/blocked source must not kill the batch
        print(f"    ! enumeration failed ({type(exc).__name__}): {str(exc)[:140]}")
        return {"region": region, "show": show_name, "enumerated": 0, "downloaded": 0,
                "dup_id": 0, "dup_audio": 0, "exists": 0, "too_long": 0, "failed": 0,
                "error": str(exc)[:200]}
    print(f"    {len(videos)} video(s) enumerated; prefix={prefix}")
    if not dry_run:
        show_dir.mkdir(parents=True, exist_ok=True)
        # heal orphans from an earlier crashed run: any wav missing its full triplet is
        # removed so it is re-fetched cleanly (a bare wav would otherwise linger forever,
        # since next_index advances past it).
        cleaned = 0
        for w in show_dir.glob("*.wav"):
            j, r = w.with_suffix(".json"), w.with_suffix(".rttm")
            if not j.exists() or (diarize_audio and not r.exists()):
                for p in (w, j, r):
                    p.unlink(missing_ok=True)
                cleaned += 1
        if cleaned:
            print(f"    ~ cleaned {cleaned} orphan wav(s) from a previous run")

    idx = next_index(show_dir, prefix) if show_dir.exists() else 1
    stats = {"region": region, "show": show_name, "enumerated": len(videos),
             "downloaded": 0, "dup_id": 0, "dup_audio": 0, "exists": 0, "too_long": 0, "failed": 0}

    for v in videos:
        vid, vurl, dur = v["id"], v["url"], v.get("duration")
        if max_duration_s and dur and dur > max_duration_s:
            stats["too_long"] += 1
            continue
        if vid and vid in seen_ids:
            stats["dup_id"] += 1
            continue
        # Name every file by the YouTube id, so <id>.wav / <id>.json / <id>.rttm sit together
        # and the filename IS the provenance. The old prefix+counter is only a fallback for the
        # rare entry with no id.
        basename = _ILLEGAL.sub("_", vid) if vid else f"{prefix}_{idx:03d}"
        dest = show_dir / f"{basename}.wav"
        jsn, rttm = dest.with_suffix(".json"), dest.with_suffix(".rttm")
        # "complete" = wav + json (+ rttm when diarizing). A bare wav is an orphan from a
        # crashed run -> wipe and redo it cleanly (self-heal) rather than skip it forever.
        if dest.exists() and jsn.exists() and (not diarize_audio or rttm.exists()):
            stats["exists"] += 1
            idx += 1
            continue
        if dry_run:
            print(f"    [dry-run] would download {vid} -> {basename}.wav  ({v.get('title')})")
            idx += 1
            continue
        for p in (dest, jsn, rttm):  # clear any partial leftovers for this basename
            p.unlink(missing_ok=True)

        try:
            info = download_audio(vurl, dest, sample_rate=config.SAMPLE_RATE, channels=config.CHANNELS)
            if info is None:
                stats["failed"] += 1
                continue
            h = audio_sha1(dest)
            rel = str(dest.relative_to(root))
            if h in seen_hashes and seen_hashes[h] != rel:
                dest.unlink(missing_ok=True)
                stats["dup_audio"] += 1
                print(f"    ~ audio duplicate of {seen_hashes[h]} — discarded {basename}")
                continue
            write_sidecar(jsn, basename=basename, info=info, show_name=show_name, source=source,
                          dialect_folder=dialect, macro_class=macro, sample_rate=config.SAMPLE_RATE,
                          channels=config.CHANNELS, lang_flag=lang_flag, source_url=url)
            if diarize_audio:
                diarize.diarize_episode(dest, file_id=basename, pipeline=pipeline, out_dir=show_dir)
            # only mark seen once the full triplet is on disk
            seen_hashes[h] = rel
            if vid:
                seen_ids.add(vid)
            stats["downloaded"] += 1
            idx += 1
            print(f"    + {basename}  ({info.get('title','')[:60]})")
        except Exception as exc:  # noqa: BLE001 — one bad video (incl. a diarize error) must not kill the batch
            for p in (dest, jsn, rttm):  # leave no partial triplet behind
                p.unlink(missing_ok=True)
            stats["failed"] += 1
            print(f"    ! processing failed for {basename} ({type(exc).__name__}): {str(exc)[:120]}")
            continue

    return stats


def harvest_registry(rows: list[dict], root: str | Path, *, hf_token: str | None = None,
                     limit: int | None = None, max_duration_s: float | None = None,
                     diarize_audio: bool = True, dry_run: bool = False, on_progress=None) -> list[dict]:
    """Harvest every approved registry row into the source tree. Returns per-source stats.

    ``on_progress(done, total, msg)`` is called once per row (so loose-video downloads — one row
    per video — report per video; playlist rows report per show)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    seen_ids, seen_hashes = load_seen(root)
    print(f"[seen] {len(seen_ids)} known youtube_ids, {len(seen_hashes)} audio hashes")

    pipeline = None
    if diarize_audio and not dry_run:
        print("[diarize] loading pyannote pipeline once ...")
        pipeline = diarize._load_pipeline(hf_token)

    summaries: list[dict] = []
    n = len(rows)
    try:
        for i, entry in enumerate(rows):
            if on_progress:
                on_progress(i, n, f"downloading {entry.get('show') or entry.get('source') or '…'}")
            summaries.append(harvest_source(
                entry, root, pipeline=pipeline, seen_ids=seen_ids, seen_hashes=seen_hashes,
                limit=limit, max_duration_s=max_duration_s, diarize_audio=diarize_audio,
                dry_run=dry_run))
            if on_progress:
                s = summaries[-1]
                on_progress(i + 1, n, f"{s['show']}: +{s['downloaded']} new")
    finally:
        if not dry_run:
            save_seen(root, seen_ids, seen_hashes)
    return summaries


def _norm_rel(rel) -> str:
    """Normalize a '<dialect>/<show>' rel-path to forward slashes, no surrounding slashes."""
    return str(rel).replace("\\", "/").strip("/")


def _read_sidecar_field(json_path, field: str) -> str:
    try:
        return str(json.loads(Path(json_path).read_text(encoding="utf-8")).get(field) or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def diarize_shows(root: str | Path, rel_paths=None, *, hf_token: str | None = None,
                  force: bool = False, dry_run: bool = False, on_progress=None) -> dict:
    """Diarize on-disk shows — the decoupled "diarize later" path the app drives.

    ``rel_paths`` = an iterable of ``'<dialect>/<show>'`` folder paths to target, or ``None``
    for every show on disk. Targeting by **rel-path** (not ``playlist_id``) is what makes this
    work for LEGACY shows whose sidecars predate ``playlist_id``. For each targeted show it
    writes the sibling ``<basename>.rttm`` for every wav that lacks one (the exact call
    ``harvest_source`` uses).

    Idempotent by default (wavs that already have an .rttm are skipped). ``force=True``
    re-diarizes EVERY wav, deleting any existing .rttm first (use after changing diarization
    settings). Returns ``{shows, to_diarize, diarized, failed, missing}`` where ``missing`` =
    requested rel-paths with no matching show on disk.
    """
    want = {_norm_rel(r) for r in rel_paths} if rel_paths is not None else None
    try:
        resources = harvest.enumerate_resources(root)
    except Exception:  # noqa: BLE001
        resources = []

    targets: list[tuple[Path, list[Path]]] = []   # (show_dir, [wavs to diarize])
    matched: set[str] = set()
    for r in resources:
        rel = _norm_rel(r["rel_path"])
        if want is not None and rel not in want:
            continue
        matched.add(rel)
        eps = r["episodes"]
        wavs = [Path(e["audio_path"]) for e in eps
                if force or not e.get("rttm_path")]
        if wavs:
            targets.append((Path(eps[0]["audio_path"]).parent, wavs))

    stats = {"shows": len(targets), "to_diarize": sum(len(w) for _, w in targets),
             "diarized": 0, "failed": 0,
             "missing": sorted(want - matched) if want is not None else []}
    if dry_run or not targets:
        return stats

    print(f"[diarize] loading pyannote pipeline once for {stats['to_diarize']} wav(s) "
          f"(force={force}) ...")
    if on_progress:
        on_progress(0, stats["to_diarize"], "loading diarization model…")
    pipeline = diarize._load_pipeline(hf_token)
    done = 0
    for show_dir, wavs in targets:
        for wav in wavs:
            try:
                if force:
                    (show_dir / f"{wav.stem}.rttm").unlink(missing_ok=True)  # diarize_episode skips if present
                diarize.diarize_episode(wav, wav.stem, pipeline=pipeline, out_dir=show_dir)
                stats["diarized"] += 1
                print(f"    + rttm {wav.stem}")
            except Exception as exc:  # noqa: BLE001 — one bad episode must not kill the batch
                stats["failed"] += 1
                print(f"    ! diarize failed {wav.name} ({type(exc).__name__}): {str(exc)[:120]}")
            done += 1
            if on_progress:
                on_progress(done, stats["to_diarize"], wav.stem)
    return stats


def reset_shows(root: str | Path, rel_paths, *, dry_run: bool = False) -> dict:
    """Force-re-download prep: wipe the given on-disk shows so a later harvest re-fetches them.

    Deletes each show's wav/json/rttm triplets AND purges their ``youtube_id``s + audio hashes
    from the de-dup seen-manifest (``.harvest_seen.json``) — otherwise ``harvest_source`` would
    skip the videos as ``dup_id`` / ``dup_audio`` and the re-download would be a no-op. Only the
    named shows are touched; the rest of the corpus and its seen-state are left intact.

    ``rel_paths`` = iterable of ``'<dialect>/<show>'``. Returns ``{shows, files, ids_purged}``.
    """
    root = Path(root)
    want = {_norm_rel(r) for r in rel_paths}
    ids_drop: set[str] = set()
    rels_drop: set[str] = set()
    n_files = n_shows = 0
    for rel in want:
        sd = root / rel
        if not sd.exists() or not sd.is_dir():
            continue
        n_shows += 1
        for j in sd.glob("*.json"):
            yid = _read_sidecar_field(j, "youtube_id")
            if yid and yid.lower() != "unknown":
                ids_drop.add(yid)
        for w in sd.glob("*.wav"):
            try:
                rels_drop.add(_norm_rel(w.relative_to(root)))
            except Exception:  # noqa: BLE001
                pass
        for pat in ("*.wav", "*.json", "*.rttm"):
            for f in sd.glob(pat):
                n_files += 1
                if not dry_run:
                    f.unlink(missing_ok=True)

    if not dry_run and (ids_drop or rels_drop):
        seen_ids, seen_hashes = load_seen(root)
        seen_ids -= ids_drop
        seen_hashes = {h: rel for h, rel in seen_hashes.items()
                       if _norm_rel(rel) not in rels_drop}
        save_seen(root, seen_ids, seen_hashes)
    return {"shows": n_shows, "files": n_files, "ids_purged": len(ids_drop)}


def forget_videos(root: str | Path, *, ids=(), rel_prefixes=()) -> dict:
    """Forget videos in the de-dup manifest so they download fresh next time — the case
    ``reset_shows`` can't reach because the folder was **already deleted by hand** (so there are
    no sidecars left to read the ids from).

    Pass the playlist's ``ids`` (re-enumerated from YouTube) and/or ``rel_prefixes`` (its
    ``'<dialect>/<show>'`` folder) to also drop stale audio hashes under it. Returns
    ``{"ids": dropped_id_count, "hashes": dropped_hash_count}``.
    """
    root = Path(root)
    want_ids = {str(i).strip() for i in ids if str(i).strip()}
    prefixes = [_norm_rel(r) for r in rel_prefixes if str(r).strip()]
    seen_ids, seen_hashes = load_seen(root)
    n_ids = len(seen_ids & want_ids)
    seen_ids -= want_ids
    n_h = 0
    if prefixes:
        kept = {}
        for h, rel in seen_hashes.items():
            nr = _norm_rel(rel)
            if any(nr == p or nr.startswith(p + "/") for p in prefixes):
                n_h += 1
            else:
                kept[h] = rel
        seen_hashes = kept
    save_seen(root, seen_ids, seen_hashes)
    return {"ids": n_ids, "hashes": n_h}


def clean_videos(root: str | Path, ids, *, dry_run: bool = False) -> dict:
    """Delete the on-disk files for specific YouTube ids AND forget them in the de-dup manifest,
    so they download fresh next time — the per-video equivalent of ``reset_shows`` (used by the
    loose-videos cleaner).

    Files are matched by each sidecar's ``youtube_id`` (not the filename), so it works whatever
    the wav is named. Returns ``{"files": deleted_file_count, "ids": id_count}``.
    """
    root = Path(root)
    want = {str(i).strip() for i in ids if str(i).strip()}
    if not want:
        return {"files": 0, "ids": 0}
    targets = [jf for jf in root.rglob("*.json")
               if jf.name != SEEN_NAME and _read_sidecar_field(jf, "youtube_id") in want]
    rels_drop, n_files = set(), 0
    for jf in targets:
        stem = jf.with_suffix("")
        try:
            rels_drop.add(_norm_rel(stem.with_suffix(".wav").relative_to(root)))
        except Exception:  # noqa: BLE001
            pass
        for ext in (".wav", ".json", ".rttm"):
            f = stem.with_suffix(ext)
            if f.exists():
                n_files += 1
                if not dry_run:
                    f.unlink(missing_ok=True)
    if not dry_run:
        seen_ids, seen_hashes = load_seen(root)
        seen_ids -= want
        seen_hashes = {h: rel for h, rel in seen_hashes.items() if _norm_rel(rel) not in rels_drop}
        save_seen(root, seen_ids, seen_hashes)
    return {"files": n_files, "ids": len(want)}


def delete_resource(root: str | Path, rel_path: str, *, region: str = "", playlist: str = "") -> dict:
    """Completely remove a dead playlist: its source folder ``<root>/<rel_path>``, its corpus
    outputs ``outputs/corpus/<region>/<playlist>`` (clustering manifests + actor clips), and its
    ids/hashes from the de-dup manifest — so a fresh download later isn't skipped as a duplicate.

    Returns ``{"source_deleted", "corpus_deleted", "ids_purged"}``.
    """
    root = Path(root)
    src = root / _norm_rel(rel_path)
    r = reset_shows(root, [rel_path])             # purge ids/hashes from sidecars, then drop files
    src_deleted = False
    if src.exists():
        shutil.rmtree(src, ignore_errors=True)    # remove the (now-empty) source folder itself
        src_deleted = True
    corpus_deleted = False
    if region and playlist:
        cdir = config.CORPUS_DIR / region / playlist
        if cdir.exists():
            shutil.rmtree(cdir, ignore_errors=True)
            corpus_deleted = True
    return {"source_deleted": src_deleted, "corpus_deleted": corpus_deleted,
            "ids_purged": r["ids_purged"]}

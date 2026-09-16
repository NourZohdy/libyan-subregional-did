"""Phase-1 harvest: enumerate playlists and download full episodes once, kept forever.

Full episodes land in ``RAW_AUDIO_DIR`` at 16 kHz mono and are NEVER auto-deleted or
re-downloaded (resumable: existing files are skipped). Only per-segment clips are
refused — those are never written here. Requires yt-dlp + ffmpeg on PATH.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections import defaultdict
from pathlib import Path

from libyan_did.shared.config import CHANNELS, RAW_AUDIO_DIR, REGION_ALIASES, REGIONS, SAMPLE_RATE

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")


def stable_file_id(video_url: str) -> str:
    """Short deterministic id from a video URL or file path (used as file_id)."""
    return hashlib.sha1(video_url.encode("utf-8")).hexdigest()[:12]


def enumerate_playlist(playlist_url: str) -> list[str]:
    """Return the list of video URLs in a playlist via yt-dlp flat extraction."""
    import yt_dlp  # lazy

    opts = {"quiet": True, "extract_flat": "in_playlist", "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
    entries = info.get("entries") or []
    urls = []
    for e in entries:
        if not e:
            continue
        urls.append(e.get("url") or e.get("webpage_url") or e.get("id"))
    return [u for u in urls if u]


def download_episode(video_url: str, out_dir: Path = RAW_AUDIO_DIR) -> Path:
    """Download a single video's audio to 16 kHz mono WAV; skip if present."""
    import yt_dlp  # lazy

    out_dir.mkdir(parents=True, exist_ok=True)
    fid = stable_file_id(video_url)
    final = out_dir / f"{fid}.wav"
    if final.exists():
        return final  # resumable: never re-download

    tmpl = str(out_dir / f"{fid}.%(ext)s")
    opts = {
        "quiet": True,
        "format": "bestaudio/best",
        "outtmpl": tmpl,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "postprocessor_args": ["-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS)],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([video_url])
    return final


# --------------------------------------------------------------------------- #
# Local-folder ingestion (use already-downloaded WAVs instead of YouTube)
# --------------------------------------------------------------------------- #
def _normalize_macro(name: str) -> str:
    """Folder name -> 'Libyan' / 'Non-Libyan'."""
    s = name.strip().lower().replace("-", "_")
    return "Non-Libyan" if s in {"non_libyan", "nonlibyan"} else "Libyan"


# Bag folders hold unrelated single videos, not one channel's show; each WAV is its own source.
_BAG_FOLDERS = {"single_videos", "unknown"}

# Sidecar channel_or_author values that are NOT a real channel — treat as missing so the
# WAV keys on its (unique) basename instead of colliding every placeholder into one "channel".
_PLACEHOLDER_KEYS = {"unknown", "none", "null", "n/a", "na", "-", ""}


def _read_sidecar_field(json_path, field: str) -> str:
    """Read one field from a WAV's sibling .json sidecar; '' if missing/unreadable."""
    if not json_path:
        return ""
    import json  # lazy
    try:
        return str(json.loads(Path(json_path).read_text(encoding="utf-8")).get(field) or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _normalize_region(name: str) -> str:
    """Folder name -> canonical region label (case-insensitive match to config.REGIONS)."""
    s = name.strip().lower().replace("-", "_")
    for r in REGIONS:
        if r.lower() == s:
            return r
    if s in {"non_libyan", "nonlibyan"}:
        return "Non-Libyan"
    return name.strip().title()


def canonical_region(name: str) -> str:
    """On-disk dialect folder name -> canonical region (config.REGION_ALIASES first).

    Folders are never renamed; this is only for output labels. Falls back to an exact
    REGIONS match, then to title-case so unknown folders still produce a stable label.
    """
    s = name.strip().lower().replace("-", "_")
    if s in REGION_ALIASES:
        return REGION_ALIASES[s]
    for r in REGIONS:
        if r.lower() == s:
            return r
    if s in {"non_libyan", "nonlibyan"}:
        return "Non-Libyan"
    return name.strip().title()


def harvest_local(root: str | Path) -> list[dict]:
    """Enumerate already-downloaded audio under a ``<macro_class>/<sub_cat>/<playlist>/``
    folder tree — no download, files are read in place.

    Returns episode records with the same shape as :func:`harvest`, plus an explicit
    ``macro`` field from the top-level folder. ``master_audio_path`` points at the
    original local file, so Phase 1/2 slice straight from it.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"LOCAL_AUDIO_ROOT does not exist: {root}")

    records: list[dict] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue
        rel = path.relative_to(root).parts
        macro_raw = rel[0] if len(rel) >= 1 else "unknown"
        sub_raw = rel[1] if len(rel) >= 2 else "unknown"
        playlist = rel[2] if len(rel) >= 3 else (rel[-2] if len(rel) >= 2 else "unknown")
        abspath = str(path.resolve())
        records.append({
            "file_id": stable_file_id(abspath),
            "video_url": "",
            "audio_path": abspath,
            "region": _normalize_region(sub_raw),
            "source": playlist,
            "playlist": playlist,
            "macro": _normalize_macro(macro_raw),
        })
    return records


def enumerate_resources(root: str | Path) -> list[dict]:
    """Group already-downloaded audio under ``<root>/<dialect>/<playlist>/`` into
    per-playlist *resources*, attaching each WAV's sibling ``.rttm`` (so no
    re-diarization is needed) and ``.json`` sidecar.

    ``root`` is the dialect parent, e.g. ``D:\\Research\\DID_Set\\libyan``. Folder names
    are NEVER renamed; ``region`` is the canonical label used for outputs. Each resource:

        {region, region_raw, playlist, macro, rel_path, episodes:[
            {file_id, audio_path, rttm_path, json_path, basename}, ...]}

    Only ``.wav`` files are ingested; playlists with no ``.wav`` are skipped.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"resource root does not exist: {root}")
    macro = _normalize_macro(root.name)

    resources: list[dict] = []
    for dialect_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        region = canonical_region(dialect_dir.name)
        for show_dir in sorted(p for p in dialect_dir.iterdir() if p.is_dir()):
            episodes: list[dict] = []
            for wav in sorted(show_dir.glob("*.wav")):
                rttm = wav.with_suffix(".rttm")
                jsn = wav.with_suffix(".json")
                episodes.append({
                    "file_id": stable_file_id(str(wav.resolve())),
                    "audio_path": str(wav.resolve()),
                    "rttm_path": str(rttm.resolve()) if rttm.exists() else None,
                    "json_path": str(jsn.resolve()) if jsn.exists() else None,
                    "basename": wav.stem,
                })
            if not episodes:
                continue
            if show_dir.name.lower() in _BAG_FOLDERS:
                # Bag folder: key each WAV on its sidecar channel_or_author so videos from
                # the SAME channel group into one resource (one source), while a
                # placeholder/empty value falls back to the WAV's unique basename so
                # unrelated lone videos do NOT collapse into one fake "channel" (FR-005).
                groups: dict[str, list[dict]] = defaultdict(list)
                for ep in episodes:
                    raw = (_read_sidecar_field(ep["json_path"], "channel_or_author") or "").strip()
                    key = raw if raw.lower() not in _PLACEHOLDER_KEYS else ep["basename"]
                    groups[key].append(ep)
                for key, eps in groups.items():
                    resources.append({
                        "region": region,
                        "region_raw": dialect_dir.name,
                        "playlist": key,
                        "macro": macro,
                        "rel_path": f"{dialect_dir.name}/{key}",
                        "episodes": eps,
                    })
            else:
                resources.append({
                    "region": region,
                    "region_raw": dialect_dir.name,
                    "playlist": show_dir.name,
                    "macro": macro,
                    "rel_path": f"{dialect_dir.name}/{show_dir.name}",
                    "episodes": episodes,
                })
    return _disambiguate_keys(resources)


def _disambiguate_keys(resources: list[dict]) -> list[dict]:
    """Guarantee every resource has a unique ``(region, playlist)`` — that pair is the
    output-directory key (``resource_paths``), so a duplicate makes two sources write the
    SAME dir and the second silently clobbers the first (observed: a lone bag video whose
    channel_or_author equals a real show's folder name, and same-channel bag videos).

    The largest resource keeps the clean key; each colliding sibling is suffixed with its
    first episode's basename (unique, deterministic). No episode is dropped or merged.
    """
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in resources:
        by_key[(r["region"], r["playlist"])].append(r)
    for (region, playlist), group in by_key.items():
        if len(group) < 2:
            continue
        group.sort(key=lambda r: (-len(r["episodes"]), r["episodes"][0]["basename"]))
        for r in group[1:]:  # keep the largest as-is; disambiguate the rest
            suffix = r["episodes"][0]["basename"]
            r["playlist"] = f"{playlist}__{suffix}"
            r["rel_path"] = f"{r['region_raw']}/{r['playlist']}"
    return resources


def harvest(download_matrix: list[dict], out_dir: Path = RAW_AUDIO_DIR) -> list[dict]:
    """Download all playlists; return episode records for the manifest pipeline.

    Each record: {file_id, video_url, audio_path, region, source, playlist}.
    """
    records: list[dict] = []
    for entry in download_matrix:
        playlist = entry["playlist_url"]
        region = entry["region"]
        source = entry["source"]
        for video_url in enumerate_playlist(playlist):
            audio_path = download_episode(video_url, out_dir)
            records.append({
                "file_id": stable_file_id(video_url),
                "video_url": video_url,
                "audio_path": str(audio_path),
                "region": region,
                "source": source,
                "playlist": playlist,
            })
    return records

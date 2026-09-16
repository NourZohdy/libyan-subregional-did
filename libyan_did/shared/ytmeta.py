"""Cheap, bounded YouTube *metadata* helpers (no audio download).

Used by the discovery-side scripts to enrich the review registry:
  * ``channel_video_count`` / ``playlist_video_count`` — how many videos.
  * ``channel_playlist_count`` — how many playlists (shows) a channel has.
  * ``list_channel_playlists`` — the channel's playlists (id + title + size),
    the raw material for splitting a multi-region TV channel into per-show rows.

Everything is ``yt-dlp --flat-playlist`` (metadata only, never downloads media) and
*bounded*: counts cap out (e.g. ``"500+"``) so a giant national channel can't stall a
batch, and every call has a hard subprocess timeout. Failures degrade to ``""`` / ``[]``
rather than raising, so one dead channel never kills an enrichment run.
"""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

YTDLP = "yt-dlp"

# Real *show* playlists we keep when splitting a channel. `PL…` = ordinary playlist,
# `OLAK5uy_…` = auto Album/Topic show. We deliberately DROP `UU…` (the whole uploads feed),
# `RD…` (mixes), `LL`/`FL` (liked/favourites) — those are not shows.
_PLAYLIST_PREFIXES = ("PL", "OLAK5uy_")


def _run(cmd: list[str], timeout: int) -> str:
    # Force yt-dlp (itself a Python program) to write UTF-8 to the pipe, so Arabic titles from
    # `--print` survive on Windows — otherwise the default pipe encoding mangles them to '?'
    # before we ever decode. (`--dump-single-json` paths were already safe via \u escapes.)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=timeout, env=env)
        return out.stdout or ""
    except Exception:  # noqa: BLE001 — a dead/slow source must not kill the batch
        return ""


def channel_base(channel_id: str = "", url: str = "") -> str:
    """Canonical channel base URL from a channel_id (preferred) or an existing url."""
    cid = (channel_id or "").strip()
    if cid.startswith("UC"):
        return f"https://www.youtube.com/channel/{cid}"
    return (url or "").strip().rstrip("/")


def count_entries(url: str, *, cap: int = 500, timeout: int = 200) -> str:
    """Number of videos a channel-tab/playlist URL lists, bounded to ``cap`` (``"500+"`` if hit)."""
    if not url:
        return ""
    cmd = [YTDLP, url, "--flat-playlist", "--no-warnings", "--ignore-errors",
           "--playlist-end", str(cap), "--print", "%(id)s"]
    ids = [ln for ln in _run(cmd, timeout).splitlines() if ln.strip()]
    n = len(ids)
    if n == 0:
        return ""
    return f"{cap}+" if n >= cap else str(n)


def channel_video_count(channel_id: str = "", url: str = "", *, cap: int = 500) -> str:
    base = channel_base(channel_id, url)
    if not base:
        return ""
    # The /videos tab excludes Shorts/Live; good enough as a size signal for review.
    return count_entries(f"{base}/videos", cap=cap)


def channel_title(channel_id: str = "", url: str = "", *, timeout: int = 120) -> str:
    """The channel's display name (or '' on failure). Metadata only, bounded, fail-soft.

    Reads the channel page's top-level JSON (``--playlist-end 1`` so it never enumerates a whole
    channel); the name lives in ``channel`` / ``uploader`` / ``title``.
    """
    base = channel_base(channel_id, url)
    if not base:
        return ""
    cmd = [YTDLP, base, "--flat-playlist", "--playlist-end", "1", "--no-warnings",
           "--ignore-errors", "--dump-single-json"]
    raw = _run(cmd, timeout)
    if not raw.strip():
        return ""
    try:
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        return ""
    return (data.get("channel") or data.get("uploader") or data.get("title") or "").strip()


def playlist_video_count(url: str, *, cap: int = 500) -> str:
    return count_entries(url, cap=cap)


def list_channel_playlists(channel_id: str = "", url: str = "", *,
                           cap: int = 200, timeout: int = 200) -> list[dict]:
    """A channel's playlists as ``[{id, title}]`` (each playlist == one show).

    This is the input to the multi-region "expand a TV channel into its shows" flow:
    each playlist becomes a candidate row the reviewer can region-tag on its own.
    """
    base = channel_base(channel_id, url)
    if not base:
        return []
    cmd = [YTDLP, f"{base}/playlists", "--flat-playlist", "--no-warnings", "--ignore-errors",
           "--playlist-end", str(cap), "--print", "%(id)s\t%(title)s"]
    out: list[dict] = []
    for line in _run(cmd, timeout).splitlines():
        parts = line.split("\t")
        pid = parts[0].strip() if parts else ""
        if not pid.startswith(_PLAYLIST_PREFIXES):
            continue
        title = parts[1].strip() if len(parts) > 1 else ""
        out.append({"id": pid, "title": title})
    return out


def channel_playlist_count(channel_id: str = "", url: str = "", *, cap: int = 200) -> str:
    pls = list_channel_playlists(channel_id, url, cap=cap)
    n = len(pls)
    if n == 0:
        return ""
    return f"{cap}+" if n >= cap else str(n)


def list_channel_videos(channel_id: str = "", url: str = "", *, cap: int = 500,
                        timeout: int = 300) -> list[dict]:
    """A channel's uploads as ``[{id, title, url, duration, upload_date}]`` (metadata only).

    Reads the ``/videos`` tab flat (no download), bounded to ``cap``. ``duration`` /
    ``upload_date`` may come back blank in flat mode; callers format defensively. Fail-soft -> ``[]``.
    """
    base = channel_base(channel_id, url)
    if not base:
        return []
    cmd = [YTDLP, f"{base}/videos", "--flat-playlist", "--no-warnings", "--ignore-errors",
           "--playlist-end", str(cap),
           "--print", "%(id)s\t%(title)s\t%(duration)s\t%(upload_date)s"]
    out: list[dict] = []
    for line in _run(cmd, timeout).splitlines():
        parts = line.split("\t")
        vid = parts[0].strip() if parts else ""
        if not vid:
            continue
        out.append({
            "id": vid,
            "title": parts[1].strip() if len(parts) > 1 else "",
            "url": f"https://www.youtube.com/watch?v={vid}",
            "duration": parts[2].strip() if len(parts) > 2 else "",
            "upload_date": parts[3].strip() if len(parts) > 3 else "",
        })
    return out


def playlist_video_ids(url: str, *, cap: int = 500, timeout: int = 300) -> list[str]:
    """Every video id in a playlist (flat, bounded, metadata only). Fail-soft -> ``[]``."""
    if not url:
        return []
    cmd = [YTDLP, url, "--flat-playlist", "--no-warnings", "--ignore-errors",
           "--playlist-end", str(cap), "--print", "%(id)s"]
    return [ln.strip() for ln in _run(cmd, timeout).splitlines() if ln.strip()]


def playlist_meta(url: str, *, cap: int = 300, n_titles: int = 6,
                  timeout: int = 200) -> dict:
    """One pass over a playlist -> ``{title, video_count, sample_titles}``.

    Reads the name from the playlist's **top-level** metadata via ``--dump-single-json``: the
    channel ``/playlists`` tab returns blank titles in flat mode, and the per-entry
    ``playlist_title`` field also goes blank when YouTube throttles a parallel batch — but the
    top-level ``title`` survives. Count + sample video titles come from the same call. Fail-soft.
    """
    out = {"title": "", "video_count": "", "sample_titles": []}
    if not url:
        return out
    cmd = [YTDLP, url, "--flat-playlist", "--no-warnings", "--ignore-errors",
           "--playlist-end", str(cap), "--dump-single-json"]
    raw = _run(cmd, timeout)
    if not raw.strip():
        return out
    try:
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        return out
    out["title"] = (data.get("title") or "").strip()
    entries = [e for e in (data.get("entries") or []) if e]
    n = len(entries)
    out["video_count"] = "" if n == 0 else (f"{cap}+" if n >= cap else str(n))
    out["sample_titles"] = [(e.get("title") or "").strip()
                            for e in entries[:n_titles] if (e.get("title") or "").strip()]
    return out


def list_channel_playlists_with_counts(channel_id: str = "", url: str = "", *, cap: int = 200,
                                       n_titles: int = 6, max_workers: int = 4) -> list[dict]:
    """A channel's show playlists, each with its real title, video count + sample video titles.

    Returns ``[{id, title, url, video_count, sample_titles}]``. The title comes from the
    playlist itself (``playlist_meta``), because the channel ``/playlists`` tab returns blank
    titles in flat mode. Fetched concurrently; all fail-soft.
    """
    pls = list_channel_playlists(channel_id, url, cap=cap)

    def _one(p: dict) -> dict:
        purl = f"https://www.youtube.com/playlist?list={p['id']}"
        meta = playlist_meta(purl, n_titles=n_titles)
        return {"id": p["id"], "title": meta["title"] or p.get("title", ""), "url": purl,
                "video_count": meta["video_count"], "sample_titles": meta["sample_titles"]}

    if not pls:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_one, pls))

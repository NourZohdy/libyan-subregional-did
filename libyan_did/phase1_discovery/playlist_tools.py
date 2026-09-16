"""Helpers for the Playlist browser (window 2): build playlist rows + match the local corpus.

Window 2 lists an approved channel's playlists (live from YouTube) and lets the human pick which
to harvest. Picked playlists become ``url_type=playlist`` registry rows, region inherited from the
channel. To avoid re-downloading, each playlist is matched **by name** against the on-disk corpus
(``harvest.enumerate_resources``) — show folders are named after the show/playlist, so a normalized
name match tells us "you already have this one".
"""

from __future__ import annotations

import re

from libyan_did.shared import harvest
from libyan_did.shared import registry

# Auto-generated feed/section "playlists" we never want as candidate rows. NOTE: a blank title
# is NOT junk — the channel /playlists tab often returns blank titles for real shows, so those
# are kept (shown as "(untitled)" / named once their real title is fetched).
_JUNK_TITLES = {"shorts", "live", "videos", "popular uploads", "uploads", "for you"}


def is_junk_playlist(title: str) -> bool:
    """A playlist title that is a feed/section, not a real show (blank is allowed)."""
    return (title or "").strip().lower() in _JUNK_TITLES


def playlist_url(playlist: dict) -> str:
    return playlist.get("url") or f"https://www.youtube.com/playlist?list={playlist['id']}"


def make_playlist_row(channel_row: dict, playlist: dict, *, region: str,
                      status: str = "approved") -> dict:
    """Build one ``url_type=playlist`` registry row from a channel + one of its playlists.

    ``region`` is passed in explicitly (window 2 defaults it to the channel's region and lets the
    reviewer override per playlist).
    """
    src = channel_row.get("source", "") or ""
    cid = (channel_row.get("channel_id") or "").strip()
    row = {c: "" for c in registry.COLS}
    row.update({
        "url": playlist_url(playlist),
        "url_type": "playlist",
        "region": region,
        "source": playlist.get("title") or f"{src} — playlist",
        "channel_id": cid,
        "video_count": str(playlist.get("video_count", "") or ""),
        "playlist_count": "",
        "format_type": channel_row.get("format_type", "") or "",
        "lang_flag": channel_row.get("lang_flag", "") or "arabic",
        "confidence": "high",
        "evidence": f"show playlist of «{src}» — picked in Playlist browser",
        "discovery_path": f"channel:{cid or src}",
        "status": status,
        "notes": "picked playlist to harvest",
    })
    return row


# --------------------------------------------------------------------------- #
# Already-downloaded matching: YouTube playlist title  <->  local show folder
# --------------------------------------------------------------------------- #
def normalize_name(s: str) -> str:
    """Lowercase, keep alphanumerics + Arabic letters, collapse the rest to single spaces."""
    s = (s or "").lower()
    s = re.sub(r"[^0-9a-z؀-ۿ]+", " ", s)
    return " ".join(s.split())


def local_show_index(root: str) -> dict[str, dict]:
    """``{normalized_show_name: {region, episodes, raw}}`` from the on-disk corpus.

    Fail-soft (missing/unreadable root -> ``{}``) so the browser still works without the corpus.
    """
    index: dict[str, dict] = {}
    try:
        resources = harvest.enumerate_resources(root)
    except Exception:  # noqa: BLE001
        return index
    for r in resources:
        key = normalize_name(r["playlist"])
        if not key:
            continue
        n = len(r["episodes"])
        if key in index:
            index[key]["episodes"] += n
        else:
            index[key] = {"region": r["region"], "episodes": n, "raw": r["playlist"]}
    return index


def match_local(title: str, index: dict[str, dict]) -> dict | None:
    """A local show that matches this playlist title by name, or ``None``.

    Exact normalized match first, then containment either way (with a length floor so short
    generic tokens don't match everything).
    """
    key = normalize_name(title)
    if not key or not index:
        return None
    if key in index:
        return index[key]
    for k, v in index.items():
        if len(min(k, key, key=len)) >= 5 and (k in key or key in k):
            return v
    return None


def channel_has_local(channel_source: str, index: dict[str, dict]) -> bool:
    """Whether any local show folder name matches this channel's name (the fallback flag)."""
    return match_local(channel_source, index) is not None

"""Add from channel (window 3) — pick an approved channel and harvest its shows.

See an approved channel's YouTube playlists with their live on-disk status, and
**Download** / **Diarize** the ones you want.

Rules (no surprises):
  • A playlist already downloaded is **not re-downloaded**; one already diarized is **not
    re-diarized**. Tick **Force re-download** / **Force re-diarize** to override.
  • Download then diarize are separate steps — download now, diarize whenever.

Everything already on disk lives in **My corpus** (where you also cluster shows); clustering a
show happens there, and speaker review in the separate Speaker Validation app.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from libyan_did.shared import acquire
from libyan_did.shared import config
from libyan_did.shared import pstate
from libyan_did.shared import registry
from libyan_did.shared import ytmeta
from libyan_did.phase1_discovery import jobs
from libyan_did.phase1_discovery import playlist_tools

REGISTRY = config.DISCOVERY_DIR / "candidates.psv"
CACHE_FILE = config.DISCOVERY_DIR / "channel_playlists.json"
DEFAULT_ROOT = config.LOCAL_AUDIO_ROOT or r"D:\Research\DID_Set\libyan"

_BADGE = {"picked": "⚪ picked", "downloaded": "📥 downloaded",
          "diarized": "🗣 diarized", "clustered": "🔗 clustered"}


def load_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


@st.cache_data(show_spinner=False)
def load_disk(root: str) -> pd.DataFrame:
    """On-disk pipeline state for every show (synced from the corpus tree). Cached; the
    Download / Diarize actions call ``load_disk.clear()`` so the next render re-scans."""
    return pstate.sync_from_disk(root)


def _int(x) -> int:
    try:
        return int(str(x).strip())
    except Exception:  # noqa: BLE001
        return 0


def _dates(lo: str, hi: str) -> str:
    f = lambda d: f"{d[:4]}-{d[4:6]}" if len(str(d)) >= 6 else str(d)
    lo, hi = str(lo or ""), str(hi or "")
    if lo and hi:
        return f(lo) if lo == hi else f"{f(lo)} → {f(hi)}"
    return f(lo or hi)


def run_diarize(root: str, rel_paths: list[str], hf_token: str, force: bool) -> None:
    """Validate, then run diarization as a background job (live bar; UI stays free)."""
    if not hf_token.strip():
        st.error("Diarization needs an **HF token** — add one in the sidebar.")
        return
    rel_paths = [r for r in rel_paths if r]
    if not rel_paths:
        st.warning("None of the picked shows are downloaded yet — download them first.")
        return

    def target(report):
        s = acquire.diarize_shows(root, rel_paths, hf_token=hf_token.strip(), force=force,
                                  on_progress=report)
        if s["to_diarize"] == 0:
            return "nothing to diarize — every episode already has an .rttm"
        return (f"wrote .rttm for {s['diarized']} episode(s) across {s['shows']} show(s)"
                + (f"; {s['failed']} failed" if s.get("failed") else ""))

    jobs.start(f"Diarize{' (FORCE)' if force else ''}", target)


st.title("➕ Add from channel — harvest an approved channel's shows")

with st.sidebar:
    st.header("Settings")
    root = st.text_input("Corpus root", value=DEFAULT_ROOT,
                         help="Folder your shows download into / are read from.")
    hf_token = st.text_input("HF token (for diarization)", value=os.environ.get("HF_TOKEN", ""),
                             type="password", help="Needed only when you diarize.")

disk = load_disk(root) if root.strip() else pstate.read()
if "rel_path" not in disk.columns:     # heal a stale st.cache_data from an older app version
    load_disk.clear()
    disk = load_disk(root) if root.strip() else pstate.read()
disk = disk.reindex(columns=pstate.COLS, fill_value="")   # never KeyError on an expected column
if root.strip():
    c = pstate.summary(disk)
    st.sidebar.caption(f"📥 {c.get('downloaded',0)} · 🗣 {c.get('diarized',0)} · "
                       f"🔗 {c.get('clustered',0)} · ⚪ {c.get('picked',0)} tracked")
    if st.sidebar.button("🔄 Rescan disk"):
        load_disk.clear(); st.rerun()

# Disk indices for the channel-join: by playlist_id (new harvests) AND by normalized show
# name (LEGACY shows have no playlist_id, so name is the only bridge to the live YouTube list).
on_disk = disk[disk["stage"].isin(["downloaded", "diarized", "clustered"])]
by_pid = {str(r["playlist_id"]): r for _, r in on_disk.iterrows()
          if str(r["playlist_id"]) and not str(r["playlist_id"]).startswith("name:")}
by_name = {playlist_tools.normalize_name(r["title"]): r for _, r in on_disk.iterrows()
           if str(r["title"]).strip()}

df = registry.read(REGISTRY)
channels = df[(df["url_type"].str.strip() == "channel")
              & (df["status"].str.strip() == "approved")]
if channels.empty:
    st.warning("No **approved** channels yet. Approve Libyan channels in **Review channels** first.")
    st.stop()

if "pl_cache" not in st.session_state:
    st.session_state.pl_cache = load_cache()

labels = {f"{(r['source'] or r['url'])[:50]}  ·  {r['region']}": r["url"]
          for _, r in channels.iterrows()}
choice = st.selectbox("Approved channel", list(labels.keys()))
sel_url = labels[choice]
ch = df[df["url"].str.strip() == sel_url].iloc[0].to_dict()
cid = (ch.get("channel_id") or "").strip()
ch_region = ch.get("region", "Unknown")
cache_key = cid or sel_url

cc = st.columns([1, 5])
refresh = cc[0].button("🔄 refresh", help="Re-fetch this channel's playlists from YouTube")
cc[1].caption(f"`{sel_url}`  ·  channel region **{ch_region}** (new shows inherit it)")

cache = st.session_state.pl_cache
if refresh or cache_key not in cache:
    with st.spinner("Fetching playlists via yt-dlp (metadata only)…"):
        cache[cache_key] = ytmeta.list_channel_playlists_with_counts(cid, sel_url)
    save_cache(cache)
playlists = [p for p in cache.get(cache_key, [])
             if not playlist_tools.is_junk_playlist(p.get("title", ""))]

LOOSE_PREFIX = "[loose] "   # synthetic show name for a channel's not-in-any-playlist videos


def _fmt_dur(d) -> str:
    try:
        s = int(float(d))
    except Exception:  # noqa: BLE001 — "NA"/blank in flat mode
        return ""
    return f"{s // 60}:{s % 60:02d}"


def _fmt_date(d) -> str:
    d = str(d or "").strip()
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 and d.isdigit() else ""


def _record_picks(picked_df) -> list[dict]:
    """Record picked shows in candidates.psv + playlist_state.csv; return harvest entries."""
    entries = [playlist_tools.make_playlist_row(
                   ch, {"title": r["playlist"], "url": r["open"], "video_count": r["videos"]},
                   region=r["region"], status="approved")
               for _, r in picked_df.iterrows()]
    fresh = registry.read(REGISTRY)
    have = set(fresh["url"].str.strip())
    new_rows = [e for e in entries if e["url"] not in have]
    if new_rows:
        out = pd.concat([fresh, pd.DataFrame(new_rows, columns=registry.COLS)], ignore_index=True)
        registry.write(out, REGISTRY, backup=REGISTRY.with_suffix(".psv.bak"))
    pstate.upsert([{
        "playlist_id": str(r["_pid"]), "channel_id": cid, "title": r["playlist"],
        "region": r["region"], "url": r["open"], "stage": "picked", "n_videos": str(r["videos"]),
    } for _, r in picked_df.iterrows() if str(r["_pid"])])
    return entries


tab_pl, tab_loose = st.tabs([f"📃 Playlists ({len(playlists)})", "🎞 Loose videos"])

# ============================= 📃 Playlists (shows) ============================= #
with tab_pl:
    if not playlists:
        vids = (ch.get("video_count") or "").strip() or "?"
        st.info(f"📺 This channel has **no playlists** — its ~{vids} videos aren't grouped into "
                "shows. Grab them individually in the **🎞 Loose videos** tab.")
        st.link_button("▶ open channel on YouTube", sel_url)
    else:
        rows = []
        for p in playlists:
            url = playlist_tools.playlist_url(p)
            pid = str(p.get("id", "") or "")
            match = by_pid.get(pid)
            if match is None:
                match = by_name.get(playlist_tools.normalize_name(p.get("title", "")))
            n_vid = str(p.get("video_count", "") or "")
            if match is not None:
                stage = match["stage"]
                dl = f"{_int(match['n_downloaded'])}/{n_vid}" if n_vid else str(_int(match["n_downloaded"]))
                dia = f"{_int(match['n_diarized'])}/{_int(match['n_downloaded'])}"
                dates = _dates(match["upload_date_min"], match["upload_date_max"])
                rel = match["rel_path"]
            else:
                stage, dl, dia, dates, rel = "", "", "", "", ""
            rows.append({
                "pick": False, "playlist": p.get("title") or "(untitled)", "videos": n_vid,
                "status": _BADGE.get(stage, "— new —"), "downloaded": dl, "diarized": dia,
                "dates": dates, "open": url,
                "region": ch_region if ch_region in registry.REGION_OPTS else "Unknown",
                "_pid": pid, "_rel": rel,
            })
        view = pd.DataFrame(rows)

        st.caption("Each row shows its live on-disk **status** (legacy downloads included). Tick "
                   "shows, then **Download** new ones or **Diarize** downloaded ones. Already-done "
                   "work is skipped unless you tick a **Force** box.")
        edited = st.data_editor(
            view, hide_index=True, use_container_width=True, num_rows="fixed", key="add_editor",
            column_order=["pick", "playlist", "videos", "status", "downloaded", "diarized",
                          "dates", "open", "region"],
            disabled=["playlist", "videos", "status", "downloaded", "diarized", "dates", "open"],
            column_config={
                "pick": st.column_config.CheckboxColumn("pick", width="small"),
                "playlist": st.column_config.TextColumn("playlist (show)", width="large"),
                "videos": st.column_config.TextColumn("videos", width="small"),
                "status": st.column_config.TextColumn("status", width="small",
                                                      help="live on-disk state"),
                "downloaded": st.column_config.TextColumn("downloaded", width="small"),
                "diarized": st.column_config.TextColumn("diarized", width="small"),
                "dates": st.column_config.TextColumn("upload dates", width="small"),
                "open": st.column_config.LinkColumn("link", display_text="▶ open", width="small"),
                "region": st.column_config.SelectboxColumn("region", options=registry.REGION_OPTS,
                                                           width="small", required=True),
                "_pid": None, "_rel": None,
            },
        )
        picked = edited[edited["pick"]]
        n_pick = int(len(picked))

        cdl, cdia, cclean = st.columns(3)
        with cclean:
            st.caption("Deleted the files but it still shows as downloaded?")
            if st.button(f"🧹 Clean selected ({n_pick})", disabled=n_pick == 0 or jobs.busy(),
                         help="Delete the downloaded files AND erase them from history "
                              "(.harvest_seen.json + state), so Download re-fetches them cleanly. "
                              "Works even if you already deleted the folder by hand."):
                if not root.strip():
                    st.error("Set a **Corpus root** in the sidebar first.")
                    st.stop()
                n_files = n_ids = 0
                with st.spinner("Cleaning files + history…"):
                    for _, r in picked.iterrows():
                        rel, purl = r["_rel"], r["open"]
                        folder = Path(root) / rel if rel else None
                        if folder is not None and folder.exists():
                            res = acquire.reset_shows(root, [rel])
                            n_files += res["files"]
                            n_ids += res["ids_purged"]
                        else:   # folder already deleted by hand: forget via the live playlist
                            ids = ytmeta.playlist_video_ids(purl) if purl else []
                            res = acquire.forget_videos(root, ids=ids,
                                                        rel_prefixes=[rel] if rel else [])
                            n_ids += res["ids"]
                load_disk.clear()
                st.success(f"Cleaned **{n_files}** file(s) · forgot **{n_ids}** video id(s). "
                           "Hit **⬇️ Download** to re-fetch cleanly.")
                st.rerun()
        with cdl:
            force_dl = st.checkbox("Force re-download (re-fetch even if already downloaded)",
                                   key="add_force_dl")
            if st.button(f"⬇️ Download selected ({n_pick})", type="primary",
                         disabled=n_pick == 0 or jobs.busy()):
                if not root.strip():
                    st.error("Set a **Corpus root** in the sidebar first.")
                    st.stop()
                entries = _record_picks(picked)
                force_rels = ([r["_rel"] for _, r in picked.iterrows() if r["_rel"]]
                              if force_dl else [])

                def target(report):
                    if force_rels:
                        acquire.reset_shows(root, force_rels)
                    summaries = acquire.harvest_registry(entries, root, diarize_audio=False,
                                                         on_progress=report)
                    got = sum(s["downloaded"] for s in summaries)
                    skip = sum(s["dup_id"] + s["exists"] + s["dup_audio"] for s in summaries)
                    fail = sum(s["failed"] for s in summaries)
                    return (f"downloaded {got} new · skipped {skip} already-present"
                            + (f" · {fail} failed" if fail else ""))

                jobs.start("Download", target)
        with cdia:
            force_dia2 = st.checkbox("Force re-diarize (redo even if already diarized)",
                                     key="add_force_dia")
            if st.button(f"🗣 Diarize selected ({n_pick})", disabled=n_pick == 0 or jobs.busy()):
                _record_picks(picked)
                rels = [r["_rel"] for _, r in picked.iterrows() if r["_rel"]]
                run_diarize(root, rels, hf_token, force_dia2)

# ============================== 🎞 Loose videos ============================== #
with tab_loose:
    st.caption("Every video uploaded to this channel — **including ones not in any playlist**. "
               "Pick any (preview or open them first), then download the lot into ONE folder "
               f"named `{LOOSE_PREFIX}<channel>`, so they live together as a single resource.")
    lk = f"loose::{cache_key}"
    exc = st.checkbox("Exclude videos already inside one of this channel's playlists "
                      "(slower — scans the playlists)", key=f"loose_exc_{cache_key}")
    if st.button("🔎 Find videos", key=f"loose_find_{cache_key}"):
        with st.spinner("Listing channel videos via yt-dlp (metadata only)…"):
            found = ytmeta.list_channel_videos(cid, sel_url)
            if exc and playlists:
                in_pl: set[str] = set()
                for p in playlists:
                    in_pl |= set(ytmeta.playlist_video_ids(playlist_tools.playlist_url(p)))
                found = [v for v in found if v["id"] not in in_pl]
            seen: set[str] = set()
            if root.strip():
                try:
                    seen, _ = acquire.load_seen(root)
                except Exception:  # noqa: BLE001
                    seen = set()
        st.session_state[lk] = {"vids": found, "seen": sorted(seen)}

    blob = st.session_state.get(lk)
    if blob is None:
        st.info("Click **🔎 Find videos** to list this channel's uploads.")
    elif not blob["vids"]:
        st.warning("No videos found (or every upload is already inside a playlist).")
    else:
        vids = blob["vids"]
        seen = set(blob["seen"])
        default_region = ch_region if ch_region in registry.REGION_OPTS else "Unknown"
        lrows = [{
            "pick": False,
            "preview": False,
            "title": v["title"] or "(untitled)",
            "duration": _fmt_dur(v["duration"]),
            "uploaded": _fmt_date(v["upload_date"]),
            "status": "📥 downloaded" if v["id"] in seen else "— new —",
            "open": v["url"],
            "region": default_region,
            "_url": v["url"],
            "_id": v["id"],
        } for v in vids]
        ledited = st.data_editor(
            pd.DataFrame(lrows), hide_index=True, use_container_width=True, num_rows="fixed",
            key=f"loose_editor_{cache_key}",
            column_order=["pick", "title", "duration", "uploaded", "status", "open",
                          "preview", "region"],
            disabled=["title", "duration", "uploaded", "status", "open"],
            column_config={
                "pick": st.column_config.CheckboxColumn("pick", width="small",
                                                        help="select to download"),
                "title": st.column_config.TextColumn("video", width="large"),
                "duration": st.column_config.TextColumn("len", width="small"),
                "uploaded": st.column_config.TextColumn("uploaded", width="small"),
                "status": st.column_config.TextColumn("status", width="small"),
                "open": st.column_config.LinkColumn("link", display_text="▶ open", width="small"),
                "preview": st.column_config.CheckboxColumn(
                    "▶ play", width="small", help="tick to watch it inside the app, below"),
                "region": st.column_config.SelectboxColumn(
                    "region", options=registry.REGION_OPTS, width="small", required=True),
                "_url": None, "_id": None,
            },
        )
        # tick "▶ play" on any row(s) to watch them right here, in-app
        for _, r in ledited[ledited["preview"]].iterrows():
            st.caption(str(r["title"])[:90])
            st.video(r["_url"])

        lpicked = ledited[ledited["pick"]]
        n_lpick = int(len(lpicked))
        channel_name = (ch.get("source") or sel_url)[:80]
        show_name = f"{LOOSE_PREFIX}{channel_name}"
        st.caption(f"Selected videos download into **`{show_name}`** under each row's **region** "
                   "(default = the channel's) — diarize / cluster it later in **My corpus**.")
        dlc, clc = st.columns(2)
        with dlc:
            if st.button(f"⬇️ Download selected ({n_lpick}) into one folder", type="primary",
                         disabled=n_lpick == 0 or jobs.busy(), key=f"loose_dl_{cache_key}"):
                if not root.strip():
                    st.error("Set a **Corpus root** in the sidebar first.")
                    st.stop()
                entries = [{"url": r["_url"], "region": r["region"], "source": channel_name,
                            "show": show_name, "lang_flag": ch.get("lang_flag") or "arabic"}
                           for _, r in lpicked.iterrows()]

                def target(report):
                    summaries = acquire.harvest_registry(entries, root, diarize_audio=False,
                                                         on_progress=report)
                    got = sum(s["downloaded"] for s in summaries)
                    skip = sum(s["dup_id"] + s["exists"] + s["dup_audio"] for s in summaries)
                    fail = sum(s["failed"] for s in summaries)
                    return (f"{got} new into {show_name} · skipped {skip}"
                            + (f" · {fail} failed" if fail else ""))

                jobs.start("Download loose", target)
        with clc:
            if st.button(f"🧹 Clean selected ({n_lpick})", disabled=n_lpick == 0 or jobs.busy(),
                         key=f"loose_clean_{cache_key}",
                         help="Delete these videos' files AND forget them in history, so Download "
                              "re-fetches them cleanly. Works even if you deleted the files by hand."):
                if not root.strip():
                    st.error("Set a **Corpus root** in the sidebar first.")
                    st.stop()
                ids = [r["_id"] for _, r in lpicked.iterrows() if r.get("_id")]
                with st.spinner("Cleaning files + history…"):
                    res = acquire.clean_videos(root, ids)
                load_disk.clear()
                st.session_state[lk] = {"vids": vids, "seen": sorted(set(blob["seen"]) - set(ids))}
                st.success(f"Cleaned **{res['files']}** file(s) · forgot **{res['ids']}** id(s). "
                           "Hit **⬇️ Download** to re-fetch cleanly.")
                st.rerun()

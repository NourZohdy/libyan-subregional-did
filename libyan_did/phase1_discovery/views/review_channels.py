"""Review channels — the human gate BEFORE harvest.

Discovery writes candidate channels into ``data/candidates.psv``. Here you eyeball each one
*before* any audio is downloaded: open it on YouTube, confirm it is real Libyan dialect, set its
province (region), fix the language flag, and mark it **approved** (harvest it) or **rejected**
(ignore it). Edits auto-save back to the same PSV (a one-time ``.bak`` is kept) and are resumable.

    region   : Tripolitania / Cyrenaica / Fezzan, or Non-Libyan / Unknown.
    lang_flag: arabic | code-switch:<lang> | minority:<lang>  (south: Tebu/Tuareg/Amazigh)
    status   : pending -> approved (harvest it) | rejected (ignore it)

Approved channels then appear in **Add from channel**, where you pick their shows. If discovery
left some channel names blank, **🔄 Rescan channel names** fetches them from YouTube.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st

from libyan_did.shared import config
from libyan_did.shared import registry
from libyan_did.shared import ytmeta

REGISTRY = config.DISCOVERY_DIR / "candidates.psv"
REGION_OPTS = registry.REGION_OPTS
STATUS_OPTS = registry.STATUS_OPTS
COLS = registry.COLS

# Colour cue carried on the status value itself (the editable grid can't paint row backgrounds).
STATUS_EMOJI = {"pending": "⏳ pending", "approved": "✅ approved", "rejected": "❌ rejected"}
STATUS_PLAIN = {v: k for k, v in STATUS_EMOJI.items()}
STATUS_DISPLAY = [STATUS_EMOJI[s] for s in STATUS_OPTS]


def load_registry(path):
    return registry.read(path)[COLS]


def save_registry(df, path):
    registry.write(df, path, backup=path.with_suffix(".psv.bak"))


def rescan_channel_names(force: bool) -> int:
    """Fetch missing (or, if ``force``, all) channel display names from YouTube; write them back.

    Re-reads the PSV fresh so a concurrent discovery run is never clobbered. Returns how many
    names were filled in."""
    fresh = registry.read(REGISTRY)
    is_ch = fresh["url_type"].str.strip() == "channel"
    need = is_ch if force else (is_ch & (fresh["source"].str.strip() == ""))
    targets = list(fresh[need].index)
    if not targets:
        return 0

    def _one(idx):
        row = fresh.loc[idx]
        return idx, ytmeta.channel_title(str(row.get("channel_id", "")), str(row.get("url", "")))

    updated = 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        for idx, name in pool.map(_one, targets):
            if name:
                fresh.at[idx, "source"] = name
                updated += 1
    if updated:
        save_registry(fresh, REGISTRY)
    return updated


st.title("🎬 Review channels — approve Libyan sources before harvest")

if "df" not in st.session_state:
    st.session_state.df = load_registry(REGISTRY)
df = st.session_state.df

if df.empty:
    st.warning(f"No candidates found at {REGISTRY}. Run the discovery agent first.")
    st.stop()

# ---- sidebar: filters + name rescan; live metrics are filled at the END (after edits) ---- #
with st.sidebar:
    st.header("Channels")
    metrics_ph = st.container()      # progress metrics, filled live below
    st.divider()
    region_ph = st.container()       # approved-by-region, filled live below
    st.divider()
    status_filter = st.multiselect("Show status", STATUS_OPTS, default=STATUS_OPTS,
                                   format_func=lambda s: STATUS_EMOJI.get(s, s))
    region_filter = st.multiselect("Show region", REGION_OPTS, default=REGION_OPTS)
    st.divider()
    st.caption("Channel names missing? Fetch them from YouTube (metadata only).")
    force_names = st.checkbox("re-fetch all (not just blanks)", key="force_names")
    if st.button("🔄 Rescan channel names"):
        with st.spinner("Fetching channel names via yt-dlp…"):
            n = rescan_channel_names(force_names)
        st.session_state.pop("df", None)         # reload with the new names
        st.session_state.pop("view_key", None)   # rebuild the grid
        st.success(f"Filled in {n} channel name(s)." if n else "Nothing to update.")
        st.rerun()

st.caption("Mark each channel **✅ approved** (Libyan, has Libyan shows) or **❌ rejected** (not "
           "Libyan); fix **region** / **lang_flag** if wrong; jot a note. Edits save instantly — "
           "no page jump. Approved channels → pick their shows in **Add from channel**.")

mask = ((df["url_type"].str.strip() == "channel")
        & df["status"].isin(status_filter) & df["region"].isin(region_filter))

# CRUCIAL for no-jump: build the grid's data ONCE per filter and reuse the SAME object across
# reruns. If we rebuilt it from df every run, applying your 1st edit would change the data the
# grid sees and Streamlit would remount it (scroll jump) on your 2nd edit. The grid overlays your
# in-session edits on this stable base; the on-disk file + the counters below hold the true state.
editor_key = "editor::" + "|".join(sorted(status_filter)) + "::" + "|".join(sorted(region_filter))
if st.session_state.get("view_key") != editor_key:
    base = df[mask].copy()
    base["status"] = base["status"].map(lambda s: STATUS_EMOJI.get(s, s))   # ✅/⏳/❌ colour cue
    st.session_state["view_base"] = base
    st.session_state["view_key"] = editor_key
view = st.session_state["view_base"]

colcfg = {
    "url": st.column_config.LinkColumn("YouTube", display_text="▶ open", width="small"),
    "source": st.column_config.TextColumn("source", width="medium"),
    "region": st.column_config.SelectboxColumn("region", options=REGION_OPTS, width="small", required=True),
    "status": st.column_config.SelectboxColumn("status", options=STATUS_DISPLAY, width="small", required=True),
    "lang_flag": st.column_config.TextColumn("lang_flag", width="small"),
    "format_type": st.column_config.TextColumn("format", width="small"),
    "video_count": st.column_config.TextColumn("vids", width="small", help="videos on the channel"),
    "playlist_count": st.column_config.TextColumn("lists", width="small",
                                                  help="playlists (shows) on the channel"),
    "confidence": st.column_config.TextColumn("conf", width="small"),
    "evidence": st.column_config.TextColumn("evidence (why this region)", width="large"),
    "discovery_path": st.column_config.TextColumn("found via", width="medium"),
    "notes": st.column_config.TextColumn("discovery note", width="medium"),
    "review_note": st.column_config.TextColumn("✍️ your note", width="medium"),
    "channel_id": None,   # hidden
    "url_type": None,     # hidden
}
order = ["url", "source", "region", "status", "video_count", "playlist_count", "lang_flag",
         "format_type", "confidence", "evidence", "discovery_path", "review_note", "notes"]

edited = st.data_editor(
    view, column_config=colcfg, column_order=order, hide_index=True,
    use_container_width=True, num_rows="fixed", key=editor_key,
    disabled=["url", "source", "evidence", "discovery_path", "format_type", "video_count",
              "playlist_count", "confidence", "notes"],
)

# Persist edits WITHOUT a reload: map the emoji status back to plain, sync into the in-memory
# frame (so the sidebar counters update live) AND into the on-disk registry (re-read first, so a
# concurrent discovery run / the Add-from-channel page is never clobbered). No st.rerun() => no jump.
cols4 = ["region", "status", "lang_flag", "review_note"]
e_idx = edited.copy()
e_idx["status"] = e_idx["status"].map(lambda s: STATUS_PLAIN.get(s, s))
e_idx = e_idx.set_index("url")[cols4]
cur = st.session_state.df.set_index("url").reindex(e_idx.index)[cols4]
changed_urls = list(e_idx.index[(e_idx != cur).any(axis=1)])
if changed_urls:
    dfu = st.session_state.df
    fresh = registry.read(REGISTRY)
    for url in changed_urls:
        for col in cols4:
            val = e_idx.at[url, col]
            dfu.loc[dfu["url"] == url, col] = val
            fresh.loc[fresh["url"] == url, col] = val
    save_registry(fresh, REGISTRY)

# ---- now fill the sidebar metrics from the (updated) frame — live, no reload ---- #
ch = st.session_state.df[st.session_state.df["url_type"].str.strip() == "channel"]
n = len(ch)
appr = int((ch["status"] == "approved").sum())
rej = int((ch["status"] == "rejected").sum())
pend = n - appr - rej
with metrics_ph:
    st.metric("Total channels", n)
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ approved", appr)
    c2.metric("❌ rejected", rej)
    c3.metric("⏳ pending", pend)
    st.progress((appr + rej) / n if n else 0.0, text=f"{appr + rej}/{n} reviewed")
    st.caption("Approved channels → pick their shows in **Add from channel**.")
with region_ph:
    st.caption("Approved channels, by region")
    appr_df = ch[ch["status"] == "approved"]
    for reg in REGION_OPTS:
        st.write(f"- {reg}: **{int((appr_df['region'] == reg).sum())}**")

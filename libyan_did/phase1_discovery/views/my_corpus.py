"""My corpus (window 2) — everything already on disk, and the per-show work you can run on it.

Read live from the corpus tree (NOT a fragile name guess): each show's episode count, how many
are diarized, whether it has been clustered, and its upload-date range. Works for LEGACY
downloads too (matched by on-disk folder, no playlist_id needed).

Tick shows and run either step on them:
  • 🗣 **Diarize**  — find speaker turns (writes a sibling ``.rttm`` per episode). Needed before
    clustering. Older shows downloaded before diarization land here.
  • 🔗 **Cluster**  — group each diarized show's segments into speakers (embed → within-playlist
    AHC → clips). Only runs on shows that are 100% diarized. Already-clustered shows are skipped
    unless you tick **Force re-cluster**.

To download NEW shows from a channel, use **Add from channel**. To check a clustered show's
speakers, use **Validate clustering**.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from libyan_did.shared import acquire
from libyan_did.shared import config
from libyan_did.shared import harvest
from libyan_did.shared import pstate
from libyan_did.shared import resource
from libyan_did.phase1_discovery import jobs

DEFAULT_ROOT = config.LOCAL_AUDIO_ROOT or r"D:\Research\DID_Set\libyan"

_BADGE = {"picked": "⚪ picked", "downloaded": "📥 downloaded",
          "diarized": "🗣 diarized", "clustered": "🔗 clustered"}
_DIARIZED_STAGES = ("diarized", "clustered")   # 100% diarized -> clusterable


@st.cache_data(show_spinner=False)
def load_disk(root: str) -> pd.DataFrame:
    """On-disk pipeline state for every show (synced from the corpus tree). Cached; the
    Diarize / Cluster actions call ``load_disk.clear()`` so the next render re-scans."""
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


def run_cluster(root: str, rel_paths: list[str], force: bool) -> None:
    """Validate + gate to fully-diarized shows, then cluster them as a background job.

    Per-playlist ``resource.process_resource`` (embed → AHC → clips), one show at a time, with a
    per-show progress bar."""
    rel_paths = [r for r in rel_paths if r]
    if not rel_paths:
        st.warning("Pick at least one diarized show to cluster.")
        return
    try:
        by_rel = {r["rel_path"]: r for r in harvest.enumerate_resources(root)}
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the corpus tree at {root}: {type(exc).__name__}: {exc}")
        return
    picked = [by_rel[r] for r in rel_paths if r in by_rel]
    runnable = [res for res in picked
                if res["episodes"] and all(e.get("rttm_path") for e in res["episodes"])]
    not_ready = [res["rel_path"] for res in picked if res not in runnable]
    if not_ready:
        st.warning("Skipped (not 100% diarized — run **🗣 Diarize** first): "
                   + ", ".join(not_ready))
    if not runnable:
        st.info("Nothing clusterable in this selection.")
        return

    def target(report):
        done, failed, total = [], 0, len(runnable)
        for i, res in enumerate(runnable):
            report(i, total, f"clustering {res['rel_path']}")
            try:
                done.append(resource.process_resource(res, hf_token=None, force=force, verbose=True))
            except Exception:  # noqa: BLE001 — surface in the summary, keep going
                failed += 1
            report(i + 1, total, f"clustered {res['rel_path']}")
        actors = sum(_int(m.get("n_actors")) for m in done)
        pairs = sum(_int(m.get("n_merge_candidates")) for m in done)
        return (f"{len(done)} show(s) · {actors} actor(s) · {pairs} gray-zone pair(s) "
                "to check in Validate clustering" + (f" · {failed} failed" if failed else ""))

    jobs.start(f"Cluster{' (FORCE)' if force else ''}", target)


st.title("📂 My corpus — what you have, and diarizing / clustering it")

if _msg := st.session_state.pop("_deleted_msg", None):
    st.success(_msg)

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

on_disk = disk[disk["stage"].isin(["downloaded", "diarized", "clustered"])]

if on_disk.empty:
    st.info("Nothing downloaded under this root yet. Use **➕ Add from channel** to harvest "
            "shows, or point the sidebar **Corpus root** at your existing corpus.")
    st.stop()

eps = on_disk["n_downloaded"].map(_int).sum()
n_dia = int(on_disk["stage"].isin(_DIARIZED_STAGES).sum())
n_clu = int((on_disk["stage"] == "clustered").sum())
m = st.columns(4)
m[0].metric("Shows", len(on_disk))
m[1].metric("Episodes", int(eps))
m[2].metric("Diarized shows", n_dia)
m[3].metric("Clustered shows", n_clu)

regions = ["(all)"] + sorted(x for x in on_disk["region"].unique() if x)
rsel = st.selectbox("Region", regions, key="inv_region")
sub = on_disk if rsel == "(all)" else on_disk[on_disk["region"] == rsel]

rows = [{
    "pick": False,
    "show": r["title"] or r["rel_path"],
    "region": r["region"],
    "status": _BADGE.get(r["stage"], ""),
    "episodes": _int(r["n_downloaded"]),
    "diarized": f"{_int(r['n_diarized'])}/{_int(r['n_downloaded'])}",
    "actors": r["n_actors"] if r["stage"] == "clustered" else "",
    "dates": _dates(r["upload_date_min"], r["upload_date_max"]),
    "_rel": r["rel_path"],
    "_stage": r["stage"],
} for _, r in sub.iterrows()]
inv = pd.DataFrame(rows)

st.caption("Everything already on disk. Tick shows, then **🗣 Diarize** the ones still missing "
           "speaker turns, or **🔗 Cluster** the diarized ones to find their speakers.")
ed = st.data_editor(
    inv, hide_index=True, use_container_width=True, num_rows="fixed", key="inv_editor",
    column_order=["pick", "show", "region", "status", "episodes", "diarized", "actors", "dates"],
    disabled=["show", "region", "status", "episodes", "diarized", "actors", "dates"],
    column_config={
        "pick": st.column_config.CheckboxColumn("pick", width="small"),
        "show": st.column_config.TextColumn("show (playlist)", width="large"),
        "episodes": st.column_config.NumberColumn("episodes", width="small"),
        "diarized": st.column_config.TextColumn("diarized", width="small",
                                                help="episodes with a speaker-turn .rttm"),
        "actors": st.column_config.TextColumn("actors", width="small",
                                              help="speakers found (clustered shows)"),
        "dates": st.column_config.TextColumn("upload dates", width="small"),
        "_rel": None, "_stage": None,
    },
)
picks = ed[ed["pick"]]
rels = picks["_rel"].tolist()
cluster_rels = picks[picks["_stage"].isin(_DIARIZED_STAGES)]["_rel"].tolist()

del_items = [(r["_rel"], r["region"], r["_rel"].split("/")[-1], r["show"])
             for _, r in picks.iterrows() if r["_rel"]]


@st.dialog("Delete playlist folder(s)?")
def confirm_delete(items):
    st.warning(f"Permanently delete **{len(items)}** playlist folder(s)? This removes the **audio "
               "files**, the **clustering outputs**, and the **download history** — it **cannot be "
               "undone**.")
    for rel, region, playlist, title in items:
        st.write(f"- **{title}**  ·  `{rel}`")
    c1, c2 = st.columns(2)
    if c1.button("🗑 Delete permanently", type="primary", use_container_width=True):
        for rel, region, playlist, title in items:
            acquire.delete_resource(root, rel, region=region, playlist=playlist)
        load_disk.clear()
        st.session_state["_deleted_msg"] = f"Deleted {len(items)} playlist folder(s)."
        st.rerun()
    if c2.button("Cancel", use_container_width=True):
        st.rerun()


cdia, cclu, cdel = st.columns(3)
with cdia:
    force_dia = st.checkbox("Force re-diarize (redo even if already diarized)", key="inv_force_dia")
    if st.button(f"🗣 Diarize selected ({len(rels)})", type="primary",
                 disabled=not rels or jobs.busy()):
        run_diarize(root, rels, hf_token, force_dia)
with cclu:
    force_clu = st.checkbox("Force re-cluster (redo even if already clustered)", key="inv_force_clu")
    if st.button(f"🔗 Cluster selected ({len(cluster_rels)})",
                 disabled=not cluster_rels or jobs.busy(),
                 help="Only fully-diarized shows can be clustered."):
        run_cluster(root, cluster_rels, force_clu)
with cdel:
    st.caption("Dead playlists? Remove them for good.")
    if st.button(f"🗑 Delete selected ({len(del_items)})", disabled=not del_items or jobs.busy(),
                 help="Permanently delete the picked playlist folders: audio + clustering outputs "
                      "+ download history. Asks for confirmation first."):
        confirm_delete(del_items)

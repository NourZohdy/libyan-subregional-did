"""Validate clustering (window 4) — check ONE playlist's speaker clustering by ear.

Per-playlist clustering (run from **My corpus**) is confident about most speakers but flags the
gray-zone pairs it was unsure about into ``merge_candidates.csv``. Those are the only uncertain
part of a playlist's clustering, so reviewing them IS validating the clustering: pick a clustered
playlist, listen to each gray-zone pair, and say **Same** / **Different** / **Unsure**.

Verdicts are written to ``outputs/gold/merge_decisions.csv`` (via ``shared.gold``) — the SAME
file the corpus-wide Speaker Validation app reads, so a within-playlist decision made here is
already counted downstream.

Scope: within ONE playlist only. Linking the same person ACROSS playlists/regions (the global
clustering) is a separate later step — it is NOT done here.

Run: ``python -m streamlit run libyan_did/phase1_discovery/review_app.py`` → this page.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from libyan_did.shared import config
from libyan_did.shared import gold
from libyan_did.shared import resource

_TRUE = {"true", "1", "1.0", "yes"}
_CLIPS_PER_ACTOR = 3


@st.cache_data(show_spinner=False)
def clip_bytes(path: str, start: float, end: float) -> bytes:
    """Slice one segment straight from the master episode wav — the corpus is timestamp-only
    (no physical clips on disk), so playback is cut on the fly. Cached so re-visiting a pair
    is instant and we don't re-read the same audio each rerun."""
    import io

    import soundfile as sf
    sr = sf.info(path).samplerate
    data, _ = sf.read(path, start=int(start * sr), stop=int(end * sr),
                      dtype="float32", always_2d=False)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV")
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def clustered_playlists() -> list[dict]:
    """Every clustered show (has a metadata.json), newest first by clustering time."""
    out = []
    for meta_path in config.CORPUS_DIR.glob("*/*/metadata.json"):
        region, playlist = meta_path.parent.parent.name, meta_path.parent.name
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        out.append({"region": region, "playlist": playlist, "meta": meta,
                    "n_pairs": _int(meta.get("n_merge_candidates")),
                    "processed_at": str(meta.get("processed_at", ""))})
    return sorted(out, key=lambda d: d["processed_at"], reverse=True)


def _int(x) -> int:
    try:
        return int(float(str(x).strip()))
    except Exception:  # noqa: BLE001
        return 0


def actor_segments(final: pd.DataFrame, aid: int, k: int = _CLIPS_PER_ACTOR) -> list[tuple]:
    """Up to k of the actor's longest segments as (master_audio_path, start, end)."""
    if not len(final):
        return []
    g = final[final["global_actor"] == aid].sort_values("duration", ascending=False).head(k)
    return [(str(r["master_audio_path"]), float(r["start_time"]), float(r["end_time"]))
            for _, r in g.iterrows()]


def play_actor(final: pd.DataFrame, aid: int, label: str) -> None:
    segs = actor_segments(final, aid)
    st.markdown(f"**actor_{aid}** · {label}")
    if not segs:
        st.caption("(no segments for this actor)")
        return
    for col, (path, s, e) in zip(st.columns(len(segs)), segs):
        try:
            col.audio(clip_bytes(path, s, e), format="audio/wav")
        except Exception as exc:  # noqa: BLE001 — a moved/missing episode mustn't kill the page
            col.caption(f"⚠️ {Path(path).name}: {type(exc).__name__}")


def pair_pager(n: int, key: str, sig: str) -> int:
    """One-pair-at-a-time navigation: ⬅ Previous · pair x of n · Next ➡. Resets to the
    first pair whenever the selected playlist (``sig``) changes."""
    if st.session_state.get(f"{key}_sig") != sig:
        st.session_state[f"{key}_sig"] = sig
        st.session_state[key] = 0
    pos = min(max(int(st.session_state.get(key, 0)), 0), n - 1)
    st.session_state[key] = pos
    c1, c2, c3 = st.columns([1, 2, 1], vertical_alignment="center")
    if c1.button("⬅ Previous", disabled=pos == 0, use_container_width=True, key=f"{key}_prev"):
        st.session_state[key] = pos - 1; st.rerun()
    c2.markdown(f"<div style='text-align:center'>pair <b>{pos + 1}</b> of {n}</div>",
                unsafe_allow_html=True)
    if c3.button("Next ➡", disabled=pos == n - 1, use_container_width=True, key=f"{key}_next"):
        st.session_state[key] = pos + 1; st.rerun()
    return pos


st.title("🔎 Validate clustering — check one playlist's speakers")

validator = st.sidebar.text_input("Validator name", value=st.session_state.get("validator", ""))
st.session_state.validator = validator

items = clustered_playlists()
if not items:
    st.warning("No clustered show yet. In **My corpus**, pick a diarized show and hit "
               "**🔗 Cluster selected**, then come back.")
    st.stop()

if st.sidebar.button("🔄 Rescan clustered shows"):
    clustered_playlists.clear(); st.rerun()

# Pick a region first, then a playlist within it.
regions = sorted({it["region"] for it in items})
csel = st.columns([1, 2])
region = csel[0].selectbox("Region", regions, key="vc_region")
in_region = [it for it in items if it["region"] == region]
pick_i = csel[1].selectbox(
    "Clustered playlist", range(len(in_region)), key=f"vc_pick_{region}",
    format_func=lambda i: f"{in_region[i]['playlist']}  ·  "
                          f"{in_region[i]['n_pairs']} gray-zone pair(s)")
it = in_region[pick_i]
region, playlist, meta = it["region"], it["playlist"], it["meta"]
paths = resource.resource_paths(region, playlist)

# Loaded once; playback is sliced from these timestamps (master_audio_path + start/end).
final = pd.read_csv(paths["final"]) if paths["final"].exists() else pd.DataFrame()
if len(final):
    final["global_actor"] = pd.to_numeric(final["global_actor"], errors="coerce").fillna(-1).astype(int)

m = st.columns(4)
m[0].metric("Actors", meta.get("n_actors", 0))
m[1].metric("Segments kept", f"{meta.get('n_segments_kept', 0)}/{meta.get('n_segments_total', 0)}")
m[2].metric("Kept minutes", f"{meta.get('kept_minutes', 0):.0f}")
m[3].metric("Gray-zone pairs", it["n_pairs"])
tc = meta.get("tier_counts") or {}
if tc:
    st.caption("Tiers: " + " · ".join(f"**{k}** {v}" for k, v in tc.items()))

decs = gold.load_merge_decisions()

# ===================== gray-zone pair review — ONE pair at a time ===================== #
st.divider()
st.subheader("🔀 Same speaker? — the pairs clustering was unsure about")
mc_path = paths["merge_candidates"]
mc = pd.read_csv(mc_path) if mc_path.exists() else pd.DataFrame()
if mc.empty:
    st.success("No gray-zone pairs for this playlist — its clustering had no ambiguous merges. ✅")
else:
    def _uid(aid) -> str:
        return f"{region}/{playlist}#{int(aid)}"

    keys = [(_uid(r["actor_a"]), _uid(r["actor_b"])) for _, r in mc.iterrows()]
    n_done = sum(1 for k in keys if k in decs)
    st.progress(n_done / len(mc), text=f"{n_done}/{len(mc)} pairs decided")

    pos = pair_pager(len(mc), "vc_pos", f"{region}/{playlist}")
    r = mc.iloc[pos]
    a_uid, b_uid = keys[pos]
    prev = decs.get((a_uid, b_uid))
    st.markdown(f"similarity **{float(r['similarity']):.3f}**  ·  "
                + (f"✅ decided: **{prev['decision']}**" if prev else "⏳ undecided"))

    ca, cb = st.columns(2)
    with ca:
        play_actor(final, int(r["actor_a"]), "A")
    with cb:
        play_actor(final, int(r["actor_b"]), "B")

    def record(decision: str) -> None:
        decs[(a_uid, b_uid)] = {
            "actor_a": a_uid, "actor_b": b_uid, "similarity": float(r["similarity"]),
            "kind": "within", "cross_region": False, "decision": decision,
            "resolved_region": "", "notes": "", "validator": validator,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        gold.save_merge_decisions(decs)
        st.session_state["vc_pos"] = min(pos + 1, len(mc) - 1)   # auto-advance to the next pair
        st.rerun()

    b1, b2, b3 = st.columns(3)
    if b1.button("✅ Same person", type="primary", use_container_width=True):
        record("same")
    if b2.button("❌ Different", use_container_width=True):
        record("different")
    if b3.button("🤷 Unsure", use_container_width=True):
        record("unsure")

# ===================== listen to ONE speaker on demand (not all at once) =============== #
st.divider()
with st.expander("🎧 Listen to one speaker", expanded=False):
    if not len(final):
        st.caption("No final_manifest.csv for this playlist.")
    else:
        excl = final["is_excluded"].astype(str).str.strip().str.lower().isin(_TRUE)
        kept = final[(~excl) & (final["tier"].isin(config.KEEP_TIERS)) & (final["global_actor"] >= 0)]
        if kept.empty:
            st.caption("No kept actors to preview (every segment was excluded by the content gates).")
        else:
            stats = (kept.groupby("global_actor")
                     .agg(n=("duration", "size"), mins=("duration", lambda s: s.sum() / 60.0))
                     .sort_values("n", ascending=False))
            opts = [int(a) for a in stats.index]
            sel = st.selectbox(
                "Speaker", opts, key="vc_listen",
                format_func=lambda a: f"actor_{a} · {int(stats.at[a, 'n'])} seg · "
                                      f"{stats.at[a, 'mins']:.1f} min")
            play_actor(final, int(sel),
                       f"{int(stats.at[sel, 'n'])} seg · {stats.at[sel, 'mins']:.1f} min")

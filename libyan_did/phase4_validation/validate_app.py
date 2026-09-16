"""Streamlit speaker-validation workbench v3 (single-validator protocol).

Three windows (sidebar), in recommended order:

1. 🔀 Merge pairs — same/different verdicts on gray-zone actor pairs (within-playlist
   merge_candidates.csv + cross-playlist link_candidates.csv). Done FIRST: one later
   speaker verdict then covers the whole merged group, and cross-region "same"
   verdicts resolve the dialect label before speaker review.
2. 🎧 Review     — every ABOVE-FLOOR speaker, ranked by triage risk. Fast-first:
   segments auto-play as one stream, ✅ Keep / ❌ Reject in one keystroke
   (→ / ←), full pagination (⏮ ⬅ go-to ➡ ⏭, ↑/↓), and a "Detailed verdict"
   expander for the nuanced cases. Goal = 100% human coverage (no sampling).

Dust floor (config.DUST_MIN_SEGMENTS / DUST_MIN_SECONDS, on the auto-linked
speaker_global aggregate): sub-floor speakers are `insufficient_evidence` by
rule — never queued, their merge pairs hidden, their segments dropped from the
export. The rule overrides human keeps; decisions stay on file.
3. 📊 Dashboard  — the complete progress report: KPIs, charts (decisions, progress
   over time, per-region outcomes, risk histogram, corpus-composition donut),
   machine-vs-human accuracy, find & fix for any recorded decision, and export.

Decision taxonomy v3 — one decision + one optional reason:

    verified        correct region label                    reason: —
    wrong_region    Libyan, but the OTHER region            reason: the correct region (required)
    non_libyan      another dialect / language              reason: which one
    msa             formal Arabic                           reason: pure_msa / code_switching
    bad_audio       unusable audio                          reason: music/noise/crosstalk/unintelligible/other
    mixed_cluster   several different voices                reason: —
    unsure          cannot decide                           reason: —

Persistence (resumable, outputs/gold/). v2 files (corrected_label/is_msa, old
category names) are migrated LOSSLESSLY on first load, with a timestamped backup:
    speaker_verifications.csv   one row per decided actor_uid
    merge_decisions.csv         one row per decided pair (+ resolved_region)

Run AFTER combine_corpus.py:
    python -m streamlit run app/validate_speakers.py
"""

from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from libyan_did.shared import config
from libyan_did.shared import gold
from libyan_did.shared.verified_manifest import build_verified_manifest  # noqa: E402

MANIFEST = config.CORPUS_MANIFEST
SCORES_CSV = config.METADATA_DIR / "actor_scores.csv"
LINK_CANDS_CSV = config.METADATA_DIR / "link_candidates.csv"
VERIF_CSV = config.GOLD_DIR / "speaker_verifications.csv"
MERGE_CSV = gold.MERGE_CSV
VERIFIED_MANIFEST = config.MANIFEST_DIR / "verified_manifest.csv"

DECISIONS = ["verified", "wrong_region", "non_libyan", "msa",
             "bad_audio", "mixed_cluster", "unsure"]
KEEP_DECISIONS = ("verified", "wrong_region")          # kept in the final corpus
LABEL_JUDGED = ("verified", "wrong_region", "non_libyan", "msa")  # judged the label
REGION_CHOICES = list(config.REGIONS)
NON_LIBYAN_CHOICES = [
    "Egyptian", "Tunisian", "Algerian", "Moroccan", "Mauritanian", "Sudanese",
    "Levantine (Syrian/Lebanese)", "Jordanian/Palestinian", "Iraqi",
    "Gulf (UAE/Kuwait/Qatar/Bahrain)", "Saudi", "Yemeni", "Omani",
    "Non-Arabic", "Mixed/Unsure",
]
BAD_AUDIO_REASONS = ["", "music", "noise", "crosstalk", "unintelligible", "other"]
MSA_REASONS = ["", "pure_msa", "code_switching"]
REJECT_AS_CHOICES = ["bad_audio", "non_libyan", "msa", "mixed_cluster"]
_TRUE = {"true", "1", "1.0", "yes"}

VERIF_COLUMNS = [
    "actor_uid", "speaker_global", "region", "playlist", "tier", "machine_label",
    "n_segments", "minutes", "risk_score", "decision", "reason",
    "notes", "mode", "validator", "timestamp",
]
MERGE_DEC_COLUMNS = gold.MERGE_DEC_COLUMNS

# v2 -> v3 decision migration (lossless: the dropped sub-categories become reasons)
LEGACY_DECISION_MAP = {
    "music_singing": ("bad_audio", "music"),
    "noise": ("bad_audio", "noise"),
    "overlap_crosstalk": ("bad_audio", "crosstalk"),
    "unintelligible": ("bad_audio", "unintelligible"),
    "rejected": ("bad_audio", ""),
}


def _s(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v)


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


# --------------------------------------------------------------------------- #
# Data (cached; mtime in the key so a rebuilt manifest invalidates the cache)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_manifest(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["is_excluded"] = df["is_excluded"].astype(str).str.strip().str.lower().isin(_TRUE)
    df = df[(~df["is_excluded"]) & (df["global_actor"] >= 0)].copy()
    if "speaker_global" not in df.columns:
        df["speaker_global"] = df["actor_uid"]
    return df


@st.cache_data(show_spinner=False)
def build_actors(path: str, mtime: float) -> pd.DataFrame:
    """One row per actor_uid with stats, triage scores and the dust floor.

    The floor (config.DUST_MIN_SEGMENTS / DUST_MIN_SECONDS) is measured on the
    AUTO-LINKED identity (speaker_global): an actor whose linked identity totals
    fewer segments AND seconds than the floor is `below_floor` and leaves the
    protocol entirely (queue, merge pairs, dashboard population, export keep).
    """
    df = load_manifest(path, mtime)
    rows = []
    for actor, g in df.groupby("actor_uid"):
        rows.append({
            "actor_uid": str(actor),
            "speaker_global": str(g["speaker_global"].iat[0]),
            "tier": g["tier"].mode().iat[0] if not g["tier"].mode().empty else "",
            "region": g["region"].mode().iat[0],
            "playlist": g["playlist"].mode().iat[0],
            "machine_label": g["region"].mode().iat[0],
            "n_segments": int(len(g)),
            "minutes": round(g["duration"].sum() / 60.0, 1),
            "seconds": float(g["duration"].sum()),
        })
    actors = pd.DataFrame(rows)
    ev = actors.groupby("speaker_global").agg(
        evidence_segments=("n_segments", "sum"), evidence_seconds=("seconds", "sum"))
    actors = actors.merge(ev, on="speaker_global", how="left")
    actors["below_floor"] = (
        (actors["evidence_segments"] < config.DUST_MIN_SEGMENTS)
        | (actors["evidence_seconds"] < config.DUST_MIN_SECONDS))
    if SCORES_CSV.exists():
        sc = pd.read_csv(SCORES_CSV)
        keep = [c for c in ["actor_uid", "cohesion", "mean_music_prob", "music_frac",
                            "lib_prob", "msa_prob", "top_dialect", "top_dialect_prob",
                            "arabic_prob", "gender_mix", "gender_majority",
                            "risk_score"] if c in sc.columns]
        actors = actors.merge(sc[keep], on="actor_uid", how="left")
    if "risk_score" not in actors.columns:
        actors["risk_score"] = np.nan
    actors["risk_score"] = actors["risk_score"].fillna(0.5)
    return actors.sort_values(["risk_score", "n_segments"],
                              ascending=[False, False]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def actor_samples(path: str, mtime: float, actor: str, k: int) -> list[dict]:
    """k clips spread across the actor's TIMELINE (start/middle/end slices),
    preferring clean + long within each slice (a pure cleanest+longest sampler
    preferentially plays theme songs)."""
    g = load_manifest(path, mtime)
    g = g[g["actor_uid"].astype(str) == str(actor)].copy()
    g = g.sort_values(["file_id", "start_time"]).reset_index(drop=True)
    if len(g) == 0:
        return []
    g["_clean"] = (g["quality"] == "clean").astype(int) if "quality" in g.columns else 1
    k = min(k, len(g))
    picked = []
    n = len(g)
    for t in range(k):
        lo, hi = int(t * n / k), max(int((t + 1) * n / k), int(t * n / k) + 1)
        tert = g.iloc[lo:hi]
        tert = tert[~tert.index.isin([p.name for p in picked])]
        if len(tert) == 0:
            continue
        best = tert.sort_values(["_clean", "duration"], ascending=[False, False]).iloc[0]
        picked.append(best)
    return [{
        "path": r["master_audio_path"], "start": float(r["start_time"]),
        "end": float(r["end_time"]), "file_id": str(r["file_id"]),
        "dur": float(r["duration"]), "quality": _s(r.get("quality", "")),
    } for r in picked]


@st.cache_data(show_spinner=False)
def clip_bytes(path: str, start: float, end: float) -> bytes:
    import soundfile as sf
    sr = sf.info(path).samplerate
    data, _ = sf.read(path, start=int(start * sr), stop=int(end * sr),
                      dtype="float32", always_2d=False)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV")
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def concat_clip_bytes(path: str, mtime: float, actor: str, k: int) -> bytes:
    """All k sample clips of an actor joined into ONE wav (0.35 s gaps) so the
    review window can play them as a single stream."""
    import soundfile as sf
    samples = actor_samples(path, mtime, actor, k)
    if not samples:
        return b""
    parts, sr0 = [], None
    for s in samples:
        sr = sf.info(s["path"]).samplerate
        data, _ = sf.read(s["path"], start=int(s["start"] * sr),
                          stop=int(s["end"] * sr), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr0 is None:
            sr0 = sr
        if sr != sr0:  # corpus contract is uniform 16 kHz; skip odd ones defensively
            continue
        parts.append(data)
        parts.append(np.zeros(int(0.35 * sr0), dtype=np.float32))
    if not parts:
        return b""
    buf = io.BytesIO()
    sf.write(buf, np.concatenate(parts), sr0, format="WAV")
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def load_merge_candidates(_mtime: float) -> pd.DataFrame:
    """Within-playlist merge candidates (per-resource CSVs) + cross-playlist link
    candidates, normalized to actor_uid pairs."""
    rows = []
    for f in sorted(config.CORPUS_DIR.glob("*/*/merge_candidates.csv")):
        region, playlist = f.parent.parent.name, f.parent.name
        try:
            c = pd.read_csv(f)
        except Exception:  # noqa: BLE001
            continue
        for _, r in c.iterrows():
            rows.append({
                "actor_a": f"{region}/{playlist}#{int(r['actor_a'])}",
                "actor_b": f"{region}/{playlist}#{int(r['actor_b'])}",
                "similarity": float(r["similarity"]), "kind": "within",
                "cross_region": False,
            })
    if LINK_CANDS_CSV.exists():
        c = pd.read_csv(LINK_CANDS_CSV)
        for _, r in c.iterrows():
            rows.append({
                "actor_a": str(r["actor_a"]), "actor_b": str(r["actor_b"]),
                "similarity": float(r["similarity"]), "kind": "cross",
                "cross_region": bool(r.get("cross_region", False)),
            })
    out = pd.DataFrame(rows, columns=["actor_a", "actor_b", "similarity",
                                      "kind", "cross_region"])
    return out.sort_values("similarity", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Persistence + schema migration
# --------------------------------------------------------------------------- #
def _migrate_v2(v: pd.DataFrame) -> pd.DataFrame:
    """v2 schema (corrected_label / is_msa / 11 categories) -> v3, losslessly."""
    rows = []
    for _, r in v.iterrows():
        dec = _s(r.get("decision"))
        corrected = _s(r.get("corrected_label"))
        notes = _s(r.get("notes"))
        reason = ""
        if dec in LEGACY_DECISION_MAP:
            dec, reason = LEGACY_DECISION_MAP[dec]
        elif dec == "wrong_dialect":
            dec = "wrong_region"
            reason = corrected if corrected in REGION_CHOICES else ""
            if corrected and not reason:
                notes = (notes + f" [old label: {corrected}]").strip()
        elif dec == "non_libyan":
            reason = corrected if corrected in NON_LIBYAN_CHOICES else ""
            if corrected and not reason:
                notes = (notes + f" [old label: {corrected}]").strip()
        if _s(r.get("is_msa")).lower() in _TRUE and dec != "msa":
            notes = (notes + " [uses MSA]").strip()
        d = r.to_dict()
        d["decision"], d["reason"], d["notes"] = dec, reason, notes
        rows.append(d)
    return pd.DataFrame(rows).reindex(columns=VERIF_COLUMNS)


def load_verifs() -> dict:
    if not VERIF_CSV.exists():
        return {}
    v = pd.read_csv(VERIF_CSV)
    if "actor_uid" not in v.columns:
        # Pre-rebuild schema (global_actor IDs): those IDs no longer exist.
        VERIF_CSV.rename(VERIF_CSV.with_name("speaker_verifications_legacy.csv"))
        return {}
    if "reason" not in v.columns:  # v2 -> v3, with a timestamped backup
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        backup = VERIF_CSV.with_name(f"speaker_verifications_v2_backup_{stamp}.csv")
        v.to_csv(backup, index=False)
        v = _migrate_v2(v)
        v.to_csv(VERIF_CSV, index=False)
    return {str(r["actor_uid"]): r.to_dict() for _, r in v.iterrows()}


def save_verifs(verifs: dict) -> None:
    VERIF_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(verifs.values())).reindex(columns=VERIF_COLUMNS).to_csv(
        VERIF_CSV, index=False)


load_merge_decisions = gold.load_merge_decisions
save_merge_decisions = gold.save_merge_decisions


def export_verified_manifest() -> Path:
    """Propagate speaker decisions + merge decisions onto every segment row.

    Thin wrapper over ``src.verified_manifest.build_verified_manifest`` so the
    Export button and ``scripts/freeze_verified_manifest.py`` share one source of
    truth (no logic drift between the app and the CLI freeze).
    """
    df = pd.read_csv(MANIFEST)
    out = build_verified_manifest(
        df, load_verifs(), load_merge_decisions(),
        regions=config.REGIONS, keep_decisions=KEEP_DECISIONS,
        min_segments=config.DUST_MIN_SEGMENTS, min_seconds=config.DUST_MIN_SECONDS)
    out.to_csv(VERIFIED_MANIFEST, index=False)
    return VERIFIED_MANIFEST


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def _traffic(v: float, good_high: bool, t_good: float, t_warn: float) -> str:
    if good_high:
        return "🟢" if v >= t_good else ("🟠" if v >= t_warn else "🔴")
    return "🟢" if v <= t_good else ("🟠" if v <= t_warn else "🔴")


def score_chips(row) -> None:
    """Big, colour-coded metric row — the triage scores at a glance."""
    specs = [
        ("risk_score", "risk", False, 0.5, 0.7),
        ("lib_prob", "P(LIB)", True, 0.5, 0.25),
        ("arabic_prob", "P(Arabic)", True, 0.9, 0.6),
        ("msa_prob", "P(MSA)", False, 0.3, 0.6),
        ("music_frac", "music", False, 0.2, 0.5),
        ("cohesion", "cohesion", True, 0.75, 0.6),
    ]
    gm = row.get("gender_mix")
    if gm is not None and pd.notna(gm):  # only once build_gender_flags.py has run
        specs.append(("gender_mix", "⚧ mix", False, 0.10, 0.25))
    cols = st.columns(len(specs) + 1)
    for col, (c, label, good_high, t_good, t_warn) in zip(cols, specs):
        v = row.get(c)
        if v is None or pd.isna(v):
            col.metric(label, "—")
        else:
            col.metric(f"{_traffic(float(v), good_high, t_good, t_warn)} {label}",
                       f"{float(v):.2f}")
    td = _s(row.get("top_dialect"))
    tdp = row.get("top_dialect_prob")
    cols[-1].metric("top dialect",
                    f"{td} {float(tdp):.2f}" if td and pd.notna(tdp) else (td or "—"))


def play_actor(uid: str, k: int, keyp: str = "", per_row: int = 4) -> None:
    samples = actor_samples(str(MANIFEST), _mtime(MANIFEST), str(uid), k)
    if not samples:
        st.warning("no playable segments")
        return
    for i in range(0, len(samples), per_row):
        chunk = samples[i:i + per_row]
        for col, s in zip(st.columns(per_row), chunk):
            with col:
                st.caption(f"ep {s['file_id'][:8]} · {s['dur']:.1f}s · {s['quality']}")
                st.audio(clip_bytes(s["path"], s["start"], s["end"]), format="audio/wav")


def pager(n: int, key: str, sig: str, noun: str) -> int:
    """Full pagination bar: ⏮ ⬅ [go to # / n] ➡ ⏭. 1-based display, 0-based
    return. One source of truth (st.session_state[key]); resets to the first
    item whenever the filter signature changes."""
    gkey = f"{key}_goto"
    if st.session_state.get(f"{key}_sig") != sig:
        st.session_state[f"{key}_sig"] = sig
        st.session_state[key] = 0
    pos = min(max(int(st.session_state.get(key, 0)), 0), n - 1)
    st.session_state[key] = pos
    st.session_state[gkey] = pos + 1  # keep the go-to box in lockstep

    def _go(target: int):
        st.session_state[key] = target % n
        st.session_state[gkey] = (target % n) + 1

    def _typed():
        st.session_state[key] = int(st.session_state[gkey]) - 1

    c1, c2, c3, c4, c5 = st.columns([1, 1, 2, 1, 1], vertical_alignment="bottom")
    c1.button("⏮ First", key=f"{key}_first", width="stretch",
              disabled=pos == 0, on_click=_go, args=(0,))
    c2.button("⬅ Back", key=f"{key}_back", width="stretch",
              on_click=_go, args=(pos - 1,))
    c3.number_input(f"go to {noun} # (of {n})", 1, n, key=gkey, on_change=_typed)
    c4.button("Next ➡", key=f"{key}_next", width="stretch",
              on_click=_go, args=(pos + 1,))
    c5.button("⏭ Last", key=f"{key}_last", width="stretch",
              disabled=pos == n - 1, on_click=_go, args=(n - 1,))
    return pos


def advance_after_decision(key: str, view: str, n: int) -> None:
    """In 'Not done' the decided item leaves the list and the next slides into
    this slot; in 'Done'/'All' the list keeps it, so step forward (wrap)."""
    if view != "Not done":
        st.session_state[key] = (int(st.session_state.get(key, 0)) + 1) % n


def reason_widget(dec: str, prev_reason: str, keyp: str) -> str:
    """The decision-dependent optional reason sub-field."""
    if dec == "wrong_region":
        return st.radio("Correct Libyan region (required)", REGION_CHOICES,
                        horizontal=True, key=f"rs_{keyp}",
                        index=REGION_CHOICES.index(prev_reason)
                        if prev_reason in REGION_CHOICES else 0)
    if dec == "non_libyan":
        opts = [""] + NON_LIBYAN_CHOICES
        return st.selectbox("Which dialect / language?", opts, key=f"rs_{keyp}",
                            index=opts.index(prev_reason) if prev_reason in opts else 0)
    if dec == "bad_audio":
        return st.selectbox("Why is the audio bad? (optional)", BAD_AUDIO_REASONS,
                            key=f"rs_{keyp}",
                            index=BAD_AUDIO_REASONS.index(prev_reason)
                            if prev_reason in BAD_AUDIO_REASONS else 0)
    if dec == "msa":
        return st.selectbox("MSA kind (optional)", MSA_REASONS, key=f"rs_{keyp}",
                            index=MSA_REASONS.index(prev_reason)
                            if prev_reason in MSA_REASONS else 0)
    return ""


def make_verif_record(uid: str, row, decision: str, *, reason: str = "",
                      notes: str = "", mode: str = "", validator: str = "") -> dict:
    return {
        "actor_uid": uid, "speaker_global": row.get("speaker_global", uid),
        "region": row["region"], "playlist": row["playlist"], "tier": row.get("tier", ""),
        "machine_label": row["machine_label"], "n_segments": int(row["n_segments"]),
        "minutes": row["minutes"], "risk_score": float(row.get("risk_score", np.nan)),
        "decision": decision, "reason": reason, "notes": notes,
        "mode": mode, "validator": validator,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def detailed_verdict_form(uid: str, row, prev: dict, mode: str, validator: str,
                          on_save=None) -> None:
    """The full decision radio + reason + notes (the nuanced 10%)."""
    keyp = f"{mode}_{uid}"
    dec = st.radio(
        "Decision", DECISIONS, horizontal=True, key=f"dec_{keyp}",
        index=DECISIONS.index(prev["decision"]) if prev.get("decision") in DECISIONS else 0,
        help="verified = correct region · wrong_region = Libyan, other region · "
             "non_libyan = other dialect/language · msa = formal Arabic · "
             "bad_audio = unusable audio · mixed_cluster = several voices")
    reason = reason_widget(dec, _s(prev.get("reason")), keyp)
    notes = st.text_input("Notes", value=_s(prev.get("notes")), key=f"notes_{keyp}")
    if st.button("💾 Save verdict", type="primary", key=f"sv_{keyp}"):
        st.session_state.verifs[uid] = make_verif_record(
            uid, row, dec, reason=reason, notes=notes, mode=mode, validator=validator)
        save_verifs(st.session_state.verifs)
        if on_save:
            on_save()
        st.rerun()


def keyboard_shortcuts(mapping: dict[str, str]) -> None:
    """Bind keyboard keys to clicking the Streamlit button whose label contains
    the given text.

    Streamlit destroys this component's iframe on every rerun, so the handler
    must NOT reach back through ``window.parent`` at keypress time (that points
    at a dead iframe after the first rerun). Instead the parent document and the
    key map are captured directly in the closure, and each render swaps the old
    listener for a fresh one (streamlit-shortcuts pattern).
    """
    rules = ",".join(f'"{k}":"{v}"' for k, v in mapping.items())
    nonce = "_".join(sorted(mapping.values()))  # html change forces re-mount per page
    components.html(f"""
    <script>
    (function() {{  // nonce: {nonce}
      const pwin = window.parent;
      const pdoc = pwin.document;
      const rules = {{{rules}}};
      const handler = (e) => {{
        const t = e.target || {{}};
        const tag = (t.tagName || '').toLowerCase();
        if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
        if (t.isContentEditable) return;
        const role = t.getAttribute ? (t.getAttribute('role') || '') : '';
        if (['radio', 'slider', 'listbox', 'combobox', 'option'].includes(role)) return;
        const want = rules[e.key];
        if (!want) return;
        const btn = Array.from(pdoc.querySelectorAll('button'))
                         .find(b => (b.innerText || '').includes(want));
        if (btn) {{ e.preventDefault(); e.stopPropagation(); btn.click(); }}
      }};
      if (pwin._kbHandler) pdoc.removeEventListener('keydown', pwin._kbHandler, true);
      pwin._kbHandler = handler;
      pdoc.addEventListener('keydown', handler, true);
    }})();
    </script>
    """, height=0)


# --------------------------------------------------------------------------- #
# App init + sidebar
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Speaker Validation", layout="wide")

if not MANIFEST.exists():
    st.error(f"No manifest at {MANIFEST}. Run run_resources.py then combine_corpus.py.")
    st.stop()

all_actors = build_actors(str(MANIFEST), _mtime(MANIFEST))
dust = all_actors[all_actors["below_floor"]]
actors = all_actors[~all_actors["below_floor"]].reset_index(drop=True)
if "verifs" not in st.session_state:
    st.session_state.verifs = load_verifs()
if "merge_decs" not in st.session_state:
    st.session_state.merge_decs = load_merge_decisions()
verifs = st.session_state.verifs
merge_decs = st.session_state.merge_decs

st.sidebar.title("🎧 Speaker Validation")
validator = st.sidebar.text_input("Validator name",
                                  value=st.session_state.get("validator", ""))
st.session_state.validator = validator
window = st.sidebar.radio("Window", ["🔀 Merge pairs", "🎧 Review", "📊 Dashboard"])
st.sidebar.caption("Recommended order: Merge pairs → Review → Dashboard/export.")

regions = ["(all)"] + sorted(actors["region"].unique().tolist())
region_pick = st.sidebar.selectbox("Region filter", regions, index=0)
risk_min = st.sidebar.slider("Risk ≥ (filter)", 0.0, 1.0, 0.0, 0.05,
                             help="Show only speakers at or above this risk. The "
                                  "list is always sorted riskiest-first, so 0.0 "
                                  "simply reviews everyone in risk order.")
min_segs = st.sidebar.number_input("Min segments per speaker", 0, 1000, 0,
                                   help="Hide tiny 'dust' actors; review the "
                                        "substantial speakers first.")
k_samples = st.sidebar.slider("Clips per speaker", 1, 10, 3,
                              help="Sample segments per speaker, spread across "
                                   "the speaker's timeline.")

flt = actors if region_pick == "(all)" else actors[actors["region"] == region_pick]
if min_segs:
    flt = flt[flt["n_segments"] >= int(min_segs)]
queue_df = flt[flt["risk_score"] >= risk_min].reset_index(drop=True)
n_done = sum(1 for u in queue_df["actor_uid"] if str(u) in verifs)
st.sidebar.progress(n_done / len(queue_df) if len(queue_df) else 1.0,
                    text=f"speakers (filtered): {n_done}/{len(queue_df)} done")
if len(dust):
    st.sidebar.caption(
        f"🧹 {len(dust)} dust speakers (< {config.DUST_MIN_SEGMENTS} segs / "
        f"< {config.DUST_MIN_SECONDS:.0f} s linked evidence) excluded by rule — "
        f"{dust['minutes'].sum() / 60:.1f} h, never queued.")

mtime_cands = _mtime(LINK_CANDS_CSV)
all_cands = load_merge_candidates(mtime_cands)
dust_uids = set(dust["actor_uid"].astype(str))
n_pairs_dusted = int((all_cands["actor_a"].isin(dust_uids)
                      | all_cands["actor_b"].isin(dust_uids)).sum())
all_cands = all_cands[~all_cands["actor_a"].isin(dust_uids)
                      & ~all_cands["actor_b"].isin(dust_uids)].reset_index(drop=True)
n_pairs_left = sum(1 for _, r in all_cands.iterrows()
                   if (str(r["actor_a"]), str(r["actor_b"])) not in merge_decs)
if n_pairs_left and window == "🎧 Review":
    st.sidebar.warning(f"🔀 {n_pairs_left} merge pair(s) still undecided — "
                       "deciding those first avoids reviewing the same person twice.")


def undecided_pairs_for(uid: str) -> pd.DataFrame:
    m = all_cands[(all_cands["actor_a"] == uid) | (all_cands["actor_b"] == uid)]
    return m[[not ((str(r["actor_a"]), str(r["actor_b"])) in merge_decs)
              for _, r in m.iterrows()]] if len(m) else m


# =========================== 🔀 Merge pairs ================================ #
if window == "🔀 Merge pairs":
    cands = all_cands
    if region_pick != "(all)":
        cands = cands[cands["actor_a"].str.startswith(region_pick + "/")
                      | cands["actor_b"].str.startswith(region_pick + "/")]
    kind_pick = st.sidebar.selectbox("Pair kind", ["(all)", "within", "cross",
                                                   "cross-region only"])
    if kind_pick == "within":
        cands = cands[cands["kind"] == "within"]
    elif kind_pick == "cross":
        cands = cands[cands["kind"] == "cross"]
    elif kind_pick == "cross-region only":
        cands = cands[cands["cross_region"]]
    cands = cands.reset_index(drop=True)
    keys = [(str(r["actor_a"]), str(r["actor_b"])) for _, r in cands.iterrows()]
    pending = [i for i, k_ in enumerate(keys) if k_ not in merge_decs]
    done_idx = [i for i, k_ in enumerate(keys) if k_ in merge_decs]
    view = st.sidebar.radio("Show", ["Not done", "Done", "All"], horizontal=True)
    scope = ("all regions" if region_pick == "(all)" else region_pick) + \
            ("" if kind_pick == "(all)" else f" · {kind_pick}")
    st.sidebar.progress(len(done_idx) / len(cands) if len(cands) else 1.0,
                        text=f"pairs ({scope}): {len(done_idx)}/{len(cands)} done "
                             f"· {len(pending)} left")
    if n_pairs_dusted:
        st.sidebar.caption(f"🧹 {n_pairs_dusted} pair(s) hidden — one side is a "
                           "dust speaker the floor already excludes, so the "
                           "verdict cannot change the corpus.")
    order = {"Not done": pending, "Done": done_idx,
             "All": list(range(len(cands)))}[view]
    if not order:
        st.success({"Not done": "No merge pairs left in this filter. 🎉  "
                                "Next: 🎧 Review.",
                    "Done": "No pairs decided yet in this filter.",
                    "All": "No merge pairs in this filter."}[view])
        st.stop()

    sig = f"{region_pick}|{kind_pick}|{view}"
    pos = pager(len(order), "pair_pos", sig, "pair")
    pair = cands.iloc[order[pos]]
    a, b = str(pair["actor_a"]), str(pair["actor_b"])
    region_a, region_b = a.split("/")[0], b.split("/")[0]

    st.subheader(f"Same speaker?  ·  similarity {pair['similarity']:.3f}  ·  "
                 f"{pair['kind']}  ·  **{pos + 1}/{len(order)}** ({view.lower()})"
                 + ("  ·  ⚠️ CROSS-REGION" if pair["cross_region"] else ""))
    prev_pair = merge_decs.get((a, b))
    if prev_pair:
        rr = _s(prev_pair.get("resolved_region"))
        st.info(f"✅ already decided: **{prev_pair.get('decision')}**"
                + (f" → {rr}" if rr else "")
                + f" · {prev_pair.get('timestamp')} — a new verdict below "
                  "OVERWRITES it.")
    if pair["cross_region"]:
        st.warning("These actors carry DIFFERENT region labels. If they are the same "
                   "person, pick the correct region below before saving 'Same'.")
    ca, cb = st.columns(2)
    with ca:
        ra = actors[actors["actor_uid"] == a]
        st.markdown(f"**A: {a}**")
        if len(ra):
            st.caption(f"{int(ra['n_segments'].iat[0])} seg · {ra['minutes'].iat[0]} min "
                       f"· risk {ra['risk_score'].iat[0]:.2f}")
        play_actor(a, k=k_samples, keyp=f"pa_{a}", per_row=2)
    with cb:
        rb = actors[actors["actor_uid"] == b]
        st.markdown(f"**B: {b}**")
        if len(rb):
            st.caption(f"{int(rb['n_segments'].iat[0])} seg · {rb['minutes'].iat[0]} min "
                       f"· risk {rb['risk_score'].iat[0]:.2f}")
        play_actor(b, k=k_samples, keyp=f"pb_{b}", per_row=2)

    # Dialect of the (single) speaker if A and B are the same person.
    region_opts = list(dict.fromkeys(  # ordered unique: pair's regions first
        [region_a, region_b] + REGION_CHOICES)) + ["Non-Libyan", "Unsure"]
    prev_rr = _s((prev_pair or {}).get("resolved_region"))
    resolved = st.radio("If SAME person — correct dialect/region:",
                        region_opts, horizontal=True, key=f"rr_{a}_{b}",
                        index=region_opts.index(prev_rr) if prev_rr in region_opts else 0)
    notes = st.text_input("Notes", value=_s((prev_pair or {}).get("notes")),
                          key=f"pair_notes_{a}_{b}")

    def record_pair(decision: str):
        merge_decs[(a, b)] = {
            "actor_a": a, "actor_b": b, "similarity": float(pair["similarity"]),
            "kind": pair["kind"], "cross_region": bool(pair["cross_region"]),
            "decision": decision,
            "resolved_region": resolved if decision == "same" else "",
            "notes": notes, "validator": validator,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        save_merge_decisions(merge_decs)
        advance_after_decision("pair_pos", view, len(order))
        st.rerun()

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("✅ Same person", type="primary", width="stretch"):
            record_pair("same")
    with c2:
        if st.button("❌ Different", width="stretch"):
            record_pair("different")
    with c3:
        if st.button("🤷 Unsure (U)", width="stretch"):
            record_pair("unsure")
    keyboard_shortcuts({"ArrowRight": "✅ Same person", "ArrowLeft": "❌ Different",
                        "u": "🤷 Unsure", "ArrowUp": "⬅ Back", "ArrowDown": "Next ➡"})
    st.caption("⌨️ → = Same · ← = Different · U = Unsure · ↑ = Back · ↓ = Next")

# =========================== 🎧 Review ===================================== #
elif window == "🎧 Review":
    k_play = st.sidebar.slider("Segments to auto-play", 1, 10, 3,
                               help="Played back-to-back as one stream.")
    autoplay = st.sidebar.toggle("Auto-play on open", value=True)
    reject_as = st.sidebar.selectbox(
        "'Reject' records as", REJECT_AS_CHOICES,
        help="The decision saved by ← / ❌ Reject. Reasons/notes for individual "
             "speakers go through the Detailed verdict panel.")
    view = st.sidebar.radio("Show", ["Not done", "Done", "All"], horizontal=True)

    all_uids = [str(u) for u in queue_df["actor_uid"]]
    done_uids = [u for u in all_uids if u in verifs]
    not_done = [u for u in all_uids if u not in verifs]
    items = {"Not done": not_done, "Done": done_uids, "All": all_uids}[view]
    st.sidebar.caption(f"{len(not_done)} not done · {len(done_uids)} done "
                       f"· {len(all_uids)} total")
    if not items:
        st.success({"Not done": "Everything in this filter is reviewed. 🎉",
                    "Done": "Nothing decided yet in this filter.",
                    "All": "No speakers in this filter."}[view])
        st.stop()

    sig = f"{region_pick}|{risk_min}|{min_segs}|{view}"
    pos = pager(len(items), "rev_pos", sig, "speaker")
    uid = items[pos]
    row = queue_df[queue_df["actor_uid"] == uid].iloc[0]

    st.subheader(f"{row['region']} · {row['playlist']} · {uid}")
    st.caption(f"machine label **{row['machine_label']}** · tier {row['tier']} · "
               f"{row['n_segments']} seg · {row['minutes']} min · "
               f"**{pos + 1}/{len(items)}** ({view.lower()})")
    prev = verifs.get(uid, {})
    if prev:
        st.info(f"✅ already decided: **{prev.get('decision')}**"
                + (f" → {prev.get('reason')}" if _s(prev.get('reason')) else "")
                + f" · {prev.get('timestamp')} — a new verdict OVERWRITES it.")
    score_chips(row)
    gm = row.get("gender_mix")
    if gm is not None and pd.notna(gm) and float(gm) >= 0.25:
        st.warning(f"⚧ possible mixed-gender cluster: {float(gm):.0%} of segments "
                   "predict the minority gender — listen for two different "
                   "voices (decision: mixed_cluster).")
    partners = actors[(actors["speaker_global"] == row["speaker_global"])
                      & (actors["actor_uid"] != uid)]
    for _, p in partners.iterrows():
        pdec = verifs.get(str(p["actor_uid"]), {}).get("decision")
        st.info(f"linked actor **{p['actor_uid']}** ({p['n_segments']} seg)"
                + (f" — already decided: **{pdec}**" if pdec else " — not yet reviewed"))
    up = undecided_pairs_for(uid)
    if len(up):
        st.warning(f"🔀 this speaker is in {len(up)} undecided merge pair(s) — "
                   "consider deciding those in Merge pairs first.")

    audio = concat_clip_bytes(str(MANIFEST), _mtime(MANIFEST), uid, k_play)
    if audio:
        st.audio(audio, format="audio/wav", autoplay=autoplay)
        if autoplay:
            st.caption("▶ auto-playing — if silent, click once anywhere on the page "
                       "(browsers block autoplay before the first interaction).")
    else:
        st.warning("no playable segments")

    def record_quick(decision: str):
        st.session_state.verifs[uid] = make_verif_record(
            uid, row, decision, mode="review", validator=validator)
        save_verifs(st.session_state.verifs)
        advance_after_decision("rev_pos", view, len(items))
        st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        if st.button(f"❌ Reject ({reject_as})", width="stretch"):
            record_quick(reject_as)
    with c2:
        if st.button("✅ Keep (verified)", type="primary", width="stretch"):
            record_quick("verified")
    with st.expander("✏️ Detailed verdict (wrong region · dialect · reason · notes)"):
        detailed_verdict_form(
            uid, row, prev, "review", validator,
            on_save=lambda: advance_after_decision("rev_pos", view, len(items)))
        st.markdown("**Individual segments**")
        play_actor(uid, k=k_samples, keyp="rev", per_row=4)
    keyboard_shortcuts({"ArrowRight": "✅ Keep", "ArrowLeft": "❌ Reject",
                        "ArrowUp": "⬅ Back", "ArrowDown": "Next ➡"})
    st.markdown("### ⌨️  → = Keep  ·  ← = Reject  ·  ↑ = Back  ·  ↓ = Next")

# =========================== 📊 Dashboard ================================== #
else:
    v = pd.DataFrame(list(verifs.values()))
    n_dust_decided = 0
    if len(v):
        active_uids = set(actors["actor_uid"].astype(str))
        n_dust_decided = int((~v["actor_uid"].astype(str).isin(active_uids)).sum())
        v = v[v["actor_uid"].astype(str).isin(active_uids)].reset_index(drop=True)
        v = v.merge(actors[["actor_uid", "top_dialect"]], on="actor_uid", how="left")
    decided_uids = set(v["actor_uid"].astype(str)) if len(v) else set()
    A = actors.copy()
    A["status"] = np.where(A["actor_uid"].isin(decided_uids), "validated", "pending")
    A = A.merge(v[["actor_uid", "decision", "reason"]], on="actor_uid", how="left") \
        if len(v) else A.assign(decision=np.nan, reason="")

    st.title("📊 Validation dashboard")

    # ---- KPI row ----
    kept = v[v["decision"].isin(KEEP_DECISIONS)] if len(v) else v
    removed = v[~v["decision"].isin(KEEP_DECISIONS)] if len(v) else v
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Coverage", f"{len(v) / len(actors) * 100:.1f}%" if len(actors) else "—",
              f"{len(v)}/{len(actors)} speakers", delta_color="off",
              help="Of the above-floor population (dust speakers are excluded "
                   "by rule, not reviewed).")
    c2.metric("Hours validated-kept",
              f"{kept['minutes'].sum() / 60:.1f} h" if len(kept) else "0 h",
              f"{len(kept)} speakers", delta_color="off")
    c3.metric("Hours excluded by review",
              f"{removed['minutes'].sum() / 60:.1f} h" if len(removed) else "0 h",
              f"{len(removed)} speakers", delta_color="off")
    c4.metric("Excluded by floor",
              f"{dust['minutes'].sum() / 60:.1f} h",
              f"{len(dust)} dust speakers", delta_color="off",
              help=f"insufficient_evidence rule: linked identity has "
                   f"< {config.DUST_MIN_SEGMENTS} segments or "
                   f"< {config.DUST_MIN_SECONDS:.0f} s of speech.")
    c5.metric("Merge pairs done", len(all_cands) - n_pairs_left,
              f"of {len(all_cands)}", delta_color="off")
    pend_h = A.loc[A["status"] == "pending", "minutes"].sum() / 60
    c6.metric("Hours pending review", f"{pend_h:.1f} h",
              f"{int((A['status'] == 'pending').sum())} speakers", delta_color="off")
    if n_dust_decided:
        st.caption(f"ℹ️ {n_dust_decided} earlier decision(s) are on speakers now "
                   "below the dust floor (or gone from the manifest) — kept on "
                   "file in speaker_verifications.csv, excluded from all numbers "
                   "above by the insufficient_evidence rule.")

    if not len(v):
        st.info("No decisions recorded yet — the charts will appear as you review.")
        st.stop()

    # ---- Machine vs human ----
    st.markdown("## 🤖 Machine vs human")
    judged = v[v["decision"] != "unsure"]
    m1, m2, m3 = st.columns(3)
    lbl = judged[judged["decision"].isin(LABEL_JUDGED)]
    if len(lbl):
        prov_acc = (lbl["decision"] == "verified").mean()
        m1.metric("Provenance label accuracy", f"{prov_acc * 100:.1f}%",
                  f"{int((lbl['decision'] == 'verified').sum())}/{len(lbl)} "
                  "labels survived review", delta_color="off",
                  help="Share of reviewed speakers whose playlist-derived region "
                       "label was confirmed correct (bad_audio / mixed_cluster "
                       "verdicts are not label judgements and are excluded).")
    hub = judged.dropna(subset=["top_dialect"])
    hub = hub[hub["top_dialect"].astype(str) != ""]
    if len(hub):
        model_lib = hub["top_dialect"].astype(str) == "LIB"
        human_lib = hub["decision"].isin(KEEP_DECISIONS)
        agree = (model_lib == human_lib).mean()
        m2.metric("HuBERT dialect-model agreement", f"{agree * 100:.1f}%",
                  f"on {len(hub)} decided speakers", delta_color="off",
                  help="Model says Libyan = its top predicted dialect is LIB; "
                       "human says Libyan = verified or wrong_region.")
        conf = pd.crosstab(
            np.where(human_lib, "human: Libyan", "human: not Libyan"),
            np.where(model_lib, "model: LIB", "model: other"))
        m3.markdown("**Confusion matrix**")
        m3.dataframe(conf, width="stretch")
    rk = v[v["decision"].isin(KEEP_DECISIONS)]["risk_score"].mean()
    rr_ = v[~v["decision"].isin(KEEP_DECISIONS) & (v["decision"] != "unsure")]["risk_score"].mean()
    if pd.notna(rk) and pd.notna(rr_):
        st.caption(f"Risk ranking sanity check: speakers you rejected averaged risk "
                   f"**{rr_:.2f}** vs **{rk:.2f}** for speakers you kept — "
                   + ("the triage ranking is working. ✅" if rr_ > rk
                      else "⚠️ rejected speakers do NOT score riskier; "
                           "the triage ranking may need attention."))

    # ---- Corpus composition donut ----
    st.markdown("## 🥧 Final corpus composition")
    weight = st.radio("Weight by", ["hours", "speakers"], horizontal=True)
    comp = A[(A["status"] == "pending") | (A["decision"].isin(KEEP_DECISIONS))].copy()
    comp["final_region"] = np.where(
        (comp["decision"] == "wrong_region") & comp["reason"].isin(REGION_CHOICES),
        comp["reason"], comp["region"])
    comp["w"] = comp["minutes"] / 60.0 if weight == "hours" else 1.0
    comp["slice"] = comp["final_region"] + " · " + comp["status"]
    donut_df = comp.groupby(["final_region", "status", "slice"])["w"].sum().reset_index()
    base_colors = {"Tripolitania": ("#1f77b4", "#aec7e8"),
                   "Cyrenaica": ("#2ca02c", "#98df8a"),
                   "Fezzan": ("#d62728", "#ff9896")}
    domain, range_ = [], []
    for reg, (full, faded) in base_colors.items():
        domain += [f"{reg} · validated", f"{reg} · pending"]
        range_ += [full, faded]
    donut = (alt.Chart(donut_df).mark_arc(innerRadius=70).encode(
        theta=alt.Theta("w:Q"),
        color=alt.Color("slice:N", scale=alt.Scale(domain=domain, range=range_),
                        legend=alt.Legend(title=f"region · status ({weight})")),
        tooltip=["final_region", "status", alt.Tooltip("w:Q", format=".1f",
                                                       title=weight)])
        .properties(height=360))
    cd1, cd2 = st.columns([3, 2])
    cd1.altair_chart(donut, width="stretch")
    with cd2:
        tot = donut_df["w"].sum()
        for reg in base_colors:
            sub = donut_df[donut_df["final_region"] == reg]
            val = sub.loc[sub["status"] == "validated", "w"].sum()
            pen = sub.loc[sub["status"] == "pending", "w"].sum()
            unit = "h" if weight == "hours" else "spk"
            st.markdown(f"**{reg}** — {(val + pen) / tot * 100:.1f}% of corpus · "
                        f"{val:.0f} {unit} validated, {pen:.0f} {unit} pending")
        rem_h = removed["minutes"].sum() / 60
        st.caption(f"(not shown: {rem_h:.1f} h excluded by review across "
                   f"{len(removed)} speakers · {dust['minutes'].sum() / 60:.1f} h "
                   f"excluded by the dust floor across {len(dust)} speakers)")

    # ---- Charts ----
    st.markdown("## 📈 Progress & outcomes")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Decisions breakdown**")
        dd = v.copy()
        dd["reason"] = dd["reason"].fillna("").replace("", "(none)")
        st.altair_chart(
            alt.Chart(dd).mark_bar().encode(
                y=alt.Y("decision:N", sort="-x"),
                x=alt.X("count():Q", title="speakers"),
                color=alt.Color("reason:N", legend=alt.Legend(title="reason")),
                tooltip=["decision", "reason", "count()"]).properties(height=260),
            width="stretch")
    with g2:
        st.markdown("**Per-region outcomes**")
        st.altair_chart(
            alt.Chart(v).mark_bar().encode(
                x=alt.X("region:N", title=None),
                y=alt.Y("count():Q", title="speakers"),
                color=alt.Color("decision:N"),
                tooltip=["region", "decision", "count()"]).properties(height=260),
            width="stretch")
    g3, g4 = st.columns(2)
    with g3:
        st.markdown("**Validation progress over time**")
        tt = v.copy()
        tt["date"] = pd.to_datetime(tt["timestamp"]).dt.floor("h")
        prog = tt.groupby("date").size().reset_index(name="decisions")
        prog["cumulative"] = prog["decisions"].cumsum()
        bars = alt.Chart(prog).mark_bar(opacity=0.6).encode(
            x=alt.X("date:T", title=None), y=alt.Y("decisions:Q"))
        line = alt.Chart(prog).mark_line(color="#d62728", point=True).encode(
            x="date:T", y=alt.Y("cumulative:Q", title="cumulative"))
        st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent")
                        .properties(height=260), width="stretch")
    with g4:
        st.markdown("**Risk score: decided vs pending**")
        st.altair_chart(
            alt.Chart(A).mark_bar(opacity=0.65).encode(
                x=alt.X("risk_score:Q", bin=alt.Bin(maxbins=25), title="risk score"),
                y=alt.Y("count():Q", stack=None, title="speakers"),
                color=alt.Color("status:N",
                                scale=alt.Scale(domain=["validated", "pending"],
                                                range=["#2ca02c", "#bbbbbb"])),
                tooltip=["status", "count()"]).properties(height=260),
            width="stretch")

    # ---- Tables ----
    st.markdown("## 🧾 Tables")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Decision × region**")
        st.dataframe(pd.crosstab(v["decision"], v["region"], margins=True,
                                 margins_name="total"), width="stretch")
    with t2:
        m = pd.DataFrame(list(merge_decs.values()))
        st.markdown("**Merge decisions**")
        if len(m):
            st.dataframe(pd.crosstab(m["kind"], m["decision"], margins=True,
                                     margins_name="total"), width="stretch")
        else:
            st.caption("none yet")
    st.markdown("**Most recent decisions**")
    st.dataframe(v.sort_values("timestamp", ascending=False)
                 [["actor_uid", "region", "playlist", "decision", "reason",
                   "mode", "timestamp"]].head(20),
                 width="stretch", hide_index=True)

    # ---- Find & fix ----
    st.markdown("## 🔍 Find & fix")
    fc1, fc2, fc3 = st.columns(3)
    dec_pick = fc1.multiselect("Decision", sorted(v["decision"].dropna().unique()))
    reg_pick2 = fc2.multiselect("Region", sorted(v["region"].dropna().unique()))
    search = fc3.text_input("Search actor_uid / playlist / notes")
    show = v.copy()
    if dec_pick:
        show = show[show["decision"].isin(dec_pick)]
    if reg_pick2:
        show = show[show["region"].isin(reg_pick2)]
    if search:
        s = search.strip()
        mask = (show["actor_uid"].astype(str).str.contains(s, case=False)
                | show["playlist"].astype(str).str.contains(s, case=False)
                | show["notes"].astype(str).str.contains(s, case=False))
        show = show[mask]
    show = show.sort_values("timestamp", ascending=False)
    st.caption(f"{len(show)} of {len(v)} decisions match")
    st.dataframe(show[["actor_uid", "region", "playlist", "n_segments", "minutes",
                       "risk_score", "decision", "reason", "notes", "timestamp"]],
                 width="stretch", hide_index=True, height=280)
    pick = st.selectbox("Open a speaker decision to re-check / fix",
                        ["(none)"] + show["actor_uid"].astype(str).tolist())
    if pick != "(none)":
        arow = actors[actors["actor_uid"] == pick]
        prev = verifs.get(pick, {})
        if len(arow) == 0:
            st.error("Speaker not in the current manifest (rebuilt since this decision?).")
        else:
            rowx = arow.iloc[0]
            st.markdown(f"**{rowx['region']} · {rowx['playlist']} · {pick}** — recorded "
                        f"**{prev.get('decision')}** at {prev.get('timestamp')}")
            score_chips(rowx)
            play_actor(pick, k=k_samples, keyp="fix", per_row=4)
            detailed_verdict_form(pick, rowx, prev, "fix", validator)
            if st.button("🗑 Delete this decision (back to unreviewed)"):
                verifs.pop(pick, None)
                save_verifs(verifs)
                st.rerun()
    m = pd.DataFrame(list(merge_decs.values()))
    if len(m):
        pair_labels = [f"{r['actor_a']}  ↔  {r['actor_b']}  ({r['decision']})"
                       for _, r in m.iterrows()]
        ppick = st.selectbox("Delete a pair decision (it returns to the merge queue)",
                             ["(none)"] + pair_labels)
        if ppick != "(none)" and st.button("🗑 Delete pair decision"):
            i = pair_labels.index(ppick)
            r = m.iloc[i]
            merge_decs.pop((str(r["actor_a"]), str(r["actor_b"])), None)
            save_merge_decisions(merge_decs)
            st.rerun()

    # ---- Export ----
    st.markdown("## 📤 Export")
    e1, e2, e3 = st.columns(3)
    with e1:
        if st.button("💾 Export verified manifest", type="primary", width="stretch"):
            path = export_verified_manifest()
            out = pd.read_csv(path)
            st.success(f"Wrote {path} — verified_keep on "
                       f"{int(out['verified_keep'].sum())}/{len(out)} segments, "
                       f"{out['speaker_final'].nunique()} final speakers")
    with e2:
        st.download_button("⬇ speaker_verifications.csv",
                           v.reindex(columns=VERIF_COLUMNS + ["top_dialect"])
                           .to_csv(index=False).encode("utf-8-sig"),
                           "speaker_verifications.csv", "text/csv", width="stretch")
    with e3:
        m = pd.DataFrame(list(merge_decs.values()))
        st.download_button("⬇ merge_decisions.csv",
                           m.to_csv(index=False).encode("utf-8-sig"),
                           "merge_decisions.csv", "text/csv", width="stretch",
                           disabled=not len(m))

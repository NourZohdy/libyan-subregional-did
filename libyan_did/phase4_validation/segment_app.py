"""Streamlit interface for the segment-level purity audit (camera-ready P1-2).

One test segment at a time, blind to the released label. Four independent yes/no properties
plus a region judgement; `indeterminate` when the clip cannot be judged at all.
Writes only outputs/segment_audit/decisions_nour.csv, append-and-fsync.

    streamlit run libyan_did/phase4_validation/segment_app.py --server.port 8503
"""

import io

import streamlit as st

from libyan_did.phase4_validation import segment_audit

OUT_DIR = segment_audit.DEFAULT_OUT_DIR

st.set_page_config(page_title="Segment audit", layout="centered")


# Copied from blind_app.py rather than imported: that module runs its page at import time.
@st.cache_data(show_spinner=False)
def clip_bytes(path: str, start: float, end: float) -> bytes:
    import soundfile as sf

    with sf.SoundFile(path) as f:
        f.seek(int(start * f.samplerate))
        data = f.read(max(0, int((end - start) * f.samplerate)), dtype="float32")
        rate = f.samplerate
    buf = io.BytesIO()
    sf.write(buf, data, rate, format="WAV")
    return buf.getvalue()

queue = segment_audit.load_blind_queue(OUT_DIR)
done = segment_audit.completed_item_ids(OUT_DIR)
remaining = queue[~queue["item_id"].isin(done)]
if remaining.empty:
    st.success(f"Complete: {len(queue)} segments judged.")
    st.stop()

item = remaining.iloc[0]
key = item["item_id"]
st.markdown(f"**Segment {int(item['serve_order'])} of {len(queue)}** · `{key}` · "
            f"{item['end_time'] - item['start_time']:.1f} s")

try:
    audio = clip_bytes(item["master_audio_path"], float(item["start_time"]), float(item["end_time"]))
    st.audio(audio, format="audio/wav", autoplay=True)
except Exception as exc:  # unplayable: record it as indeterminate, do not stall the session
    st.error(f"Cannot play this clip ({type(exc).__name__}).")

with st.form(key=f"form_{key}", clear_on_submit=True):
    st.caption("Properties (tick all that apply)")
    is_msa = st.checkbox("MSA (Modern Standard Arabic)")
    is_cs = st.checkbox("Code-switching (Arabic ↔ another language)")
    is_mixed = st.checkbox("Mixed speakers (more than one voice)")
    region = st.radio("Region", segment_audit.REGION_JUDGEMENTS, horizontal=True)
    indeterminate = st.checkbox("Indeterminate — cannot judge this clip at all")
    note = st.text_input("Note (optional)")
    submitted = st.form_submit_button("Submit", use_container_width=True, type="primary")

if submitted:
    try:
        segment_audit.append_decision(OUT_DIR, key, is_msa, is_cs, is_mixed, region,
                                      indeterminate, note)
    except ValueError as exc:
        st.error(f"Submission refused: {exc}")
        st.stop()
    st.rerun()

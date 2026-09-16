"""Streamlit interface for the blind second-annotator review."""

import atexit
import io
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from libyan_did.phase4_validation import blind_study


OUT_DIR = blind_study.DEFAULT_OUT_DIR
PLACEHOLDER = "— select —"

st.set_page_config(page_title="Blind review", layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.4rem;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 0.75rem;
    }
    div[data-testid="stAudio"] {
        margin: 0;
    }
    [class*="st-key-judgement_0_"] button,
    [class*="st-key-judgement_1_"] button,
    [class*="st-key-judgement_2_"] button {
        min-height: 3.2rem;
        font-size: 1.05rem;
        font-weight: 600;
    }
    [class*="st-key-judgement_0_"] button {
        background-color: #1f6feb !important;
        color: #ffffff !important;
        border-color: #1f6feb !important;
    }
    [class*="st-key-judgement_0_"] button:hover {
        background-color: #1859bd !important;
        color: #ffffff !important;
        border-color: #1859bd !important;
    }
    [class*="st-key-judgement_1_"] button {
        background-color: #1a7f37 !important;
        color: #ffffff !important;
        border-color: #1a7f37 !important;
    }
    [class*="st-key-judgement_1_"] button:hover {
        background-color: #146c2e !important;
        color: #ffffff !important;
        border-color: #146c2e !important;
    }
    [class*="st-key-judgement_2_"] button {
        background-color: #a45b02 !important;
        color: #ffffff !important;
        border-color: #a45b02 !important;
    }
    [class*="st-key-judgement_2_"] button:hover {
        background-color: #824800 !important;
        color: #ffffff !important;
        border-color: #824800 !important;
    }
    [class*="st-key-judgement_3_"] button,
    [class*="st-key-judgement_4_"] button,
    [class*="st-key-judgement_5_"] button,
    [class*="st-key-judgement_6_"] button {
        background-color: #57606a !important;
        color: #ffffff !important;
        border-color: #57606a !important;
    }
    [class*="st-key-judgement_3_"] button:hover,
    [class*="st-key-judgement_4_"] button:hover,
    [class*="st-key-judgement_5_"] button:hover,
    [class*="st-key-judgement_6_"] button:hover {
        background-color: #424a53 !important;
        color: #ffffff !important;
        border-color: #424a53 !important;
    }
    [class*="st-key-judgement_7_"] button {
        background-color: #a40e26 !important;
        color: #ffffff !important;
        border-color: #a40e26 !important;
    }
    [class*="st-key-judgement_7_"] button:hover {
        background-color: #820b1e !important;
        color: #ffffff !important;
        border-color: #820b1e !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def lock_holder() -> tuple[str, str]:
    """(pid, iso timestamp) recorded in the lock file, or ("", "") if unreadable."""
    path = blind_study.lock_path(OUT_DIR)
    try:
        pid, _, stamp = path.read_text(encoding="ascii").partition("\n")
    except OSError:
        return "", ""
    return pid, stamp


def unlink_owned_lock() -> None:
    path = blind_study.lock_path(OUT_DIR)
    if path.exists() and lock_holder()[0] == str(os.getpid()):
        path.unlink()


def release_lock() -> None:
    if st.session_state.get("lock_owned"):
        unlink_owned_lock()
        st.session_state["lock_owned"] = False


def write_lock(descriptor: int) -> None:
    try:
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        os.write(descriptor, f"{os.getpid()}\n{stamp}".encode("ascii"))
    finally:
        os.close(descriptor)
    st.session_state["lock_owned"] = True
    atexit.register(unlink_owned_lock)


def acquire_lock() -> None:
    blind_study.ensure_out_dir(OUT_DIR)
    path = blind_study.lock_path(OUT_DIR)
    try:
        write_lock(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return
    except FileExistsError:
        pass
    # A lock outlives the browser session that took it, so a closed tab would
    # otherwise block resumption forever (SC-007). Offer an explicit takeover
    # instead of guessing whether the recorded PID is still alive: duplicate and
    # interleaved writes are already prevented by append_decision.
    pid, stamp = lock_holder()
    st.error(f"A session is already running (pid {pid or 'unknown'}, started {stamp or 'unknown'}).")
    st.write("If that session was closed without ending it, take it over to resume.")
    if st.button("Take over this session"):
        path.unlink(missing_ok=True)
        st.rerun()
    st.stop()


@st.cache_data(show_spinner=False)
def load_audio_manifest() -> pd.DataFrame:
    return pd.read_csv(
        blind_study.MANIFEST_PATH,
        usecols=[
            "speaker_final",
            "file_id",
            "start_time",
            "end_time",
            "duration",
            "quality",
            "master_audio_path",
        ],
    )


@st.cache_data(show_spinner=False)
def clip_bytes(path: str, start: float, end: float) -> bytes:
    import soundfile as sf

    with sf.SoundFile(path) as audio:
        audio.seek(int(start * audio.samplerate))
        data = audio.read(
            max(0, int((end - start) * audio.samplerate)), dtype="float32"
        )
        sample_rate = audio.samplerate
    buffer = io.BytesIO()
    sf.write(buffer, data, sample_rate, format="WAV")
    return buffer.getvalue()


try:
    roster = blind_study.load_precommit(OUT_DIR)["annotator_roster"]
except (FileNotFoundError, KeyError):
    st.error("Run sample first.")
    st.stop()

if "annotator" not in st.session_state:
    choice = st.selectbox("Annotator", [PLACEHOLDER, *roster])
    if choice == PLACEHOLDER:
        st.stop()
    if choice not in roster or choice == blind_study.EXCLUDED_ANNOTATOR:
        st.error("Identity is not allowed.")
        st.stop()
    acquire_lock()
    st.session_state["annotator"] = choice

annotator = st.session_state["annotator"]
if not st.session_state.get("lock_owned"):
    acquire_lock()

queue = blind_study.load_blind_queue(OUT_DIR)
completed = blind_study.completed_item_ids(OUT_DIR, annotator)
remaining = queue[~queue["item_id"].isin(completed)]
if remaining.empty:
    release_lock()
    st.success("Complete.")
    st.stop()

item = remaining.iloc[0]
item_key = item["item_id"]
counter_column, end_column = st.columns([5, 1], vertical_alignment="center")
counter_column.markdown(
    f"**Item {int(item['serve_order'])} of {len(queue)}** · `{item_key}`"
)
if end_column.button("End session", use_container_width=True):
    release_lock()
    del st.session_state["annotator"]
    st.rerun()

# Round 0 is the frozen timeline-spread protocol, so the primary evidence still
# matches what the first annotator worked from (FR-011). Later rounds are random
# draws the annotator asked for.
round_key = f"shuffle_{item_key}"
shuffle_round = st.session_state.get(round_key, 0)
manifest = load_audio_manifest()
shown = st.session_state.setdefault(f"shown_{item_key}", {})
if shuffle_round in shown:
    clips = shown[shuffle_round]
elif shuffle_round == 0:
    clips = blind_study.select_clips(manifest, item["speaker_final"])
else:
    exclude = {
        (str(clip["path"]), float(clip["start"]))
        for round_number, round_clips in shown.items()
        if round_number < shuffle_round
        for clip in round_clips
    }
    clips = blind_study.sample_clips(
        manifest,
        item["speaker_final"],
        seed=int(item_key[1:]) * 1000 + shuffle_round,
        exclude=exclude,
    )

rendered_audio = []
for clip in clips:
    try:
        audio = clip_bytes(clip["path"], clip["start"], clip["end"])
    except Exception:
        continue
    rendered_audio.append(audio)

# Keyed by round so a plain rerun re-counts the same round instead of adding to
# it; the total is every clip actually presented for this item (FR-016).
heard = st.session_state.setdefault(f"heard_{item_key}", {})
heard[shuffle_round] = len(rendered_audio)
clips_presented = sum(heard.values())
# Every clip drawn counts as shown, even one that failed to decode, so it is
# never re-drawn and cannot keep the button alive forever.
shown[shuffle_round] = clips

speaker_segments = manifest[
    manifest["speaker_final"].astype(str) == str(item["speaker_final"])
]
all_segment_keys = set(
    zip(
        speaker_segments["master_audio_path"].astype(str),
        speaker_segments["start_time"].astype(float),
    )
)
shown_segment_keys = {
    (str(clip["path"]), float(clip["start"]))
    for round_clips in shown.values()
    for clip in round_clips
}
remaining_segments = all_segment_keys - shown_segment_keys

if st.button(
    "Play 10 different clips",
    key=f"play_different_{item_key}",
    disabled=not remaining_segments,
):
    st.session_state[round_key] = shuffle_round + 1
    st.rerun()
if not remaining_segments:
    st.caption(f"All {len(all_segment_keys)} segments for this speaker have been played.")

grid_key = f"clip_grid_{item_key}_{shuffle_round}"
with st.container(key=grid_key):
    for row_start in range(0, len(rendered_audio), 5):
        columns = st.columns(5)
        for index, audio in enumerate(
            rendered_audio[row_start : row_start + 5], start=row_start + 1
        ):
            with columns[index - row_start - 1]:
                # Only the first clip autoplays; ten at once would be unlistenable.
                st.audio(audio, format="audio/wav", autoplay=index == 1)

# Parent-DOM chaining is version-fragile convenience; the manual grid is the fallback.
st.components.v1.html(
    f"""
    <script>
    (() => {{
        try {{
            const parentDocument = window.parent.document;
            const gridSelector = ".st-key-{grid_key}";
            const expectedCount = {len(rendered_audio)};

            const containerFor = (player) =>
                player.closest('[data-testid="stAudio"]') || player.parentElement;
            const clearHighlight = (player) => {{
                const container = containerFor(player);
                if (!container) return;
                container.style.outline = "";
                container.style.outlineOffset = "";
                container.style.borderRadius = "";
            }};
            const highlight = (player) => {{
                const container = containerFor(player);
                if (!container) return;
                container.style.outline = "3px solid #f2cc60";
                container.style.outlineOffset = "2px";
                container.style.borderRadius = "0.5rem";
            }};
            const currentGridFor = (player) =>
                player.closest('[class*="st-key-clip_grid_"]');

            const wireGrid = () => {{
                const grid = parentDocument.querySelector(gridSelector);
                if (!grid) return false;
                const players = Array.from(grid.querySelectorAll("audio"));
                if (!players.length) return expectedCount === 0;

                players.forEach((player) => {{
                    if (!player.paused && !player.ended) highlight(player);
                    if (player.dataset.blindChainWired) return;
                    player.dataset.blindChainWired = "true";

                    player.addEventListener("play", () => {{
                        try {{
                            const currentGrid = currentGridFor(player);
                            if (currentGrid) {{
                                currentGrid.querySelectorAll("audio").forEach(clearHighlight);
                            }}
                            highlight(player);
                        }} catch (_) {{}}
                    }});
                    player.addEventListener("pause", () => {{
                        try {{ clearHighlight(player); }} catch (_) {{}}
                    }});
                    player.addEventListener("ended", () => {{
                        try {{
                            clearHighlight(player);
                            const currentGrid = currentGridFor(player);
                            if (!currentGrid) return;
                            const currentPlayers = Array.from(
                                currentGrid.querySelectorAll("audio")
                            );
                            const nextPlayer = currentPlayers[
                                currentPlayers.indexOf(player) + 1
                            ];
                            if (!nextPlayer) return;
                            const playAttempt = nextPlayer.play();
                            if (playAttempt) playAttempt.catch(() => {{}});
                        }} catch (_) {{}}
                    }});
                }});
                return players.length >= expectedCount;
            }};

            if (!wireGrid()) {{
                const observer = new MutationObserver(() => {{
                    try {{
                        if (wireGrid()) observer.disconnect();
                    }} catch (_) {{
                        observer.disconnect();
                    }}
                }});
                observer.observe(parentDocument.body, {{ childList: true, subtree: true }});
                window.setTimeout(() => observer.disconnect(), 3000);
            }}
        }} catch (_) {{}}
    }})();
    </script>
    """,
    height=0,
)

note_key = f"note_{item_key}"
with st.expander("Note (optional)"):
    st.text_input("Note", key=note_key, label_visibility="collapsed")

judgement = None
st.caption("Region")
for index, (column, option) in enumerate(
    zip(st.columns(3), blind_study.JUDGEMENTS[:3])
):
    if column.button(
        option,
        key=f"judgement_{index}_{item_key}",
        use_container_width=True,
    ):
        judgement = option

st.caption("Other")
for index, (column, option) in enumerate(
    zip(st.columns(5), blind_study.JUDGEMENTS[3:]), start=3
):
    if column.button(
        option,
        key=f"judgement_{index}_{item_key}",
        use_container_width=True,
    ):
        judgement = option

if judgement is not None:
    try:
        blind_study.append_decision(
            OUT_DIR,
            annotator,
            item["item_id"],
            judgement,
            st.session_state.get(note_key, ""),
            clips_presented,
        )
    except ValueError as exc:
        st.error(f"Submission refused: {exc}")
        st.stop()
    st.rerun()

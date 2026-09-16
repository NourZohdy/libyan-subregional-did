"""Timestamp-manifest layer — lifted near-verbatim from the author's posted
"Incremental Blueprint Enricher".

Turns pyannote RTTM + retained episode audio into a ``master_segments.csv`` manifest
**without writing any per-segment WAV**. Pure-Python and numpy only, so the merge/split
logic is unit-testable here without audio/models.

Manifest schema (one row per kept segment):
    file_id, speaker_id, start_time, end_time, duration,
    macro_class, sub_cat, playlist, source, master_audio_path, is_excluded,
    dialect_strength, quality

``dialect_strength`` (CAFE dialect-intensity panel, plan #6) and ``quality``
(MASC clean/noisy flag, plan #6) are initialised here so the schema is stable
from Phase 1 onward; ``quality`` is filled by the embedding stage (where audio is
already in memory) and ``dialect_strength`` by the future MSA/DID gate (plan #2).
See ``src/quality.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from libyan_did.shared.config import (
    GAP_TOLERANCE,
    MAX_DURATION,
    MIN_DURATION,
    SAMPLE_RATE,
    SPLIT_WINDOW_S,
)

MANIFEST_COLUMNS = [
    "file_id", "speaker_id", "start_time", "end_time", "duration",
    "macro_class", "sub_cat", "playlist", "source", "master_audio_path",
    "is_excluded", "dialect_strength", "quality",
]


@dataclass
class Turn:
    """A diarization turn for a single speaker."""
    speaker_id: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


# --------------------------------------------------------------------------- #
# Audio loading (lazy soundfile; keeps the split logic numpy-pure and testable)
# --------------------------------------------------------------------------- #
def load_mono_waveform(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an audio file as a 1-D float32 mono array; returns ``(waveform, sr)``.

    Used by Phase 1 to hand the episode audio to the energy-valley splitter. soundfile
    is imported lazily so the merge/split functions stay importable without it.
    """
    import soundfile as sf  # lazy

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (frames, channels)
    return data.mean(axis=1), sr


# --------------------------------------------------------------------------- #
# RTTM parsing
# --------------------------------------------------------------------------- #
def parse_rttm(rttm_path: str | Path) -> list[Turn]:
    """Parse a pyannote RTTM file into a list of Turns, sorted by start time.

    RTTM SPEAKER line format (whitespace separated):
        SPEAKER <file> <chan> <start> <dur> <NA> <NA> <speaker> <NA> <NA>
    """
    turns: list[Turn] = []
    with open(rttm_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if not parts or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            dur = float(parts[4])
            speaker = parts[7]
            turns.append(Turn(speaker_id=speaker, start=start, end=start + dur))
    turns.sort(key=lambda t: (t.start, t.end))
    return turns


# --------------------------------------------------------------------------- #
# Overlap exclusion (drop cross-talk: frames with >=2 simultaneous speakers)
# --------------------------------------------------------------------------- #
def exclusive_turns(turns: list[Turn]) -> list[Turn]:
    """Keep only sub-intervals covered by exactly one speaker.

    Overlapped-speech regions (>=2 distinct speakers active) are dropped entirely, since
    cross-talk contaminates speaker embeddings. A turn that spans another speaker's
    interjection is split around the overlapped slice and its clean pieces re-merged.
    No-op when the diarization is already overlap-free.
    """
    if not turns:
        return []
    points = sorted({t.start for t in turns} | {t.end for t in turns})
    pieces: list[Turn] = []
    for a, b in zip(points, points[1:]):
        if b - a <= 0:
            continue
        mid = 0.5 * (a + b)
        speakers = {t.speaker_id for t in turns if t.start <= mid < t.end}
        if len(speakers) == 1:
            pieces.append(Turn(next(iter(speakers)), a, b))
    pieces.sort(key=lambda t: (t.start, t.end))
    merged: list[Turn] = []
    for p in pieces:
        if merged and merged[-1].speaker_id == p.speaker_id and abs(p.start - merged[-1].end) < 1e-6:
            merged[-1].end = p.end
        else:
            merged.append(Turn(p.speaker_id, p.start, p.end))
    return merged


# --------------------------------------------------------------------------- #
# Same-speaker gap merge
# --------------------------------------------------------------------------- #
def merge_turns(turns: list[Turn], gap_tolerance: float = GAP_TOLERANCE) -> list[Turn]:
    """Merge consecutive same-speaker turns separated by <= gap_tolerance seconds.

    Turns are grouped per speaker so an interjection by speaker B between two
    speaker-A turns does not block the A-A merge.
    """
    by_speaker: dict[str, list[Turn]] = {}
    for t in turns:
        by_speaker.setdefault(t.speaker_id, []).append(t)

    merged: list[Turn] = []
    for speaker, spk_turns in by_speaker.items():
        spk_turns.sort(key=lambda t: t.start)
        cur = Turn(speaker, spk_turns[0].start, spk_turns[0].end)
        for nxt in spk_turns[1:]:
            if nxt.start - cur.end <= gap_tolerance:
                cur.end = max(cur.end, nxt.end)
            else:
                merged.append(cur)
                cur = Turn(speaker, nxt.start, nxt.end)
        merged.append(cur)
    merged.sort(key=lambda t: (t.start, t.end))
    return merged


# --------------------------------------------------------------------------- #
# Energy-valley split point
# --------------------------------------------------------------------------- #
def find_best_split_time(
    waveform: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    window_s: float = SPLIT_WINDOW_S,
    min_duration: float = MIN_DURATION,
    max_duration: float = MAX_DURATION,
) -> float:
    """Return the timestamp (seconds) of the quietest short-time window inside the
    *splittable* range ``[start_s + min_duration, min(start_s + max_duration, end_s)]``.

    The split is placed at the deepest natural pause within the legal range, so it
    lands on a real silence rather than being forced to the ``max_duration`` boundary.
    Anchoring the search lower bound at ``min_duration`` also guarantees the resulting
    chunk is at least ``min_duration`` long — so no separate "too short" length guard
    (which would otherwise hard-cut mid-utterance) is needed.

    Mirrors the posted splitter, including the edge-safety fallback: if the search
    region is narrower than ``window_s`` (only happens at the true end of the audio),
    return its left edge.
    """
    n = waveform.shape[-1]
    max_audio_sec = n / sr
    min_search = min(start_s + min_duration, max_audio_sec)
    max_search = min(start_s + max_duration, end_s, max_audio_sec)

    if max_search - min_search < window_s:
        return min_search

    win = max(1, int(window_s * sr))
    step = max(1, win // 2)
    i0 = int(min_search * sr)
    i1 = int(max_search * sr)

    wav = waveform[0] if waveform.ndim > 1 else waveform  # mono 1-D view
    region = np.ascontiguousarray(wav[i0:i1])
    if region.shape[0] < win:
        return min_search

    # Vectorized short-time energy via a strided sliding window (no torch needed).
    starts = np.arange(0, region.shape[0] - win + 1, step)
    frames = np.lib.stride_tricks.sliding_window_view(region, win)[::step]
    energies = np.mean(frames.astype(np.float64) ** 2, axis=1)
    quiet_center = int(starts[int(np.argmin(energies))]) + win // 2
    return min_search + quiet_center / sr


# --------------------------------------------------------------------------- #
# Duration-bounded splitting
# --------------------------------------------------------------------------- #
def split_long_turn(
    turn: Turn,
    waveform: np.ndarray | None = None,
    sr: int = SAMPLE_RATE,
    min_duration: float = MIN_DURATION,
    max_duration: float = MAX_DURATION,
) -> list[tuple[float, float]]:
    """Split a turn into [start, end] spans each <= max_duration.

    When a waveform is given, each cut is placed at the deepest energy valley within
    ``[cur_start + min_duration, cur_start + max_duration]`` — a natural pause — and is
    guaranteed to advance by at least ``min_duration`` (so no hard boundary cut is
    forced). Without a waveform, splits fall at a hard max_duration stride. Trailing
    spans shorter than min_duration are dropped.
    """
    spans: list[tuple[float, float]] = []
    cur_start = turn.start
    while turn.end - cur_start > max_duration:
        if waveform is not None:
            cut = find_best_split_time(
                waveform, sr, cur_start, turn.end,
                min_duration=min_duration, max_duration=max_duration,
            )
        else:
            cut = cur_start + max_duration
        if cut <= cur_start:  # defensive: never stall at a degenerate edge
            cut = cur_start + max_duration
        spans.append((cur_start, cut))
        cur_start = cut

    if turn.end - cur_start >= min_duration:
        spans.append((cur_start, turn.end))
    return spans


def turns_to_rows(
    turns: list[Turn],
    *,
    file_id: str,
    macro_class: str,
    sub_cat: str,
    playlist: str,
    source: str,
    master_audio_path: str,
    waveform: np.ndarray | None = None,
    sr: int = SAMPLE_RATE,
) -> list[dict]:
    """Convert merged turns into manifest rows (no audio written).

    macro_class -> e.g. Libyan / Non-Libyan ; sub_cat -> region (Tripolitania/...).
    """
    rows: list[dict] = []
    for t in turns:
        for (s, e) in split_long_turn(t, waveform=waveform, sr=sr):
            rows.append({
                "file_id": file_id,
                "speaker_id": t.speaker_id,
                "start_time": round(s, 3),
                "end_time": round(e, 3),
                "duration": round(e - s, 3),
                "macro_class": macro_class,
                "sub_cat": sub_cat,
                "playlist": playlist,
                "source": source,
                "master_audio_path": master_audio_path,
                "is_excluded": False,
                # Placeholders; filled downstream (quality by embed stage,
                # dialect_strength by the MSA/DID gate). See src/quality.py.
                "dialect_strength": float("nan"),
                "quality": "unknown",
            })
    return rows


def process_file_metadata(
    rttm_path: str | Path,
    *,
    file_id: str,
    macro_class: str,
    sub_cat: str,
    playlist: str,
    source: str,
    master_audio_path: str,
    waveform: np.ndarray | None = None,
    sr: int = SAMPLE_RATE,
    gap_tolerance: float = GAP_TOLERANCE,
) -> list[dict]:
    """Full RTTM -> manifest-rows path for one episode: parse, merge, split."""
    turns = parse_rttm(rttm_path)
    turns = exclusive_turns(turns)  # drop overlapped cross-talk before merge/split
    merged = merge_turns(turns, gap_tolerance=gap_tolerance)
    return turns_to_rows(
        merged,
        file_id=file_id,
        macro_class=macro_class,
        sub_cat=sub_cat,
        playlist=playlist,
        source=source,
        master_audio_path=master_audio_path,
        waveform=waveform,
        sr=sr,
    )

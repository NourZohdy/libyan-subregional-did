"""Phase-2 export: materialize final clips for kept rows; write reject list.

Final clips are the ONLY audio written. They are sliced directly from the retained
episodes (no re-download). Set config.MATERIALIZE_CLIPS=False to keep the manifest +
timestamps as the deliverable instead.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from libyan_did.shared.config import (
    CLIPS_DIR,
    KEEP_TIERS,
    MATERIALIZE_CLIPS,
    REJECTS_DIR,
    SAMPLE_RATE,
)


def write_reject_list(df: pd.DataFrame, out_dir: Path = REJECTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rejects = df[df["is_excluded"]]
    path = out_dir / "reject_list.csv"
    rejects.to_csv(path, index=False)
    return path


def _export_clip(wav, start_s: float, end_s: float, dest: Path):
    import soundfile as sf
    s = int(start_s * SAMPLE_RATE)
    e = int(end_s * SAMPLE_RATE)
    sf.write(str(dest), wav[s:e], SAMPLE_RATE)


def export_clips(df: pd.DataFrame, out_dir: Path = CLIPS_DIR) -> list[Path]:
    """Slice kept rows from retained episodes into Tier/<global_actor>/ folders."""
    if not MATERIALIZE_CLIPS:
        return []
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    kept = df[(~df["is_excluded"]) & (df["tier"].isin(KEEP_TIERS))]
    written: list[Path] = []
    for audio_path, g in kept.groupby("master_audio_path"):
        wav, sr = sf.read(audio_path)
        for i, row in g.iterrows():
            folder = out_dir / str(row["tier"]) / f"actor_{int(row['global_actor'])}"
            folder.mkdir(parents=True, exist_ok=True)
            dest = folder / f"{row['file_id']}_{i}.wav"
            _export_clip(wav, row["start_time"], row["end_time"], dest)
            written.append(dest)
    return written


def export_resource_clips(
    df: pd.DataFrame,
    *,
    dest_dir: Path,
    clean_existing: bool = True,
) -> list[Path]:
    """Slice ONE resource's kept rows into ``<dest_dir>/actor_<id>/`` — the SAME
    per-playlist folder that holds its manifests/metadata (one self-contained unit).

    ``clean_existing`` removes only the ``actor_*`` subfolders first (NEVER the
    manifest files), so stale clips from an earlier (re-tiered) run don't linger.
    """
    if not MATERIALIZE_CLIPS:
        return []
    import soundfile as sf

    dest_dir = Path(dest_dir)
    if clean_existing and dest_dir.exists():
        for sub in dest_dir.glob("actor_*"):
            if sub.is_dir():
                shutil.rmtree(sub)
    kept = df[(~df["is_excluded"]) & (df["tier"].isin(KEEP_TIERS))]
    written: list[Path] = []
    for audio_path, g in kept.groupby("master_audio_path"):
        wav, _ = sf.read(audio_path)
        for i, row in g.iterrows():
            folder = dest_dir / f"actor_{int(row['global_actor'])}"
            folder.mkdir(parents=True, exist_ok=True)
            dest = folder / f"{row['file_id']}_{i}.wav"
            _export_clip(wav, row["start_time"], row["end_time"], dest)
            written.append(dest)
    return written


def export_all(df: pd.DataFrame) -> dict:
    """Write reject list + final clips; return paths summary."""
    return {
        "reject_list": write_reject_list(df),
        "clips": export_clips(df),
    }

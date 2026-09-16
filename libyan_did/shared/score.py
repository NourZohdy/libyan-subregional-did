"""Per-segment music/singing scoring with PANNs CNN14 (Kong et al. 2020).

Why this exists: Silero VAD classifies *singing* as speech, so the speech-ratio
content gate misses theme songs and musical interludes — exactly the "actor that is
actually the show jingle" failure observed in the corpus. CNN14 is an AudioSet-trained
tagger whose Music / Singing class probabilities separate sung from spoken audio.

Run AFTER embedding (a separate pass, so re-scoring never forces re-embedding) and
BEFORE (re)clustering: writes a ``music_prob`` column into each resource's
``master_segments.csv``, which ``cluster.apply_content_gates`` thresholds at
``config.MUSIC_REJECT_PROB`` (exclude_reason = "music").

The model expects 32 kHz audio; 16 kHz corpus slices are resampled up. The checkpoint
(~340 MB) is downloaded to ``~/panns_data/`` on first use and cached process-wide.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from libyan_did.shared.config import PANNS_SAMPLE_RATE

_TAGGER = None
_MUSIC_IDX: list[int] = []
_SPEECH_IDX: int = -1

SCORE_BATCH = 16  # 10 s @ 32 kHz x 16 fits comfortably in 8 GB VRAM


def _device() -> str:
    import torch  # lazy
    return "cuda" if torch.cuda.is_available() else "cpu"


def _tagger():
    """Process-wide cached CNN14 tagger + indices of the music-ish classes."""
    global _TAGGER, _MUSIC_IDX, _SPEECH_IDX
    if _TAGGER is None:
        import torch  # lazy
        from panns_inference import AudioTagging, labels  # lazy

        ckpt = Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"
        if not ckpt.exists() or ckpt.stat().st_size < 100_000_000:
            raise RuntimeError(
                f"PANNs CNN14 checkpoint missing/corrupt at {ckpt} (~327 MB). "
                "panns_inference's own downloader fails on Windows TLS; fetch it from "
                "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth")
        # panns_inference calls torch.load without weights_only=False; torch>=2.6
        # defaults to weights_only=True. The Zenodo checkpoint is a trusted academic
        # artifact, so allow the legacy pickle for this one load.
        orig_load = torch.load
        torch.load = lambda *a, **k: orig_load(*a, **{**k, "weights_only": False})
        try:
            _TAGGER = AudioTagging(checkpoint_path=None, device=_device())
        finally:
            torch.load = orig_load
        _MUSIC_IDX = [labels.index("Music"), labels.index("Singing")]
        _SPEECH_IDX = labels.index("Speech")
    return _TAGGER


def _load_audio_32k(path: str) -> np.ndarray:
    """Decode an episode to mono float32 at the CNN14 rate (32 kHz)."""
    import soundfile as sf  # lazy
    import torch  # lazy
    import torchaudio  # lazy

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = torch.from_numpy(data.mean(axis=1))
    if sr != PANNS_SAMPLE_RATE:
        mono = torchaudio.functional.resample(mono, sr, PANNS_SAMPLE_RATE)
    return mono.numpy()


def _music_probs(batch: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(max(P[Music], P[Singing]), P[Speech]) per variable-length slice."""
    tagger = _tagger()
    maxlen = max(s.shape[0] for s in batch)
    arr = np.zeros((len(batch), maxlen), dtype=np.float32)
    for k, s in enumerate(batch):
        arr[k, : s.shape[0]] = s
    clipwise, _ = tagger.inference(arr)
    return clipwise[:, _MUSIC_IDX].max(axis=1), clipwise[:, _SPEECH_IDX]


def score_resource_music(manifest_csv: Path, *, force: bool = False) -> dict:
    """Fill the ``music_prob`` column of one resource's master_segments.csv.

    Incremental: a resource whose column is already fully populated is skipped
    unless ``force``. Each episode is decoded once. Returns a status summary.
    """
    df = pd.read_csv(manifest_csv)
    if len(df) == 0:
        return {"status": "empty", "n": 0}
    done = all(c in df.columns and df[c].notna().all()
               for c in ("music_prob", "panns_speech_prob"))
    if done and not force:
        return {"status": "skipped", "n": len(df)}

    probs = np.full(len(df), np.nan, dtype=np.float64)
    speech = np.full(len(df), np.nan, dtype=np.float64)
    for audio_path, g in df.groupby("master_audio_path"):
        wav = _load_audio_32k(audio_path)
        idxs, batch = [], []

        def flush():
            if batch:
                m, sp = _music_probs(batch)
                probs[idxs] = m
                speech[idxs] = sp
                idxs.clear()
                batch.clear()

        for idx, row in g.iterrows():
            s = int(row["start_time"] * PANNS_SAMPLE_RATE)
            e = int(row["end_time"] * PANNS_SAMPLE_RATE)
            sl = wav[s:e]
            if sl.shape[0] < PANNS_SAMPLE_RATE // 10:  # <0.1 s: nothing to score
                probs[idx] = 0.0
                speech[idx] = 0.0
                continue
            idxs.append(idx)
            batch.append(sl)
            if len(batch) >= SCORE_BATCH:
                flush()
        flush()

    df["music_prob"] = np.round(probs, 4)
    df["panns_speech_prob"] = np.round(speech, 4)
    df.to_csv(manifest_csv, index=False)
    n_hi = int(((df["music_prob"].fillna(0) >= 0.5)
                & (df["panns_speech_prob"].fillna(1) < 0.5)).sum())
    return {"status": "scored", "n": len(df), "n_musicish": n_hi}

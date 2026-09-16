"""Phase-1 embedding enrichment over master_segments.csv.

Loads each retained episode ONCE and slices every manifest row's [start,end] span
**in memory** -> ECAPA 192-D L2-normalized embedding. Appends to embeddings.npy aligned
row-for-row with the manifest. No per-segment WAV is ever written.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from libyan_did.shared.config import (
    ECAPA_SOURCE,
    EMBEDDING_DIM,
    EMBEDDINGS_NPY,
    MASTER_SEGMENTS_CSV,
    SAMPLE_RATE,
)


def _device() -> str:
    import torch  # lazy
    return "cuda" if torch.cuda.is_available() else "cpu"


_ENCODER = None  # module-level cache: load ECAPA ONCE per process, reuse across resources


def _load_encoder():
    """Return a process-wide cached ECAPA encoder.

    Re-creating the encoder for every resource (186 playlists) reloads the model onto
    the GPU each time and steadily consumes VRAM until embedding OOMs. Caching it loads
    once and reuses it for the whole run.
    """
    global _ENCODER
    if _ENCODER is None:
        from speechbrain.inference.speaker import EncoderClassifier  # lazy
        _ENCODER = EncoderClassifier.from_hparams(
            source=ECAPA_SOURCE, run_opts={"device": _device()})
    return _ENCODER


def _load_audio(path: str):
    """Load a WAV fully in memory via soundfile (avoids torchcodec, which a Windows
    Application Control policy blocks). Returns a (1, T) mono tensor at SAMPLE_RATE."""
    import soundfile as sf  # lazy
    import torch  # lazy

    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (frames, channels)
    wav = torch.from_numpy(data.T).contiguous()  # -> (channels, frames)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        import torchaudio  # lazy; functional.resample is pure-DSP, no codec needed
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav  # (1, T)


def embed_manifest(
    manifest_csv: Path = MASTER_SEGMENTS_CSV,
    out_npy: Path = EMBEDDINGS_NPY,
) -> np.ndarray:
    """Compute row-aligned embeddings for the manifest; returns the (N, D) array.

    Side effect: while each slice is already in memory we also compute, with no extra
    audio decode, three content columns written back into ``manifest_csv``:
      * ``snr_db``       — SNR proxy in dB (src/quality.py)
      * ``quality``      — MASC-style clean/noisy flag from that SNR
      * ``speech_ratio`` — Silero VAD speech fraction (src/vad.py), for the music/noise gate
    """
    import torch  # lazy

    from libyan_did.shared.quality import classify_quality, estimate_snr_db  # pure-numpy, no heavy dep
    from libyan_did.shared.vad import speech_ratio

    df = pd.read_csv(manifest_csv)
    encoder = _load_encoder()
    device = _device()
    embeddings = np.zeros((len(df), EMBEDDING_DIM), dtype=np.float32)
    snr = np.full(len(df), np.nan, dtype=np.float64)
    quality = np.full(len(df), "unknown", dtype=object)
    sratio = np.full(len(df), np.nan, dtype=np.float64)
    empty = np.zeros(len(df), dtype=bool)

    # Group by episode so each file is decoded once.
    for audio_path, g in df.groupby("master_audio_path"):
        wav = _load_audio(audio_path)
        for idx, row in g.iterrows():
            s = int(row["start_time"] * SAMPLE_RATE)
            e = int(row["end_time"] * SAMPLE_RATE)
            slice_wav = wav[:, s:e]
            if slice_wav.shape[1] == 0:
                # No audio under this span (timestamps past EOF): the row has a zero
                # embedding, so it must never reach centroid/clustering math.
                empty[idx] = True
                continue
            sl = slice_wav.squeeze(0).cpu().numpy()
            snr_db = estimate_snr_db(sl, sr=SAMPLE_RATE)
            snr[idx] = snr_db
            quality[idx] = classify_quality(snr_db)
            sratio[idx] = speech_ratio(sl, sr=SAMPLE_RATE)
            with torch.no_grad():
                emb = encoder.encode_batch(slice_wav.to(device)).squeeze().cpu().numpy()
            norm = np.linalg.norm(emb)
            embeddings[idx] = emb / norm if norm > 0 else emb

    df["snr_db"] = snr
    df["quality"] = quality
    df["speech_ratio"] = sratio
    if "exclude_reason" not in df.columns:
        df["exclude_reason"] = ""
    df.loc[empty, "is_excluded"] = True
    df.loc[empty & (df["exclude_reason"].fillna("") == ""), "exclude_reason"] = "empty_slice"
    df.to_csv(manifest_csv, index=False)

    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, embeddings)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # release per-slice tensors between resources
    return embeddings

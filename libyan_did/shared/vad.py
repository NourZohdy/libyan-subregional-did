"""Silero VAD speech-density helper (content filter, plan #2).

Computes the fraction of a segment that is *actual speech*. Music, singing, applause,
crowd noise, and silence-padded turns that the diarizer mislabeled as a speaker turn
score low here and are dropped by the Phase-2 content gate (``MIN_SPEECH_RATIO``).

The Silero model is tiny and loaded once (module-level cache). Pure helper: callers
pass an in-memory mono float32 array (the embed stage already has the slice decoded).
"""

from __future__ import annotations

import numpy as np

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from silero_vad import load_silero_vad  # lazy
        _MODEL = load_silero_vad()
    return _MODEL


def speech_ratio(wav_1d: np.ndarray, sr: int = 16000, threshold: float = 0.5) -> float:
    """Fraction of ``wav_1d`` Silero VAD marks as speech, in ``[0, 1]``.

    Returns 0.0 for empty/too-short input. 16 kHz mono expected.
    """
    import torch  # lazy
    from silero_vad import get_speech_timestamps  # lazy

    n = int(np.asarray(wav_1d).shape[-1])
    if n < 512:  # Silero needs at least one 512-sample window at 16 kHz
        return 0.0
    t = torch.from_numpy(np.ascontiguousarray(wav_1d, dtype=np.float32))
    ts = get_speech_timestamps(t, _model(), sampling_rate=sr, threshold=threshold)
    speech = sum(seg["end"] - seg["start"] for seg in ts)
    return float(speech) / float(n)

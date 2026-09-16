"""Per-segment quality + dialect-strength fields (plan recommendation #6).

Two manifest fields are populated here:

* ``quality`` — a MASC-style ``clean``/``noisy`` flag, so a clean test set can be
  carved out. Derived from a lightweight SNR *proxy* computed directly on the
  in-memory waveform slice (no extra model, no heavy deps).
* ``dialect_strength`` — CAFE's dialect-intensity / confidence panel. There is no
  DID/MSA classifier in-repo, so this stays a documented placeholder (NaN) until a
  scorer is plugged into :func:`assign_dialect_strength`. That hook is the wiring
  point for the MSA gate (plan recommendation #2) and doubles as the MSA-vs-dialect
  signal once a model exists.

Everything here is pure numpy/pandas so it is unit-testable without audio or torch.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from libyan_did.shared.config import SAMPLE_RATE, SNR_CLEAN_THRESHOLD_DB


# --------------------------------------------------------------------------- #
# SNR proxy + clean/noisy flag
# --------------------------------------------------------------------------- #
def estimate_snr_db(
    wav: np.ndarray,
    sr: int = SAMPLE_RATE,
    frame_ms: float = 25.0,
) -> float:
    """Estimate a per-segment SNR *proxy* in dB from a mono waveform.

    Frames the signal into ``frame_ms`` windows, takes each frame's RMS in dB, and
    returns ``P90 - P10`` of those frame energies — the active-speech level above the
    quiet (noise-floor) level. This is a cheap, model-free percentile SNR estimate in
    the spirit of NIST/WADA-style measures; it is a proxy, not a calibrated SNR, and
    is used only to threshold clean vs. noisy.

    Returns ``0.0`` for empty or sub-2-frame inputs (too short to estimate).
    """
    wav = np.asarray(wav, dtype=np.float64).ravel()
    if wav.size == 0:
        return 0.0
    n = max(1, int(sr * frame_ms / 1000.0))
    nframes = wav.size // n
    if nframes < 2:
        return 0.0
    frames = wav[: nframes * n].reshape(nframes, n)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    noise = float(np.percentile(db, 10))
    signal = float(np.percentile(db, 90))
    return signal - noise


def classify_quality(
    snr_db: float,
    threshold: float = SNR_CLEAN_THRESHOLD_DB,
) -> str:
    """Map an SNR proxy (dB) to a MASC-style ``"clean"``/``"noisy"`` flag."""
    return "clean" if snr_db >= threshold else "noisy"


def quality_for_slice(
    wav_slice: np.ndarray,
    sr: int = SAMPLE_RATE,
    threshold: float = SNR_CLEAN_THRESHOLD_DB,
) -> str:
    """Convenience: SNR proxy → flag for one in-memory waveform slice."""
    return classify_quality(estimate_snr_db(wav_slice, sr=sr), threshold=threshold)


# --------------------------------------------------------------------------- #
# dialect_strength hook (filled by a future MSA/DID scorer; plan #2)
# --------------------------------------------------------------------------- #
def assign_dialect_strength(
    df,
    *,
    scorer: Optional[Callable[[int], float]] = None,
    embeddings: Optional[np.ndarray] = None,
):
    """Populate the ``dialect_strength`` column.

    ``scorer`` is a callable mapping a manifest row index to a confidence in
    ``[0, 1]`` (e.g. a softmax margin from a DID/MSA classifier). When ``embeddings``
    is given it is passed through so a scorer can close over it. When no scorer is
    supplied the column is left as ``NaN`` — an explicit "not yet measured" marker
    rather than a fabricated number. This is the wiring point for recommendation #2.

    Returns a copy of ``df`` with ``dialect_strength`` set.
    """
    out = df.copy()
    if "dialect_strength" not in out.columns:
        out["dialect_strength"] = float("nan")
    if scorer is None:
        return out
    out["dialect_strength"] = [float(scorer(i)) for i in range(len(out))]
    return out

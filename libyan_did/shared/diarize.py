"""Phase-1 diarization: pyannote -> tiny .rttm per episode (no per-segment WAV).

Optionally trims non-speech inside turns with Silero VAD. RTTM is small text metadata
that the segment layer parses into the timestamp manifest.
"""

from __future__ import annotations

from pathlib import Path

from libyan_did.shared.config import PYANNOTE_PIPELINE, RTTM_DIR


def _load_pipeline(auth_token: str | None = None):
    import torch  # lazy
    from pyannote.audio import Pipeline  # lazy

    pipeline = Pipeline.from_pretrained(PYANNOTE_PIPELINE, token=auth_token)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    return pipeline


def _load_audio(audio_path: str | Path) -> dict:
    """Load a WAV fully in memory and hand pyannote a ``{waveform, sample_rate}`` dict.

    Avoids pyannote's file-path decoding, which routes through torchcodec/ffmpeg DLLs
    that a Windows Application Control policy can block (WinError 4551). soundfile reads
    the 16 kHz mono WAVs produced by the harvest step reliably.
    """
    import soundfile as sf  # lazy
    import torch  # lazy

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)  # (frames, channels)
    waveform = torch.from_numpy(data.T).contiguous()  # -> (channels, frames)
    return {"waveform": waveform, "sample_rate": sr}


def diarize_episode(
    audio_path: str | Path,
    file_id: str,
    *,
    pipeline=None,
    auth_token: str | None = None,
    out_dir: Path = RTTM_DIR,
) -> Path:
    """Run diarization on one episode and write ``<file_id>.rttm``; skip if present."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rttm_path = out_dir / f"{file_id}.rttm"
    if rttm_path.exists():
        return rttm_path  # resumable

    if pipeline is None:
        pipeline = _load_pipeline(auth_token)

    result = pipeline(_load_audio(audio_path))
    # community-1 returns a wrapper exposing the Annotation under .speaker_diarization;
    # 3.x returns the Annotation directly.
    annotation = getattr(result, "speaker_diarization", result)
    with open(rttm_path, "w", encoding="utf-8") as fh:
        annotation.write_rttm(fh)
    return rttm_path


def diarize_all(episode_records: list[dict], auth_token: str | None = None) -> list[dict]:
    """Diarize every harvested episode; annotate each record with ``rttm_path``."""
    pipeline = _load_pipeline(auth_token)
    for rec in episode_records:
        rec["rttm_path"] = str(
            diarize_episode(rec["audio_path"], rec["file_id"], pipeline=pipeline)
        )
    return episode_records

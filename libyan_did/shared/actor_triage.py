"""Per-ACTOR triage scores for risk-ranked human validation (ranking only — these
scores NEVER auto-reject anything; rejection is the per-segment content gates' job).

For every actor (clustered speaker) this computes:

* ``cohesion``        — mean cosine similarity of the actor's segment embeddings to
                        its centroid (low = impure cluster, likely a clustering error)
* ``mean_music_prob`` / ``music_frac`` — PANNs aggregate (jingle-cluster signal)
* ``mean_speech_ratio`` / ``mean_snr_db`` — existing per-segment QA aggregates
* ``lib_prob`` / ``msa_prob`` / ``top_dialect`` — HuBERT Arabic-dialect classifier
  (config.HUBERT_DIALECT_MODEL; 17 dialects + MSA incl. an explicit LIB class),
  averaged over up to ``ACTOR_SCORE_MAX_SEGMENTS`` sampled segments
* ``arabic_prob``     — VoxLingua107 ECAPA LID P(Arabic): catches foreign-language
                        guests, which an Arabic-dialect model cannot represent
* ``risk_score``      — max of the component risks; the validation app reviews
                        actors in descending risk order

Output: ``outputs/metadata/actor_scores.csv`` keyed by actor_uid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from libyan_did.shared import config

SCORE_COLUMNS = [
    "actor_uid", "region", "playlist", "global_actor", "n_segments", "minutes",
    "cohesion", "mean_music_prob", "music_frac", "mean_speech_ratio", "mean_snr_db",
    "lib_prob", "msa_prob", "top_dialect", "top_dialect_prob", "arabic_prob",
    "n_scored_segments", "risk_score",
]

_TRUE = {"true", "1", "1.0", "yes"}
_HUBERT = None  # (feature_extractor, model, lib_idx, msa_idx, id2label)
_LID = None     # (classifier, arabic_idx)


def _device() -> str:
    import torch  # lazy
    return "cuda" if torch.cuda.is_available() else "cpu"


def _hubert():
    global _HUBERT
    if _HUBERT is None:
        import torch  # noqa: F401  (lazy)
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        fe = AutoFeatureExtractor.from_pretrained(config.HUBERT_DIALECT_MODEL)
        model = (AutoModelForAudioClassification
                 .from_pretrained(config.HUBERT_DIALECT_MODEL).to(_device()).eval())
        label2id = {v: k for k, v in model.config.id2label.items()}
        _HUBERT = (fe, model, label2id["LIB"], label2id["MSA"], model.config.id2label)
    return _HUBERT


def _lid():
    global _LID
    if _LID is None:
        from speechbrain.inference.classifiers import EncoderClassifier
        clf = EncoderClassifier.from_hparams(source=config.LID_MODEL,
                                             run_opts={"device": _device()})
        lab2ind = clf.hparams.label_encoder.lab2ind
        ar = next((i for lab, i in lab2ind.items() if str(lab).split(":")[0].strip() == "ar"),
                  None)
        if ar is None:
            raise RuntimeError(f"no 'ar' label in {config.LID_MODEL} label set")
        _LID = (clf, int(ar))
    return _LID


def _sample_rows(g: pd.DataFrame, k: int) -> pd.DataFrame:
    """Up to k rows per actor: clean + long first, spread across episodes."""
    g = g.copy()
    g["_clean"] = (g["quality"] == "clean").astype(int) if "quality" in g.columns else 1
    g = g.sort_values(["_clean", "duration"], ascending=[False, False])
    picked_idx, seen = [], set()
    for idx, r in g.iterrows():
        if r["file_id"] not in seen:
            picked_idx.append(idx)
            seen.add(r["file_id"])
        if len(picked_idx) >= k:
            break
    for idx in g.index:
        if len(picked_idx) >= k:
            break
        if idx not in picked_idx:
            picked_idx.append(idx)
    return g.loc[picked_idx]


def _dialect_and_lid_probs(slices: list[np.ndarray]) -> dict:
    """Run HuBERT + LID over the sampled slices; average the posteriors."""
    import torch

    fe, model, lib_idx, msa_idx, id2label = _hubert()
    clf, ar_idx = _lid()
    device = _device()

    inputs = fe(slices, sampling_rate=config.SAMPLE_RATE,
                return_tensors="pt", padding=True)
    mask = inputs.get("attention_mask")
    with torch.no_grad():
        logits = model(inputs.input_values.to(device),
                       attention_mask=mask.to(device) if mask is not None else None).logits
        probs = torch.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()

    ar_probs = []
    with torch.no_grad():
        for sl in slices:
            t = torch.from_numpy(sl).unsqueeze(0)
            out_prob, _, _, _ = clf.classify_batch(t)
            ar_probs.append(float(out_prob.exp()[0, ar_idx]))

    top = int(np.argmax(probs))
    return {
        "lib_prob": float(probs[lib_idx]),
        "msa_prob": float(probs[msa_idx]),
        "top_dialect": id2label[top],
        "top_dialect_prob": float(probs[top]),
        "arabic_prob": float(np.mean(ar_probs)),
    }


def score_resource_actors(
    final_csv: Path,
    emb_npy: Path,
    region: str,
    playlist: str,
    *,
    max_segments: int = config.ACTOR_SCORE_MAX_SEGMENTS,
) -> list[dict]:
    """Triage rows for every actor of one resource (episodes decoded once)."""
    import soundfile as sf  # lazy

    df = pd.read_csv(final_csv)
    if len(df) == 0:
        return []
    emb = np.load(emb_npy)
    df["is_excluded"] = df["is_excluded"].astype(str).str.strip().str.lower().isin(_TRUE)
    active = df[(~df["is_excluded"]) & (df["global_actor"] >= 0)]
    if len(active) == 0:
        return []

    # Sample rows for ALL actors first, grouped by episode -> decode each file once.
    sampled: dict[int, pd.DataFrame] = {
        int(a): _sample_rows(g, max_segments)
        for a, g in active.groupby("global_actor")
    }
    need = pd.concat(sampled.values())
    slices: dict[int, np.ndarray] = {}
    for path, g in need.groupby("master_audio_path"):
        wav, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = wav.mean(axis=1)
        for idx, row in g.iterrows():
            s = int(row["start_time"] * sr)
            e = int(row["end_time"] * sr)
            sl = mono[s:e]
            if sr != config.SAMPLE_RATE and sl.shape[0]:
                import torch
                import torchaudio
                sl = torchaudio.functional.resample(
                    torch.from_numpy(sl), sr, config.SAMPLE_RATE).numpy()
            if sl.shape[0] >= config.SAMPLE_RATE:  # >=1 s usable
                slices[idx] = sl

    rows = []
    for actor, g in active.groupby("global_actor"):
        actor = int(actor)
        idx = list(g.index)
        centroid = emb[idx].mean(axis=0)
        centroid /= (np.linalg.norm(centroid) or 1.0)
        cohesion = float((emb[idx] @ centroid).mean())

        music = g["music_prob"].astype(float) if "music_prob" in g.columns else pd.Series(dtype=float)
        row = {
            "actor_uid": f"{region}/{playlist}#{actor}",
            "region": region, "playlist": playlist, "global_actor": actor,
            "n_segments": len(g),
            "minutes": round(g["duration"].sum() / 60.0, 2),
            "cohesion": round(cohesion, 4),
            "mean_music_prob": round(float(music.mean()), 4) if len(music) else np.nan,
            "music_frac": (round(float((music >= 0.5).mean()), 4) if len(music) else np.nan),
            "mean_speech_ratio": (round(float(g["speech_ratio"].mean()), 4)
                                  if "speech_ratio" in g.columns else np.nan),
            "mean_snr_db": (round(float(g["snr_db"].mean()), 2)
                            if "snr_db" in g.columns else np.nan),
        }

        actor_slices = [slices[i] for i in sampled[actor].index if i in slices]
        if actor_slices:
            row.update(_dialect_and_lid_probs(actor_slices))
            row["n_scored_segments"] = len(actor_slices)
        else:
            row.update({"lib_prob": np.nan, "msa_prob": np.nan, "top_dialect": "",
                        "top_dialect_prob": np.nan, "arabic_prob": np.nan})
            row["n_scored_segments"] = 0

        r_dialect = 1.0 - row["lib_prob"] if pd.notna(row["lib_prob"]) else 0.5
        r_lid = 1.0 - row["arabic_prob"] if pd.notna(row["arabic_prob"]) else 0.5
        r_music = row["mean_music_prob"] if pd.notna(row["mean_music_prob"]) else 0.0
        r_cohesion = 1.0 - row["cohesion"]
        row["risk_score"] = round(max(r_dialect, r_lid, r_music, r_cohesion), 4)
        rows.append(row)
    return rows

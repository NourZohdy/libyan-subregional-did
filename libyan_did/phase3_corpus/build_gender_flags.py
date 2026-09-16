"""Gender-mix cluster-purity flag (ranking only — never auto-rejects).

Three stages (resumable; the expensive one checkpoints to disk):

1. TEACHER  — sample config.GENDER_PROBE_SAMPLES active segments (stratified by
   region, clean + >=2 s preferred), label them with the audeering age+gender
   model -> outputs/metadata/gender_probe_labels.csv. Skipped if that file
   already exists (delete it or --force-teacher to relabel).
2. PROBE    — fit a logistic regression on the labeled segments' cached ECAPA
   embeddings (high-margin male/female labels only) and report held-out
   accuracy. Embeddings encode gender strongly, so this scales the teacher's
   labels to ALL 84k segments at zero inference cost.
3. FLAG     — predict gender for every active segment, aggregate per actor_uid:
   gender_mix = minority-gender fraction. Write gender_mix / gender_majority
   into actor_scores.csv (timestamped backup first) and lift
   risk_score = max(risk_score_base, min(1, 2*gender_mix)) for actors with
   >= config.GENDER_MIX_MIN_SEGMENTS segments. Idempotent: the pre-flag risk is
   preserved in risk_score_base.

Usage:
    python scripts/build_gender_flags.py
    python scripts/build_gender_flags.py --probe-samples 500 --limit 20   # smoke
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


from libyan_did.shared import config

SCORES_CSV = config.METADATA_DIR / "actor_scores.csv"
LABELS_CSV = config.METADATA_DIR / "gender_probe_labels.csv"
SEED = 1337
_TRUE = {"true", "1", "1.0", "yes"}


def iter_resources(limit: int | None = None):
    finals = sorted(config.CORPUS_DIR.glob("*/*/final_manifest.csv"))
    if limit:
        finals = finals[:limit]
    for fpath in finals:
        emb_path = fpath.parent / "embeddings.npy"
        if not emb_path.exists():
            continue
        df = pd.read_csv(fpath)
        if len(df) == 0:
            continue
        df["is_excluded"] = (df["is_excluded"].astype(str).str.strip()
                             .str.lower().isin(_TRUE))
        yield fpath.parent.parent.name, fpath.parent.name, fpath, emb_path, df


def collect_active(limit: int | None) -> pd.DataFrame:
    """One row per ACTIVE segment corpus-wide, with its embedding address."""
    rows = []
    for region, playlist, fpath, emb_path, df in iter_resources(limit):
        active = df[(~df["is_excluded"]) & (df["global_actor"] >= 0)]
        for idx, r in active.iterrows():
            rows.append({
                "region": region, "playlist": playlist,
                "emb_path": str(emb_path), "row": int(idx),
                "actor_uid": f"{region}/{playlist}#{int(r['global_actor'])}",
                "master_audio_path": r["master_audio_path"],
                "start": float(r["start_time"]), "end": float(r["end_time"]),
                "duration": float(r["duration"]),
                "clean": int(r.get("quality", "") == "clean"),
            })
    return pd.DataFrame(rows)


def run_teacher(segs: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """Stage 1: stratified sample -> audeering labels -> LABELS_CSV."""
    import soundfile as sf

    from libyan_did.shared.age_gender import AgeGenderScorer

    rng = np.random.RandomState(SEED)
    per_region = max(1, n_samples // len(config.REGIONS))
    picked = []
    for region, g in segs.groupby("region"):
        g = g[g["duration"] >= 2.0]
        g = g.sort_values("clean", ascending=False)
        k = min(per_region, len(g))
        # clean-first pool of 2k size, random inside it (avoids one-show bias)
        pool = g.head(min(len(g), k * 3))
        picked.append(pool.iloc[rng.permutation(len(pool))[:k]])
    sample = pd.concat(picked).reset_index(drop=True)
    print(f"teacher: labeling {len(sample)} segments "
          f"({dict(sample['region'].value_counts())})")

    scorer = AgeGenderScorer()
    results = [None] * len(sample)
    done = 0
    for path, g in sample.groupby("master_audio_path"):
        wav, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = wav.mean(axis=1)
        slices, locs = [], []
        for i, r in g.iterrows():
            sl = mono[int(r["start"] * sr):int(r["end"] * sr)]
            if sr != config.SAMPLE_RATE and sl.shape[0]:
                import torch
                import torchaudio
                sl = torchaudio.functional.resample(
                    torch.from_numpy(sl), sr, config.SAMPLE_RATE).numpy()
            if sl.shape[0] >= config.SAMPLE_RATE:
                slices.append(sl)
                locs.append(i)
        if slices:
            for loc, pred in zip(locs, scorer.predict(slices)):
                results[loc] = pred
        done += len(slices)
        if done % 200 < len(slices):
            print(f"  {done}/{len(sample)} labeled")

    keep = [i for i, p in enumerate(results) if p is not None]
    out = sample.iloc[keep].reset_index(drop=True)
    preds = pd.DataFrame([results[i] for i in keep])
    out = pd.concat([out, preds], axis=1)
    config.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(LABELS_CSV, index=False)
    print(f"-> {LABELS_CSV} ({len(out)} labeled)")
    return out


def load_embeddings(rows: pd.DataFrame) -> np.ndarray:
    """Fetch ECAPA embeddings for (emb_path, row) addresses, file by file."""
    out = np.zeros((len(rows), config.EMBEDDING_DIM), dtype=np.float32)
    for path, g in rows.groupby("emb_path"):
        emb = np.load(path)
        out[np.asarray(g.index)] = emb[g["row"].to_numpy()]
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n == 0, 1.0, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-samples", type=int, default=config.GENDER_PROBE_SAMPLES)
    ap.add_argument("--limit", type=int, default=None, help="first N resources (smoke)")
    ap.add_argument("--force-teacher", action="store_true")
    args = ap.parse_args()

    segs = collect_active(args.limit)
    print(f"active segments: {len(segs):,} across "
          f"{segs['actor_uid'].nunique():,} actors")

    # ---- stage 1: teacher labels (checkpointed) ----
    if LABELS_CSV.exists() and not args.force_teacher:
        labels = pd.read_csv(LABELS_CSV)
        print(f"teacher: reusing {LABELS_CSV.name} ({len(labels)} labels)")
    else:
        labels = run_teacher(segs, args.probe_samples)

    # ---- stage 2: logistic probe on ECAPA embeddings ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    lab = labels[(labels["gender"].isin(["male", "female"]))
                 & ((labels["female"] - labels["male"]).abs()
                    >= config.GENDER_PROBE_MIN_MARGIN)].reset_index(drop=True)
    print(f"probe: {len(lab)} high-margin labels "
          f"({dict(lab['gender'].value_counts())})")
    X = load_embeddings(lab)
    y = (lab["gender"] == "male").astype(int).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED,
                                          stratify=y)
    probe = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    acc = probe.score(Xte, yte)
    print(f"probe held-out accuracy: {acc:.1%} (n_test={len(yte)})")
    if acc < 0.90:
        print("!! probe accuracy below 0.90 — gender_mix would be noise; aborting "
              "before touching actor_scores.csv")
        sys.exit(1)

    # ---- stage 3: flag every actor ----
    segs = segs.reset_index(drop=True)
    p_male = np.zeros(len(segs), dtype=np.float32)
    for path, g in segs.groupby("emb_path"):
        emb = np.load(path)
        e = emb[g["row"].to_numpy()]
        n = np.linalg.norm(e, axis=1, keepdims=True)
        p_male[np.asarray(g.index)] = probe.predict_proba(
            e / np.where(n == 0, 1.0, n))[:, 1]
    segs["is_male"] = p_male >= 0.5

    agg = segs.groupby("actor_uid")["is_male"].agg(["mean", "size"])
    agg["gender_mix"] = np.minimum(agg["mean"], 1.0 - agg["mean"]).round(4)
    agg["gender_majority"] = np.where(agg["mean"] >= 0.5, "male", "female")
    agg = agg.rename(columns={"size": "gender_n"}).drop(columns=["mean"])

    sc = pd.read_csv(SCORES_CSV)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = SCORES_CSV.with_name(f"actor_scores_pre_gender_{stamp}.csv")
    sc.to_csv(backup, index=False)
    if "risk_score_base" not in sc.columns:
        sc["risk_score_base"] = sc["risk_score"]
    sc = sc.drop(columns=[c for c in ("gender_mix", "gender_majority", "gender_n")
                          if c in sc.columns])
    sc = sc.merge(agg, on="actor_uid", how="left")
    penalty = np.where(
        (sc["gender_n"].fillna(0) >= config.GENDER_MIX_MIN_SEGMENTS),
        np.minimum(1.0, 2.0 * sc["gender_mix"].fillna(0.0)), 0.0)
    sc["risk_score"] = np.maximum(sc["risk_score_base"], penalty).round(4)
    sc.to_csv(SCORES_CSV, index=False)

    flagged = sc[(sc["gender_mix"].fillna(0) >= 0.25)
                 & (sc["gender_n"].fillna(0) >= config.GENDER_MIX_MIN_SEGMENTS)]
    print(f"-> {SCORES_CSV} updated (backup: {backup.name})")
    print(f"flagged actors (gender_mix>=0.25, n>={config.GENDER_MIX_MIN_SEGMENTS}): "
          f"{len(flagged)}")
    if len(flagged):
        print(flagged.sort_values("gender_mix", ascending=False)
              [["actor_uid", "n_segments", "gender_mix", "gender_majority",
                "risk_score_base", "risk_score"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()

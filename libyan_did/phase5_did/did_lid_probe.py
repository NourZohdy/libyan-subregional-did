"""E5: frozen VoxLingua107 ECAPA-LID embedding probe for 3-way sub-regional DID.

E2 used the speaker-recognition ECAPA (spkrec-ecapa-voxceleb, 192-d), whose t-SNE
shows it organizes audio by voice. E5 swaps in the VoxLingua107 *language-ID* ECAPA
(speechbrain/lang-id-voxlingua107-ecapa): the same ECAPA-TDNN architecture, but
pretrained to tell 107 languages apart rather than speakers, so its 256-d embedding
should carry more dialect signal. We keep the encoder FROZEN and train the same
logistic-regression probe as E2/E3 on top. This isolates the backbone effect; the
end-to-end fine-tuned counterpart is E6 (did_finetune_lid.py).

Audio is read once per source file with soundfile, resampled to 16 kHz, sliced, and
pushed through the encoder; the 256-d embeddings are cached to
outputs/manifests/lid_ecapa_emb.npy (row-aligned to splits_manifest.csv) so re-runs
skip the GPU pass. Results append next to the E0/E2/E3 rows.

Run (diarization env, from libyan_did_pipeline/):
    python scripts/did_lid_probe.py                  # full run (GPU, ~20-40 min)
    python scripts/did_lid_probe.py --limit-files 5  # smoke test (no cache write)
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from libyan_did.shared import config

SPLITS = config.MANIFEST_DIR / "splits_manifest.csv"
TABLES = config.TABLES_DIR
SEED = 1337
REGIONS = list(config.REGIONS)
SR = config.SAMPLE_RATE
LID_DIM = 256
MODEL_NAME = "LID-ECAPA frozen+probe (E5)"
CACHE = config.MANIFEST_DIR / "lid_ecapa_emb.npy"
SAVEDIR = config.MANIFEST_DIR / "sb_lid_voxlingua107"


def _amp(device):
    """fp16 autocast on CUDA, no-op on CPU."""
    import torch
    return torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else contextlib.nullcontext()


def load_lid_mods(device: str):
    """Load the VoxLingua107 ECAPA mods (COPY strategy: Windows symlink needs admin).

    Returns (compute_features, mean_var_norm, embedding_model). The 107-way classifier
    is dropped. Shared by E5 (frozen) and E6 (fine-tuned).
    """
    import torch  # noqa: F401
    from speechbrain.inference.classifiers import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    clf = EncoderClassifier.from_hparams(
        source=config.LID_MODEL, savedir=str(SAVEDIR),
        run_opts={"device": device}, local_strategy=LocalStrategy.COPY)
    return clf.mods.compute_features, clf.mods.mean_var_norm, clf.mods.embedding_model


def extract_features(work: pd.DataFrame, batch_size: int, max_seconds: float,
                     limit_files: int | None) -> np.ndarray:
    import soundfile as sf
    import torch
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_features, mean_var_norm, embedding_model = load_lid_mods(device)
    embedding_model.eval()
    X = np.zeros((len(work), LID_DIM), dtype=np.float32)
    print(f"model {config.LID_MODEL} | dim {LID_DIM} | device {device}")

    files = list(work.groupby("master_audio_path"))
    if limit_files:
        files = files[:limit_files]
    max_len = int(max_seconds * SR)
    done = 0
    for fi, (path, g) in enumerate(files):
        try:
            wav, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception as e:
            print(f"  ! read fail {path}: {e}")
            continue
        mono = wav.mean(axis=1)
        if sr != SR:
            mono = torchaudio.functional.resample(torch.from_numpy(mono), sr, SR).numpy()
        idxs, slices = [], []
        for ridx, r in g.iterrows():
            a, b = int(r["start_time"] * SR), int(r["end_time"] * SR)
            sl = mono[a:b][:max_len]
            if sl.shape[0] < 640:                      # guard against degenerate slices
                sl = np.pad(sl, (0, 640 - sl.shape[0]))
            idxs.append(ridx); slices.append(sl)
        # batch this file's slices through the encoder (pad + relative lengths)
        for s in range(0, len(slices), batch_size):
            chunk = slices[s:s + batch_size]
            ci = idxs[s:s + batch_size]
            tmax = max(x.shape[0] for x in chunk)
            padded = np.zeros((len(chunk), tmax), dtype=np.float32)
            rel = np.empty(len(chunk), dtype=np.float32)
            for j, x in enumerate(chunk):
                padded[j, : x.shape[0]] = x
                rel[j] = x.shape[0] / tmax
            wavb = torch.from_numpy(padded).to(device)
            wl = torch.from_numpy(rel).to(device)
            with torch.no_grad(), _amp(device):
                feats = compute_features(wavb)
                feats = mean_var_norm(feats, wl)
                emb = embedding_model(feats, wl).squeeze(1).float()   # (b, 256)
            X[np.asarray(ci)] = emb.cpu().numpy()
        if device == "cuda":
            torch.cuda.empty_cache()                        # reset fragmentation across files
        done += len(slices)
        if fi % 50 == 0 or fi == len(files) - 1:
            print(f"  files {fi+1}/{len(files)} | {done} segments")
    return X


def _scores(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score
    return (accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, labels=REGIONS, average="macro"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-seconds", type=float, default=15.0)
    ap.add_argument("--limit-files", type=int, default=None)
    args = ap.parse_args()

    sp = pd.read_csv(SPLITS)
    work = sp[["file_id", "start_time", "end_time", "master_audio_path",
               "final_dialect", "split", "speaker_final"]].reset_index(drop=True)
    work = work.dropna(subset=["master_audio_path", "final_dialect", "split"])
    bad = set(work["final_dialect"].unique()) - set(REGIONS)
    assert not bad, f"unexpected labels in final_dialect: {bad}"
    work = work.reset_index(drop=True)

    if CACHE.exists() and not args.limit_files:
        print(f"loading cached features {CACHE.name}")
        X = np.load(CACHE)
        assert X.shape == (len(work), LID_DIM), f"cache shape {X.shape} != {(len(work), LID_DIM)}"
    else:
        X = extract_features(work, args.batch_size, args.max_seconds, args.limit_files)
        if args.limit_files:
            print("smoke test done (features not cached); rerun without --limit-files")
            return
        np.save(CACHE, X)
        print(f"cached -> {CACHE}")

    # L2-normalize features for the linear probe
    n = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(n == 0, 1.0, n)
    y = work["final_dialect"].to_numpy()
    split = work["split"].to_numpy()
    spk = work["speaker_final"].to_numpy()
    tr, dv, te = split == "train", split == "dev", split == "test"

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    clf = LogisticRegression(max_iter=5000, C=1.0).fit(X[tr], y[tr])
    pred = {s: clf.predict(X[m]) for s, m in [("dev", dv), ("test", te)]}

    rows = []
    for s, m in [("dev", dv), ("test", te)]:
        acc, mf1 = _scores(y[m], pred[s])
        rows.append({"model": MODEL_NAME, "level": "segment", "split": s,
                     "accuracy": round(acc, 4), "macro_f1": round(mf1, 4)})
    for s, m in [("dev", dv), ("test", te)]:
        df = pd.DataFrame({"spk": spk[m], "true": y[m], "pred": pred[s]})
        gg = df.groupby("spk").agg(true=("true", "first"),
                                   pred=("pred", lambda x: x.value_counts().idxmax()))
        acc, mf1 = _scores(gg["true"].to_numpy(), gg["pred"].to_numpy())
        rows.append({"model": MODEL_NAME, "level": "speaker", "split": s,
                     "accuracy": round(acc, 4), "macro_f1": round(mf1, 4)})
    new = pd.DataFrame(rows)

    TABLES.mkdir(parents=True, exist_ok=True)
    rp = TABLES / "did_results.csv"
    if rp.exists():
        old = pd.read_csv(rp)
        old = old[old["model"] != MODEL_NAME]              # idempotent re-run
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(rp, index=False)

    cm = confusion_matrix(y[te], pred["test"], labels=REGIONS)
    pd.DataFrame(cm, index=[f"true_{r}" for r in REGIONS],
                 columns=[f"pred_{r}" for r in REGIONS]).to_csv(
        TABLES / "did_confusion_test_lid_ecapa.csv")

    print("\n==== E5 (frozen LID-ECAPA probe) ====")
    print(new[new["model"] == MODEL_NAME].to_string(index=False))
    print("\n==== TEST per-class (segment) ====")
    print(classification_report(y[te], pred["test"], labels=REGIONS, digits=3))
    print("wrote: did_results.csv, did_confusion_test_lid_ecapa.csv")


if __name__ == "__main__":
    main()

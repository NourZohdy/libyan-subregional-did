"""E3: frozen self-supervised encoder probe for 3-way sub-regional DID.

Extracts mean-pooled hidden states from a FROZEN pretrained SSL speech encoder
(default facebook/wav2vec2-xls-r-300m) for every split segment, then trains the
same logistic-regression probe as E2 on top. No fine-tuning. This feature is more
phonetic than the ECAPA speaker embedding, so it tests whether dialect cues beyond
speaker identity help.

Audio is read once per source file (1,281 files), resampled to 16 kHz, sliced, and
pushed through the encoder in batches; features are cached to
outputs/manifests/ssl_<tag>.npy so re-runs skip the GPU pass. Results are appended
to outputs/tables/did_results.csv next to the E0/E2 rows.

Run (diarization env, from libyan_did_pipeline/):
    python scripts/did_ssl_probe.py                      # full run (GPU, ~30-60 min)
    python scripts/did_ssl_probe.py --limit-files 5      # smoke test (no cache write)
"""

from __future__ import annotations

import argparse
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


def extract_features(work: pd.DataFrame, model_id: str, batch_size: int,
                     max_seconds: float, limit_files: int | None) -> np.ndarray:
    import soundfile as sf
    import torch
    import torchaudio
    from transformers import AutoFeatureExtractor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    H = model.config.hidden_size
    X = np.zeros((len(work), H), dtype=np.float32)
    print(f"model {model_id} | hidden {H} | device {device}")

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
            if sl.shape[0] < 640:                      # too short for the conv stack
                sl = np.pad(sl, (0, 640 - sl.shape[0]))
            idxs.append(ridx); slices.append(sl)
        # batch this file's slices through the encoder
        for s in range(0, len(slices), batch_size):
            chunk = slices[s:s + batch_size]
            ci = idxs[s:s + batch_size]
            inp = fe(chunk, sampling_rate=SR, return_tensors="pt", padding=True)
            iv = inp.input_values.to(device)
            am = inp.get("attention_mask")
            am = am.to(device) if am is not None else None
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                out = model(iv, attention_mask=am).last_hidden_state.float()
            if am is not None:
                olen = model._get_feat_extract_output_lengths(am.sum(-1)).long()
                mask = (torch.arange(out.size(1), device=device)[None, :]
                        < olen[:, None]).unsqueeze(-1).float()
                feat = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            else:
                feat = out.mean(1)
            X[np.asarray(ci)] = feat.cpu().numpy()
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
    ap.add_argument("--model", default="facebook/wav2vec2-xls-r-300m")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-seconds", type=float, default=15.0)
    ap.add_argument("--limit-files", type=int, default=None)
    args = ap.parse_args()
    tag = args.model.split("/")[-1].replace(".", "_")
    cache = config.MANIFEST_DIR / f"ssl_{tag}.npy"

    sp = pd.read_csv(SPLITS)
    work = sp[["file_id", "start_time", "end_time", "master_audio_path",
               "final_dialect", "split", "speaker_final"]].reset_index(drop=True)

    if cache.exists() and not args.limit_files:
        print(f"loading cached features {cache.name}")
        X = np.load(cache)
    else:
        X = extract_features(work, args.model, args.batch_size, args.max_seconds,
                             args.limit_files)
        if args.limit_files:
            print("smoke test done (features not cached); rerun without --limit-files")
            return
        np.save(cache, X)
        print(f"cached -> {cache}")

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
    name = f"{tag}+probe (E3)"
    for s, m in [("dev", dv), ("test", te)]:
        acc, mf1 = _scores(y[m], pred[s])
        rows.append({"model": name, "level": "segment", "split": s,
                     "accuracy": round(acc, 4), "macro_f1": round(mf1, 4)})
    for s, m in [("dev", dv), ("test", te)]:
        df = pd.DataFrame({"spk": spk[m], "true": y[m], "pred": pred[s]})
        gg = df.groupby("spk").agg(true=("true", "first"),
                                   pred=("pred", lambda x: x.value_counts().idxmax()))
        acc, mf1 = _scores(gg["true"].to_numpy(), gg["pred"].to_numpy())
        rows.append({"model": name, "level": "speaker", "split": s,
                     "accuracy": round(acc, 4), "macro_f1": round(mf1, 4)})
    new = pd.DataFrame(rows)

    rp = TABLES / "did_results.csv"
    if rp.exists():
        old = pd.read_csv(rp)
        old = old[old["model"] != name]                # idempotent re-run
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(rp, index=False)

    cm = confusion_matrix(y[te], pred["test"], labels=REGIONS)
    pd.DataFrame(cm, index=[f"true_{r}" for r in REGIONS],
                 columns=[f"pred_{r}" for r in REGIONS]).to_csv(
        TABLES / f"did_confusion_test_{tag}.csv")

    print("\n==== E3 results ====")
    print(new[new["model"] == name].to_string(index=False))
    print("\n==== TEST per-class (segment) ====")
    print(classification_report(y[te], pred["test"], labels=REGIONS, digits=3))


if __name__ == "__main__":
    main()

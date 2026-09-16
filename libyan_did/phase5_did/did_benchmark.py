"""E0+E2: 3-way sub-regional dialect-ID benchmark on cached ECAPA embeddings.

Classifies each verified segment as Tripolitania / Cyrenaica / Fezzan using the
already-cached ECAPA speaker embeddings (no inference). Trains on the `train`
split, reports on `dev` and `test` at BOTH segment level and speaker level
(majority vote per speaker_final). Reports accuracy, macro-F1, per-class F1 and a
confusion matrix, against a majority-class floor (E0).

Reuses the embedding-address loading pattern from scripts/build_gender_flags.py:
each per-playlist embeddings.npy is row-aligned with its final_manifest.csv, so a
segment's embedding is addressed by (emb_path, row). Labels + split come from
splits_manifest.csv, joined on (file_id, start_time, end_time).

Run (diarization env, from libyan_did_pipeline/):
    python scripts/did_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from libyan_did.shared import config

SPLITS = config.MANIFEST_DIR / "splits_manifest.csv"
TABLES = config.TABLES_DIR
SEED = 1337
REGIONS = list(config.REGIONS)


def build_address_table() -> pd.DataFrame:
    """(file_id, start_time, end_time) -> (emb_path, emb_row) for every segment.

    emb_row is the positional row in that playlist's embeddings.npy, which is
    row-aligned with its final_manifest.csv (see embed.embed_manifest).
    """
    parts = []
    for fpath in sorted(config.CORPUS_DIR.glob("*/*/final_manifest.csv")):
        emb_path = fpath.parent / "embeddings.npy"
        if not emb_path.exists():
            continue
        df = pd.read_csv(fpath)
        if len(df) == 0:
            continue
        df = df.reset_index(drop=True)
        parts.append(pd.DataFrame({
            "file_id": df["file_id"].astype(str),
            "start_time": df["start_time"].round(3),
            "end_time": df["end_time"].round(3),
            "emb_path": str(emb_path),
            "emb_row": np.arange(len(df), dtype=int),
        }))
    return pd.concat(parts, ignore_index=True)


def load_embeddings(m: pd.DataFrame) -> np.ndarray:
    """Fetch + L2-normalize embeddings for an (emb_path, emb_row)-addressed frame."""
    X = np.zeros((len(m), config.EMBEDDING_DIM), dtype=np.float32)
    for path, g in m.groupby("emb_path"):
        emb = np.load(path)
        X[g.index.to_numpy()] = emb[g["emb_row"].to_numpy()]
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n == 0, 1.0, n)


def _scores(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score
    return (accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, labels=REGIONS, average="macro"))


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix

    # ---- assemble (embedding, label, split, speaker) ----
    sp = pd.read_csv(SPLITS)
    sp["file_id"] = sp["file_id"].astype(str)
    sp["start_time"] = sp["start_time"].round(3)
    sp["end_time"] = sp["end_time"].round(3)
    addr = build_address_table()
    m = sp.merge(addr, on=["file_id", "start_time", "end_time"], how="left")
    miss = int(m["emb_path"].isna().sum())
    print(f"segments: {len(m):,} | embedding join misses: {miss} "
          f"({miss/len(m)*100:.2f}%)")
    m = m.dropna(subset=["emb_path"]).reset_index(drop=True)
    m["emb_row"] = m["emb_row"].astype(int)

    X = load_embeddings(m)
    y = m["final_dialect"].to_numpy()
    split = m["split"].to_numpy()
    spk = m["speaker_final"].to_numpy()
    tr, dv, te = split == "train", split == "dev", split == "test"
    print(f"train {tr.sum():,} | dev {dv.sum():,} | test {te.sum():,} segments")
    for s, mask in [("train", tr), ("dev", dv), ("test", te)]:
        vc = pd.Series(y[mask]).value_counts().reindex(REGIONS).to_dict()
        print(f"  {s} region segs: {vc}")

    rows = []  # results table

    # ---- E0: majority-class floor ----
    maj = pd.Series(y[tr]).value_counts().idxmax()
    for s, mask in [("dev", dv), ("test", te)]:
        acc, mf1 = _scores(y[mask], np.full(mask.sum(), maj))
        rows.append({"model": "majority (E0)", "level": "segment", "split": s,
                     "accuracy": round(acc, 4), "macro_f1": round(mf1, 4)})

    # ---- E2: ECAPA + logistic regression (segment level) ----
    clf = LogisticRegression(max_iter=5000, C=1.0).fit(X[tr], y[tr])
    pred = {s: clf.predict(X[mask]) for s, mask in [("dev", dv), ("test", te)]}
    for s, mask in [("dev", dv), ("test", te)]:
        acc, mf1 = _scores(y[mask], pred[s])
        rows.append({"model": "ECAPA+logreg (E2)", "level": "segment", "split": s,
                     "accuracy": round(acc, 4), "macro_f1": round(mf1, 4)})

    # ---- E2: speaker-level (majority vote over each speaker's segments) ----
    for s, mask in [("dev", dv), ("test", te)]:
        df = pd.DataFrame({"spk": spk[mask], "true": y[mask], "pred": pred[s]})
        g = df.groupby("spk").agg(true=("true", "first"),
                                  pred=("pred", lambda x: x.value_counts().idxmax()))
        acc, mf1 = _scores(g["true"].to_numpy(), g["pred"].to_numpy())
        rows.append({"model": "ECAPA+logreg (E2)", "level": "speaker", "split": s,
                     "accuracy": round(acc, 4), "macro_f1": round(mf1, 4)})

    res = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    res.to_csv(TABLES / "did_results.csv", index=False)

    # ---- TEST confusion matrix + per-class report (segment level) ----
    cm = confusion_matrix(y[te], pred["test"], labels=REGIONS)
    cm_df = pd.DataFrame(cm, index=[f"true_{r}" for r in REGIONS],
                         columns=[f"pred_{r}" for r in REGIONS])
    cm_df.to_csv(TABLES / "did_confusion_test.csv")

    print("\n==== RESULTS (3-way sub-regional DID) ====")
    print(res.to_string(index=False))
    print("\n==== TEST confusion (segment level, rows=true) ====")
    print(cm_df.to_string())
    print("\n==== TEST per-class report (segment level) ====")
    print(classification_report(y[te], pred["test"], labels=REGIONS, digits=3))
    print(f"\ntables -> {TABLES/'did_results.csv'} , {TABLES/'did_confusion_test.csv'}")


if __name__ == "__main__":
    main()

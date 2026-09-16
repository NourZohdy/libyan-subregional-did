"""Benchmark extensions on cached features (no GPU re-extraction).

Adds to the E0/E2/E3 benchmark, all from cache:
  - ECAPA + linear SVM head            (E2 classifier ablation)
  - Fusion: ECAPA (192-d) + XLS-R (1024-d) -> logistic probe   (E4)
  - Per-class precision/recall/F1 (E2, E3, fusion) on the test split
  - Duration-stratified accuracy buckets (E2, fusion)
  - Bootstrap 95% CIs for segment-level macro-F1 (E2, E3, fusion)
  - Provenance-label accuracy from the human decisions (for the record)

Features come from cache:
  ECAPA : per-playlist embeddings.npy via the (file_id,start,end) address table
  XLS-R : outputs/manifests/ssl_wav2vec2-xls-r-300m.npy, row-aligned to splits_manifest
New rows are APPENDED to did_results.csv; the published E0/E2/E3 rows are untouched.
Per-class for E2/E3 is read from the published confusion CSVs so it matches Table 4 exactly.

Run (diarization env, from libyan_did_pipeline/):
    python scripts/did_extra.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from libyan_did.shared import config
from libyan_did.phase5_did.did_benchmark import (build_address_table, load_embeddings, _scores,  # noqa: E402
                           SPLITS, TABLES, REGIONS)

SEED = 1337
SSL_CACHE = config.MANIFEST_DIR / "ssl_wav2vec2-xls-r-300m.npy"


def perclass_from_confusion(path: Path, model: str) -> list[dict]:
    """precision/recall/F1 per region from a saved confusion CSV (rows = true)."""
    cm = pd.read_csv(path, index_col=0).to_numpy().astype(float)
    rows, ps, rs, fs = [], [], [], []
    for i, r in enumerate(REGIONS):
        tp = cm[i, i]
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"model": model, "region": r, "precision": round(prec, 3),
                     "recall": round(rec, 3), "f1": round(f1, 3)})
        ps.append(prec); rs.append(rec); fs.append(f1)
    rows.append({"model": model, "region": "Macro", "precision": round(np.mean(ps), 3),
                 "recall": round(np.mean(rs), 3), "f1": round(np.mean(fs), 3)})
    return rows


def boot_ci(y_true, y_pred, n=1000, seed=SEED):
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    N = len(y_true)
    vals = np.empty(n)
    for k in range(n):
        idx = rng.integers(0, N, N)
        vals[k] = f1_score(y_true[idx], y_pred[idx], labels=REGIONS, average="macro")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.metrics import confusion_matrix

    # ---- assemble aligned features (ECAPA + XLS-R on the same rows) ----
    sp = pd.read_csv(SPLITS)
    sp["_row"] = np.arange(len(sp))                       # row index into XLS-R cache
    sp["file_id"] = sp["file_id"].astype(str)
    sp["start_time"] = sp["start_time"].round(3)
    sp["end_time"] = sp["end_time"].round(3)
    addr = build_address_table().drop_duplicates(["file_id", "start_time", "end_time"])
    m = sp.merge(addr, on=["file_id", "start_time", "end_time"], how="left")
    miss = int(m["emb_path"].isna().sum())
    print(f"segments {len(m):,} | ECAPA join misses {miss} ({miss/len(m)*100:.2f}%)")
    m = m.dropna(subset=["emb_path"]).reset_index(drop=True)
    m["emb_row"] = m["emb_row"].astype(int)

    Xe = load_embeddings(m)                               # ECAPA, L2-normed
    ssl = np.load(SSL_CACHE)
    Xs = ssl[m["_row"].to_numpy()]
    ns = np.linalg.norm(Xs, axis=1, keepdims=True)
    Xs = Xs / np.where(ns == 0, 1.0, ns)                 # XLS-R, L2-normed
    Xf = np.concatenate([Xe, Xs], axis=1)                # fusion

    y = m["final_dialect"].to_numpy()
    split = m["split"].to_numpy()
    spk = m["speaker_final"].to_numpy()
    dur = m["duration"].to_numpy()
    tr, dv, te = split == "train", split == "dev", split == "test"
    print(f"train {tr.sum():,} dev {dv.sum():,} test {te.sum():,}")

    def predict(clf, X):
        return {s: clf.predict(X[mask]) for s, mask in [("dev", dv), ("test", te)]}

    def result_rows(name, preds):
        out = []
        for s, mask in [("dev", dv), ("test", te)]:
            acc, mf1 = _scores(y[mask], preds[s])
            out.append({"model": name, "level": "segment", "split": s,
                        "accuracy": round(acc, 4), "macro_f1": round(mf1, 4)})
        for s, mask in [("dev", dv), ("test", te)]:
            d = pd.DataFrame({"spk": spk[mask], "true": y[mask], "pred": preds[s]})
            g = d.groupby("spk").agg(true=("true", "first"),
                                     pred=("pred", lambda x: x.value_counts().idxmax()))
            acc, mf1 = _scores(g["true"].to_numpy(), g["pred"].to_numpy())
            out.append({"model": name, "level": "speaker", "split": s,
                        "accuracy": round(acc, 4), "macro_f1": round(mf1, 4)})
        return out

    # ---- models ----
    e2 = predict(LogisticRegression(max_iter=5000, C=1.0).fit(Xe[tr], y[tr]), Xe)
    e3 = predict(LogisticRegression(max_iter=5000, C=1.0).fit(Xs[tr], y[tr]), Xs)
    svm = predict(LinearSVC(C=1.0, max_iter=5000).fit(Xe[tr], y[tr]), Xe)
    fus = predict(LogisticRegression(max_iter=5000, C=1.0).fit(Xf[tr], y[tr]), Xf)

    print("\n-- sanity (segment, test) vs published --")
    for nm, pr in [("E2", e2), ("E3", e3)]:
        acc, mf1 = _scores(y[te], pr["test"])
        print(f"  {nm}: acc {acc:.4f} macroF1 {mf1:.4f}")

    # ---- append new model rows (SVM, fusion) to did_results.csv ----
    new = result_rows("ECAPA+SVM", svm) + result_rows("Fusion ECAPA+XLS-R (E4)", fus)
    new_df = pd.DataFrame(new)
    rp = TABLES / "did_results.csv"
    old = pd.read_csv(rp)
    drop = set(new_df["model"].unique()) | {"Fusion ECAPA+XLS-R (E5)"}  # +legacy E5 label
    old = old[~old["model"].isin(drop)]                          # idempotent re-run
    pd.concat([old, new_df], ignore_index=True).to_csv(rp, index=False)
    print(f"\nappended to {rp.name}:")
    print(new_df.to_string(index=False))

    # ---- fusion confusion (test, segment) ----
    cm = confusion_matrix(y[te], fus["test"], labels=REGIONS)
    pd.DataFrame(cm, index=[f"true_{r}" for r in REGIONS],
                 columns=[f"pred_{r}" for r in REGIONS]).to_csv(
        TABLES / "did_confusion_test_fusion.csv")

    # ---- per-class P/R/F1 (E2, E3 from published confusions; fusion fresh) ----
    pc = []
    pc += perclass_from_confusion(TABLES / "did_confusion_test.csv", "ECAPA+probe (E2)")
    pc += perclass_from_confusion(
        TABLES / "did_confusion_test_wav2vec2-xls-r-300m.csv", "XLS-R+probe (E3)")
    pc += perclass_from_confusion(TABLES / "did_confusion_test_fusion.csv",
                                  "Fusion ECAPA+XLS-R (E4)")
    pc_df = pd.DataFrame(pc)
    pc_df.to_csv(TABLES / "did_perclass_test.csv", index=False)
    print("\n-- per-class (test, segment) --")
    print(pc_df.to_string(index=False))

    # ---- duration-stratified accuracy (E2, fusion; test, segment) ----
    edges = [3.0, 5.0, 7.0, 10.01]
    labels = ["3-5s", "5-7s", "7-10s"]
    bucket = pd.cut(dur[te], bins=edges, labels=labels, right=False)
    yt = y[te]
    db = []
    for nm, pr in [("ECAPA+probe (E2)", e2["test"]),
                   ("Fusion ECAPA+XLS-R (E4)", fus["test"])]:
        correct = (pr == yt)
        for lab in labels:
            sel = bucket == lab
            n = int(sel.sum())
            acc = float(correct[sel].mean()) if n else 0.0
            db.append({"model": nm, "bucket": lab, "n": n, "accuracy": round(acc, 4)})
    db_df = pd.DataFrame(db)
    db_df.to_csv(TABLES / "did_duration_buckets.csv", index=False)
    print("\n-- duration buckets (test, segment) --")
    print(db_df.to_string(index=False))

    # ---- bootstrap 95% CIs for segment-level macro-F1 ----
    ci = []
    for nm, pr in [("ECAPA+probe (E2)", e2["test"]),
                   ("XLS-R+probe (E3)", e3["test"]),
                   ("Fusion ECAPA+XLS-R (E4)", fus["test"])]:
        acc, mf1 = _scores(yt, pr)
        lo, hi = boot_ci(yt, pr)
        ci.append({"model": nm, "macro_f1": round(mf1, 4),
                   "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
    ci_df = pd.DataFrame(ci)
    ci_df.to_csv(TABLES / "did_bootstrap_ci.csv", index=False)
    print("\n-- bootstrap 95% CI (test, segment macro-F1) --")
    print(ci_df.to_string(index=False))

    # ---- provenance-label accuracy from the human decisions ----
    v = pd.read_csv(config.GOLD_DIR / "speaker_verifications.csv")
    kept = v[v["decision"].isin(["verified", "wrong_region"])]
    acc_prov = (kept["decision"] == "verified").mean()
    print(f"\nprovenance-label accuracy: {(kept['decision']=='verified').sum()}/"
          f"{len(kept)} kept speakers retained channel region = {acc_prov*100:.1f}%")
    print(f"keep rate: {len(kept)}/{len(v)} = {len(kept)/len(v)*100:.1f}%")


if __name__ == "__main__":
    main()

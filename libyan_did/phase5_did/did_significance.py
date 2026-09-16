"""Significance tests between DID models (reviewer request: prove E6 > E5, etc.).

Reproduces aligned per-segment TEST predictions for the probe models from cached
features (no GPU) and reads the fine-tuned model's saved predictions, then runs:

  - per-model macro-F1 with a percentile bootstrap 95% CI (fills the missing E5 CI)
  - McNemar's exact test on segment correctness (paired accuracy)
  - a paired bootstrap on the macro-F1 difference (segment-level, i.i.d.)
  - a speaker-clustered paired bootstrap on macro-F1 (accounts for within-speaker
    correlation; this is the honest test, since segments repeat speakers)

All probes use the exact published recipe: L2-normalized cached features, sklearn
LogisticRegression(C=1, max_iter=5000), labels = config.REGIONS, split seed 1337.
E2/E3/E4 features and rows mirror scripts/did_extra.py; E5 mirrors
scripts/did_lid_probe.py; E6 is read from did_lid_finetune_preds_test.csv.

Run (diarization env, from libyan_did_pipeline/):
    python scripts/did_significance.py

Writes outputs/tables/did_significance.csv, did_model_ci.csv, and upserts E5 into
did_bootstrap_ci.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from libyan_did.shared import config
from libyan_did.phase5_did.did_benchmark import build_address_table, load_embeddings  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

SPLITS = config.MANIFEST_DIR / "splits_manifest.csv"
TABLES = config.TABLES_DIR
SSL_CACHE = config.MANIFEST_DIR / "ssl_wav2vec2-xls-r-300m.npy"
LID_CACHE = config.MANIFEST_DIR / "lid_ecapa_emb.npy"
E6_PREDS = TABLES / "did_lid_finetune_preds_test.csv"
REGIONS = list(config.REGIONS)
SEED = 1337
KEY = ["file_id", "start_time", "end_time"]


def mf1(yt, yp):
    return f1_score(yt, yp, labels=REGIONS, average="macro")


def boot_ci(yt, yp, n=1000, seed=SEED):
    """Percentile bootstrap 95% CI for macro-F1 (i.i.d. over segments)."""
    rng = np.random.default_rng(seed)
    N = len(yt)
    vals = np.empty(n)
    for k in range(n):
        idx = rng.integers(0, N, N)
        vals[k] = mf1(yt[idx], yp[idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_boot(yt, pa, pb, n=2000, seed=SEED):
    """Paired bootstrap of macro-F1(B) - macro-F1(A), i.i.d. over segments."""
    rng = np.random.default_rng(seed)
    N = len(yt)
    d = np.empty(n)
    for k in range(n):
        idx = rng.integers(0, N, N)
        d[k] = mf1(yt[idx], pb[idx]) - mf1(yt[idx], pa[idx])
    obs = mf1(yt, pb) - mf1(yt, pa)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p_one = float(np.mean(d <= 0))            # P(B not better than A)
    return obs, float(lo), float(hi), p_one


def paired_boot_speaker(yt, pa, pb, spk, n=2000, seed=SEED):
    """Cluster (speaker) bootstrap of macro-F1(B) - macro-F1(A)."""
    rng = np.random.default_rng(seed)
    spk = np.asarray(spk)
    uniq = np.unique(spk)
    idx_by_spk = {s: np.where(spk == s)[0] for s in uniq}
    d = np.empty(n)
    for k in range(n):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_spk[s] for s in chosen])
        d[k] = mf1(yt[idx], pb[idx]) - mf1(yt[idx], pa[idx])
    obs = mf1(yt, pb) - mf1(yt, pa)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p_one = float(np.mean(d <= 0))
    return obs, float(lo), float(hi), p_one


def mcnemar(yt, pa, pb):
    """Exact McNemar on segment correctness. b01 = A wrong & B right."""
    ca, cb = (pa == yt), (pb == yt)
    b01 = int(np.sum(~ca & cb))               # A wrong, B right (favours B)
    b10 = int(np.sum(ca & ~cb))               # A right, B wrong (favours A)
    nd = b01 + b10
    p = binomtest(min(b01, b10), nd, 0.5, alternative="two-sided").pvalue if nd else 1.0
    return b01, b10, float(p)


def fit_predict(Xtr, ytr, Xte):
    clf = LogisticRegression(max_iter=5000, C=1.0).fit(Xtr, ytr)
    return clf.predict(Xte)


def probe_predictions() -> pd.DataFrame:
    """E2/E3/E4 aligned test predictions, keyed by segment, mirroring did_extra.py."""
    sp = pd.read_csv(SPLITS)
    sp["_row"] = np.arange(len(sp))
    sp["file_id"] = sp["file_id"].astype(str)
    sp["start_time"] = sp["start_time"].round(3)
    sp["end_time"] = sp["end_time"].round(3)
    addr = build_address_table().drop_duplicates(KEY)
    m = sp.merge(addr, on=KEY, how="left")
    miss = int(m["emb_path"].isna().sum())
    print(f"[probes] segments {len(m):,} | ECAPA join misses {miss} "
          f"({miss/len(m)*100:.2f}%)")
    m = m.dropna(subset=["emb_path"]).reset_index(drop=True)
    m["emb_row"] = m["emb_row"].astype(int)

    Xe = load_embeddings(m)                                  # ECAPA, L2-normed
    ssl = np.load(SSL_CACHE)
    Xs = ssl[m["_row"].to_numpy()]
    Xs = Xs / np.where((nn := np.linalg.norm(Xs, axis=1, keepdims=True)) == 0, 1.0, nn)
    Xf = np.concatenate([Xe, Xs], axis=1)                    # fusion

    y = m["final_dialect"].to_numpy()
    split = m["split"].to_numpy()
    tr, te = split == "train", split == "test"

    out = m.loc[te, KEY + ["speaker_final", "final_dialect"]].reset_index(drop=True)
    out = out.rename(columns={"final_dialect": "true"})
    out["pred_E2"] = fit_predict(Xe[tr], y[tr], Xe[te])
    out["pred_E3"] = fit_predict(Xs[tr], y[tr], Xs[te])
    out["pred_E4"] = fit_predict(Xf[tr], y[tr], Xf[te])
    for nm, col in [("E2", "pred_E2"), ("E3", "pred_E3"), ("E4", "pred_E4")]:
        print(f"[probes] {nm} test macro-F1 {mf1(out['true'].to_numpy(), out[col].to_numpy()):.4f}")
    return out


def e5_predictions() -> pd.DataFrame:
    """E5 (frozen LID-ECAPA probe) aligned test predictions, mirroring did_lid_probe.py."""
    sp = pd.read_csv(SPLITS)
    cols = ["file_id", "start_time", "end_time", "master_audio_path",
            "final_dialect", "split", "speaker_final"]
    X = np.load(LID_CACHE)                                   # row-aligned to full splits
    assert X.shape[0] == len(sp), f"LID cache {X.shape} not aligned to splits {len(sp)}"
    sub = sp[cols].copy()
    mask = sub[["master_audio_path", "final_dialect", "split"]].notna().all(axis=1).to_numpy()
    work = sub[mask].reset_index(drop=True)                  # mask cache by ORIGINAL position
    X = X[mask]
    X = X / np.where((n := np.linalg.norm(X, axis=1, keepdims=True)) == 0, 1.0, n)
    work["file_id"] = work["file_id"].astype(str)
    work["start_time"] = work["start_time"].round(3)
    work["end_time"] = work["end_time"].round(3)
    y = work["final_dialect"].to_numpy()
    split = work["split"].to_numpy()
    tr, te = split == "train", split == "test"
    out = work[te][KEY + ["speaker_final", "final_dialect"]].reset_index(drop=True)
    out = out.rename(columns={"final_dialect": "true"})
    out["pred_E5"] = fit_predict(X[tr], y[tr], X[te])
    print(f"[E5] test macro-F1 {mf1(out['true'].to_numpy(), out['pred_E5'].to_numpy()):.4f}")
    return out


def e6_predictions() -> pd.DataFrame:
    """E6 saved predictions, keyed by segment via the splits test order (verified)."""
    sp = pd.read_csv(SPLITS)
    sp["file_id"] = sp["file_id"].astype(str)
    sp["start_time"] = sp["start_time"].round(3)
    sp["end_time"] = sp["end_time"].round(3)
    te = sp[sp["split"] == "test"].reset_index(drop=True)
    e6 = pd.read_csv(E6_PREDS)
    assert len(e6) == len(te), f"E6 preds {len(e6)} != test rows {len(te)}"
    # verify positional alignment via the true-label sequence
    same = (e6["true"].to_numpy() == te["final_dialect"].to_numpy())
    print(f"[E6] true-label alignment with splits test order: {same.mean()*100:.2f}% "
          f"({same.sum()}/{len(same)})")
    assert same.all(), "E6 preds are NOT in splits test order; cannot key by position"
    out = te[KEY + ["speaker_final"]].copy()
    out["true"] = e6["true"].to_numpy()
    out["pred_E6"] = e6["pred"].to_numpy()
    print(f"[E6] test macro-F1 {mf1(out['true'].to_numpy(), out['pred_E6'].to_numpy()):.4f}")
    return out


def main() -> None:
    probes = probe_predictions()
    e5 = e5_predictions()
    e6 = e6_predictions()

    # merge all on the segment key (E5/E6 full coverage; probes may miss a few)
    assert not e6[KEY].duplicated().any(), "duplicate segment keys in E6 test set"
    assert not e5[KEY].duplicated().any(), "duplicate segment keys in E5 test set"
    base = e6.merge(e5[KEY + ["pred_E5"]], on=KEY, how="inner")
    assert len(base) == len(e6), f"E5/E6 key merge changed row count {len(e6)} -> {len(base)}"
    base = base.merge(probes[KEY + ["pred_E2", "pred_E3", "pred_E4"]], on=KEY, how="left")
    # sanity: E5/E6 true must agree
    chk = e5.merge(e6[KEY + ["true"]], on=KEY, suffixes=("_e5", "_e6"))
    assert (chk["true_e5"] == chk["true_e6"]).all(), "E5/E6 true labels disagree after merge"
    print(f"\nmerged test segments: {len(base):,} "
          f"(probe-covered: {base['pred_E2'].notna().sum():,})")

    yt_full = base["true"].to_numpy()
    spk_full = base["speaker_final"].to_numpy()

    # ---- per-model macro-F1 + bootstrap CI (i.i.d. segments, n=1000, seed 1337) ----
    model_ci = []
    for nm, col in [("ECAPA+probe (E2)", "pred_E2"),
                    ("XLS-R+probe (E3)", "pred_E3"),
                    ("Fusion ECAPA+XLS-R (E4)", "pred_E4"),
                    ("LID-ECAPA frozen+probe (E5)", "pred_E5"),
                    ("LID-ECAPA-FT (E6)", "pred_E6")]:
        sub = base.dropna(subset=[col])
        yt, yp = sub["true"].to_numpy(), sub[col].to_numpy()
        f1 = mf1(yt, yp)
        lo, hi = boot_ci(yt, yp)
        model_ci.append({"model": nm, "macro_f1": round(f1, 4),
                         "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "n": len(sub)})
    ci_df = pd.DataFrame(model_ci)
    print("\n==== per-model macro-F1 + 95% bootstrap CI (test, segment) ====")
    print(ci_df.to_string(index=False))

    # ---- paired comparisons ----
    comps = [("E2", "pred_E2", "E5", "pred_E5"),
             ("E3", "pred_E3", "E5", "pred_E5"),
             ("E4", "pred_E4", "E5", "pred_E5"),
             ("E5", "pred_E5", "E6", "pred_E6"),
             ("E4", "pred_E4", "E6", "pred_E6"),
             ("E2", "pred_E2", "E6", "pred_E6")]
    rows = []
    for na, ca, nb, cb in comps:
        sub = base.dropna(subset=[ca, cb])
        yt = sub["true"].to_numpy()
        pa, pb = sub[ca].to_numpy(), sub[cb].to_numpy()
        spk = sub["speaker_final"].to_numpy()
        d, lo, hi, p1 = paired_boot(yt, pa, pb)
        sd, slo, shi, sp1 = paired_boot_speaker(yt, pa, pb, spk)
        b01, b10, pm = mcnemar(yt, pa, pb)
        rows.append({
            "comparison": f"{na} vs {nb}", "n": len(sub),
            "f1_a": round(mf1(yt, pa), 4), "f1_b": round(mf1(yt, pb), 4),
            "delta_f1": round(d, 4),
            "seg_boot_lo": round(lo, 4), "seg_boot_hi": round(hi, 4), "seg_p": round(p1, 4),
            "spk_boot_lo": round(slo, 4), "spk_boot_hi": round(shi, 4), "spk_p": round(sp1, 4),
            "mcnemar_b_a_wrong_b_right": b01, "mcnemar_b_a_right_b_wrong": b10,
            "mcnemar_p": f"{pm:.2e}",
        })
    sig = pd.DataFrame(rows)
    print("\n==== paired significance tests (B - A; positive favours B) ====")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(sig.to_string(index=False))

    # ---- write outputs ----
    TABLES.mkdir(parents=True, exist_ok=True)
    sig.to_csv(TABLES / "did_significance.csv", index=False)
    ci_df.to_csv(TABLES / "did_model_ci.csv", index=False)

    # upsert E5 into the published bootstrap CI table (keep existing rows/order)
    bpath = TABLES / "did_bootstrap_ci.csv"
    e5row = ci_df[ci_df["model"].str.contains("(E5)", regex=False)][["model", "macro_f1", "ci_lo", "ci_hi"]]
    e5row = e5row.assign(model="LID-ECAPA frozen+probe (E5)")
    if bpath.exists():
        b = pd.read_csv(bpath)
        b = b[~b["model"].str.contains(r"\(E5\)", regex=True)]
        b = pd.concat([b, e5row], ignore_index=True)
    else:
        b = e5row
    b.to_csv(bpath, index=False)

    print(f"\nwrote: {TABLES/'did_significance.csv'}")
    print(f"wrote: {TABLES/'did_model_ci.csv'}")
    print(f"upserted E5 into: {bpath}")


if __name__ == "__main__":
    main()

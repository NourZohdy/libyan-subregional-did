"""Channel-disjoint robustness check for the sub-regional DID benchmark.

Our released splits are speaker-disjoint: a speaker (after cross-episode merging)
is in exactly one split. They are NOT channel-disjoint -- a YouTube channel
(playlist) can contribute speakers to more than one split, so a model could read
channel/recording cues that happen to correlate with region instead of dialect.

This script rebuilds a channel-disjoint, region-stratified split (whole channels
assigned to one split, 80/10/10 by duration per region) and re-runs the two main
models on the SAME cached features:
  E2  ECAPA (192-d) + logistic regression
  E4  Fusion ECAPA + frozen XLS-R (1216-d) + logistic regression
It reports test macro-F1 and per-class F1 under the speaker-disjoint split (the
published numbers) next to the channel-disjoint split, so the gap is the channel
leakage. Caveat: Fezzan has only 43 channels, so its channel-disjoint test set is
small and noisy; this is a robustness check, not a replacement benchmark.

Output: outputs/tables/did_channel_disjoint.csv

Run (diarization env, from libyan_did_pipeline/):
    python scripts/did_channel_disjoint.py
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
TARGETS = {"train": 0.8, "dev": 0.1, "test": 0.1}


def assign_channels(df: pd.DataFrame) -> dict[str, str]:
    """Greedy region-stratified channel->split map (whole channels, 80/10/10 by duration).

    Each playlist is assigned by its dominant region (the region holding most of its
    duration). Within a region, channels are taken largest-first and dropped into the
    split currently furthest below its duration target -- the same rule build_splits
    uses for speakers, with the grouping key swapped to playlist.
    """
    dur = df.groupby("playlist")["duration"].sum().rename("dur")
    dom = (df.groupby(["playlist", "final_dialect"])["duration"].sum().reset_index()
           .sort_values(["duration", "final_dialect"], ascending=[False, True])
           .groupby("playlist").tail(1).set_index("playlist")["final_dialect"].rename("region"))
    pl = pd.concat([dur, dom], axis=1).reset_index()

    assign: dict[str, str] = {}
    for region, grp in pl.groupby("region"):
        grp = grp.sort_values(["dur", "playlist"], ascending=[False, True])
        total = float(grp["dur"].sum())
        acc = {s: 0.0 for s in TARGETS}
        for _, row in grp.iterrows():
            deficit = {s: TARGETS[s] * total - acc[s] for s in TARGETS}
            s = max(deficit, key=deficit.get)
            assign[row["playlist"]] = s
            acc[s] += float(row["dur"])
    return assign


def per_class(y_true, y_pred) -> list[dict]:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=REGIONS).astype(float)
    out, ps, rs, fs = [], [], [], []
    for i, r in enumerate(REGIONS):
        tp = cm[i, i]
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out.append({"region": r, "precision": round(prec, 3),
                    "recall": round(rec, 3), "f1": round(f1, 3)})
        ps.append(prec); rs.append(rec); fs.append(f1)
    out.append({"region": "Macro", "precision": round(np.mean(ps), 3),
                "recall": round(np.mean(rs), 3), "f1": round(np.mean(fs), 3)})
    return out


def main() -> None:
    from sklearn.linear_model import LogisticRegression

    # ---- assemble aligned features (ECAPA + XLS-R on the same rows) ----
    sp = pd.read_csv(SPLITS)
    sp["_row"] = np.arange(len(sp))                       # row index into XLS-R cache
    sp["file_id"] = sp["file_id"].astype(str)
    sp["start_time"] = sp["start_time"].round(3)
    sp["end_time"] = sp["end_time"].round(3)
    addr = build_address_table().drop_duplicates(["file_id", "start_time", "end_time"])
    m = sp.merge(addr, on=["file_id", "start_time", "end_time"], how="left")
    m = m.dropna(subset=["emb_path"]).reset_index(drop=True)
    m["emb_row"] = m["emb_row"].astype(int)
    print(f"segments with features: {len(m):,}")

    Xe = load_embeddings(m)                               # ECAPA, L2-normed
    ssl = np.load(SSL_CACHE)
    Xs = ssl[m["_row"].to_numpy()]
    Xs = Xs / np.where(np.linalg.norm(Xs, axis=1, keepdims=True) == 0, 1.0,
                       np.linalg.norm(Xs, axis=1, keepdims=True))
    Xf = np.concatenate([Xe, Xs], axis=1)                # fusion
    y = m["final_dialect"].to_numpy()

    # ---- the two splits over the SAME rows ----
    spk_split = m["split"].to_numpy()                    # published speaker-disjoint
    assign = assign_channels(m[["playlist", "final_dialect", "duration"]])
    chan_split = m["playlist"].map(assign).to_numpy()    # new channel-disjoint

    # ---- disjointness checks ----
    pl_by_split = {s: set(m.loc[chan_split == s, "playlist"]) for s in TARGETS}
    assert not (pl_by_split["train"] & pl_by_split["test"]), "channel leak train/test!"
    assert not (pl_by_split["train"] & pl_by_split["dev"]), "channel leak train/dev!"
    assert not (pl_by_split["dev"] & pl_by_split["test"]), "channel leak dev/test!"
    # how much channel overlap the speaker-disjoint split actually has, for the paper
    spk_tr_ch = set(m.loc[spk_split == "train", "playlist"])
    spk_te = m[spk_split == "test"]
    seg_share = m.loc[spk_split == "test", "playlist"].isin(spk_tr_ch).mean()
    spk_share = (spk_te.groupby("speaker_final")["playlist"]
                 .first().isin(spk_tr_ch).mean())
    print(f"\nspeaker-disjoint split: {seg_share*100:.0f}% of test segments and "
          f"{spk_share*100:.0f}% of test speakers come from a channel also in train")

    print("\nsegments per region x split:")
    for name, splt in [("speaker_disjoint", spk_split), ("channel_disjoint", chan_split)]:
        comp = (pd.crosstab(splt, y).reindex(index=["train", "dev", "test"], columns=REGIONS)
                .fillna(0).astype(int))
        print(f"  [{name}]\n{comp.to_string()}")
        nchan = {s: len(pl_by_split[s]) for s in TARGETS} if name == "channel_disjoint" else None
        if nchan:
            print(f"   channels per split: {nchan}")

    # ---- fit + evaluate both models under both splits ----
    rows = []
    summary = []
    for split_name, splt in [("speaker_disjoint", spk_split), ("channel_disjoint", chan_split)]:
        tr, te = splt == "train", splt == "test"
        for model, X in [("ECAPA (E2)", Xe), ("Fusion (E4)", Xf)]:
            clf = LogisticRegression(max_iter=5000, C=1.0).fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            acc, mf1 = _scores(y[te], pred)
            summary.append({"split_type": split_name, "model": model,
                            "n_train": int(tr.sum()), "n_test": int(te.sum()),
                            "accuracy": round(acc, 4), "macro_f1": round(mf1, 4)})
            for pc in per_class(y[te], pred):
                rows.append({"split_type": split_name, "model": model, **pc})

    sm = pd.DataFrame(summary)
    out = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES / "did_channel_disjoint.csv", index=False)

    print("\n==== summary (test) ====")
    print(sm.to_string(index=False))
    print("\n==== macro-F1 drop (speaker-disjoint -> channel-disjoint) ====")
    piv = sm.pivot(index="model", columns="split_type", values="macro_f1")
    piv["drop"] = (piv["speaker_disjoint"] - piv["channel_disjoint"]).round(4)
    print(piv.to_string())
    print("\n==== per-class (channel-disjoint test) ====")
    print(out[out["split_type"] == "channel_disjoint"].to_string(index=False))
    print(f"\nwrote {TABLES/'did_channel_disjoint.csv'}")


if __name__ == "__main__":
    main()

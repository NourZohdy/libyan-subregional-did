"""Honest LDA (discriminant) projection of E6 embeddings over all 645 speakers.

LDA axes are fit on TRAIN SEGMENTS only (~46k points >> 256 dims, so well-posed and
not overfit, unlike the earlier centroid-fit version). All 645 speaker centroids are
then projected onto the 2 discriminant axes. This is a supervised view: it shows how
linearly separable the regions are in E6's space (the model itself ends in a linear
head), and because the axes are train-only, separation of dev/test centroids is real
generalization, not fitting.

Run (diarization env, from libyan_did_pipeline/):
    python scripts/make_e6_lda_all.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from libyan_did.shared import config

FIG_DIR = config.OUTPUTS / "figures" / "paper"
PAPER_FIG = config.CORPUS_PAPER_DIR / "figures"
E6_EMB = config.MANIFEST_DIR / "e6_all_emb.npy"
E6_IDX = config.MANIFEST_DIR / "e6_all_index.csv"
REGIONS = list(config.REGIONS)
COLORS = {"Tripolitania": "#2980b9", "Cyrenaica": "#27ae60", "Fezzan": "#e67e22"}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.metrics import silhouette_score

    X = np.load(E6_EMB)
    X = X / np.where((n := np.linalg.norm(X, axis=1, keepdims=True)) == 0, 1.0, n)
    idx = pd.read_csv(E6_IDX)

    # fit LDA on TRAIN segments (well-posed: ~46k samples, 256 dims)
    tr = (idx["split"] == "train").to_numpy()
    lda = LDA(n_components=2).fit(X[tr], idx.loc[tr, "region"].to_numpy())
    print(f"LDA fit on {int(tr.sum()):,} train segments")

    # project per-speaker centroids (mean embedding -> LDA is linear, so == mean proj)
    cents, regs, splits = [], [], []
    for s, g in idx.groupby("speaker_final").groups.items():
        g = list(g)
        c = X[g].mean(axis=0)
        cents.append(c); regs.append(idx.loc[g[0], "region"])
        splits.append(idx.loc[g[0], "split"])
    C = np.array(cents); R = np.array(regs); S = np.array(splits)
    Z = lda.transform(C)
    print(f"{len(C)} speaker centroids")
    print(f"LDA-projection region silhouette: {silhouette_score(Z, R):+.4f} "
          f"(optimistic: axes chosen to separate)")
    # held-out-only silhouette in the projection (honest generalization read)
    ho = S != "train"
    print(f"  dev+test only ({int(ho.sum())} spk): {silhouette_score(Z[ho], R[ho]):+.4f}")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for r in REGIONS:
        m = R == r
        ax.scatter(Z[m, 0], Z[m, 1], s=24, c=COLORS[r],
                   label=f"{r} (n={int(m.sum())})", edgecolors="black",
                   linewidths=0.3, alpha=0.85)
    ax.set_xlabel("LD1"); ax.set_ylabel("LD2")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Fine-tuned E6 speaker embeddings (LDA projection)", fontsize=12)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"C8j_lda_e6_all.{ext}", dpi=150, bbox_inches="tight")
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIG_DIR / "C8j_lda_e6_all.pdf", PAPER_FIG / "C8j_lda_e6_all.pdf")
    print(f"saved C8j_lda_e6_all -> {FIG_DIR}")


if __name__ == "__main__":
    main()

"""Region t-SNE of the fine-tuned E6 encoder over ALL 645 speakers.

One point per speaker_final (centroid of its E6 segment embeddings), colored by
region. Uses cached e6_all_emb.npy (extract_e6_all_emb.py). Prints region silhouette.

Run (diarization env, from libyan_did_pipeline/):
    python scripts/make_e6_tsne_all.py
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
SEED = 42


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    X = np.load(E6_EMB)
    X = X / np.where((n := np.linalg.norm(X, axis=1, keepdims=True)) == 0, 1.0, n)
    idx = pd.read_csv(E6_IDX)

    cents, regs, spks = [], [], []
    for s, g in idx.groupby("speaker_final").groups.items():
        g = list(g)
        c = X[g].mean(axis=0); c /= (np.linalg.norm(c) or 1.0)
        cents.append(c); regs.append(idx.loc[g[0], "region"]); spks.append(s)
    C = np.array(cents); R = np.array(regs)
    print(f"{len(C)} speaker centroids")
    print(f"region silhouette (cosine): {silhouette_score(C, R, metric='cosine'):+.4f}")

    xy = TSNE(n_components=2, metric="cosine", init="pca", perplexity=30,
              random_state=SEED).fit_transform(C)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for r in REGIONS:
        m = R == r
        ax.scatter(xy[m, 0], xy[m, 1], s=24, c=COLORS[r],
                   label=f"{r} (n={int(m.sum())})", edgecolors="black",
                   linewidths=0.3, alpha=0.85)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"C8h_tsne_e6_all.{ext}", dpi=150, bbox_inches="tight")
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIG_DIR / "C8h_tsne_e6_all.pdf", PAPER_FIG / "C8h_tsne_e6_all.pdf")
    print(f"saved C8h_tsne_e6_all -> {FIG_DIR} and {PAPER_FIG}")


if __name__ == "__main__":
    main()

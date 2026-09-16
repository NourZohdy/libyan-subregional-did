"""Region-colored t-SNE of ECAPA speaker embeddings (per-speaker centroids).

One point per verified speaker (speaker_final), colored by final_dialect, to show
how separable the three regions are in the speaker-embedding space the E2 probe uses.
Saves figures/paper/C8_tsne_region.{pdf,png} and copies the PDF into manuscripts/corpus_manuscripts/corpus_paper/figures/.

Run (diarization env, from libyan_did_pipeline/):
    python scripts/make_tsne_region.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from libyan_did.shared import config
from libyan_did.phase5_did.did_benchmark import build_address_table, load_embeddings, SPLITS, REGIONS  # noqa: E402

SEED = 42
FIG_DIR = config.OUTPUTS / "figures" / "paper"
PAPER_FIG = config.CORPUS_PAPER_DIR / "figures"
COLORS = {"Tripolitania": "#2980b9", "Cyrenaica": "#e74c3c", "Fezzan": "#f39c12"}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    sp = pd.read_csv(SPLITS)
    sp["file_id"] = sp["file_id"].astype(str)
    sp["start_time"] = sp["start_time"].round(3)
    sp["end_time"] = sp["end_time"].round(3)
    addr = build_address_table().drop_duplicates(["file_id", "start_time", "end_time"])
    m = (sp.merge(addr, on=["file_id", "start_time", "end_time"], how="left")
         .dropna(subset=["emb_path"]).reset_index(drop=True))
    m["emb_row"] = m["emb_row"].astype(int)
    X = load_embeddings(m)

    # one renormalized mean embedding per verified speaker, with its region
    cents, regs = [], []
    for _, idx in m.groupby("speaker_final").groups.items():
        idx = list(idx)
        c = X[idx].mean(axis=0)
        c /= (np.linalg.norm(c) or 1.0)
        cents.append(c)
        regs.append(m.loc[idx[0], "final_dialect"])
    C = np.array(cents)
    regs = np.array(regs)
    print(f"{len(C)} speaker centroids")

    xy = TSNE(n_components=2, metric="cosine", init="pca", perplexity=30,
              random_state=SEED).fit_transform(C)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for r in REGIONS:
        msk = regs == r
        ax.scatter(xy[msk, 0], xy[msk, 1], s=28, c=COLORS[r],
                   label=f"{r} (n={int(msk.sum())})", edgecolors="black",
                   linewidths=0.3, alpha=0.85)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.set_title("t-SNE of ECAPA speaker centroids by region")
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"C8_tsne_region.{ext}", dpi=150, bbox_inches="tight")
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIG_DIR / "C8_tsne_region.pdf", PAPER_FIG / "C8_tsne_region.pdf")
    print(f"saved C8_tsne_region -> {FIG_DIR} and {PAPER_FIG}")


if __name__ == "__main__":
    main()

"""Two-panel region-colored t-SNE: speaker-ECAPA (E2) vs LID-ECAPA (E5).

One point per verified speaker (speaker_final), colored by final_dialect, computed
on the SAME speaker set for both panels so the only difference is the encoder. The
left panel is the speaker-recognition ECAPA the E2 probe uses (organizes by voice);
the right panel is the VoxLingua107 language-ID ECAPA the E5 probe uses. Visualizes
the paper's core claim: swapping the encoder surfaces the regional cue.

Saves figures/paper/C8b_tsne_compare.{pdf,png} and copies the PDF to manuscripts/corpus_manuscripts/corpus_paper/figures/.

Run (diarization env, from libyan_did_pipeline/):
    python scripts/make_tsne_compare.py
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
LID_CACHE = config.MANIFEST_DIR / "lid_ecapa_emb.npy"
COLORS = {"Tripolitania": "#2980b9", "Cyrenaica": "#e74c3c", "Fezzan": "#f39c12"}
KEY = ["file_id", "start_time", "end_time"]


def centroids(X: np.ndarray, m: pd.DataFrame):
    """Per-speaker_final renormalized mean embedding + its region, in a fixed order."""
    cents, regs, spk = [], [], []
    for s, idx in m.groupby("speaker_final").groups.items():
        idx = list(idx)
        c = X[idx].mean(axis=0)
        c /= (np.linalg.norm(c) or 1.0)
        cents.append(c)
        regs.append(m.loc[idx[0], "final_dialect"])
        spk.append(s)
    return np.array(cents), np.array(regs), np.array(spk)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    sp = pd.read_csv(SPLITS)
    sp["_row"] = np.arange(len(sp))                        # index into the LID cache
    sp["file_id"] = sp["file_id"].astype(str)
    sp["start_time"] = sp["start_time"].round(3)
    sp["end_time"] = sp["end_time"].round(3)
    addr = build_address_table().drop_duplicates(KEY)
    m = (sp.merge(addr, on=KEY, how="left")
         .dropna(subset=["emb_path"]).reset_index(drop=True))
    m["emb_row"] = m["emb_row"].astype(int)

    # ECAPA (192-d, L2-normed by load_embeddings)
    Xe = load_embeddings(m)
    # LID-ECAPA (256-d), addressed by original splits row, then L2-normed
    lid = np.load(LID_CACHE)
    assert lid.shape[0] == len(sp), f"LID cache {lid.shape} not aligned to splits {len(sp)}"
    Xl = lid[m["_row"].to_numpy()]
    Xl = Xl / np.where((n := np.linalg.norm(Xl, axis=1, keepdims=True)) == 0, 1.0, n)

    Ce, regs_e, spk_e = centroids(Xe, m)
    Cl, regs_l, spk_l = centroids(Xl, m)
    assert (spk_e == spk_l).all(), "speaker order differs between panels"
    regs = regs_e
    print(f"{len(Ce)} speaker centroids (same set in both panels)")

    def embed(C):
        return TSNE(n_components=2, metric="cosine", init="pca", perplexity=30,
                    random_state=SEED).fit_transform(C)

    xy_e, xy_l = embed(Ce), embed(Cl)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, xy, title in [(axes[0], xy_e, "speaker-recognition ECAPA (E2)"),
                          (axes[1], xy_l, "language-identification ECAPA (E5)")]:
        for r in REGIONS:
            msk = regs == r
            ax.scatter(xy[msk, 0], xy[msk, 1], s=28, c=COLORS[r],
                       label=f"{r} (n={int(msk.sum())})", edgecolors="black",
                       linewidths=0.3, alpha=0.85)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=12)
    axes[0].legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"C8b_tsne_compare.{ext}", dpi=150, bbox_inches="tight")
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIG_DIR / "C8b_tsne_compare.pdf", PAPER_FIG / "C8b_tsne_compare.pdf")
    print(f"saved C8b_tsne_compare -> {FIG_DIR} and {PAPER_FIG}")


if __name__ == "__main__":
    main()

"""Two-panel LDA projection: speaker-ECAPA (E2) vs LID-ECAPA (E5).

Unlike t-SNE (unsupervised geometry), this projects per-speaker centroids onto the
2 linear discriminant axes that best separate the three regions -- the geometric
counterpart of the linear probe. LDA axes are fit on TRAIN speakers only (no test
leakage); all 645 speakers are then plotted. If the LID encoder makes regions more
linearly separable (0.660 vs 0.525 macro-F1), this is where it shows.

Saves figures/paper/C8c_lda_compare.{pdf,png} and copies the PDF to manuscripts/corpus_manuscripts/corpus_paper/figures/.

Run (diarization env, from libyan_did_pipeline/):
    python scripts/make_lda_compare.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from libyan_did.shared import config
from libyan_did.phase5_did.did_benchmark import build_address_table, load_embeddings, SPLITS, REGIONS  # noqa: E402

FIG_DIR = config.OUTPUTS / "figures" / "paper"
PAPER_FIG = config.CORPUS_PAPER_DIR / "figures"
LID_CACHE = config.MANIFEST_DIR / "lid_ecapa_emb.npy"
COLORS = {"Tripolitania": "#2980b9", "Cyrenaica": "#e74c3c", "Fezzan": "#f39c12"}
KEY = ["file_id", "start_time", "end_time"]


def centroids(X, m):
    cents, regs, spk, splits = [], [], [], []
    for s, idx in m.groupby("speaker_final").groups.items():
        idx = list(idx)
        c = X[idx].mean(axis=0)
        c /= (np.linalg.norm(c) or 1.0)
        cents.append(c); regs.append(m.loc[idx[0], "final_dialect"])
        spk.append(s); splits.append(m.loc[idx[0], "split"])
    return np.array(cents), np.array(regs), np.array(spk), np.array(splits)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    sp = pd.read_csv(SPLITS)
    sp["_row"] = np.arange(len(sp))
    sp["file_id"] = sp["file_id"].astype(str)
    sp["start_time"] = sp["start_time"].round(3)
    sp["end_time"] = sp["end_time"].round(3)
    addr = build_address_table().drop_duplicates(KEY)
    m = (sp.merge(addr, on=KEY, how="left")
         .dropna(subset=["emb_path"]).reset_index(drop=True))
    m["emb_row"] = m["emb_row"].astype(int)

    Xe = load_embeddings(m)
    lid = np.load(LID_CACHE)
    Xl = lid[m["_row"].to_numpy()]
    Xl = Xl / np.where((n := np.linalg.norm(Xl, axis=1, keepdims=True)) == 0, 1.0, n)

    Ce, regs, spk_e, splits = centroids(Xe, m)
    Cl, _, spk_l, _ = centroids(Xl, m)
    assert (spk_e == spk_l).all()
    tr = splits == "train"
    print(f"{len(Ce)} speakers | train {tr.sum()} used to fit LDA axes")

    def project(C):
        z = LDA(n_components=2).fit(C[tr], regs[tr]).transform(C)
        return z

    Ze, Zl = project(Ce), project(Cl)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, Z, title in [(axes[0], Ze, "speaker-recognition ECAPA (E2)"),
                         (axes[1], Zl, "language-identification ECAPA (E5)")]:
        for r in REGIONS:
            msk = regs == r
            ax.scatter(Z[msk, 0], Z[msk, 1], s=26, c=COLORS[r],
                       label=f"{r} (n={int(msk.sum())})", edgecolors="black",
                       linewidths=0.3, alpha=0.85)
        ax.set_xlabel("LD1"); ax.set_ylabel("LD2")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=12)
    axes[0].legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"C8c_lda_compare.{ext}", dpi=150, bbox_inches="tight")
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIG_DIR / "C8c_lda_compare.pdf", PAPER_FIG / "C8c_lda_compare.pdf")
    print(f"saved C8c_lda_compare -> {FIG_DIR} and {PAPER_FIG}")


if __name__ == "__main__":
    main()

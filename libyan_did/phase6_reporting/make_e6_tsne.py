"""Three-panel region t-SNE on the SAME held-out test speakers:
speaker-ECAPA (E2) -> frozen LID-ECAPA (E5) -> fine-tuned LID-ECAPA (E6).

E2/E5 come from caches; E6 from extract_e6_test_emb.py. One point per test
speaker_final (centroid of its segment embeddings), colored by region. Shows what
each encoder organizes by: voice (E2), language (E5), region after task fine-tuning
(E6). Silhouette(region) is printed per panel as a quantitative companion.

Run (diarization env, from libyan_did_pipeline/):
    python scripts/make_e6_tsne.py
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
E6_EMB = config.MANIFEST_DIR / "e6_test_emb.npy"
E6_IDX = config.MANIFEST_DIR / "e6_test_index.csv"
COLORS = {"Tripolitania": "#2980b9", "Cyrenaica": "#e74c3c", "Fezzan": "#f39c12"}
KEY = ["file_id", "start_time", "end_time"]
SEED = 42


def centroids_from(X, spk_arr, reg_arr):
    """Per-speaker renormalized mean embedding, returned in sorted-speaker order."""
    df = pd.DataFrame({"spk": spk_arr, "reg": reg_arr})
    cents, regs, spks = [], [], []
    for s, idx in df.groupby("spk").groups.items():
        idx = list(idx)
        c = X[idx].mean(axis=0)
        c /= (np.linalg.norm(c) or 1.0)
        cents.append(c); regs.append(df.loc[idx[0], "reg"]); spks.append(s)
    order = np.argsort(spks)
    return (np.array(cents)[order], np.array(regs)[order], np.array(spks)[order])


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    sp = pd.read_csv(SPLITS)
    sp["_row"] = np.arange(len(sp))
    sp["file_id"] = sp["file_id"].astype(str)
    sp["start_time"] = sp["start_time"].round(3)
    sp["end_time"] = sp["end_time"].round(3)

    # ---- E2: speaker-ECAPA (test segments via address table) ----
    addr = build_address_table().drop_duplicates(KEY)
    m = (sp.merge(addr, on=KEY, how="left")
         .dropna(subset=["emb_path"]).reset_index(drop=True))
    m["emb_row"] = m["emb_row"].astype(int)
    mte = m[m["split"] == "test"].reset_index(drop=True)
    Xe = load_embeddings(mte)
    Ce, Re, Se = centroids_from(Xe, mte["speaker_final"].to_numpy(),
                                mte["final_dialect"].to_numpy())

    # ---- E5: frozen LID (test rows from cache) ----
    lid = np.load(LID_CACHE)
    spte = sp[sp["split"] == "test"].reset_index(drop=True)
    Xl = lid[spte["_row"].to_numpy()]
    Xl = Xl / np.where((n := np.linalg.norm(Xl, axis=1, keepdims=True)) == 0, 1.0, n)
    Cl, Rl, Sl = centroids_from(Xl, spte["speaker_final"].to_numpy(),
                                spte["final_dialect"].to_numpy())

    # ---- E6: fine-tuned LID (extracted test embeddings) ----
    X6 = np.load(E6_EMB)
    X6 = X6 / np.where((n := np.linalg.norm(X6, axis=1, keepdims=True)) == 0, 1.0, n)
    idx6 = pd.read_csv(E6_IDX)
    C6, R6, S6 = centroids_from(X6, idx6["speaker_final"].to_numpy(),
                                idx6["region"].to_numpy())

    print(f"test speakers per panel: E2={len(Ce)} E5={len(Cl)} E6={len(C6)}")
    assert (Se == Sl).all() and (Sl == S6).all(), "test speaker sets differ across panels"

    panels = [("speaker-recognition ECAPA (E2)", Ce, Re),
              ("frozen language-ID ECAPA (E5)", Cl, Rl),
              ("fine-tuned language-ID ECAPA (E6)", C6, R6)]

    print("\nregion silhouette (cosine, test-speaker centroids):")
    for name, C, R in panels:
        print(f"  {name:38s} {silhouette_score(C, R, metric='cosine'):+.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, C, R) in zip(axes, panels):
        xy = TSNE(n_components=2, metric="cosine", init="pca", perplexity=30,
                  random_state=SEED).fit_transform(C)
        for r in REGIONS:
            msk = R == r
            ax.scatter(xy[msk, 0], xy[msk, 1], s=30, c=COLORS[r],
                       label=f"{r} (n={int(msk.sum())})", edgecolors="black",
                       linewidths=0.3, alpha=0.85)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=12)
    axes[0].legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"C8d_tsne_e2_e5_e6.{ext}", dpi=150, bbox_inches="tight")
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIG_DIR / "C8d_tsne_e2_e5_e6.pdf", PAPER_FIG / "C8d_tsne_e2_e5_e6.pdf")
    print(f"\nsaved C8d_tsne_e2_e5_e6 -> {FIG_DIR} and {PAPER_FIG}")


if __name__ == "__main__":
    main()

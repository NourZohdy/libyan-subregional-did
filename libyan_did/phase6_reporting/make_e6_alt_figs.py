"""Alternative 'what E6 learns' figures (cleaner than a speaker scatter).

Reuses cached e6_test_emb.npy (raw, pre-head) + the checkpoint's linear head, so the
logits reproduced here are exactly the model's. No GPU.

A) Ternary/simplex plot of per-speaker mean softmax, colored by TRUE region: confident
   regions sit near their corner; the diffuse region drifts to the middle.
B) Region-prototype cosine-similarity 3x3 heatmap (mean E6 embedding per region):
   shows which regions sit closest (supports 'Fezzan leans toward Tripolitania').

Run (diarization env, from libyan_did_pipeline/):
    python scripts/make_e6_alt_figs.py
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
E6_EMB = config.MANIFEST_DIR / "e6_test_emb.npy"
E6_IDX = config.MANIFEST_DIR / "e6_test_index.csv"
CKPT = config.MANIFEST_DIR / "did_finetune_lid_speaker_best.pt"
E6_PREDS = config.TABLES_DIR / "did_lid_finetune_preds_test.csv"
REGIONS = list(config.REGIONS)
COLORS = {"Tripolitania": "#2980b9", "Cyrenaica": "#e74c3c", "Fezzan": "#f39c12"}
SHORT = {"Tripolitania": "Trip.", "Cyrenaica": "Cyr.", "Fezzan": "Fez."}


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main() -> None:
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X = np.load(E6_EMB)                                  # raw, pre-head
    idx = pd.read_csv(E6_IDX)
    sd = torch.load(CKPT, map_location="cpu")
    W = sd["head.weight"].numpy()                       # (3, 256)
    b = sd["head.bias"].numpy()                         # (3,)
    logits = X @ W.T + b                                 # exact model logits
    probs = softmax(logits)

    # sanity: argmax must match the saved E6 test predictions
    pred_idx = logits.argmax(1)
    pred_lab = np.array([REGIONS[i] for i in pred_idx])
    saved = pd.read_csv(E6_PREDS)
    agree = (pred_lab == saved["pred"].to_numpy()).mean()
    print(f"reconstructed-logit argmax vs saved E6 preds: {agree*100:.2f}% agreement")

    df = pd.DataFrame({"spk": idx["speaker_final"], "true": idx["region"]})
    for j, r in enumerate(REGIONS):
        df[r] = probs[:, j]
    g = df.groupby("spk").agg({**{r: "mean" for r in REGIONS}, "true": "first"})
    P = g[REGIONS].to_numpy()
    P = P / P.sum(1, keepdims=True)
    true = g["true"].to_numpy()
    print(f"{len(g)} test speakers")

    # ---- A) ternary plot ----
    # corners: Tripolitania=(0,0), Cyrenaica=(1,0), Fezzan=(0.5, h)
    h = np.sqrt(3) / 2
    corners = np.array([[0, 0], [1, 0], [0.5, h]])
    xy = P @ corners
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    tri = np.vstack([corners, corners[0]])
    ax.plot(tri[:, 0], tri[:, 1], color="0.6", lw=1.0, zorder=1)
    for r in REGIONS:
        m = true == r
        ax.scatter(xy[m, 0], xy[m, 1], s=30, c=COLORS[r], edgecolors="black",
                   linewidths=0.3, alpha=0.85, label=f"{r} (n={int(m.sum())})", zorder=3)
    off = [(-0.04, -0.05), (0.04, -0.05), (0.0, 0.04)]
    for (cx, cy), r, (dx, dy) in zip(corners, REGIONS, off):
        ax.text(cx + dx, cy + dy, SHORT[r], ha="center", va="center",
                fontsize=12, fontweight="bold", color=COLORS[r])
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title("E6 softmax over test speakers (true region = color)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"C8f_ternary.{ext}", dpi=150, bbox_inches="tight")
    shutil.copy(FIG_DIR / "C8f_ternary.pdf", PAPER_FIG / "C8f_ternary.pdf")

    # ---- B) region-prototype cosine similarity ----
    Xn = X / np.where((n := np.linalg.norm(X, axis=1, keepdims=True)) == 0, 1.0, n)
    protos = np.array([Xn[idx["region"].to_numpy() == r].mean(0) for r in REGIONS])
    protos = protos / np.linalg.norm(protos, axis=1, keepdims=True)
    S = protos @ protos.T
    fig2, ax2 = plt.subplots(figsize=(4.2, 3.8))
    im = ax2.imshow(S, cmap="viridis", vmin=S[~np.eye(3, dtype=bool)].min() - 0.02,
                    vmax=1.0)
    ax2.set_xticks(range(3)); ax2.set_yticks(range(3))
    ax2.set_xticklabels([SHORT[r] for r in REGIONS])
    ax2.set_yticklabels([SHORT[r] for r in REGIONS])
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center",
                     color="white" if S[i, j] < 0.7 else "black", fontsize=11)
    ax2.set_title("E6 region-prototype cosine similarity")
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(FIG_DIR / f"C8g_protosim.{ext}", dpi=150, bbox_inches="tight")
    shutil.copy(FIG_DIR / "C8g_protosim.pdf", PAPER_FIG / "C8g_protosim.pdf")

    # off-diagonal readout for the text
    pairs = [(0, 1), (0, 2), (1, 2)]
    print("region-prototype cosine similarity (off-diagonal):")
    for i, j in pairs:
        print(f"  {SHORT[REGIONS[i]]}-{SHORT[REGIONS[j]]}: {S[i, j]:.3f}")
    print("saved C8f_ternary, C8g_protosim ->", FIG_DIR)


if __name__ == "__main__":
    main()

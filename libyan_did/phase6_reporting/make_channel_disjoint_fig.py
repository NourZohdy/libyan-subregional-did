"""Channel-disjoint robustness figure (C9) for the paper.

Reads outputs/tables/did_channel_disjoint.csv (written by did_channel_disjoint.py)
and draws per-region F1 for the Fusion (E4) model under the speaker-disjoint split
(the published setting) next to the channel-disjoint split, so the per-region drop
-- and Fezzan's collapse -- is visible at a glance. ECAPA (E2) follows the same
pattern; the full table is in the appendix.

Writes manuscripts/corpus_manuscripts/corpus_paper/figures/C9_channel_disjoint.{pdf,png}.

Run (diarization env, from libyan_did_pipeline/):
    python scripts/make_channel_disjoint_fig.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
from libyan_did.shared import config

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(context="paper", style="whitegrid", font_scale=1.15)

SRC = config.TABLES_DIR / "did_channel_disjoint.csv"
PAPER_FIG = config.CORPUS_PAPER_DIR / "figures"
ORDER = ["Tripolitania", "Cyrenaica", "Fezzan", "Macro"]
MODEL = "LID-ECAPA-FT (E6)"


def main() -> None:
    df = pd.read_csv(SRC)
    d = df[df["model"] == MODEL].copy()
    d["split_type"] = d["split_type"].map(
        {"speaker_disjoint": "Speaker-disjoint", "channel_disjoint": "Channel-disjoint"})

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    sns.barplot(data=d, x="region", y="f1", hue="split_type", order=ORDER,
                hue_order=["Speaker-disjoint", "Channel-disjoint"],
                palette={"Speaker-disjoint": "#3498db", "Channel-disjoint": "#e74c3c"},
                edgecolor="black", ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=8, fontweight="bold", padding=1)
    ax.set_ylim(0, 0.95)
    ax.set_xlabel("")
    ax.set_ylabel("Test F1")
    ax.set_title("LID-ECAPA-FT (E6): channel-disjoint vs speaker-disjoint")
    ax.legend(title="", loc="upper right", frameon=True)
    fig.tight_layout()

    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG / "C9_channel_disjoint.pdf", bbox_inches="tight")
    fig.savefig(PAPER_FIG / "C9_channel_disjoint.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PAPER_FIG/'C9_channel_disjoint.pdf'}")


if __name__ == "__main__":
    main()
